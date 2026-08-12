"""End-to-end integration against a stub OpenAI-compatible server.

The unit suite injects a fake backend callable, which proves the logic but not
the wiring. This module drives the *real* ``openai`` client — the same adapter
``ui/app.py`` builds — against a local HTTP server that speaks the subset of the
OpenAI API vLLM implements, including server-sent-event streaming.

Skipped automatically when the optional ``openai`` dependency is absent, so the
dependency-free core suite still runs anywhere.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

openai = pytest.importorskip("openai", reason="optional UI dependency")

from llmbox import ContextBudget, PromptCache, ResilientChatClient, build_client  # noqa: E402

MODEL = "llama-3-8b-instruct"


class _Handler(BaseHTTPRequestHandler):
    """Minimal stand-in for vLLM's OpenAI-compatible server."""

    received: list[dict] = []

    def log_message(self, *args):  # silence the test output
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append(request)

        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # Exactly the chunk shapes vLLM emits, including a role-only opener
            # and a content-free finish chunk.
            frames = [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
                {"choices": [{"index": 0, "delta": {"content": " from"}}]},
                {"choices": [{"index": 0, "delta": {"content": " vLLM"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for frame in frames:
                frame.update({"id": "1", "object": "chat.completion.chunk", "created": 0, "model": MODEL})
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._json(
            {
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from vLLM"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        )


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


@pytest.fixture
def backend(server):
    """The exact adapter ui/app.py builds around the OpenAI client."""
    _Handler.received.clear()
    client = openai.OpenAI(base_url=server, api_key="not-needed", timeout=30.0)

    def _backend(payload: dict, stream: bool):
        return client.chat.completions.create(**payload)

    return _backend


def test_model_discovery_matches_the_ui_path(server):
    client = openai.OpenAI(base_url=server, api_key="not-needed", timeout=30.0)
    assert client.models.list().data[0].id == MODEL


def test_non_streaming_round_trip(backend):
    client = build_client(backend, MODEL, max_model_len=2048)
    result = client.chat([{"role": "user", "content": "hi"}], system="Be brief.")
    assert result.text == "Hello from vLLM"
    # Server-reported usage is trusted over local estimates.
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 4

    sent = _Handler.received[0]
    assert sent["model"] == MODEL
    assert sent["messages"][0] == {"role": "system", "content": "Be brief."}


def test_streaming_round_trip_through_the_real_sdk(backend):
    """Proves the SDK's chunk objects (not dicts) normalise correctly."""
    client = ResilientChatClient(
        backend=backend, model=MODEL, budget=ContextBudget(2048, 256), cache=PromptCache()
    )
    handle = client.stream([{"role": "user", "content": "hi"}])
    assert "".join(handle) == "Hello from vLLM"
    assert handle.result.text == "Hello from vLLM"
    assert handle.result.completion_tokens > 0
    assert _Handler.received[0]["stream"] is True


def test_cache_prevents_a_second_round_trip(backend):
    client = build_client(backend, MODEL, max_model_len=2048)
    messages = [{"role": "user", "content": "what is k3s?"}]
    first = client.chat(messages, temperature=0.0)
    second = client.chat(messages, temperature=0.0)
    assert first.text == second.text
    assert second.cached
    assert len(_Handler.received) == 1  # only one request ever hit the server
