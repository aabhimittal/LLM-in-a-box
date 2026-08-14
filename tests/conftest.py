"""Shared test fixtures.

Everything time-dependent in ``llmbox`` takes an injectable clock so the test
suite is deterministic — no ``sleep``, no flakes, no wall-clock dependence.
"""

from __future__ import annotations

import pytest


class FakeClock:
    """Monotonic clock under test control."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        """Drop-in for ``time.sleep`` that advances this clock instead."""
        self.advance(seconds)


class RecordingBackend:
    """Fake vLLM backend that records calls and replays scripted responses.

    ``script`` is a list of either response payloads or exceptions; the last
    entry repeats once exhausted.
    """

    def __init__(self, script=None, text: str = "hello world"):
        self.calls: list[dict] = []
        self.script = list(script) if script is not None else [chat_response(text)]

    def __call__(self, payload: dict, stream: bool):
        self.calls.append(payload)
        index = min(len(self.calls) - 1, len(self.script) - 1)
        item = self.script[index]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(payload, stream)
        return item

    @property
    def call_count(self) -> int:
        return len(self.calls)


def chat_response(text: str, prompt_tokens: int | None = None, completion_tokens: int | None = None):
    """Build an OpenAI-shaped non-streaming response."""
    response: dict = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if prompt_tokens is not None or completion_tokens is not None:
        response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    return response


def stream_chunks(*texts: str):
    """Build a realistic streaming chunk sequence.

    Mirrors what vLLM actually emits: a role-only opening chunk, content
    chunks, a finish-reason chunk with no content, and a trailing usage-only
    chunk with an empty ``choices`` list.
    """
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
    for text in texts:
        chunks.append({"choices": [{"delta": {"content": text}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    chunks.append({"choices": [], "usage": {"completion_tokens": len(texts)}})
    return chunks


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
