"""A tiny HTTP endpoint so llmbox metrics are actually scrapable.

Without this, everything :mod:`llmbox.observability` records — cache hit rate,
guardrail blocks, rate-limit rejections, shed requests, end-to-end latency —
lives only inside the process. vLLM's own ``/metrics`` shows GPU-side load but
cannot show requests that were *rejected before reaching it*, which is precisely
what you need when users report failures the model server never saw.

Implemented on ``http.server`` to keep the package dependency-free. It serves a
handful of tiny responses on a scrape interval; a production WSGI stack would be
more machinery than the job needs.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .observability import Metrics

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _make_handler(metrics: Metrics, health: Callable[[], bool] | None):
    class Handler(BaseHTTPRequestHandler):
        # Quiet: request logging is the scraper's job, and one line per scrape
        # every 15s is pure noise in pod logs.
        def log_message(self, *args):
            pass

        def _respond(self, status: int, body: str, content_type: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def do_GET(self):  # noqa: N802 - http.server API
            if self.path.split("?")[0] == "/metrics":
                self._respond(200, metrics.render(), PROMETHEUS_CONTENT_TYPE)
            elif self.path.split("?")[0] in ("/healthz", "/health"):
                ok = True if health is None else bool(health())
                self._respond(
                    200 if ok else 503,
                    "ok\n" if ok else "unhealthy\n",
                    "text/plain; charset=utf-8",
                )
            else:
                self._respond(404, "not found\n", "text/plain; charset=utf-8")

        do_HEAD = do_GET  # noqa: N815 - http.server API

    return Handler


class MetricsServer:
    """Serves ``/metrics`` and ``/healthz`` on a background thread.

    Args:
        metrics: Registry to expose.
        port: TCP port; ``0`` binds an ephemeral port (used by the tests).
        host: Bind address. Defaults to all interfaces so a Kubernetes Service
            can reach it.
        health: Optional readiness predicate for ``/healthz``.
    """

    def __init__(
        self,
        metrics: Metrics,
        port: int = 9100,
        host: str = "0.0.0.0",
        health: Callable[[], bool] | None = None,
    ):
        self.metrics = metrics
        self.host = host
        self.port = port
        self._health = health
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> int:
        """Start serving. Returns the bound port. Idempotent."""
        if self._server is not None:
            return self.port
        server = ThreadingHTTPServer((self.host, self.port), _make_handler(self.metrics, self._health))
        # Do not let an in-flight scrape keep the pod alive during shutdown.
        server.daemon_threads = True
        self._server = server
        self.port = server.server_address[1]
        self._thread = threading.Thread(
            target=server.serve_forever, name="llmbox-metrics", daemon=True
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        """Stop serving. Safe to call when not running."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> MetricsServer:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
