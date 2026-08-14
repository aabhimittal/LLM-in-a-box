"""The scrape endpoint for llmbox's own metrics."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from llmbox.metrics_server import MetricsServer
from llmbox.observability import Metrics


@pytest.fixture
def metrics() -> Metrics:
    registry = Metrics()
    registry.inc("llmbox_requests_total", labels={"outcome": "ok"}, help_text="Requests.")
    registry.observe("llmbox_request_duration_seconds", 0.25)
    return registry


@pytest.fixture
def server(metrics):
    with MetricsServer(metrics, port=0, host="127.0.0.1") as running:
        yield running


def get(server, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}", timeout=5) as response:
        return response.status, response.headers.get("Content-Type"), response.read().decode()


def test_serves_the_registry_in_prometheus_format(server):
    status, content_type, body = get(server, "/metrics")
    assert status == 200
    assert "version=0.0.4" in content_type
    assert 'llmbox_requests_total{outcome="ok"} 1' in body
    assert "llmbox_request_duration_seconds_bucket" in body


def test_reflects_later_updates(server, metrics):
    metrics.inc("llmbox_requests_total", labels={"outcome": "ok"})
    _, _, body = get(server, "/metrics")
    assert 'llmbox_requests_total{outcome="ok"} 2' in body


def test_query_string_is_ignored(server):
    assert get(server, "/metrics?foo=bar")[0] == 200


def test_healthz_defaults_to_healthy(server):
    status, _, body = get(server, "/healthz")
    assert status == 200
    assert body.strip() == "ok"


def test_healthz_reports_unready_backend(metrics):
    """Readiness should follow the model server, not just the web process."""
    ready = {"value": False}
    with MetricsServer(metrics, port=0, host="127.0.0.1", health=lambda: ready["value"]) as server:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            get(server, "/healthz")
        assert excinfo.value.code == 503

        ready["value"] = True
        assert get(server, "/healthz")[0] == 200


def test_unknown_paths_are_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, "/admin")
    assert excinfo.value.code == 404


def test_start_is_idempotent_and_stop_is_safe(metrics):
    server = MetricsServer(metrics, port=0, host="127.0.0.1")
    port = server.start()
    assert server.start() == port  # no second bind
    assert server.running
    server.stop()
    assert not server.running
    server.stop()  # stopping twice must not raise


def test_port_is_released_after_stop(metrics):
    """A leaked socket would break pod restarts with 'address already in use'."""
    server = MetricsServer(metrics, port=0, host="127.0.0.1")
    port = server.start()
    server.stop()
    rebound = MetricsServer(metrics, port=port, host="127.0.0.1")
    try:
        assert rebound.start() == port
    finally:
        rebound.stop()
