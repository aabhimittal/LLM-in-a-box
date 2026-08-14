"""Upstream queue-depth signal and adaptive shedding."""

from __future__ import annotations

import pytest

from llmbox.errors import Overloaded
from llmbox.loadsignal import (
    QueueDepthShedder,
    UpstreamLoadSignal,
    parse_prometheus_text,
)

# Trimmed but shape-accurate sample of vLLM's /metrics output.
VLLM_METRICS = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama-3-8b-instruct"} 4.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama-3-8b-instruct"} 12.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="llama-3-8b-instruct"} 0.87
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="x"} 3
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="x"} 9
vllm:time_to_first_token_seconds_sum{model_name="x"} 1.25
"""


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parses_labelled_and_bare_series():
    parsed = parse_prometheus_text(VLLM_METRICS + "bare_metric 7\n")
    assert parsed["vllm:num_requests_waiting"] == 12.0
    assert parsed["vllm:gpu_cache_usage_perc"] == 0.87
    assert parsed["bare_metric"] == 7.0


def test_sums_series_across_label_sets():
    """Two models sharing a GPU both contribute to the queue."""
    text = (
        'vllm:num_requests_waiting{model_name="a"} 3\n'
        'vllm:num_requests_waiting{model_name="b"} 4\n'
    )
    assert parse_prometheus_text(text)["vllm:num_requests_waiting"] == 7.0


def test_histogram_suffixes_are_distinct_metrics():
    parsed = parse_prometheus_text(VLLM_METRICS)
    assert "vllm:time_to_first_token_seconds_bucket" in parsed
    assert "vllm:time_to_first_token_seconds" not in parsed


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# only a comment\n",
        "\n\n   \n",
        "malformed line without value\n",
        "metric not_a_number\n",
        "metric{unclosed 5\n",
    ],
)
def test_malformed_input_never_raises(text):
    """A bad scrape must not become an exception in the request path."""
    assert isinstance(parse_prometheus_text(text), dict)


def test_special_float_values():
    parsed = parse_prometheus_text("a +Inf\nb NaN\nc -3.5\n")
    assert parsed["a"] == float("inf")
    assert "b" not in parsed  # NaN is dropped rather than poisoning comparisons
    assert parsed["c"] == -3.5


def test_trailing_timestamps_are_ignored():
    assert parse_prometheus_text("metric 5 1712345678000\n")["metric"] == 5.0


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #


def test_reads_the_configured_metric(clock):
    signal = UpstreamLoadSignal(lambda: VLLM_METRICS, clock=clock)
    assert signal.value() == 12.0


def test_readings_are_cached_for_the_ttl(clock):
    """Scraping per request would add a round trip to the hot path."""
    calls = []

    def fetch():
        calls.append(1)
        return VLLM_METRICS

    signal = UpstreamLoadSignal(fetch, ttl=5.0, clock=clock)
    for _ in range(10):
        signal.value()
    assert len(calls) == 1

    clock.advance(5.1)
    signal.value()
    assert len(calls) == 2


def test_unreachable_endpoint_reports_unknown_not_zero(clock):
    """Zero would read as 'idle' and is the opposite of what we know."""

    def fetch():
        raise ConnectionError("metrics endpoint down")

    signal = UpstreamLoadSignal(fetch, clock=clock)
    assert signal.value() is None
    assert signal.failures == 1


def test_missing_metric_reports_unknown(clock):
    """vLLM renames series between versions; that must not be read as idle."""
    signal = UpstreamLoadSignal(lambda: "other_metric 1\n", clock=clock)
    assert signal.value() is None


def test_last_known_value_is_reused_briefly_then_discarded(clock):
    """A blip should not flap the decision; a long outage should not act on stale data."""
    state = {"up": True}

    def fetch():
        if not state["up"]:
            raise ConnectionError("down")
        return VLLM_METRICS

    signal = UpstreamLoadSignal(fetch, ttl=1.0, stale_after=30.0, clock=clock)
    assert signal.value() == 12.0

    state["up"] = False
    clock.advance(2.0)
    assert signal.value() == 12.0  # brief blip: keep using the last reading

    clock.advance(40.0)
    assert signal.value() is None  # too old to trust


# --------------------------------------------------------------------------- #
# Shedder
# --------------------------------------------------------------------------- #


def test_admits_while_the_queue_is_shallow(clock):
    signal = UpstreamLoadSignal(lambda: 'vllm:num_requests_waiting{m="x"} 2\n', clock=clock)
    QueueDepthShedder(signal, max_depth=8).check()  # no raise


def test_sheds_when_the_queue_is_deep(clock):
    signal = UpstreamLoadSignal(lambda: VLLM_METRICS, clock=clock)
    shedder = QueueDepthShedder(signal, max_depth=8, retry_after=5.0)
    with pytest.raises(Overloaded) as excinfo:
        shedder.check()
    assert excinfo.value.source == "upstream_queue"
    assert excinfo.value.depth == 12.0
    assert excinfo.value.retry_after == 5.0
    assert shedder.shed == 1


def test_boundary_is_inclusive(clock):
    signal = UpstreamLoadSignal(lambda: "vllm:num_requests_waiting 8\n", clock=clock)
    QueueDepthShedder(signal, max_depth=8).check()  # exactly at the limit is fine


def test_fails_open_when_the_signal_is_unavailable(clock):
    """A monitoring outage must never become a serving outage."""

    def fetch():
        raise TimeoutError("scrape timed out")

    shedder = QueueDepthShedder(UpstreamLoadSignal(fetch, clock=clock), max_depth=0)
    shedder.check()  # must not raise
    assert shedder.shed == 0


def test_fails_open_on_a_garbage_response(clock):
    signal = UpstreamLoadSignal(lambda: "<html>502 Bad Gateway</html>", clock=clock)
    QueueDepthShedder(signal, max_depth=0).check()  # must not raise


def test_rejects_invalid_configuration(clock):
    with pytest.raises(ValueError):
        UpstreamLoadSignal(lambda: "", ttl=0, clock=clock)
    with pytest.raises(ValueError):
        QueueDepthShedder(UpstreamLoadSignal(lambda: "", clock=clock), max_depth=-1)
