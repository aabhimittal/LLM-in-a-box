"""Circuit breaking and jittered retry."""

from __future__ import annotations

import pytest

from llmbox.errors import CircuitOpen, UpstreamError
from llmbox.resilience import CircuitBreaker, CircuitState, RetryPolicy, default_retryable


def breaker(clock, **kwargs) -> CircuitBreaker:
    defaults = dict(failure_threshold=3, recovery_timeout=30.0, clock=clock)
    defaults.update(kwargs)
    return CircuitBreaker(**defaults)


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


def test_starts_closed(clock):
    assert breaker(clock).state is CircuitState.CLOSED


def test_trips_only_at_the_threshold(clock):
    cb = breaker(clock)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED
    assert cb.allow()
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert not cb.allow()
    assert cb.trips == 1


def test_success_resets_the_failure_run(clock):
    """Only *consecutive* failures should trip the breaker."""
    cb = breaker(clock)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED


def test_stays_open_for_the_recovery_timeout(clock):
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.advance(29.9)
    assert cb.state is CircuitState.OPEN
    assert cb.retry_after() == pytest.approx(0.1)
    clock.advance(0.2)
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.retry_after() == 0.0


def test_half_open_admits_a_limited_number_of_probes(clock):
    cb = breaker(clock, half_open_max_calls=1)
    for _ in range(3):
        cb.record_failure()
    clock.advance(30)
    assert cb.allow()      # the probe
    assert not cb.allow()  # everyone else keeps failing fast


def test_failed_probe_reopens_and_restarts_the_timer(clock):
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.advance(30)
    assert cb.allow()
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    clock.advance(29)
    assert cb.state is CircuitState.OPEN  # full timeout again, not a partial one
    clock.advance(2)
    assert cb.state is CircuitState.HALF_OPEN


def test_successful_probe_closes_the_circuit(clock):
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.advance(30)
    cb.allow()
    cb.record_success()
    assert cb.state is CircuitState.CLOSED
    # And the failure budget is fresh, not one failure from re-tripping.
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED


def test_call_records_outcomes_and_reraises(clock):
    cb = breaker(clock, failure_threshold=1)
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert cb.state is CircuitState.OPEN
    with pytest.raises(CircuitOpen) as excinfo:
        cb.call(lambda: "never runs")
    assert excinfo.value.retry_after == pytest.approx(30.0)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_max_calls=0)


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #


def policy(clock, **kwargs) -> RetryPolicy:
    defaults = dict(max_attempts=3, base_delay=1.0, max_delay=10.0, clock=clock, rand=lambda: 1.0)
    defaults.update(kwargs)
    return RetryPolicy(**defaults)


def test_returns_immediately_on_success(clock):
    p = policy(clock)
    assert p.run(lambda: "ok", sleep=clock.sleep) == "ok"
    assert p.attempts_made == 1


def test_retries_transient_failures_then_succeeds(clock):
    p = policy(clock)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise UpstreamError("503 loading model", status_code=503, retryable=True)
        return "ready"

    assert p.run(flaky, sleep=clock.sleep) == "ready"
    assert p.attempts_made == 3


def test_permanent_failures_are_not_retried(clock):
    """A 400 will fail identically forever; retrying only adds latency."""
    p = policy(clock)
    calls = []

    def bad_request():
        calls.append(1)
        raise UpstreamError("400 bad request", status_code=400, retryable=False)

    with pytest.raises(UpstreamError):
        p.run(bad_request, sleep=clock.sleep)
    assert len(calls) == 1


def test_exhausted_retries_raise_the_last_error(clock):
    p = policy(clock)
    with pytest.raises(UpstreamError, match="attempt"):
        p.run(
            lambda: (_ for _ in ()).throw(
                UpstreamError(f"attempt failed", status_code=503, retryable=True)
            ),
            sleep=clock.sleep,
        )
    assert p.attempts_made == 3


def test_backoff_grows_exponentially_and_is_capped(clock):
    p = policy(clock, base_delay=1.0, max_delay=4.0)
    assert [p.backoff_cap(i) for i in range(1, 6)] == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_full_jitter_spreads_delays_across_the_whole_window(clock):
    """Without jitter every replica retries in lockstep and re-saturates the GPU."""
    samples = []
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = policy(clock, rand=lambda r=r: r)
        samples.append(p.delay_for(3))
    cap = policy(clock).backoff_cap(3)
    assert samples == [0.0, cap * 0.25, cap * 0.5, cap * 0.75, cap]
    assert all(0.0 <= s <= cap for s in samples)


def test_jitter_can_be_disabled(clock):
    p = policy(clock, jitter=False)
    assert p.delay_for(2) == p.backoff_cap(2)


def test_deadline_stops_retrying_early(clock):
    """A latency budget must win over the attempt count."""
    p = policy(clock, max_attempts=10, base_delay=1.0, max_delay=100.0, jitter=False)

    def always_fails():
        raise UpstreamError("503", status_code=503, retryable=True)

    with pytest.raises(UpstreamError):
        p.run(always_fails, sleep=clock.sleep, deadline=1.5)
    # attempt 1 -> sleep 1.0 (fits); attempt 2 -> next delay 2.0 exceeds budget.
    assert p.attempts_made == 2


def test_open_circuit_is_not_retried(clock):
    """Retrying through an open breaker just burns the caller's budget."""
    p = policy(clock)
    calls = []

    def blocked():
        calls.append(1)
        raise CircuitOpen("open", retry_after=5.0)

    with pytest.raises(CircuitOpen):
        p.run(blocked, sleep=clock.sleep)
    assert len(calls) == 1


def test_default_retryable_classification():
    assert default_retryable(UpstreamError("x", retryable=True))
    assert not default_retryable(UpstreamError("x", retryable=False))
    assert default_retryable(TimeoutError())
    assert default_retryable(ConnectionError())
    assert not default_retryable(CircuitOpen("open"))
    assert not default_retryable(ValueError("bug in our own code"))


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
