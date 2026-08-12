"""Token-bucket admission control."""

from __future__ import annotations

import threading

import pytest

from llmbox.errors import RateLimitExceeded
from llmbox.ratelimit import RateLimiter, TokenBucket


def test_burst_up_to_capacity_then_denied(clock):
    bucket = TokenBucket(capacity=10, refill_per_sec=1, clock=clock)
    assert bucket.try_acquire(10)
    assert not bucket.try_acquire(1)


def test_tokens_refill_over_time(clock):
    bucket = TokenBucket(capacity=10, refill_per_sec=2, clock=clock)
    assert bucket.try_acquire(10)
    clock.advance(2.5)  # 5 tokens back
    assert bucket.try_acquire(5)
    assert not bucket.try_acquire(1)


def test_refill_never_exceeds_capacity(clock):
    bucket = TokenBucket(capacity=10, refill_per_sec=100, clock=clock)
    clock.advance(1000)
    assert bucket.tokens == 10


def test_retry_after_is_the_actual_wait(clock):
    bucket = TokenBucket(capacity=10, refill_per_sec=2, clock=clock)
    bucket.try_acquire(10)
    wait = bucket.retry_after(4)
    assert wait == pytest.approx(2.0)
    clock.advance(wait)
    assert bucket.try_acquire(4)


def test_request_larger_than_capacity_is_never_satisfiable(clock):
    """Must fail fast rather than advertise an infinite retry-after."""
    bucket = TokenBucket(capacity=10, refill_per_sec=1, clock=clock)
    assert not bucket.try_acquire(11)
    assert bucket.retry_after(11) is None


def test_non_refilling_bucket_reports_no_retry(clock):
    bucket = TokenBucket(capacity=5, refill_per_sec=0, clock=clock)
    bucket.try_acquire(5)
    assert bucket.retry_after(1) is None


def test_clock_moving_backwards_does_not_credit_tokens(clock):
    """A mis-injected or forked clock must not become a token faucet."""
    bucket = TokenBucket(capacity=10, refill_per_sec=1, clock=clock)
    bucket.try_acquire(10)
    clock.advance(-100)
    assert not bucket.try_acquire(1)


def test_zero_cost_always_admitted(clock):
    bucket = TokenBucket(capacity=1, refill_per_sec=0, clock=clock)
    bucket.try_acquire(1)
    assert bucket.try_acquire(0)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, refill_per_sec=1)
    with pytest.raises(ValueError):
        TokenBucket(capacity=1, refill_per_sec=-1)
    with pytest.raises(ValueError):
        RateLimiter(capacity=1, refill_per_sec=1, max_tenants=0)


def test_concurrent_acquire_never_oversells(clock):
    """The bucket is the only thing standing between users and a saturated GPU."""
    bucket = TokenBucket(capacity=50, refill_per_sec=0, clock=clock)
    granted = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        for _ in range(10):
            if bucket.try_acquire(1):
                granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == 50


# --------------------------------------------------------------------------- #
# Multi-tenant limiter
# --------------------------------------------------------------------------- #


def test_tenants_are_isolated(clock):
    limiter = RateLimiter(capacity=2, refill_per_sec=0, clock=clock)
    assert limiter.allow("alice", 2)
    assert not limiter.allow("alice", 1)
    assert limiter.allow("bob", 2)


def test_check_raises_with_retry_after(clock):
    limiter = RateLimiter(capacity=10, refill_per_sec=5, clock=clock)
    limiter.check("alice", 10)
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("alice", 5)
    assert excinfo.value.retry_after == pytest.approx(1.0)
    assert excinfo.value.tenant == "alice"


def test_impossible_request_reports_no_retry_after(clock):
    limiter = RateLimiter(capacity=10, refill_per_sec=5, clock=clock)
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("alice", 999)
    assert excinfo.value.retry_after is None
    assert "never be admitted" in str(excinfo.value)


def test_tenant_map_is_bounded(clock):
    """An attacker-controlled tenant key must not be an unbounded memory leak."""
    limiter = RateLimiter(capacity=1, refill_per_sec=1, max_tenants=100, clock=clock)
    for i in range(5000):
        limiter.allow(f"tenant-{i}")
    assert len(limiter) == 100
    assert limiter.evictions == 4900


def test_actively_throttled_tenant_survives_key_flooding(clock):
    """A tenant that keeps calling stays most-recently-used, so it cannot flood
    its own bucket out of the map to win a fresh allowance."""
    limiter = RateLimiter(capacity=3, refill_per_sec=0, max_tenants=10, clock=clock)
    for _ in range(3):
        assert limiter.allow("noisy")

    for i in range(500):
        limiter.allow(f"other-{i}")
        # Each call keeps "noisy" at the MRU end and must still be refused.
        assert not limiter.allow("noisy")


def test_eviction_resets_an_idle_tenant_by_design(clock):
    """Documented limitation of a bounded map, asserted so it cannot regress
    silently: a tenant idle long enough to be evicted gets a fresh bucket.

    The bound is what makes the limiter memory-safe against attacker-chosen
    keys; the mitigation for the reset is to key on authenticated identities and
    size ``max_tenants`` above the real population.
    """
    limiter = RateLimiter(capacity=3, refill_per_sec=0, max_tenants=10, clock=clock)
    for _ in range(3):
        assert limiter.allow("victim")
    assert not limiter.allow("victim")

    for i in range(50):  # evicts the idle "victim"
        limiter.allow(f"other-{i}")

    assert limiter.allow("victim")  # fresh bucket — the trade-off, made explicit
