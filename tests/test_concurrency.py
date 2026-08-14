"""Bulkhead: bounded in-flight concurrency."""

from __future__ import annotations

import threading
import time

import pytest

from llmbox.concurrency import Bulkhead
from llmbox.errors import Overloaded


def test_admits_up_to_the_limit():
    bulkhead = Bulkhead(max_concurrent=3)
    assert all(bulkhead.try_acquire() for _ in range(3))
    assert not bulkhead.try_acquire()
    assert bulkhead.in_flight == 3
    assert bulkhead.available == 0
    assert bulkhead.rejections == 1


def test_release_frees_a_slot():
    bulkhead = Bulkhead(max_concurrent=1)
    assert bulkhead.try_acquire()
    assert not bulkhead.try_acquire()
    bulkhead.release()
    assert bulkhead.try_acquire()


def test_slot_context_manager_releases_on_success_and_failure():
    bulkhead = Bulkhead(max_concurrent=1)
    with bulkhead.slot():
        assert bulkhead.in_flight == 1
    assert bulkhead.in_flight == 0

    with pytest.raises(ValueError):
        with bulkhead.slot():
            raise ValueError("boom")
    assert bulkhead.in_flight == 0  # not leaked by the exception


def test_acquire_or_raise_reports_overload():
    bulkhead = Bulkhead(max_concurrent=1)
    bulkhead.acquire_or_raise()
    with pytest.raises(Overloaded) as excinfo:
        bulkhead.acquire_or_raise()
    assert excinfo.value.source == "bulkhead"


def test_double_release_is_a_loud_error():
    """Silently tolerating it would inflate the effective limit without trace."""
    bulkhead = Bulkhead(max_concurrent=2)
    bulkhead.try_acquire()
    bulkhead.release()
    with pytest.raises(RuntimeError, match="without a matching acquire"):
        bulkhead.release()


def test_peak_is_tracked_for_capacity_planning():
    bulkhead = Bulkhead(max_concurrent=4)
    for _ in range(3):
        bulkhead.try_acquire()
    for _ in range(3):
        bulkhead.release()
    assert bulkhead.peak_in_flight == 3
    assert bulkhead.in_flight == 0


def test_timeout_waits_for_a_slot_then_succeeds():
    bulkhead = Bulkhead(max_concurrent=1)
    bulkhead.try_acquire()

    def release_soon():
        time.sleep(0.05)
        bulkhead.release()

    threading.Thread(target=release_soon, daemon=True).start()
    assert bulkhead.try_acquire(timeout=2.0)


def test_timeout_gives_up_rather_than_blocking_forever():
    bulkhead = Bulkhead(max_concurrent=1)
    bulkhead.try_acquire()
    started = time.monotonic()
    assert not bulkhead.try_acquire(timeout=0.05)
    assert time.monotonic() - started < 1.0
    assert bulkhead.rejections == 1


def test_never_oversells_under_contention():
    """The whole point: concurrent callers must not exceed the limit."""
    bulkhead = Bulkhead(max_concurrent=5)
    observed_peak = []
    lock = threading.Lock()
    current = {"n": 0}
    barrier = threading.Barrier(40)

    def worker():
        barrier.wait()
        for _ in range(20):
            if bulkhead.try_acquire():
                with lock:
                    current["n"] += 1
                    observed_peak.append(current["n"])
                time.sleep(0.001)
                with lock:
                    current["n"] -= 1
                bulkhead.release()

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert max(observed_peak) <= 5
    assert bulkhead.in_flight == 0


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        Bulkhead(max_concurrent=0)
    with pytest.raises(ValueError):
        Bulkhead(max_concurrent=1, acquire_timeout=-1)
