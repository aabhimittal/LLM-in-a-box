"""Stall detection on token streams."""

from __future__ import annotations

import threading
import time

import pytest

from llmbox.errors import StreamStalled
from llmbox.streaming import iter_with_idle_timeout


def test_passes_through_a_healthy_stream():
    assert list(iter_with_idle_timeout(iter("abc"), idle_timeout=2.0)) == ["a", "b", "c"]


def test_empty_stream_terminates_cleanly():
    assert list(iter_with_idle_timeout(iter([]), idle_timeout=1.0)) == []


def test_stalled_stream_raises_instead_of_hanging():
    """An open-but-silent socket never raises on its own; this is the backstop."""
    release = threading.Event()

    def stalls():
        yield "first"
        release.wait(timeout=5)  # silence
        yield "never delivered"

    stream = iter_with_idle_timeout(stalls(), idle_timeout=0.1)
    assert next(stream) == "first"
    started = time.monotonic()
    with pytest.raises(StreamStalled) as excinfo:
        next(stream)
    elapsed = time.monotonic() - started

    assert 0.05 < elapsed < 3.0  # gave up promptly, did not hang
    assert excinfo.value.idle_seconds == pytest.approx(0.1)
    release.set()


def test_slow_but_steady_stream_is_not_interrupted():
    """The timer restarts per item, so a slow generation is fine."""

    def slow():
        for index in range(5):
            time.sleep(0.02)
            yield str(index)

    assert list(iter_with_idle_timeout(slow(), idle_timeout=0.5)) == list("01234")


def test_source_exception_is_propagated_unchanged():
    def failing():
        yield "partial"
        raise ConnectionError("connection reset by peer")

    stream = iter_with_idle_timeout(failing(), idle_timeout=2.0)
    assert next(stream) == "partial"
    with pytest.raises(ConnectionError, match="reset by peer"):
        next(stream)


def test_abandoning_the_stream_stops_the_pump_thread():
    """A consumer that walks away must not leave a thread draining forever."""
    before = threading.active_count()
    produced = []

    def endless():
        index = 0
        while True:
            produced.append(index)
            yield str(index)
            index += 1
            time.sleep(0.005)

    stream = iter_with_idle_timeout(endless(), idle_timeout=2.0)
    next(stream)
    stream.close()  # triggers the finally that sets the stop flag

    time.sleep(0.2)
    assert threading.active_count() <= before + 1  # pump exited (or is exiting)
    settled = len(produced)
    time.sleep(0.1)
    assert len(produced) == settled  # production actually stopped


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        list(iter_with_idle_timeout(iter([]), idle_timeout=0))
