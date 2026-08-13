"""Bulkheading: a hard ceiling on in-flight upstream requests.

The failure this prevents is the one a circuit breaker cannot see. When vLLM
saturates it does not start failing — it starts *queueing*. Requests still
succeed, just slower and slower, so error-rate-based protection never trips.
Meanwhile the caller happily opens another connection for every arriving user,
and the real queue depth becomes invisible, unbounded, and shared by everyone.

A bulkhead converts that silent latency collapse into an explicit, immediate
rejection at a depth *you* chose. Combined with the token-bucket limiter (which
bounds a single tenant's rate) it bounds total concurrent load on the GPU.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from .errors import Overloaded


class Bulkhead:
    """Bounded concurrency gate.

    Args:
        max_concurrent: Requests allowed in flight simultaneously.
        acquire_timeout: Seconds to wait for a slot before rejecting. ``0``
            (the default) rejects immediately — usually right, because queueing
            *here* just moves the invisible queue rather than removing it. A
            small non-zero value smooths brief bursts.
    """

    def __init__(self, max_concurrent: int, acquire_timeout: float = 0.0):
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if acquire_timeout < 0:
            raise ValueError("acquire_timeout must not be negative")
        self.max_concurrent = max_concurrent
        self.acquire_timeout = acquire_timeout
        self._condition = threading.Condition()
        self._in_flight = 0
        self.peak_in_flight = 0
        self.rejections = 0

    @property
    def in_flight(self) -> int:
        with self._condition:
            return self._in_flight

    @property
    def available(self) -> int:
        with self._condition:
            return self.max_concurrent - self._in_flight

    def _take_locked(self) -> None:
        self._in_flight += 1
        if self._in_flight > self.peak_in_flight:
            self.peak_in_flight = self._in_flight

    def try_acquire(self, timeout: float | None = None) -> bool:
        """Reserve a slot. Returns False rather than raising."""
        wait = self.acquire_timeout if timeout is None else timeout
        with self._condition:
            if self._in_flight < self.max_concurrent:
                self._take_locked()
                return True
            if wait <= 0:
                self.rejections += 1
                return False
            granted = self._condition.wait_for(
                lambda: self._in_flight < self.max_concurrent, timeout=wait
            )
            if granted:
                self._take_locked()
                return True
            self.rejections += 1
            return False

    def release(self) -> None:
        """Free a slot and wake one waiter."""
        with self._condition:
            if self._in_flight == 0:
                # Signals a double-release bug in the caller. Failing loudly is
                # far better than silently inflating the effective limit.
                raise RuntimeError("Bulkhead.release() called without a matching acquire")
            self._in_flight -= 1
            self._condition.notify()

    def acquire_or_raise(self) -> None:
        """Reserve a slot or raise :class:`Overloaded`."""
        if not self.try_acquire():
            raise Overloaded(
                f"at capacity: {self.max_concurrent} requests already in flight",
                source="bulkhead",
            )

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold a slot for the duration of the block."""
        self.acquire_or_raise()
        try:
            yield
        finally:
            self.release()
