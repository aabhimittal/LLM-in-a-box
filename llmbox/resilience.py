"""Circuit breaking and retry with full jitter.

A model server behaves unlike a typical HTTP backend: it is *unavailable for
minutes* at startup while weights load, and when saturated it degrades by
queueing rather than by failing fast. Both properties punish naive clients.

* Retrying into a cold server multiplies the queue depth and delays readiness.
  The circuit breaker converts a long outage into fast, cheap failures.
* Retrying without jitter synchronises every replica of the UI into a
  thundering herd on recovery. Full jitter spreads them out.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

from .errors import CircuitOpen, UpstreamError

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Fails fast while the upstream is unhealthy.

    Args:
        failure_threshold: Consecutive failures that trip the circuit.
        recovery_timeout: Seconds to stay open before probing again.
        half_open_max_calls: Concurrent probes allowed while half-open.
        clock: Monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probes = 0
        self.trips = 0

    @property
    def state(self) -> CircuitState:
        """Current state, accounting for an elapsed recovery timeout."""
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def _maybe_half_open_locked(self) -> None:
        if (
            self._state is CircuitState.OPEN
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._probes = 0

    def allow(self) -> bool:
        """Reserve permission to make one call."""
        with self._lock:
            self._maybe_half_open_locked()
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                return False
            if self._probes < self.half_open_max_calls:
                self._probes += 1
                return True
            return False

    def retry_after(self) -> float:
        """Seconds until the circuit will next admit a probe."""
        with self._lock:
            if self._state is not CircuitState.OPEN:
                return 0.0
            return max(0.0, self.recovery_timeout - (self._clock() - self._opened_at))

    def guard(self) -> None:
        """Raise :class:`CircuitOpen` unless a call is permitted."""
        if not self.allow():
            raise CircuitOpen(
                "upstream circuit is open; failing fast",
                retry_after=self.retry_after(),
            )

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._probes = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # A failed probe re-opens immediately and restarts the timer.
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._probes = 0
                self.trips += 1
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._probes = 0
                self.trips += 1

    def call(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` under the breaker, recording the outcome."""
        self.guard()
        try:
            result = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


def default_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` represents a transient failure worth retrying."""
    if isinstance(exc, CircuitOpen):
        # The breaker already decided; retrying in-process just burns the
        # caller's latency budget.
        return False
    if isinstance(exc, UpstreamError):
        return exc.retryable
    return isinstance(exc, (TimeoutError, ConnectionError))


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter and an optional wall-clock budget."""

    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 5.0
    jitter: bool = True
    retry_on: Callable[[BaseException], bool] = default_retryable
    clock: Callable[[], float] = time.monotonic
    rand: Callable[[], float] = random.random
    attempts_made: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    def backoff_cap(self, attempt: int) -> float:
        """Un-jittered ceiling for the delay after ``attempt`` failures."""
        return min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))

    def delay_for(self, attempt: int) -> float:
        cap = self.backoff_cap(attempt)
        # Full jitter (AWS architecture blog): uniform over [0, cap]. Beats
        # "equal jitter" at reducing contention on recovery.
        return self.rand() * cap if self.jitter else cap

    def run(
        self,
        fn: Callable[[], T],
        sleep: Callable[[float], None] = time.sleep,
        deadline: float | None = None,
    ) -> T:
        """Call ``fn``, retrying transient failures.

        Args:
            fn: Zero-argument callable to invoke.
            sleep: Sleep function (injectable for tests).
            deadline: Optional total budget in seconds. Retries stop once the
                *next* attempt could not start before the budget expires.
        """
        started = self.clock()
        self.attempts_made = 0
        last: BaseException | None = None

        for attempt in range(1, self.max_attempts + 1):
            self.attempts_made = attempt
            try:
                return fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if not self.retry_on(exc):
                    raise
                last = exc
                if attempt >= self.max_attempts:
                    break
                delay = self.delay_for(attempt)
                if deadline is not None:
                    elapsed = self.clock() - started
                    if elapsed + delay >= deadline:
                        break
                sleep(delay)

        assert last is not None
        raise last
