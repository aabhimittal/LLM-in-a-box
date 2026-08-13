"""Adaptive load shedding driven by the model server's own queue depth.

Every other control in this package infers upstream health indirectly — from
errors (circuit breaker) or from local concurrency (bulkhead). vLLM publishes
the truth directly: ``vllm:num_requests_waiting`` is the number of requests
admitted but not yet running. Non-zero and growing means the GPU is saturated,
and it says so *before* latency degrades enough for users to complain.

Shedding on that signal is strictly better than waiting for timeouts, with one
non-negotiable rule, enforced here and covered by tests:

    **Fail open.** If the metrics endpoint is unreachable, slow, or changes its
    metric names between vLLM versions, the signal must be treated as *unknown*
    and traffic must flow. A monitoring failure that takes down serving is a
    worse outage than the one it was meant to prevent.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from .errors import Overloaded

# vLLM's exporter prefixes its series with "vllm:".
DEFAULT_QUEUE_METRIC = "vllm:num_requests_waiting"


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse a Prometheus text exposition into ``{metric_name: summed_value}``.

    Series carrying labels (vLLM labels by ``model_name``) are summed, which is
    the correct aggregation for a queue-depth gauge across models sharing one
    GPU. Comments, blank lines and unparseable values are skipped rather than
    raising — a malformed scrape must not become an exception in the request
    path.
    """
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "name{labels} value" or "name value"; trailing timestamps are ignored.
        if "{" in line:
            name, _, rest = line.partition("{")
            _, _, after = rest.partition("}")
            value_part = after.strip()
        else:
            name, _, value_part = line.partition(" ")
        name = name.strip()
        if not name or not value_part:
            continue
        token = value_part.split()[0]
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isnan(value):
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


class UpstreamLoadSignal:
    """Cached view of one upstream metric.

    Args:
        fetch: Returns the raw Prometheus exposition text (e.g. an HTTP GET of
            vLLM's ``/metrics``).
        metric: Metric name to extract.
        ttl: Seconds to reuse a reading. Scraping on every request would add a
            network round trip to the hot path for a value that changes on the
            scale of seconds.
        stale_after: Seconds after which an un-refreshable reading is discarded
            and the signal reports "unknown" rather than acting on stale data.
    """

    def __init__(
        self,
        fetch: Callable[[], str],
        metric: str = DEFAULT_QUEUE_METRIC,
        ttl: float = 5.0,
        stale_after: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self.fetch = fetch
        self.metric = metric
        self.ttl = ttl
        self.stale_after = stale_after
        self._clock = clock
        self._lock = threading.Lock()
        self._value: float | None = None
        self._fetched_at: float | None = None
        self.failures = 0
        self.refreshes = 0

    def value(self) -> float | None:
        """Current metric value, or ``None`` when it cannot be determined."""
        now = self._clock()
        with self._lock:
            fresh = self._fetched_at is not None and (now - self._fetched_at) < self.ttl
            if fresh:
                return self._value

        # Fetch outside the lock: a slow scrape must not serialise every caller.
        try:
            text = self.fetch()
            totals = parse_prometheus_text(text)
            value = totals.get(self.metric)
        except Exception:  # noqa: BLE001 - deliberately fail open
            with self._lock:
                self.failures += 1
                if (
                    self._fetched_at is not None
                    and (now - self._fetched_at) >= self.stale_after
                ):
                    # Too old to trust; report unknown instead of shedding on it.
                    self._value = None
                return self._value

        with self._lock:
            self.refreshes += 1
            self._value = value  # None when the metric is absent -> unknown
            self._fetched_at = now
            return self._value


class QueueDepthShedder:
    """Rejects work while the upstream queue is deeper than ``max_depth``.

    ``max_depth=0`` sheds as soon as anything is waiting, which is usually too
    aggressive: a healthy server briefly queues during normal batching. Pick a
    value from observed steady-state depth.
    """

    def __init__(
        self,
        signal: UpstreamLoadSignal,
        max_depth: float = 8.0,
        retry_after: float = 5.0,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must not be negative")
        self.signal = signal
        self.max_depth = max_depth
        self.retry_after = retry_after
        self.shed = 0

    def check(self) -> None:
        """Raise :class:`Overloaded` if the upstream queue is too deep."""
        depth = self.signal.value()
        if depth is None:  # unknown -> fail open
            return
        if depth > self.max_depth:
            self.shed += 1
            raise Overloaded(
                f"upstream queue depth {depth:g} exceeds the limit of {self.max_depth:g}",
                retry_after=self.retry_after,
                source="upstream_queue",
                depth=depth,
            )
