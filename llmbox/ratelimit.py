"""Admission control via token buckets.

A single GPU serving a shared model is a hard capacity limit: one script in a
loop can starve every human user. Rate limiting belongs in front of the model,
not behind it.

Design notes for production use:

* Buckets are **per tenant** and stored in a bounded LRU map. An unbounded
  ``dict`` keyed by user id is a memory leak with an attacker-controlled key.
* Eviction is least-recently-used, so a tenant hammering the service is never
  the one evicted — evicting it would hand it a fresh full bucket, turning the
  limiter into an amplifier.
* A request whose cost exceeds the bucket capacity is rejected immediately with
  ``retry_after=None`` rather than being told to wait forever.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable

from .errors import RateLimitExceeded


class TokenBucket:
    """A classic token bucket with an injectable clock.

    Args:
        capacity: Maximum burst size, in tokens.
        refill_per_sec: Sustained rate at which tokens are replenished.
        clock: Monotonic time source (injectable for deterministic tests).
        initial: Starting token count; defaults to a full bucket.
    """

    def __init__(
        self,
        capacity: float,
        refill_per_sec: float,
        clock: Callable[[], float] = time.monotonic,
        initial: float | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_sec < 0:
            raise ValueError("refill_per_sec must not be negative")
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self._clock = clock
        self._tokens = float(capacity if initial is None else initial)
        self._last = clock()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = self._clock()
        # A monotonic clock should never go backwards, but a mis-injected clock
        # (or a clock read across a process fork) can. Clamp rather than credit
        # negative tokens.
        elapsed = max(0.0, now - self._last)
        self._last = now
        if elapsed and self.refill_per_sec:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)

    @property
    def tokens(self) -> float:
        """Current token count, after accounting for elapsed time."""
        with self._lock:
            self._refill_locked()
            return self._tokens

    def try_acquire(self, amount: float = 1.0) -> bool:
        """Consume ``amount`` tokens if available. Never blocks."""
        if amount <= 0:
            return True
        if amount > self.capacity:
            return False
        with self._lock:
            self._refill_locked()
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    def retry_after(self, amount: float = 1.0) -> float | None:
        """Seconds until ``amount`` tokens are available.

        Returns ``0.0`` if available now and ``None`` if the request can never
        be satisfied (cost exceeds capacity, or the bucket never refills).
        """
        if amount <= 0:
            return 0.0
        if amount > self.capacity:
            return None
        with self._lock:
            self._refill_locked()
            deficit = amount - self._tokens
            if deficit <= 0:
                return 0.0
            if not self.refill_per_sec:
                return None
            return deficit / self.refill_per_sec


class RateLimiter:
    """Bounded, per-tenant token-bucket rate limiter."""

    def __init__(
        self,
        capacity: float,
        refill_per_sec: float,
        max_tenants: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_tenants <= 0:
            raise ValueError("max_tenants must be positive")
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.max_tenants = max_tenants
        self._clock = clock
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lock = threading.Lock()
        self.evictions = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)

    def _bucket(self, tenant: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(tenant)
            if bucket is None:
                bucket = TokenBucket(self.capacity, self.refill_per_sec, self._clock)
                self._buckets[tenant] = bucket
                while len(self._buckets) > self.max_tenants:
                    self._buckets.popitem(last=False)  # evict least-recently-used
                    self.evictions += 1
            else:
                self._buckets.move_to_end(tenant)
            return bucket

    def allow(self, tenant: str, cost: float = 1.0) -> bool:
        """Return whether ``tenant`` may spend ``cost`` right now."""
        return self._bucket(tenant).try_acquire(cost)

    def check(self, tenant: str, cost: float = 1.0) -> None:
        """Raise :class:`RateLimitExceeded` unless ``tenant`` may proceed."""
        bucket = self._bucket(tenant)
        if bucket.try_acquire(cost):
            return

        # Distinguish the two un-satisfiable cases, which need different
        # operator responses: raise the capacity, versus wait for refill.
        if cost > self.capacity:
            raise RateLimitExceeded(
                f"request cost {cost:g} exceeds the per-tenant capacity "
                f"of {self.capacity:g}; it can never be admitted",
                retry_after=None,
                tenant=tenant,
            )
        retry_after = bucket.retry_after(cost)
        if retry_after is None:
            raise RateLimitExceeded(
                f"rate limit exhausted for {tenant!r}; this bucket does not refill",
                retry_after=None,
                tenant=tenant,
            )
        raise RateLimitExceeded(
            f"rate limit exceeded for {tenant!r}; retry in {retry_after:.2f}s",
            retry_after=retry_after,
            tenant=tenant,
        )
