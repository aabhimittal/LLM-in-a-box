"""Response caching and stampede protection.

Two independent problems, both expensive on a single-GPU deployment:

1. **Repeated identical prompts.** Health checks, retried requests and shared
   demo links produce the same prompt over and over. Serving those from memory
   frees the GPU for real work.
2. **Thundering herd.** When N users submit the same prompt simultaneously
   (a cold cache after a deploy, or a link shared in a chat room), a naive cache
   issues N identical generations. :class:`SingleFlight` collapses them into
   one upstream call whose result is shared.

Correctness note: only *deterministic* requests are cached by default. Caching a
``temperature=0.8`` completion and replaying it to every user would silently
destroy sampling diversity — a bug that looks like "the model keeps repeating
itself" and is very hard to trace back to the cache.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

_MISS = object()


@dataclass
class CacheStats:
    """Counters suitable for direct export to Prometheus."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def make_key(model: str, messages: Sequence[Mapping[str, str]], **params: Any) -> str:
    """Build a stable cache key.

    The key covers every input that can change the output: the model, the full
    message list (roles included) and the sampling parameters. Serialisation is
    canonical (sorted keys, UTF-8) so equivalent requests hash identically
    regardless of dict ordering.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": m.get("role", ""), "content": m.get("content") or ""} for m in messages
        ],
        "params": {k: params[k] for k in sorted(params) if params[k] is not None},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_deterministic(params: Mapping[str, Any]) -> bool:
    """True when the sampling parameters make generation reproducible."""
    temperature = params.get("temperature")
    if temperature is None or float(temperature) > 0.0:
        return False
    # top_p is irrelevant at temperature 0 (greedy decoding), but an explicit
    # seed with non-zero temperature is *not* treated as deterministic here:
    # vLLM's seeding is per-request and does not survive continuous batching
    # rearrangement in all versions.
    return True


class PromptCache:
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(
        self,
        max_entries: int = 512,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        cache_nondeterministic: bool = False,
    ):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.cache_nondeterministic = cache_nondeterministic
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def cacheable(self, params: Mapping[str, Any]) -> bool:
        """Whether a request with these parameters may be cached."""
        return self.cache_nondeterministic or is_deterministic(params)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value, or ``default`` on miss/expiry."""
        with self._lock:
            entry = self._entries.get(key, _MISS)
            if entry is _MISS:
                self.stats.misses += 1
                return default
            expires_at, value = entry  # type: ignore[misc]
            if self._clock() >= expires_at:
                del self._entries[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return default
            self._entries.move_to_end(key)
            self.stats.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Store ``value``, evicting the least-recently-used entry if needed."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
            self._entries[key] = (self._clock() + self.ttl_seconds, value)
            self.stats.stores += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class _Call:
    __slots__ = ("event", "value", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None


class SingleFlight:
    """Collapses concurrent identical calls into one execution.

    The first caller for a key executes ``fn``; every other caller arriving
    while it is in flight blocks and then receives the same result (or the same
    exception). This bounds GPU work by *distinct* prompts in flight rather than
    by request count.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, _Call] = {}

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._calls)

    def do(self, key: str, fn: Callable[[], Any]) -> Any:
        with self._lock:
            call = self._calls.get(key)
            leader = call is None
            if leader:
                call = _Call()
                self._calls[key] = call

        assert call is not None
        if not leader:
            call.event.wait()
            if call.error is not None:
                raise call.error
            return call.value

        try:
            call.value = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below and to waiters
            call.error = exc
        finally:
            with self._lock:
                self._calls.pop(key, None)
            call.event.set()

        if call.error is not None:
            raise call.error
        return call.value
