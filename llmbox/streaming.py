"""Stall detection for token streams.

A streaming generation that *fails* raises. A streaming generation that simply
**stops producing tokens** does not: the socket stays open and the iterator
blocks forever. The user watches a cursor blink indefinitely, the request holds
a bulkhead slot and an upstream connection, and nothing times out — HTTP
read timeouts are per-read, and a server that sends a keep-alive or nothing at
all never trips them.

This module puts a watchdog on the gap *between* tokens. The source is drained
by a helper thread so the consumer can time-box each wait, which is the only way
to bound a blocking iterator without cooperation from the producer.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterable, Iterator

from .errors import StreamStalled

_DONE = object()


def iter_with_idle_timeout(
    source: Iterable[Any],
    idle_timeout: float,
    thread_name: str = "llmbox-stream-pump",
) -> Iterator[Any]:
    """Yield from ``source``, raising :class:`StreamStalled` on a silent gap.

    Args:
        source: Any iterable, typically an SSE chunk iterator.
        idle_timeout: Maximum seconds allowed between consecutive items. The
            clock restarts on every item, so a slow-but-steady generation is
            never interrupted — only a genuinely idle one.

    Raises:
        StreamStalled: No item arrived within ``idle_timeout``.

    Notes:
        The queue is intentionally unbounded. A bounded queue would block the
        pump thread when a consumer stops reading, leaking the thread; unbounded
        lets the pump notice the stop flag and exit promptly. Token deltas are
        small, and generation rate is the real bound on volume.
    """
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")

    items: queue.Queue = queue.Queue()
    stop = threading.Event()

    def pump() -> None:
        try:
            for item in source:
                if stop.is_set():
                    break
                items.put(item)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            items.put(exc)
            return
        items.put(_DONE)

    worker = threading.Thread(target=pump, name=thread_name, daemon=True)
    worker.start()

    try:
        while True:
            try:
                item = items.get(timeout=idle_timeout)
            except queue.Empty:
                raise StreamStalled(
                    f"no token received for {idle_timeout:g}s; treating the stream as stalled",
                    idle_seconds=idle_timeout,
                ) from None
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # Runs on normal exhaustion, on error, and when a consumer abandons the
        # generator (GeneratorExit at close/GC), so the pump never outlives use.
        stop.set()
