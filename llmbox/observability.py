"""Structured logging and Prometheus-compatible metrics.

Deliberately dependency-free: a UI pod should not pull in a metrics client just
to export half a dozen series. The registry renders the Prometheus text
exposition format directly, so it can be served from any HTTP handler.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

DEFAULT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

_LABEL_ESCAPES = str.maketrans({"\\": "\\\\", "\n": "\\n", '"': '\\"'})


def new_request_id() -> str:
    """Short, collision-resistant id for correlating logs across components."""
    return uuid.uuid4().hex[:16]


def _render_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{str(v).translate(_LABEL_ESCAPES)}"' for k, v in sorted(labels.items())
    )
    return "{" + inner + "}"


@dataclass
class _Series:
    name: str
    kind: str
    help: str
    values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)
    # Histogram-only state.
    buckets: tuple[float, ...] = ()
    counts: dict[tuple[tuple[str, str], ...], list[int]] = field(default_factory=dict)
    sums: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)


class Metrics:
    """Minimal in-process metrics registry (counters, gauges, histograms)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._series: dict[str, _Series] = {}

    def _get(self, name: str, kind: str, help_text: str, buckets: tuple[float, ...] = ()) -> _Series:
        series = self._series.get(name)
        if series is None:
            series = _Series(name=name, kind=kind, help=help_text, buckets=buckets)
            self._series[name] = series
        elif series.kind != kind:
            raise ValueError(f"metric {name!r} already registered as {series.kind}")
        return series

    @staticmethod
    def _key(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))

    def inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        with self._lock:
            series = self._get(name, "counter", help_text)
            key = self._key(labels)
            series.values[key] = series.values.get(key, 0.0) + value

    def gauge(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        with self._lock:
            series = self._get(name, "gauge", help_text)
            series.values[self._key(labels)] = value

    def observe(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
        buckets: Iterable[float] = DEFAULT_BUCKETS,
    ) -> None:
        with self._lock:
            series = self._get(name, "histogram", help_text, tuple(buckets))
            key = self._key(labels)
            counts = series.counts.setdefault(key, [0] * (len(series.buckets) + 1))
            series.sums[key] = series.sums.get(key, 0.0) + value
            placed = False
            for index, bound in enumerate(series.buckets):
                if value <= bound:
                    counts[index] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1

    def value(self, name: str, labels: Mapping[str, str] | None = None) -> float:
        """Read a counter/gauge value (primarily for tests)."""
        with self._lock:
            series = self._series.get(name)
            if series is None:
                return 0.0
            return series.values.get(self._key(labels), 0.0)

    def render(self) -> str:
        """Render every series in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._series):
                series = self._series[name]
                if series.help:
                    lines.append(f"# HELP {name} {series.help}")
                lines.append(f"# TYPE {name} {series.kind}")
                if series.kind == "histogram":
                    for key in sorted(series.counts):
                        labels = dict(key)
                        cumulative = 0
                        counts = series.counts[key]
                        for index, bound in enumerate(series.buckets):
                            cumulative += counts[index]
                            bucket_labels = {**labels, "le": _fmt(bound)}
                            lines.append(
                                f"{name}_bucket{_render_labels(bucket_labels)} {cumulative}"
                            )
                        cumulative += counts[-1]
                        lines.append(
                            f"{name}_bucket{_render_labels({**labels, 'le': '+Inf'})} {cumulative}"
                        )
                        lines.append(f"{name}_sum{_render_labels(labels)} {_fmt(series.sums[key])}")
                        lines.append(f"{name}_count{_render_labels(labels)} {cumulative}")
                else:
                    for key in sorted(series.values):
                        lines.append(
                            f"{name}{_render_labels(dict(key))} {_fmt(series.values[key])}"
                        )
        return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)


class StructuredLogger:
    """Emits one JSON object per event.

    Never log raw prompt text through this: pass already-redacted content. The
    logger enforces a length cap so a 100k-character prompt cannot blow up the
    log pipeline.
    """

    def __init__(self, logger: logging.Logger | None = None, max_field_chars: int = 512):
        self._logger = logger or logging.getLogger("llmbox")
        self.max_field_chars = max_field_chars

    def _clip(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self.max_field_chars:
            return value[: self.max_field_chars] + f"…(+{len(value) - self.max_field_chars})"
        return value

    def emit(self, event: str, level: int = logging.INFO, **fields: Any) -> str:
        record = {"event": event, "ts": round(time.time(), 3)}
        record.update({k: self._clip(v) for k, v in fields.items()})
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        self._logger.log(level, line)
        return line
