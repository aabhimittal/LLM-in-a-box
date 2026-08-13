"""llmbox — the serving edge for LLM-in-a-Box.

A dependency-free toolkit that sits between a chat front-end and a vLLM server
and supplies the things a raw OpenAI client does not: context budgeting,
token-priced admission control, deterministic response caching with stampede
protection, PII redaction, prompt-injection screening, circuit breaking with
jittered retries, and structured telemetry.

Typical use::

    from llmbox import build_client

    client = build_client(backend=my_backend, model="llama-3-8b-instruct",
                          max_model_len=8192)
    result = client.chat([{"role": "user", "content": "Hello"}])
    print(result.text, result.prompt_tokens, result.cached)
"""

from __future__ import annotations

from typing import Any, Callable

from .cache import CacheStats, PromptCache, SingleFlight, is_deterministic, make_key
from .concurrency import Bulkhead
from .client import (
    ChatResult,
    ResilientChatClient,
    StreamHandle,
    classify_exception,
    extract_text,
    extract_usage,
    iter_stream_deltas,
)
from .context import ContextBudget, FitResult, truncate_to_tokens
from .errors import (
    CircuitOpen,
    ContextOverflow,
    GuardrailBlocked,
    LLMBoxError,
    Overloaded,
    RateLimitExceeded,
    StreamInterrupted,
    StreamStalled,
    UpstreamError,
)
from .guardrails import (
    Finding,
    InjectionDetector,
    InjectionVerdict,
    RedactionResult,
    Redactor,
    luhn_valid,
    strip_invisible,
)
from .loadsignal import (
    DEFAULT_QUEUE_METRIC,
    QueueDepthShedder,
    UpstreamLoadSignal,
    parse_prometheus_text,
)
from .metrics_server import MetricsServer
from .observability import Metrics, StructuredLogger, new_request_id
from .ratelimit import RateLimiter, TokenBucket
from .resilience import CircuitBreaker, CircuitState, RetryPolicy, default_retryable
from .streaming import iter_with_idle_timeout
from .tokens import count_message_tokens, estimate_tokens, make_counter

__version__ = "0.2.0"

__all__ = [
    "Bulkhead",
    "CacheStats",
    "ChatResult",
    "CircuitBreaker",
    "CircuitOpen",
    "CircuitState",
    "ContextBudget",
    "ContextOverflow",
    "DEFAULT_QUEUE_METRIC",
    "Finding",
    "FitResult",
    "GuardrailBlocked",
    "InjectionDetector",
    "InjectionVerdict",
    "LLMBoxError",
    "Metrics",
    "MetricsServer",
    "Overloaded",
    "PromptCache",
    "RateLimitExceeded",
    "QueueDepthShedder",
    "RateLimiter",
    "RedactionResult",
    "Redactor",
    "ResilientChatClient",
    "RetryPolicy",
    "SingleFlight",
    "StreamHandle",
    "StreamInterrupted",
    "StreamStalled",
    "StructuredLogger",
    "TokenBucket",
    "UpstreamError",
    "UpstreamLoadSignal",
    "build_client",
    "classify_exception",
    "count_message_tokens",
    "default_retryable",
    "estimate_tokens",
    "extract_text",
    "extract_usage",
    "is_deterministic",
    "iter_with_idle_timeout",
    "iter_stream_deltas",
    "luhn_valid",
    "make_counter",
    "make_key",
    "new_request_id",
    "parse_prometheus_text",
    "strip_invisible",
    "truncate_to_tokens",
]


def build_client(
    backend: Callable[[dict, bool], Any],
    model: str,
    *,
    max_model_len: int = 8192,
    max_output_tokens: int = 512,
    tokens_per_minute: float = 60_000,
    burst_tokens: float = 20_000,
    cache_entries: int = 512,
    cache_ttl: float = 300.0,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    max_attempts: int = 3,
    max_concurrent: int = 16,
    stream_idle_timeout: float | None = 60.0,
    metrics_url: str | None = None,
    max_queue_depth: float = 8.0,
    **overrides: Any,
) -> ResilientChatClient:
    """Assemble a :class:`ResilientChatClient` with production-shaped defaults.

    The rate limiter is priced in tokens per minute so a few very large prompts
    cannot slip past a request-count limit and monopolise the GPU, and the
    bulkhead caps total in-flight work so a burst is rejected outright rather
    than silently queueing on the GPU.

    Args:
        max_concurrent: Upstream requests allowed in flight at once.
        stream_idle_timeout: Seconds of silence before a stream is treated as
            stalled. ``None`` disables the watchdog.
        metrics_url: vLLM ``/metrics`` URL. When provided, requests are shed
            while the server's own queue is deeper than ``max_queue_depth``.
            Fails open if the endpoint is unreachable.
    """
    shedder = None
    if metrics_url:
        # Imported lazily: the core package stays dependency-free and this is
        # the only place that needs a network client at all.
        from urllib.request import urlopen

        def _fetch() -> str:
            with urlopen(metrics_url, timeout=2.0) as response:  # noqa: S310
                return response.read().decode("utf-8", "replace")

        shedder = QueueDepthShedder(
            UpstreamLoadSignal(_fetch), max_depth=max_queue_depth
        )

    return ResilientChatClient(
        backend=backend,
        model=model,
        budget=ContextBudget(max_model_len, max_output_tokens),
        cache=PromptCache(max_entries=cache_entries, ttl_seconds=cache_ttl),
        limiter=RateLimiter(capacity=burst_tokens, refill_per_sec=tokens_per_minute / 60.0),
        breaker=CircuitBreaker(
            failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
        ),
        retry=RetryPolicy(max_attempts=max_attempts),
        redactor=Redactor(),
        detector=InjectionDetector(),
        single_flight=SingleFlight(),
        bulkhead=Bulkhead(max_concurrent=max_concurrent),
        shedder=shedder,
        stream_idle_timeout=stream_idle_timeout,
        **overrides,
    )
