"""The serving edge: one call path that composes every safeguard.

Order of operations, and why:

1. **Guardrails** — cheapest check, and the only one that must run before the
   text is used for anything else (including logging).
2. **Context budgeting** — determines the real token cost, which the rate
   limiter needs. Also guarantees the request *can* succeed.
3. **Rate limiting** — priced in tokens, not requests. A 6k-token prompt costs
   ~60x a 100-token one and should be charged as such; request-count limits let
   a handful of huge prompts monopolise the GPU while looking compliant.
4. **Cache** — only consulted for deterministic requests.
5. **Single-flight** — collapses a concurrent herd onto one generation.
6. **Circuit breaker → retry** — the breaker is inside the retry loop so an
   open circuit aborts immediately instead of burning the retry budget.

The backend is an injected callable ``(payload, stream) -> response``, so this
module has no HTTP dependency and is fully testable without a model server.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from .cache import PromptCache, SingleFlight, make_key
from .context import ContextBudget
from .errors import (
    GuardrailBlocked,
    LLMBoxError,
    StreamInterrupted,
    UpstreamError,
)
from .guardrails import InjectionDetector, Redactor
from .observability import Metrics, StructuredLogger, new_request_id
from .ratelimit import RateLimiter
from .resilience import CircuitBreaker, RetryPolicy
from .tokens import estimate_tokens

Backend = Callable[[dict, bool], Any]

# Status codes worth retrying: transient overload, gateway churn, and the 503
# vLLM returns while weights are still loading.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, whichever we were handed."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def classify_exception(exc: BaseException) -> UpstreamError:
    """Normalise an arbitrary backend exception into an :class:`UpstreamError`."""
    if isinstance(exc, UpstreamError):
        return exc
    status = _field(exc, "status_code", None) or _field(exc, "status", None)
    if isinstance(status, int):
        return UpstreamError(
            f"upstream returned {status}",
            status_code=status,
            retryable=status in RETRYABLE_STATUS,
            cause=exc,
        )
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return UpstreamError(str(exc) or exc.__class__.__name__, retryable=True, cause=exc)
    return UpstreamError(str(exc) or exc.__class__.__name__, retryable=False, cause=exc)


def extract_text(response: Any) -> str:
    """Pull the assistant text out of a chat-completions response."""
    choices = _field(response, "choices") or []
    if not choices:
        raise UpstreamError("response contained no choices", retryable=True)
    message = _field(choices[0], "message")
    if message is None:
        raise UpstreamError("response choice contained no message", retryable=True)
    # A tool-call-only reply legitimately has null content.
    return _field(message, "content") or ""


def extract_usage(response: Any) -> tuple[int | None, int | None]:
    """Return ``(prompt_tokens, completion_tokens)`` when the server reports them."""
    usage = _field(response, "usage")
    if usage is None:
        return None, None
    return _field(usage, "prompt_tokens"), _field(usage, "completion_tokens")


def iter_stream_deltas(chunks: Any) -> Iterator[str]:
    """Yield text deltas from a streaming response, tolerating real-world shapes.

    Handles the chunk variants vLLM and the OpenAI SDK actually emit:

    * a first chunk carrying only ``role`` (``delta.content`` is ``None``);
    * a final chunk with an empty ``choices`` list carrying only ``usage``;
    * ``finish_reason`` chunks with no content.
    """
    for chunk in chunks:
        choices = _field(chunk, "choices") or []
        if not choices:
            continue
        delta = _field(choices[0], "delta")
        if delta is None:
            continue
        content = _field(delta, "content")
        if content:
            yield content


@dataclass
class ChatResult:
    """Everything the caller (and the audit log) needs about one request."""

    text: str = ""
    request_id: str = ""
    model: str = ""
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1
    dropped_messages: int = 0
    truncated: bool = False
    redactions: dict[str, int] = field(default_factory=dict)
    injection_score: float = 0.0


class StreamHandle:
    """Iterable stream whose :attr:`result` is populated as tokens arrive."""

    def __init__(
        self,
        source: Iterator[str],
        result: ChatResult,
        on_done: Callable[[ChatResult, str], None],
    ):
        self._source = source
        self._on_done = on_done
        self.result = result
        self._finished = False

    def __iter__(self) -> Iterator[str]:
        parts: list[str] = []
        try:
            for delta in self._source:
                parts.append(delta)
                yield delta
        except LLMBoxError as exc:
            self.result.text = "".join(parts)
            self._finish(_outcome_for(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as StreamInterrupted
            partial = "".join(parts)
            self.result.text = partial
            self._finish("stream_interrupted")
            raise StreamInterrupted(
                f"stream interrupted after {len(partial)} characters", partial=partial, cause=exc
            ) from exc
        self.result.text = "".join(parts)
        self._finish("ok")

    def _finish(self, outcome: str) -> None:
        if self._finished:
            return
        self._finished = True
        self.result.completion_tokens = estimate_tokens(self.result.text)
        self._on_done(self.result, outcome)


class ResilientChatClient:
    """Composes guardrails, budgeting, admission control and resilience."""

    def __init__(
        self,
        backend: Backend,
        model: str,
        *,
        budget: ContextBudget | None = None,
        cache: PromptCache | None = None,
        limiter: RateLimiter | None = None,
        breaker: CircuitBreaker | None = None,
        retry: RetryPolicy | None = None,
        redactor: Redactor | None = None,
        detector: InjectionDetector | None = None,
        metrics: Metrics | None = None,
        logger: StructuredLogger | None = None,
        single_flight: SingleFlight | None = None,
        injection_threshold: float = 0.6,
        redact_upstream: bool = False,
        cost_fn: Callable[[int, int], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.backend = backend
        self.model = model
        self.budget = budget
        self.cache = cache
        self.limiter = limiter
        self.breaker = breaker
        self.retry = retry
        self.redactor = redactor
        self.detector = detector
        self.metrics = metrics or Metrics()
        self.logger = logger or StructuredLogger()
        self.single_flight = single_flight
        self.injection_threshold = injection_threshold
        # Self-hosted models usually *should* see the raw prompt; redaction is
        # for logs. Enable this only when the weights are not under your control.
        self.redact_upstream = redact_upstream
        self.cost_fn = cost_fn or (lambda prompt_tokens, max_tokens: prompt_tokens + max_tokens)
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------------ #
    # Pre-flight
    # ------------------------------------------------------------------ #
    def _screen(self, messages: Sequence[Mapping[str, str]]) -> tuple[float, dict[str, int]]:
        """Run guardrails over the newest user turn."""
        newest = ""
        for message in reversed(list(messages)):
            if message.get("role") == "user":
                newest = message.get("content") or ""
                break

        score = 0.0
        if self.detector is not None and newest:
            verdict = self.detector.inspect(newest)
            score = verdict.score
            if score >= self.injection_threshold:
                self.metrics.inc(
                    "llmbox_guardrail_blocks_total",
                    labels={"reason": "prompt_injection"},
                    help_text="Requests rejected by an input guardrail.",
                )
                raise GuardrailBlocked(
                    "request rejected by prompt-injection guardrail",
                    reasons=verdict.reasons,
                    score=score,
                )

        redactions: dict[str, int] = {}
        if self.redactor is not None and newest:
            redactions = self.redactor.redact(newest).counts
            for kind, count in redactions.items():
                self.metrics.inc(
                    "llmbox_pii_redactions_total",
                    value=count,
                    labels={"kind": kind},
                    help_text="PII spans redacted from telemetry.",
                )
        return score, redactions

    def _prepare(
        self, messages: Sequence[Mapping[str, str]], system: str | None, max_tokens: int
    ) -> tuple[list[dict], int, int, bool]:
        """Apply guardrail redaction (optional) and context budgeting."""
        prepared = [
            {"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages
        ]
        if self.redact_upstream and self.redactor:
            prepared = [
                {**m, "content": self.redactor.redact(m["content"]).text} for m in prepared
            ]

        if self.budget is None:
            payload = ([{"role": "system", "content": system}] if system else []) + prepared
            from .tokens import count_message_tokens

            return payload, count_message_tokens(payload), 0, False

        fit = self.budget.fit(prepared, system=system)
        if fit.dropped_messages:
            self.metrics.inc(
                "llmbox_context_dropped_messages_total",
                value=fit.dropped_messages,
                help_text="Messages dropped to fit the context window.",
            )
        if fit.truncated:
            self.metrics.inc(
                "llmbox_context_truncations_total",
                help_text="Requests whose content was truncated to fit.",
            )
        return fit.messages, fit.prompt_tokens, fit.dropped_messages, fit.truncated

    def _invoke(self, payload: dict, stream: bool) -> Any:
        """Call the backend under the breaker and retry policy."""

        def once() -> Any:
            try:
                return self.backend(payload, stream)
            except Exception as exc:  # noqa: BLE001 - normalised for retry logic
                raise classify_exception(exc) from exc

        guarded = (lambda: self.breaker.call(once)) if self.breaker is not None else once
        if self.retry is None:
            return guarded()
        return self.retry.run(guarded, sleep=self._sleep)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system: str | None = None,
        tenant: str = "anonymous",
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **extra: Any,
    ) -> ChatResult:
        """Run one non-streaming chat completion through the full pipeline."""
        request_id = new_request_id()
        started = self._clock()
        result = ChatResult(request_id=request_id, model=self.model)

        try:
            result.injection_score, result.redactions = self._screen(messages)
            payload_messages, prompt_tokens, dropped, truncated = self._prepare(
                messages, system, max_tokens
            )
            result.prompt_tokens = prompt_tokens
            result.dropped_messages = dropped
            result.truncated = truncated

            if self.limiter is not None:
                self.limiter.check(tenant, self.cost_fn(prompt_tokens, max_tokens))

            params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, **extra}
            payload = {"model": self.model, "messages": payload_messages, **params}

            key = None
            if self.cache is not None and self.cache.cacheable(params):
                key = make_key(self.model, payload_messages, **params)
                hit = self.cache.get(key)
                if hit is not None:
                    result.text = hit
                    result.cached = True
                    result.completion_tokens = estimate_tokens(hit)
                    result.latency_ms = (self._clock() - started) * 1000
                    self._record(result, "cache_hit", tenant)
                    return result

            def call() -> str:
                response = self._invoke(payload, False)
                text = extract_text(response)
                reported_prompt, reported_completion = extract_usage(response)
                if reported_prompt:
                    result.prompt_tokens = reported_prompt
                result.completion_tokens = reported_completion or estimate_tokens(text)
                return text

            if key is not None and self.single_flight is not None:
                text = self.single_flight.do(key, call)
            else:
                text = call()

            if key is not None:
                self.cache.put(key, text)

            result.text = text
            result.attempts = self.retry.attempts_made if self.retry is not None else 1
            result.latency_ms = (self._clock() - started) * 1000
            self._record(result, "ok", tenant)
            return result

        except LLMBoxError as exc:
            result.latency_ms = (self._clock() - started) * 1000
            self._record(result, _outcome_for(exc), tenant, error=exc)
            raise

    def stream(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system: str | None = None,
        tenant: str = "anonymous",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **extra: Any,
    ) -> StreamHandle:
        """Start a streaming completion.

        Returns immediately with a :class:`StreamHandle`; the upstream call has
        already been made (so admission control and circuit breaking apply
        before the first token), but tokens are consumed lazily.
        """
        request_id = new_request_id()
        started = self._clock()
        result = ChatResult(request_id=request_id, model=self.model)

        try:
            result.injection_score, result.redactions = self._screen(messages)
            payload_messages, prompt_tokens, dropped, truncated = self._prepare(
                messages, system, max_tokens
            )
            result.prompt_tokens = prompt_tokens
            result.dropped_messages = dropped
            result.truncated = truncated

            if self.limiter is not None:
                self.limiter.check(tenant, self.cost_fn(prompt_tokens, max_tokens))

            payload = {
                "model": self.model,
                "messages": payload_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": True,
                **extra,
            }
            chunks = self._invoke(payload, True)
            result.attempts = self.retry.attempts_made if self.retry is not None else 1
        except LLMBoxError as exc:
            result.latency_ms = (self._clock() - started) * 1000
            self._record(result, _outcome_for(exc), tenant, error=exc)
            raise

        def on_done(final: ChatResult, outcome: str) -> None:
            final.latency_ms = (self._clock() - started) * 1000
            self._record(final, outcome, tenant)

        return StreamHandle(iter_stream_deltas(chunks), result, on_done)

    # ------------------------------------------------------------------ #
    def _record(
        self,
        result: ChatResult,
        outcome: str,
        tenant: str,
        error: BaseException | None = None,
    ) -> None:
        self.metrics.inc(
            "llmbox_requests_total",
            labels={"outcome": outcome, "model": self.model},
            help_text="Chat requests by outcome.",
        )
        self.metrics.observe(
            "llmbox_request_duration_seconds",
            result.latency_ms / 1000.0,
            labels={"outcome": outcome},
            help_text="End-to-end request latency.",
        )
        if result.prompt_tokens:
            self.metrics.inc(
                "llmbox_prompt_tokens_total",
                value=result.prompt_tokens,
                help_text="Prompt tokens admitted.",
            )
        if result.completion_tokens:
            self.metrics.inc(
                "llmbox_completion_tokens_total",
                value=result.completion_tokens,
                help_text="Completion tokens generated.",
            )
        self.logger.emit(
            "chat_request",
            level=logging.WARNING if error else logging.INFO,
            request_id=result.request_id,
            outcome=outcome,
            tenant=tenant,
            model=self.model,
            cached=result.cached,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=round(result.latency_ms, 2),
            attempts=result.attempts,
            dropped_messages=result.dropped_messages,
            truncated=result.truncated,
            redactions=result.redactions or None,
            injection_score=round(result.injection_score, 3) or None,
            error=str(error) if error else None,
        )


def _outcome_for(exc: BaseException) -> str:
    return {
        "RateLimitExceeded": "rate_limited",
        "CircuitOpen": "circuit_open",
        "ContextOverflow": "context_overflow",
        "GuardrailBlocked": "guardrail_blocked",
        "StreamInterrupted": "stream_interrupted",
        "UpstreamError": "upstream_error",
    }.get(type(exc).__name__, "error")
