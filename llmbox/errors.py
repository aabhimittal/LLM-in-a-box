"""Exception hierarchy for the LLM-in-a-Box serving edge.

Every error carries enough structure for a caller to decide *what to do next*
(retry, shed, surface to the user) without string-matching on messages.
"""

from __future__ import annotations


class LLMBoxError(Exception):
    """Base class for every error raised by this package."""


class RateLimitExceeded(LLMBoxError):
    """The caller exhausted its token bucket.

    ``retry_after`` is the number of seconds until the request would succeed,
    or ``None`` when the request can never succeed (cost exceeds capacity).
    """

    def __init__(self, message: str, retry_after: float | None = None, tenant: str | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.tenant = tenant


class CircuitOpen(LLMBoxError):
    """The circuit breaker is open; the upstream is presumed unhealthy."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ContextOverflow(LLMBoxError):
    """The request cannot be made to fit inside the model's context window."""

    def __init__(self, message: str, required: int | None = None, available: int | None = None):
        super().__init__(message)
        self.required = required
        self.available = available


class GuardrailBlocked(LLMBoxError):
    """A guardrail rejected the request before it reached the model."""

    def __init__(self, message: str, reasons: list[str] | None = None, score: float = 0.0):
        super().__init__(message)
        self.reasons = reasons or []
        self.score = score


class UpstreamError(LLMBoxError):
    """The model server failed.

    ``retryable`` distinguishes transient failures (connection resets, 503 while
    weights load, 429) from permanent ones (400 bad request, 401 auth), so retry
    logic never hammers a request that can only ever fail.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.cause = cause


class StreamInterrupted(UpstreamError):
    """A streaming response died part-way through.

    ``partial`` holds whatever text was successfully received, so callers can
    surface a partial answer instead of losing the whole generation.
    """

    def __init__(self, message: str, partial: str = "", cause: BaseException | None = None):
        super().__init__(message, retryable=True, cause=cause)
        self.partial = partial


class StreamStalled(StreamInterrupted):
    """A stream stopped producing tokens without closing.

    Distinct from :class:`StreamInterrupted`: nothing failed, the connection is
    simply idle. Left unchecked this hangs the caller forever, because a socket
    that is open but silent never raises.
    """

    def __init__(self, message: str, idle_seconds: float = 0.0, partial: str = ""):
        super().__init__(message, partial=partial)
        self.idle_seconds = idle_seconds


class Overloaded(LLMBoxError):
    """The request was shed to protect the model server.

    Raised when this process already has as many requests in flight as it is
    willing to hold (``source="bulkhead"``), or when the upstream's own queue is
    too deep to accept more work (``source="upstream_queue"``).
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        source: str = "bulkhead",
        depth: float | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.source = source
        self.depth = depth
