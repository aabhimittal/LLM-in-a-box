"""Context-window budgeting.

The single most common production failure for a chat front-end is unbounded
history growth: the conversation quietly accumulates until the prompt exceeds
``max_model_len`` and *every* subsequent turn fails with a 400. This module
makes the prompt fit, deterministically, before the request is ever sent.

Guarantees provided by :meth:`ContextBudget.fit`:

* the system prompt is preserved (truncated only as a last resort);
* the most recent turns are preserved in preference to older ones;
* the returned history never begins with a dangling ``assistant`` turn, which
  several chat templates reject outright;
* a single oversized message is truncated head-and-tail rather than causing the
  whole request to fail (configurable).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .errors import ContextOverflow
from .tokens import PER_MESSAGE_OVERHEAD, REPLY_PRIMING_OVERHEAD, estimate_tokens

Message = Mapping[str, str]

ELISION = "\n\n…[{n} characters elided]…\n\n"


@dataclass
class FitResult:
    """Outcome of fitting a conversation into the context window."""

    messages: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    input_budget: int = 0
    dropped_messages: int = 0
    truncated: bool = False

    @property
    def modified(self) -> bool:
        """True when the conversation had to be altered to fit."""
        return self.dropped_messages > 0 or self.truncated


def _is_joiner(ch: str) -> bool:
    """Characters that must not be separated from the grapheme they modify."""
    # Combining marks, zero-width joiner, variation selectors, regional
    # indicators (flags) and skin-tone modifiers.
    if unicodedata.combining(ch):
        return True
    cp = ord(ch)
    return (
        cp == 0x200D
        or 0xFE00 <= cp <= 0xFE0F
        or 0x1F3FB <= cp <= 0x1F3FF
        or 0x1F1E6 <= cp <= 0x1F1FF
    )


def _safe_head(text: str, idx: int) -> int:
    """Move ``idx`` backwards so ``text[:idx]`` never splits a grapheme."""
    idx = max(0, min(idx, len(text)))
    while 0 < idx < len(text) and _is_joiner(text[idx]):
        idx -= 1
    return idx


def _safe_tail(text: str, idx: int) -> int:
    """Move ``idx`` forwards so ``text[idx:]`` never starts mid-grapheme."""
    idx = max(0, min(idx, len(text)))
    while 0 < idx < len(text) and _is_joiner(text[idx]):
        idx += 1
    return idx


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    counter: Callable[[str], int] = estimate_tokens,
) -> str:
    """Shrink ``text`` until it costs at most ``max_tokens``.

    Keeps the head and the tail (instructions usually live at the front, the
    actual question at the back) and elides the middle. Uses binary search over
    the kept-character count so it works with any counter implementation,
    including exact tokenizers with non-linear behaviour.
    """
    if max_tokens <= 0:
        return ""
    if counter(text) <= max_tokens:
        return text

    lo, hi = 0, len(text) // 2
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        head_end = _safe_head(text, mid)
        tail_start = _safe_tail(text, len(text) - mid)
        if tail_start <= head_end:
            candidate = text[:head_end]
        else:
            elided = tail_start - head_end
            candidate = text[:head_end] + ELISION.format(n=elided) + text[tail_start:]
        if counter(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    # Even a zero-character keep did not fit (the elision marker alone is too
    # large); fall back to a hard character cut.
    if not best:
        for size in (len(text) // 2, 64, 16, 4, 1):
            candidate = text[: _safe_head(text, size)]
            if candidate and counter(candidate) <= max_tokens:
                return candidate
        return ""
    return best


class ContextBudget:
    """Fits a conversation into a model's context window.

    Args:
        max_model_len: The server's ``--max-model-len``.
        max_output_tokens: Tokens reserved for the completion.
        counter: Token counting function.
        safety_margin: Extra head-room absorbing estimator error and chat
            template differences. Never set this to 0 with a heuristic counter.
        on_overflow: ``"truncate"`` (default) shrinks an oversized newest
            message; ``"error"`` raises :class:`ContextOverflow` instead.
    """

    def __init__(
        self,
        max_model_len: int,
        max_output_tokens: int,
        counter: Callable[[str], int] = estimate_tokens,
        safety_margin: int = 32,
        on_overflow: str = "truncate",
    ):
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if max_output_tokens < 0:
            raise ValueError("max_output_tokens must not be negative")
        if on_overflow not in ("truncate", "error"):
            raise ValueError("on_overflow must be 'truncate' or 'error'")
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.counter = counter
        self.safety_margin = safety_margin
        self.on_overflow = on_overflow

    @property
    def input_budget(self) -> int:
        """Tokens available for the prompt after reserving room for the reply."""
        return self.max_model_len - self.max_output_tokens - self.safety_margin

    def _cost(self, content: str) -> int:
        return PER_MESSAGE_OVERHEAD + self.counter(content)

    def fit(self, history: Sequence[Message], system: str | None = None) -> FitResult:
        """Return the largest recent suffix of ``history`` that fits."""
        budget = self.input_budget
        if budget <= REPLY_PRIMING_OVERHEAD:
            raise ContextOverflow(
                "no room for any prompt: max_output_tokens + safety_margin "
                f"consume the entire {self.max_model_len}-token window",
                required=REPLY_PRIMING_OVERHEAD,
                available=budget,
            )
        budget -= REPLY_PRIMING_OVERHEAD

        result = FitResult(input_budget=self.input_budget)
        kept: list[dict] = []
        used = 0

        # 1. The system prompt is non-negotiable, but must itself fit.
        if system:
            cost = self._cost(system)
            if cost > budget:
                if self.on_overflow == "error":
                    raise ContextOverflow(
                        "system prompt alone exceeds the context window",
                        required=cost,
                        available=budget,
                    )
                # Reserve room for the newest turn: a system prompt that
                # consumes the whole window leaves the model with no question
                # to answer. Cap the reserve at half the window so a huge
                # newest message cannot starve the system prompt either.
                newest_cost = (
                    self._cost(history[-1].get("content") or "") if history else 0
                )
                reserve = min(newest_cost, budget // 2)
                shrunk = truncate_to_tokens(
                    system, max(0, budget - reserve - PER_MESSAGE_OVERHEAD), self.counter
                )
                if not shrunk:
                    raise ContextOverflow(
                        "system prompt cannot be truncated small enough to fit",
                        required=cost,
                        available=budget,
                    )
                system = shrunk
                cost = self._cost(system)
                result.truncated = True
            kept.append({"role": "system", "content": system})
            used += cost

        remaining = budget - used

        # 2. Walk newest -> oldest, keeping what fits.
        tail: list[dict] = []
        for message in reversed(list(history)):
            content = message.get("content") or ""
            cost = self._cost(content)
            if cost <= remaining:
                tail.append({"role": message.get("role", "user"), "content": content})
                remaining -= cost
            else:
                break

        dropped = len(history) - len(tail)
        tail.reverse()

        # 3. Nothing fit, but there was history: the newest message is oversized.
        if history and not tail:
            newest = history[-1]
            if self.on_overflow == "error":
                raise ContextOverflow(
                    "the most recent message alone exceeds the context window",
                    required=self._cost(newest.get("content") or ""),
                    available=remaining,
                )
            shrunk = truncate_to_tokens(
                newest.get("content") or "",
                max(0, remaining - PER_MESSAGE_OVERHEAD),
                self.counter,
            )
            if not shrunk:
                raise ContextOverflow(
                    "cannot fit even a truncated message into the context window",
                    required=PER_MESSAGE_OVERHEAD,
                    available=remaining,
                )
            tail = [{"role": newest.get("role", "user"), "content": shrunk}]
            dropped = len(history) - 1
            result.truncated = True
            remaining -= self._cost(shrunk)

        # 4. Never begin with a dangling assistant turn.
        while tail and tail[0]["role"] == "assistant":
            removed = tail.pop(0)
            remaining += self._cost(removed["content"])
            dropped += 1

        kept.extend(tail)
        result.messages = kept
        result.dropped_messages = dropped
        # An empty prompt costs nothing; otherwise charge what was consumed plus
        # the reply priming, matching ``count_message_tokens`` exactly.
        result.prompt_tokens = (
            (budget + REPLY_PRIMING_OVERHEAD) - remaining if kept else 0
        )
        return result
