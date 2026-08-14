"""Token accounting.

Exact token counts require the model's own tokenizer, which is a heavyweight
dependency we do not want in the request path of a UI pod. This module provides
a deliberately *conservative* script-aware estimator that never has to be right,
only has to avoid under-counting badly enough to overflow the context window.

If a real tokenizer is available it is used instead — see :func:`make_counter`.

Why not simply ``len(text) // 4``? Because that rule is calibrated on English.
CJK text tokenizes at roughly one token per character and emoji frequently cost
2-4 tokens each, so the naive rule under-counts such prompts by 3-5x, which in
production shows up as sporadic ``400 context length exceeded`` errors on
exactly the inputs least likely to appear in testing.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Callable, Iterable, Mapping

# Chat framing costs: each message carries role/delimiter tokens, and the
# response is primed with a few more. Mirrors OpenAI's documented accounting and
# is close enough for Llama-style chat templates.
PER_MESSAGE_OVERHEAD = 4
REPLY_PRIMING_OVERHEAD = 3

_WORD_RE = re.compile(r"\S+")

# Latin-ish text averages ~4 characters per token.
_CHARS_PER_TOKEN = 4.0
# Wide scripts (CJK, Hangul, Kana) average ~1 token per character.
_WIDE_TOKEN_COST = 1.0
# Astral-plane codepoints (most emoji) commonly cost 2+ tokens each.
_ASTRAL_TOKEN_COST = 2.0
# Miscellaneous symbols/dingbats also tokenize poorly.
_SYMBOL_TOKEN_COST = 1.0


def _is_wide(cp: int) -> bool:
    """True for East-Asian wide/fullwidth characters."""
    try:
        return unicodedata.east_asian_width(chr(cp)) in ("W", "F")
    except ValueError:  # pragma: no cover - chr() always valid for our inputs
        return False


def _is_symbol(cp: int) -> bool:
    """True for BMP symbol/dingbat ranges that tokenize like standalone units."""
    return 0x2190 <= cp <= 0x2BFF


def estimate_tokens(text: str) -> int:
    """Return a conservative token estimate for ``text``.

    The estimate is script-aware and is never less than 1 for non-empty input.
    """
    if not text:
        return 0

    total = 0.0
    for word in _WORD_RE.findall(text):
        plain = 0
        for ch in word:
            cp = ord(ch)
            if cp >= 0x10000:
                total += _ASTRAL_TOKEN_COST
            elif _is_wide(cp):
                total += _WIDE_TOKEN_COST
            elif _is_symbol(cp):
                total += _SYMBOL_TOKEN_COST
            else:
                plain += 1
        if plain:
            # Never let a word cost less than one token.
            total += max(1.0, plain / _CHARS_PER_TOKEN)

    # Newlines survive tokenization as their own units often enough to matter in
    # long structured prompts (code, markdown tables).
    total += text.count("\n") * 0.5

    return max(1, int(math.ceil(total)))


def make_counter() -> Callable[[str], int]:
    """Return the best available token counter.

    Prefers ``tiktoken`` when installed (exact for OpenAI-family BPE and a good
    proxy for Llama), otherwise falls back to :func:`estimate_tokens`. Resolved
    once at construction so the hot path never pays for the import check.
    """
    try:  # pragma: no cover - depends on optional dependency being installed
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(encoding.encode(text, disallowed_special=()))

        return _count
    except Exception:
        return estimate_tokens


def count_message_tokens(
    messages: Iterable[Mapping[str, str]],
    counter: Callable[[str], int] = estimate_tokens,
) -> int:
    """Estimate the prompt token cost of a chat ``messages`` list.

    Includes per-message framing overhead and reply priming, so the result is
    directly comparable against a model's context window.
    """
    total = 0
    count = 0
    for message in messages:
        count += 1
        total += PER_MESSAGE_OVERHEAD
        content = message.get("content") or ""
        total += counter(content)
        # A non-default role name costs a token or two of its own.
        role = message.get("role") or ""
        if role not in ("user", "assistant", "system"):
            total += counter(role)
    if count:
        total += REPLY_PRIMING_OVERHEAD
    return total
