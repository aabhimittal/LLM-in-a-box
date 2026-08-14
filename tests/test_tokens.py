"""Token estimation, with emphasis on the scripts that break naive heuristics."""

from __future__ import annotations

import pytest

from llmbox.tokens import (
    PER_MESSAGE_OVERHEAD,
    REPLY_PRIMING_OVERHEAD,
    count_message_tokens,
    estimate_tokens,
)


def test_empty_text_costs_nothing():
    assert estimate_tokens("") == 0


def test_non_empty_text_always_costs_at_least_one_token():
    for text in ("a", " ", ".", "\n", "é"):
        assert estimate_tokens(text) >= 1


def test_english_is_roughly_four_characters_per_token():
    text = "the quick brown fox jumps over the lazy dog " * 10
    estimate = estimate_tokens(text)
    naive = len(text) / 4
    # Within 2x of the classic rule of thumb for plain English.
    assert naive / 2 <= estimate <= naive * 2


@pytest.mark.parametrize(
    "text",
    [
        "人工知能は世界を変える",  # Japanese
        "机器学习模型服务",  # Chinese
        "인공지능 모델 서빙",  # Korean
    ],
)
def test_cjk_is_not_undercounted(text):
    """The len/4 rule undercounts CJK by 3-5x; ours must not."""
    naive = max(1, len(text) // 4)
    estimate = estimate_tokens(text)
    assert estimate > naive * 2
    # Roughly one token per ideograph, allowing for spaces.
    assert estimate >= len(text.replace(" ", ""))


def test_emoji_cost_more_than_one_token_each():
    assert estimate_tokens("🚀") >= 2
    # A ZWJ family sequence is several codepoints and several tokens.
    assert estimate_tokens("👨‍👩‍👧‍👦") >= 4


def test_estimate_is_monotonic_under_concatenation():
    a = "hello world"
    b = "and some more text here"
    assert estimate_tokens(a + " " + b) >= max(estimate_tokens(a), estimate_tokens(b))


def test_message_accounting_includes_framing_overhead():
    messages = [{"role": "user", "content": "hi"}]
    expected = PER_MESSAGE_OVERHEAD + estimate_tokens("hi") + REPLY_PRIMING_OVERHEAD
    assert count_message_tokens(messages) == expected


def test_empty_message_list_costs_nothing():
    assert count_message_tokens([]) == 0


def test_null_content_is_tolerated():
    """Tool-call messages legitimately carry ``content: null``."""
    assert count_message_tokens([{"role": "assistant", "content": None}]) == (
        PER_MESSAGE_OVERHEAD + REPLY_PRIMING_OVERHEAD
    )


def test_custom_role_is_charged():
    standard = count_message_tokens([{"role": "user", "content": "x"}])
    custom = count_message_tokens([{"role": "tool_result_handler", "content": "x"}])
    assert custom > standard
