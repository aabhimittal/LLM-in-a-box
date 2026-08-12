"""Context-window budgeting — the failure mode that kills long chat sessions."""

from __future__ import annotations

import random

import pytest

from llmbox.context import ContextBudget, truncate_to_tokens
from llmbox.errors import ContextOverflow
from llmbox.tokens import count_message_tokens, estimate_tokens


def budget(**kwargs) -> ContextBudget:
    defaults = dict(max_model_len=200, max_output_tokens=50, safety_margin=0)
    defaults.update(kwargs)
    return ContextBudget(**defaults)


def test_empty_history_produces_empty_prompt():
    fit = budget().fit([])
    assert fit.messages == []
    assert fit.prompt_tokens == 0
    assert not fit.modified


def test_short_conversation_is_untouched():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you?"},
    ]
    fit = budget().fit(history, system="You are helpful.")
    assert fit.messages[0]["role"] == "system"
    assert [m["content"] for m in fit.messages[1:]] == [m["content"] for m in history]
    assert not fit.modified


def test_reported_prompt_tokens_match_independent_accounting():
    history = [{"role": "user", "content": "hello there friend"}]
    fit = budget().fit(history, system="Be brief.")
    assert fit.prompt_tokens == count_message_tokens(fit.messages)


def test_oldest_turns_are_dropped_first():
    history = [{"role": "user", "content": f"message number {i} " * 5} for i in range(40)]
    fit = budget(max_model_len=200, max_output_tokens=50).fit(history)
    assert fit.dropped_messages > 0
    # Whatever survived must be the most recent suffix.
    survivors = [m["content"] for m in fit.messages]
    assert survivors == [m["content"] for m in history[-len(survivors):]]


def test_result_always_fits_the_input_budget():
    """The core invariant: fit() output never exceeds the window."""
    rng = random.Random(1234)
    for _ in range(200):
        turns = rng.randint(0, 30)
        history = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": "word " * rng.randint(1, 60),
            }
            for i in range(turns)
        ]
        system = "sys " * rng.randint(0, 20) or None
        limit = rng.choice([64, 128, 256, 512])
        b = ContextBudget(max_model_len=limit, max_output_tokens=limit // 4, safety_margin=8)
        fit = b.fit(history, system=system)
        assert count_message_tokens(fit.messages) <= b.input_budget
        assert fit.prompt_tokens == count_message_tokens(fit.messages)


def test_history_never_starts_with_a_dangling_assistant_turn():
    """Several chat templates reject a leading assistant message outright."""
    history = [
        {"role": "assistant", "content": "stale reply " * 30},
        {"role": "assistant", "content": "another " * 30},
        {"role": "user", "content": "the real question"},
    ]
    fit = budget(max_model_len=120, max_output_tokens=30).fit(history)
    non_system = [m for m in fit.messages if m["role"] != "system"]
    assert non_system
    assert non_system[0]["role"] == "user"


def test_single_oversized_message_is_truncated_not_rejected():
    history = [{"role": "user", "content": "x " * 5000}]
    fit = budget(max_model_len=200, max_output_tokens=50).fit(history)
    assert fit.truncated
    assert len(fit.messages) == 1
    assert count_message_tokens(fit.messages) <= budget().input_budget


def test_single_oversized_message_raises_in_strict_mode():
    history = [{"role": "user", "content": "x " * 5000}]
    with pytest.raises(ContextOverflow):
        budget(on_overflow="error").fit(history)


def test_system_prompt_is_truncated_only_as_a_last_resort():
    b = budget(max_model_len=120, max_output_tokens=20)
    fit = b.fit([{"role": "user", "content": "hi"}], system="rules " * 500)
    assert fit.truncated
    assert fit.messages[0]["role"] == "system"
    # Truncating the system prompt must not starve the actual question: a
    # prompt with no user turn is a request the model cannot answer.
    assert fit.messages[-1]["role"] == "user"
    assert fit.messages[-1]["content"] == "hi"
    assert count_message_tokens(fit.messages) <= b.input_budget


def test_oversized_system_prompt_raises_in_strict_mode():
    with pytest.raises(ContextOverflow):
        budget(on_overflow="error").fit([], system="rules " * 500)


def test_output_reservation_consuming_the_window_is_rejected():
    """A misconfiguration that would otherwise fail confusingly at request time."""
    b = ContextBudget(max_model_len=100, max_output_tokens=100, safety_margin=0)
    with pytest.raises(ContextOverflow) as excinfo:
        b.fit([{"role": "user", "content": "hi"}])
    assert "entire" in str(excinfo.value)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        ContextBudget(max_model_len=0, max_output_tokens=10)
    with pytest.raises(ValueError):
        ContextBudget(max_model_len=100, max_output_tokens=-1)
    with pytest.raises(ValueError):
        ContextBudget(max_model_len=100, max_output_tokens=10, on_overflow="explode")


# --------------------------------------------------------------------------- #
# truncate_to_tokens
# --------------------------------------------------------------------------- #


def test_truncation_keeps_head_and_tail():
    text = "START " + ("filler " * 500) + "END"
    out = truncate_to_tokens(text, 60)
    assert out.startswith("START")
    assert out.endswith("END")
    assert estimate_tokens(out) <= 60


def test_truncation_is_a_no_op_when_it_already_fits():
    text = "short enough"
    assert truncate_to_tokens(text, 1000) == text


def test_truncation_to_zero_budget_returns_empty():
    assert truncate_to_tokens("anything", 0) == ""


def test_truncation_never_splits_a_grapheme_cluster():
    """Cutting inside a ZWJ emoji sequence produces mojibake in the UI."""
    text = "👨‍👩‍👧‍👦" * 200
    out = truncate_to_tokens(text, 40)
    assert estimate_tokens(out) <= 40
    # No dangling zero-width joiner at either seam.
    assert not out.endswith("‍")
    assert "‍" not in (out[:1] + out[-1:])


def test_truncation_handles_combining_marks():
    text = ("é" * 400)  # e + combining acute
    out = truncate_to_tokens(text, 30)
    assert estimate_tokens(out) <= 30
    # A combining mark must never lead the result.
    assert not out.startswith("́")
