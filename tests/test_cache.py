"""Response caching, cache-key hygiene and stampede protection."""

from __future__ import annotations

import threading
import time

import pytest

from llmbox.cache import PromptCache, SingleFlight, is_deterministic, make_key

MESSAGES = [{"role": "user", "content": "what is k3s?"}]


def test_hit_and_miss(clock):
    cache = PromptCache(clock=clock)
    assert cache.get("k") is None
    cache.put("k", "value")
    assert cache.get("k") == "value"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


def test_entries_expire(clock):
    cache = PromptCache(ttl_seconds=10, clock=clock)
    cache.put("k", "value")
    clock.advance(9.9)
    assert cache.get("k") == "value"
    clock.advance(0.2)
    assert cache.get("k") is None
    assert cache.stats.expirations == 1
    assert len(cache) == 0  # expired entry is dropped, not merely hidden


def test_lru_eviction_order(clock):
    cache = PromptCache(max_entries=2, clock=clock)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # promote "a"
    cache.put("c", 3)  # evicts "b"
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.stats.evictions == 1


def test_overwriting_a_key_does_not_grow_the_cache(clock):
    cache = PromptCache(max_entries=2, clock=clock)
    for _ in range(10):
        cache.put("same", "v")
    assert len(cache) == 1


def test_falsy_values_are_cached_correctly(clock):
    """An empty completion is a real result, not a miss."""
    cache = PromptCache(clock=clock)
    cache.put("k", "")
    assert cache.get("k", default=object()) == ""


# --------------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------------- #


def test_key_is_stable_across_dict_ordering():
    a = make_key("m", MESSAGES, temperature=0, max_tokens=100, top_p=1)
    b = make_key("m", MESSAGES, top_p=1, max_tokens=100, temperature=0)
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "other-model"},
        {"temperature": 0.5},
        {"max_tokens": 200},
        {"top_p": 0.9},
    ],
)
def test_key_changes_when_any_input_changes(kwargs):
    base = dict(model="m", temperature=0.0, max_tokens=100, top_p=1.0)
    merged = {**base, **kwargs}
    a = make_key(base.pop("model"), MESSAGES, **base)
    b = make_key(merged.pop("model"), MESSAGES, **merged)
    assert a != b


def test_key_distinguishes_roles_not_just_text():
    """Same words, different speaker, different meaning — must not collide."""
    a = make_key("m", [{"role": "user", "content": "hello"}], temperature=0)
    b = make_key("m", [{"role": "assistant", "content": "hello"}], temperature=0)
    assert a != b


def test_key_handles_unicode_and_null_content():
    key = make_key("m", [{"role": "user", "content": "日本語 🚀"}, {"role": "assistant"}], temperature=0)
    assert len(key) == 64


def test_message_boundaries_are_not_ambiguous():
    """Concatenation must not make two different conversations hash alike."""
    a = make_key("m", [{"role": "user", "content": "ab"}, {"role": "user", "content": "c"}])
    b = make_key("m", [{"role": "user", "content": "a"}, {"role": "user", "content": "bc"}])
    assert a != b


# --------------------------------------------------------------------------- #
# Determinism policy
# --------------------------------------------------------------------------- #


def test_only_deterministic_requests_are_cacheable(clock):
    cache = PromptCache(clock=clock)
    assert cache.cacheable({"temperature": 0.0})
    assert not cache.cacheable({"temperature": 0.7})
    assert not cache.cacheable({})  # unspecified temperature defaults to sampling


def test_opt_in_allows_caching_sampled_responses(clock):
    cache = PromptCache(clock=clock, cache_nondeterministic=True)
    assert cache.cacheable({"temperature": 0.7})


def test_is_deterministic_rejects_seeded_sampling():
    """Seeding is not reliable under continuous batching, so it is not enough."""
    assert not is_deterministic({"temperature": 0.7, "seed": 42})
    assert is_deterministic({"temperature": 0})


# --------------------------------------------------------------------------- #
# Stampede protection
# --------------------------------------------------------------------------- #


def test_single_flight_collapses_concurrent_identical_calls():
    flight = SingleFlight()
    executions = []
    started = threading.Event()
    release = threading.Event()
    results = []

    def slow():
        executions.append(1)
        started.set()
        release.wait(timeout=5)
        return "generated"

    def worker():
        results.append(flight.do("same-key", slow))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    started.wait(timeout=5)
    time.sleep(0.05)  # let the followers pile up behind the leader
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert len(executions) == 1  # 16 users, one generation
    assert results == ["generated"] * 16
    assert flight.in_flight == 0


def test_single_flight_propagates_failure_to_all_waiters():
    flight = SingleFlight()
    started = threading.Event()
    release = threading.Event()
    errors = []

    def failing():
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("upstream exploded")

    def worker():
        try:
            flight.do("k", failing)
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    started.wait(timeout=5)
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert errors == ["upstream exploded"] * 8
    assert flight.in_flight == 0  # no leaked in-flight entry after failure


def test_single_flight_keys_are_independent():
    flight = SingleFlight()
    executions = []
    for key in ("a", "b", "a"):
        flight.do(key, lambda: executions.append(key))
    assert len(executions) == 3  # sequential calls do not share results


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        PromptCache(max_entries=0)
    with pytest.raises(ValueError):
        PromptCache(ttl_seconds=0)
