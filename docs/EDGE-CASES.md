# Industrial edge cases

This document catalogues the production failure modes LLM-in-a-Box handles, why
each one matters on a single-GPU deployment, and where it is tested. Everything
here runs in CI with no GPU and no cluster (`make test`, ~2s, 172 tests).

The implementation lives in [`llmbox/`](../llmbox); the scenario tests are in
[`tests/test_edge_cases.py`](../tests/test_edge_cases.py).

---

## 1. Cold starts

vLLM is unavailable for **minutes** after a pod starts while it downloads and
loads weights — unlike a typical HTTP backend, which is either up or down.

| Failure | Behaviour | Test |
| --- | --- | --- |
| Server returns 503 during load | Retried with exponential backoff + full jitter | `test_cold_start_is_absorbed_by_retries` |
| Load takes longer than the retry budget | Circuit opens; requests fail fast instead of deepening the queue | `test_prolonged_cold_start_trips_the_breaker_instead_of_hammering` |
| Server finishes loading | A half-open probe succeeds and the circuit closes | `test_breaker_recovers_when_the_server_finishes_loading` |

Retrying into a loading server is actively harmful: each retry lands in the same
queue the server is trying to drain. The breaker converts a long outage into
cheap, immediate failures — 50 shed requests generate **zero** backend calls.

**Full jitter** (`delay = random() * min(cap, base·2ⁿ)`) rather than fixed
backoff, so multiple UI replicas do not resynchronise into a herd on recovery.

## 2. Runaway conversations

The most common chat front-end bug: history accumulates until the prompt exceeds
`max_model_len`, after which *every* subsequent turn fails with a 400 and the
session is permanently broken.

`ContextBudget.fit()` guarantees the prompt fits, and preserves:

* the **system prompt** (truncated only as a last resort, and never to the point
  of starving the user's actual question);
* the **most recent turns** in preference to older ones;
* a history that **never starts with a dangling assistant turn** — several chat
  templates reject that outright.

| Failure | Behaviour | Test |
| --- | --- | --- |
| 200-turn conversation | Always fits; system prompt survives every trim | `test_two_hundred_turn_conversation_never_overflows` |
| User pastes a 20k-line log | Head-and-tail truncation, not rejection | `test_a_single_pasted_document_is_truncated_not_rejected` |
| System prompt larger than the window | Truncated, but reserves room for the newest turn | `test_system_prompt_is_truncated_only_as_a_last_resort` |
| `max_output_tokens` ≥ window | Rejected at construction with a clear error | `test_output_reservation_consuming_the_window_is_rejected` |

A randomised invariant test (`test_result_always_fits_the_input_budget`) checks
200 generated conversations: the result always fits, and the reported token count
always matches independent accounting.

### Tokenization is script-aware

`len(text) // 4` is calibrated on English and **undercounts CJK by 3-5x** and
emoji by 2-4x. In production that shows up as sporadic context-overflow errors on
exactly the prompts least likely to appear in testing.

| Input | Naive `len/4` | This estimator |
| --- | --- | --- |
| `"人工知能は世界を変える"` | 2 | ≥ 11 |
| `"👨‍👩‍👧‍👦"` | 1 | ≥ 4 |

Tested in `test_tokens.py`; end-to-end in
`test_multilingual_prompt_is_budgeted_by_script_not_character_count`.

Truncation is also **grapheme-safe** — cutting inside a ZWJ emoji sequence or
before a combining mark produces mojibake in the UI
(`test_truncation_never_splits_a_grapheme_cluster`).

## 3. Load: herds, whales and saturation

### Thundering herd

When a link is shared in a chat room, N users submit the same prompt within
seconds of each other and a naive cache issues N identical generations.
`SingleFlight` collapses them: **32 concurrent users → 1 generation**
(`test_thundering_herd_on_a_cold_cache_causes_one_generation`).

### Rate limiting priced in tokens, not requests

A 6,000-token prompt costs roughly 60× a 100-token one. A request-count limit
lets a handful of huge prompts monopolise the GPU while appearing compliant.
The limiter charges `prompt_tokens + max_tokens`:

```
10 small prompts  →  admitted     (a request-count limit of 10 would be at its cap)
1 whale prompt    →  refused      (it alone costs more than the 10 combined)
```

Tested in `test_token_pricing_stops_a_whale_that_request_counting_would_miss`.

### Other admission-control cases

| Failure | Behaviour | Test |
| --- | --- | --- |
| One tenant exhausts its budget | Other tenants unaffected | `test_one_abusive_tenant_does_not_starve_others` |
| Client needs to know when to retry | `retry_after` is accurate — waiting exactly that long succeeds | `test_backpressure_is_reported_with_an_actionable_retry_after` |
| Request costs more than the bucket capacity | Rejected immediately with `retry_after=None`, never "wait forever" | `test_request_larger_than_capacity_is_never_satisfiable` |
| Attacker-controlled tenant keys | Bounded LRU map; 5,000 keys → 100 retained | `test_tenant_map_is_bounded` |
| Tenant floods to reset its own bucket | Stays most-recently-used, so it stays throttled | `test_actively_throttled_tenant_survives_key_flooding` |
| Clock moves backwards | Clamped; never credits tokens | `test_clock_moving_backwards_does_not_credit_tokens` |
| 20 threads racing on one bucket | Never oversells — exactly capacity granted | `test_concurrent_acquire_never_oversells` |

> **Documented trade-off.** A bounded tenant map means a tenant idle long enough
> to be evicted receives a fresh bucket. That is the price of being memory-safe
> against attacker-chosen keys. It is asserted explicitly in
> `test_eviction_resets_an_idle_tenant_by_design` so it cannot regress silently;
> mitigate by keying on authenticated identities and sizing `max_tenants` above
> the real user population.

## 4. Caching correctness

Only **deterministic** (`temperature=0`) requests are cached by default. Caching
a sampled completion and replaying it to every user silently destroys sampling
diversity — a bug that presents as "the model keeps repeating itself" and is very
hard to trace back to the cache (`test_sampled_requests_are_never_cached`).

Cache keys cover the model, every message *including its role*, and all sampling
parameters. Message boundaries are unambiguous, so
`["ab", "c"]` and `["a", "bc"]` do not collide
(`test_message_boundaries_are_not_ambiguous`). Empty completions are cached as
real results rather than treated as misses.

Seeding is **not** treated as deterministic: vLLM's per-request seeding does not
survive continuous-batching rearrangement in all versions.

## 5. Hostile and malformed input

### Prompt injection

Scored rather than binary, so operators can tune against their own false-positive
tolerance. Input is normalised before matching, which defeats two common evasions:

* **Zero-width characters** splitting keywords (`ig​no​re all pre​vious…`) —
  stripped before matching, and their presence is itself a signal.
* **Fullwidth homoglyphs** (`ｉｇｎｏｒｅ ａｌｌ…`) — NFKC-normalised.
* **Unicode tag characters** (U+E0000 block) can encode an entire hidden ASCII
  payload; stripped (`test_unicode_tag_smuggling_is_stripped`).
* **Bidi overrides** (Trojan-Source style) — treated as invisible.
* **Chat-template tokens** (`<|im_start|>`, `[INST]`, `<<SYS>>`) let a user forge
  a system turn inside their message; weighted highest.

False positives are the real cost of guardrails, so benign prompts that merely
*mention* these concepts are explicitly tested as allowed —
`"How should I defend an LLM gateway against prompt injection?"` passes
(`test_legitimate_security_question_is_not_blocked`).

### PII redaction

Redaction runs **before logging**, not before the model: on self-hosted weights
the model should see the real prompt, while logs and traces should not. Set
`redact_upstream=True` to strip PII from the prompt itself.

Two details that are easy to get wrong:

* **Luhn validation.** Without it, any 16-digit order number or tracking id is
  flagged as a credit card. `4111111111111111` redacts; `4111111111111112`
  (same shape, bad checksum) does not.
* **Overlap resolution.** Naive sequential `re.sub` computes replacement offsets
  against the original string and mangles later matches. Matches are resolved
  into a non-overlapping set first, then the string is rebuilt once. Redaction is
  idempotent and preserves surrounding Unicode exactly.

Version strings (`vLLM 0.5.4`) are not mistaken for phone numbers; IPs beat the
phone pattern on tie-break.

### Malformed responses

Never let a response-shape surprise reach the UI as an `AttributeError`:

| Response | Behaviour | Test |
| --- | --- | --- |
| `{"choices": []}` (abort/preemption) | Retryable `UpstreamError` | `test_malformed_responses_surface_as_upstream_errors` |
| `content: null` (tool-call reply) | Empty string, no crash | `test_tool_call_reply_with_null_content_is_handled` |
| Dict *or* SDK object | Both parsed identically | `test_extract_text_from_dict_and_object` |

### Retry classification

Retrying a request that can only ever fail multiplies the damage:

* `400 context length exceeded` → **not** retried (`test_context_length_error_from_the_server_is_not_retried`)
* `500 CUDA out of memory` → retried; often succeeds on the next scheduling pass
* `CircuitOpen` → not retried; the breaker already decided

## 6. Streaming under failure

| Failure | Behaviour | Test |
| --- | --- | --- |
| Connection drops mid-generation | `StreamInterrupted` carries `partial`; the UI shows the partial answer instead of losing it | `test_stream_interruption_preserves_partial_output` |
| Role-only / finish-reason / usage-only chunks | Skipped, never rendered as text | `test_stream_normalisation_skips_non_content_chunks` |
| `delta` absent or `None` | Tolerated | `test_stream_tolerates_missing_delta_and_empty_choices` |
| User navigates away mid-stream | No exception | `test_consumer_abandoning_a_stream_does_not_raise` |
| Model emits a stop token first | Empty result is valid, not an error | `test_empty_generation_is_a_valid_result` |

Admission control and circuit breaking apply **before the first token**, so a
rate-limited stream never reaches the backend
(`test_stream_admission_control_applies_before_first_token`).

## 7. Telemetry integrity

Structured logs carry ids, counts and outcomes — **never prompt or completion
text** (`test_prompt_text_never_reaches_the_logs`). Fields are length-capped so a
100k-character prompt cannot overwhelm the log pipeline.

The Prometheus exposition output is validated for well-formedness after mixed
traffic including cache hits, rate-limit rejections and guardrail blocks
(`test_metrics_exposition_is_well_formed_after_mixed_traffic`).

## 8. Cluster-level disruption

Handled in manifests rather than code:

* **PodDisruptionBudget** — a node drain would otherwise evict vLLM and cause a
  multi-minute outage while weights reload.
* **NetworkPolicy** — without it, any pod can call the inference API directly,
  bypassing guardrails, rate limits and audit logging entirely. (Note: k3s ships
  Flannel, which does not enforce NetworkPolicy; a policy-capable CNI is
  required for these rules to take effect.)
* **Alerts** on queue depth, KV-cache saturation and p95 time-to-first-token —
  the metrics that actually predict user-visible pain, in
  `k8s/overlays/observability/`.

---

## Two bugs this suite caught during development

Recorded because they are the kind that survive code review and fail in
production:

1. **`if self.limiter:` silently disabled rate limiting.** `RateLimiter` defines
   `__len__`, so an empty limiter is falsy. Every rate-limit test failed at once;
   the fix was `is not None` on every optional component.
2. **Truncating an oversized system prompt starved the user's question.** The
   system prompt consumed the entire window, leaving no room for the actual
   message, and the request failed with `ContextOverflow`. Now a reserve is held
   back for the newest turn, capped at half the window so a huge newest message
   cannot starve the system prompt either.
