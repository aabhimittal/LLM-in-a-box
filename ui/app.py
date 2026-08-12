"""LLM-in-a-Box — Streamlit chat UI.

A chat front-end for a vLLM server exposing the OpenAI-compatible API. All the
production behaviour — context budgeting, token-priced rate limiting, response
caching, PII redaction, prompt-injection screening, circuit breaking and
retries — lives in the :mod:`llmbox` package and is exercised by its own test
suite; this module is just presentation.

Configuration (all optional except the endpoint):

    VLLM_BASE_URL              vLLM OpenAI endpoint (default http://vllm:8000/v1)
    VLLM_API_KEY               Bearer token; vLLM ignores it unless --api-key is set
    VLLM_MODEL                 Model to request; auto-discovered when unset
    APP_TITLE                  UI title

    LLMBOX_MAX_MODEL_LEN       Must match vLLM's --max-model-len (default 8192)
    LLMBOX_MAX_OUTPUT_TOKENS   Tokens reserved for the reply (default 512)
    LLMBOX_TOKENS_PER_MINUTE   Sustained per-user token budget (default 60000)
    LLMBOX_BURST_TOKENS        Per-user burst capacity (default 20000)
    LLMBOX_CACHE_TTL           Cache TTL in seconds (default 300)
    LLMBOX_CACHE_ENTRIES       Max cached responses (default 512)
    LLMBOX_INJECTION_THRESHOLD Block score in [0,1] (default 0.6)
"""

from __future__ import annotations

import logging
import os
import uuid

import streamlit as st
from openai import OpenAI, OpenAIError

from llmbox import (
    CircuitOpen,
    ContextOverflow,
    GuardrailBlocked,
    RateLimitExceeded,
    StreamInterrupted,
    UpstreamError,
    build_client,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
API_KEY = os.getenv("VLLM_API_KEY", "not-needed")
CONFIGURED_MODEL = os.getenv("VLLM_MODEL", "").strip()
APP_TITLE = os.getenv("APP_TITLE", "LLM-in-a-Box")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running fully on-premise, served by "
    "vLLM on a k3s cluster."
)


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)


@st.cache_resource(show_spinner=False)
def discover_model() -> str | None:
    """Resolve which model to request, preferring explicit configuration."""
    if CONFIGURED_MODEL:
        return CONFIGURED_MODEL
    try:
        models = get_openai_client().models.list()
        return models.data[0].id if models.data else None
    except OpenAIError:
        return None


@st.cache_resource(show_spinner=False)
def get_llm(model: str):
    """Build the resilient client once per process.

    ``cache_resource`` matters here: Streamlit re-runs this script top to bottom
    on every interaction, and the cache, rate-limiter buckets and circuit-breaker
    state must survive those re-runs to mean anything.
    """
    openai_client = get_openai_client()

    def backend(payload: dict, stream: bool):
        return openai_client.chat.completions.create(**payload)

    return build_client(
        backend=backend,
        model=model,
        max_model_len=_int_env("LLMBOX_MAX_MODEL_LEN", 8192),
        max_output_tokens=_int_env("LLMBOX_MAX_OUTPUT_TOKENS", 512),
        tokens_per_minute=_float_env("LLMBOX_TOKENS_PER_MINUTE", 60_000),
        burst_tokens=_float_env("LLMBOX_BURST_TOKENS", 20_000),
        cache_entries=_int_env("LLMBOX_CACHE_ENTRIES", 512),
        cache_ttl=_float_env("LLMBOX_CACHE_TTL", 300.0),
        injection_threshold=_float_env("LLMBOX_INJECTION_THRESHOLD", 0.6),
    )


def health() -> tuple[bool, str]:
    try:
        get_openai_client().models.list()
        return True, "connected"
    except OpenAIError as exc:
        return False, str(exc)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="centered")

# Each browser session is a rate-limiting tenant. Behind real auth, use the
# authenticated user id instead — a session id is trivially rotated.
if "tenant" not in st.session_state:
    st.session_state.tenant = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []

model = discover_model()

with st.sidebar:
    st.title(f"📦 {APP_TITLE}")
    st.caption("Open-source LLM served by vLLM on k3s.")

    healthy, detail = health()
    if healthy:
        st.success("vLLM server reachable")
    else:
        st.error("vLLM server unreachable")
        with st.expander("Details"):
            st.code(detail)

    if model:
        st.info(f"**Model:** `{model}`")
    else:
        st.warning("No model detected. Is vLLM still loading weights?")

    st.divider()
    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 64, 4096, 512, 64)
    top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
    st.caption("Temperature 0 makes the response cacheable.")

    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=120)

    if model:
        llm = get_llm(model)
        st.divider()
        st.subheader("Serving edge")
        stats = llm.cache.stats
        col_a, col_b = st.columns(2)
        col_a.metric("Cache hit rate", f"{stats.hit_rate:.0%}")
        col_b.metric("Circuit", llm.breaker.state.value)
        st.caption(
            f"cached entries: {len(llm.cache)} · hits {stats.hits} · misses {stats.misses}"
        )
        with st.expander("Prometheus metrics"):
            st.code(llm.metrics.render() or "(no samples yet)", language="text")

    st.divider()
    st.caption(f"Endpoint: `{BASE_URL}`")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask anything…")

if prompt:
    if not model:
        st.error("No model is available yet. Please wait for vLLM to finish loading.")
        st.stop()

    llm = get_llm(model)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        collected = ""
        try:
            handle = llm.stream(
                st.session_state.messages,
                system=system_prompt,
                tenant=st.session_state.tenant,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for delta in handle:
                collected += delta
                placeholder.markdown(collected + "▌")
            placeholder.markdown(collected)

            result = handle.result
            notes = []
            if result.dropped_messages:
                notes.append(f"trimmed {result.dropped_messages} older message(s) to fit the context window")
            if result.truncated:
                notes.append("truncated an oversized message")
            if result.redactions:
                kinds = ", ".join(f"{k.lower()} ×{v}" for k, v in result.redactions.items())
                notes.append(f"redacted from logs: {kinds}")
            if notes:
                st.caption("ℹ️ " + " · ".join(notes))

        except GuardrailBlocked as exc:
            placeholder.empty()
            st.warning(
                "This message was blocked by the input guardrail "
                f"(score {exc.score:.2f}: {', '.join(exc.reasons)})."
            )
            st.session_state.messages.pop()
            st.stop()
        except RateLimitExceeded as exc:
            placeholder.empty()
            wait = f" Try again in {exc.retry_after:.0f}s." if exc.retry_after else ""
            st.warning(f"You've hit the per-user token budget.{wait}")
            st.session_state.messages.pop()
            st.stop()
        except CircuitOpen as exc:
            placeholder.empty()
            st.error(
                "The model server is unhealthy and requests are being shed"
                + (f"; retrying in {exc.retry_after:.0f}s." if exc.retry_after else ".")
            )
            st.session_state.messages.pop()
            st.stop()
        except ContextOverflow as exc:
            placeholder.empty()
            st.error(f"That message cannot fit in the model's context window: {exc}")
            st.session_state.messages.pop()
            st.stop()
        except StreamInterrupted as exc:
            # Keep whatever tokens made it through rather than losing the answer.
            collected = exc.partial
            placeholder.markdown(collected)
            st.warning("The connection dropped mid-response; the answer above is partial.")
        except (UpstreamError, OpenAIError) as exc:
            placeholder.empty()
            st.error(f"Request to vLLM failed: {exc}")
            st.session_state.messages.pop()
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": collected})
