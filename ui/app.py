"""LLM-in-a-Box — Streamlit chat UI.

A lightweight chat front-end that talks to a vLLM server exposing the
OpenAI-compatible API. Designed to run either locally (pointing at a
port-forwarded vLLM service) or inside the k3s cluster alongside vLLM.

Configuration is driven entirely by environment variables so the same
image works in every environment:

    VLLM_BASE_URL   Base URL of the vLLM OpenAI endpoint
                    (default: http://vllm:8000/v1)
    VLLM_API_KEY    API key sent as a bearer token. vLLM ignores it unless
                    started with --api-key, but the OpenAI client requires a
                    non-empty value (default: "not-needed").
    VLLM_MODEL      Model name to request. When unset the app queries the
                    server's /models endpoint and uses the first one served.
    APP_TITLE       Title shown in the UI (default: "LLM-in-a-Box").
"""

from __future__ import annotations

import os

import streamlit as st
from openai import OpenAI, OpenAIError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
API_KEY = os.getenv("VLLM_API_KEY", "not-needed")
CONFIGURED_MODEL = os.getenv("VLLM_MODEL", "").strip()
APP_TITLE = os.getenv("APP_TITLE", "LLM-in-a-Box")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running fully on-premise, served by "
    "vLLM on a k3s cluster."
)


@st.cache_resource(show_spinner=False)
def get_client() -> OpenAI:
    """Return a cached OpenAI client pointed at the vLLM server."""
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)


def discover_model(client: OpenAI) -> str | None:
    """Resolve the model to use.

    Prefer the explicitly configured model, otherwise ask the server which
    models it is currently serving and take the first one.
    """
    if CONFIGURED_MODEL:
        return CONFIGURED_MODEL
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except OpenAIError:
        return None
    return None


def check_health(client: OpenAI) -> tuple[bool, str]:
    """Best-effort connectivity check against the vLLM server."""
    try:
        client.models.list()
        return True, "connected"
    except OpenAIError as exc:  # network / server errors
        return False, str(exc)


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="centered")

client = get_client()

# --------------------------------------------------------------------------- #
# Sidebar — settings & status
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title(f"📦 {APP_TITLE}")
    st.caption("Open-source LLM served by vLLM on k3s.")

    healthy, detail = check_health(client)
    if healthy:
        st.success("vLLM server reachable")
    else:
        st.error("vLLM server unreachable")
        with st.expander("Details"):
            st.code(detail)

    model = discover_model(client)
    if model:
        st.info(f"**Model:** `{model}`")
    else:
        st.warning("No model detected. Is vLLM still loading weights?")

    st.divider()
    st.subheader("Generation settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 64, 4096, 512, 64)
    top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)

    system_prompt = st.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=120,
    )

    st.divider()
    st.caption(f"Endpoint: `{BASE_URL}`")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

# --------------------------------------------------------------------------- #
# Chat state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render prior turns.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------------------------- #
# Chat input & streaming response
# --------------------------------------------------------------------------- #
prompt = st.chat_input("Ask anything…")

if prompt:
    if not model:
        st.error("No model is available yet. Please wait for vLLM to finish loading.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the request payload: system prompt + full conversation history.
    payload = [{"role": "system", "content": system_prompt}]
    payload.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        collected = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                collected += delta
                placeholder.markdown(collected + "▌")
            placeholder.markdown(collected)
        except OpenAIError as exc:
            placeholder.empty()
            st.error(f"Request to vLLM failed: {exc}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": collected})
