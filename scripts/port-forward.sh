#!/usr/bin/env bash
# Convenience wrapper to reach the UI (and optionally the raw vLLM API) locally.
set -euo pipefail

UI_PORT="${UI_PORT:-8501}"
API_PORT="${API_PORT:-8000}"

echo "==> Forwarding Streamlit UI  -> http://localhost:${UI_PORT}"
echo "==> Forwarding vLLM API      -> http://localhost:${API_PORT}/v1"
echo "    (Ctrl-C to stop)"

# Forward the vLLM API in the background, the UI in the foreground.
kubectl -n llm-in-a-box port-forward svc/vllm "${API_PORT}:8000" >/dev/null 2>&1 &
API_PF_PID=$!
trap 'kill "${API_PF_PID}" 2>/dev/null || true' EXIT

kubectl -n llm-in-a-box port-forward svc/streamlit "${UI_PORT}:8501"
