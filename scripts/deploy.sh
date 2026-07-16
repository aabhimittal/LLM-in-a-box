#!/usr/bin/env bash
# Deploy the full LLM-in-a-Box stack to the current kubectl context.
#
# Usage:
#   scripts/deploy.sh              # GPU deployment (base manifests)
#   scripts/deploy.sh cpu-test     # CPU-only smoke-test overlay
#
# Set HF_TOKEN to auto-create the HuggingFace secret for gated models:
#   HF_TOKEN=hf_xxx scripts/deploy.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-base}"

if [[ "${MODE}" == "cpu-test" ]]; then
  KUSTOMIZE_PATH="${ROOT_DIR}/k8s/overlays/cpu-test"
else
  KUSTOMIZE_PATH="${ROOT_DIR}/k8s"
fi

echo "==> Creating namespace"
kubectl apply -f "${ROOT_DIR}/k8s/namespace.yaml"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "==> Creating/updating HuggingFace token secret"
  kubectl -n llm-in-a-box create secret generic vllm-secrets \
    --from-literal=HF_TOKEN="${HF_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "==> HF_TOKEN not set — skipping secret creation."
  echo "    Gated models (meta-llama/*) will fail to download without it."
fi

echo "==> Applying manifests from ${KUSTOMIZE_PATH}"
kubectl apply -k "${KUSTOMIZE_PATH}"

echo "==> Waiting for vLLM to become ready (this can take several minutes)…"
kubectl -n llm-in-a-box rollout status deployment/vllm --timeout=900s || {
  echo "vLLM did not become ready in time. Inspect logs with:"
  echo "  kubectl -n llm-in-a-box logs deploy/vllm"
  exit 1
}

echo "==> Waiting for the Streamlit UI…"
kubectl -n llm-in-a-box rollout status deployment/streamlit --timeout=180s

cat <<'EOF'

Deployment complete!

Open the UI with a port-forward:

    scripts/port-forward.sh

…then browse to http://localhost:8501

Or, if the ingress is configured, http://llm.localhost
EOF
