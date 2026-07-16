#!/usr/bin/env bash
# End-to-end smoke test: verify the vLLM OpenAI API answers a chat request.
# Run after `scripts/port-forward.sh` (or against any reachable endpoint).
#
#   BASE_URL=http://localhost:8000/v1 scripts/smoke-test.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"

echo "==> Listing served models at ${BASE_URL}/models"
MODELS_JSON="$(curl -fsS "${BASE_URL}/models")"
echo "${MODELS_JSON}"

MODEL="$(printf '%s' "${MODELS_JSON}" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -n1)"
if [[ -z "${MODEL}" ]]; then
  echo "Could not determine a served model id from the /models response." >&2
  exit 1
fi
echo "==> Using model: ${MODEL}"

echo "==> Sending a chat completion request…"
curl -fsS "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Reply with exactly: LLM-in-a-Box is live"}],
  "max_tokens": 32,
  "temperature": 0
}
JSON
)"

echo
echo "==> Smoke test succeeded."
