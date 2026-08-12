#!/usr/bin/env bash
# Build the Streamlit UI image and, on k3s, import it into containerd so the
# cluster can run it without an external registry.
set -euo pipefail

IMAGE="${IMAGE:-llm-in-a-box/streamlit-ui:latest}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Building ${IMAGE}"
# Build context is the repo root so the llmbox package ships with the UI.
docker build -f "${ROOT_DIR}/ui/Dockerfile" -t "${IMAGE}" "${ROOT_DIR}"

# k3s uses containerd, not the Docker daemon, so a locally-built Docker image
# is not visible to the cluster by default. Import it directly when k3s is
# available; otherwise assume the image is pushed to a registry.
if command -v k3s >/dev/null 2>&1; then
  echo "==> Importing image into k3s containerd"
  docker save "${IMAGE}" | sudo k3s ctr images import -
  echo "==> Done. Image available in-cluster as ${IMAGE}"
else
  cat <<EOF
==> k3s not detected locally.
    Push the image to a registry reachable by your cluster and update
    k8s/base/streamlit/deployment.yaml (or run: kubectl -n llm-in-a-box set image \\
    deployment/streamlit streamlit=<registry>/${IMAGE}).
EOF
fi
