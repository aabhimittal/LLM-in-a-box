#!/usr/bin/env bash
# Install a single-node k3s cluster suitable for LLM-in-a-Box.
#
# For GPU workloads the node must already have the NVIDIA driver and the
# NVIDIA container toolkit installed; k3s auto-detects the nvidia container
# runtime when the toolkit is present. See docs/SETUP.md for GPU prerequisites.
set -euo pipefail

echo "==> Installing k3s (single node)…"
curl -sfL https://get.k3s.io | sh -

echo "==> Waiting for the node to become Ready…"
sudo k3s kubectl wait --for=condition=Ready node --all --timeout=180s

echo "==> Exporting kubeconfig to ./kubeconfig"
sudo cat /etc/rancher/k3s/k3s.yaml > kubeconfig
sudo chown "$(id -u):$(id -g)" kubeconfig
chmod 600 kubeconfig

cat <<'EOF'

k3s is installed.

Point kubectl at the cluster with:

    export KUBECONFIG="$(pwd)/kubeconfig"
    kubectl get nodes

For GPU support, install the NVIDIA device plugin next:

    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.1/deployments/static/nvidia-device-plugin.yml

EOF
