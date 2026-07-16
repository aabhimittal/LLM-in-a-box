# Setup & Troubleshooting

This guide covers the full prerequisites for running LLM-in-a-Box with real
Llama 3 serving on a GPU, plus how to test without one.

## Prerequisites

| Requirement | Notes |
| ----------- | ----- |
| Linux host  | Tested on Ubuntu 22.04. A VM or bare metal both work. |
| `kubectl`   | Installed automatically with k3s; or install separately. |
| `docker`    | To build the Streamlit UI image. |
| NVIDIA GPU  | Required for Llama-3-8B at usable speed (≈16&nbsp;GB VRAM). |
| HF token    | A [HuggingFace token](https://huggingface.co/settings/tokens) with access to the gated Llama 3 repo. |

## GPU prerequisites

For vLLM to use the GPU inside k3s you need three things on the node **before**
deploying:

1. **NVIDIA driver** — verify with `nvidia-smi`.
2. **NVIDIA Container Toolkit** — lets containerd run GPU containers. Install per
   the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
   k3s auto-detects the `nvidia` container runtime when the toolkit is present.
3. **NVIDIA device plugin** — advertises `nvidia.com/gpu` to the scheduler:

   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.1/deployments/static/nvidia-device-plugin.yml
   ```

Confirm the GPU is schedulable:

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
# should print "1" (or your GPU count), not null
```

## Step-by-step deployment

```bash
# 1. Cluster
make install-k3s
export KUBECONFIG="$(pwd)/kubeconfig"

# 2. GPU device plugin (see above)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.1/deployments/static/nvidia-device-plugin.yml

# 3. Build & import the UI image
make build-ui

# 4. Deploy (creates the HF secret from HF_TOKEN)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx make deploy

# 5. Access
make port-forward     # http://localhost:8501
```

### Managing the HuggingFace token manually

`make deploy` creates the secret for you when `HF_TOKEN` is set. To manage it
yourself:

```bash
kubectl -n llm-in-a-box create secret generic vllm-secrets \
  --from-literal=HF_TOKEN='hf_xxxxxxxxxxxxxxxxx'
```

Never commit the real token. `k8s/base/vllm/hf-token-secret.example.yaml` is a
template; a filled-in `hf-token-secret.yaml` is git-ignored.

## CPU-only testing

The `vllm/vllm-openai` image is CUDA-based and expects a GPU. To validate the
Kubernetes wiring without one, use the CPU overlay, which serves the tiny
ungated **TinyLlama-1.1B** model and drops the GPU request:

```bash
make build-ui
make deploy-cpu
make port-forward
```

> **Note:** running vLLM itself purely on CPU requires a CPU build of vLLM.
> The overlay handles the Kubernetes side (model choice, no GPU request); if the
> stock image fails to start without CUDA, build a CPU image following the
> [vLLM CPU installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html)
> and set it on the vLLM Deployment. This path is meant for smoke-testing the
> end-to-end flow, not for performance.

## Verifying the deployment

```bash
make status        # pods should be Running/Ready
make logs          # watch vLLM load weights: "Loading model weights..."
make smoke-test    # send a chat request to the API
```

## Troubleshooting

| Symptom | Likely cause & fix |
| ------- | ------------------ |
| vLLM pod `Pending` | No schedulable GPU. Check the device plugin and `nvidia.com/gpu` allocatable (see above). For CPU testing use `make deploy-cpu`. |
| `401`/`403` pulling the model | Missing or invalid `HF_TOKEN`, or the account lacks access to the gated repo. Request access on the model's HF page and recreate the secret. |
| Pod `CrashLoopBackOff`, OOM | Model too large for VRAM. Lower `MAX_MODEL_LEN`, reduce `GPU_MEMORY_UTILIZATION`, or use a smaller model in `vllm-config`. |
| `RuntimeError: ... /dev/shm` | Shared memory too small. The manifest mounts an 8Gi in-memory `/dev/shm`; increase `sizeLimit` if needed. |
| UI shows "vLLM server unreachable" | vLLM still loading (watch `make logs`), or `VLLM_BASE_URL` is wrong. First load can take minutes. |
| Streamlit `ImagePullBackOff` | The locally-built image wasn't imported into k3s. Re-run `make build-ui`, or push to a registry and update the Deployment image. |
| Ingress host not resolving | Add `<node-ip> llm.localhost` to `/etc/hosts`, or just use `make port-forward`. |

## Tearing down

```bash
make clean         # deletes the namespace and everything in it
# To remove k3s entirely:
/usr/local/bin/k3s-uninstall.sh
```
