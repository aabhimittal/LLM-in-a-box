# LLM-in-a-Box 📦

Deploy an open-source LLM (Llama 3 by default) on a **k3s** cluster, serve it
with **vLLM**, and chat with it through a simple **Streamlit** UI — all
self-hosted, no external API calls.

```
                 ┌──────────────────────────── k3s cluster ───────────────────────────┐
                 │  namespace: llm-in-a-box                                             │
                 │                                                                      │
  Browser  ──────┼──► Ingress (Traefik) ──► Streamlit UI ──► vLLM (OpenAI API) ──► GPU  │
  :8501 /        │                          Deployment       Deployment                 │
  llm.localhost  │                          Service :8501    Service :8000              │
                 │                                            PVC (model cache)          │
                 └──────────────────────────────────────────────────────────────────────┘
```

* **vLLM** serves any HuggingFace-compatible model behind an
  **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`, …).
* **Streamlit** provides a streaming chat interface and talks to vLLM using the
  standard `openai` Python client — so you can point it at any OpenAI-compatible
  backend.
* Everything is packaged as plain Kubernetes manifests plus a Kustomize base and
  a CPU-only test overlay.

---

## Repository layout

```
.
├── ui/                       # Streamlit chat application
│   ├── app.py                #   streaming chat UI (OpenAI client -> vLLM)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .streamlit/config.toml
├── k8s/                      # Kubernetes / k3s manifests
│   ├── namespace.yaml
│   ├── vllm/                 #   model server (Deployment, Service, PVC, Config, Secret)
│   ├── streamlit/            #   UI (Deployment, Service, Config)
│   ├── ingress.yaml          #   Traefik ingress for the UI
│   ├── kustomization.yaml    #   base: `kubectl apply -k k8s/`
│   └── overlays/cpu-test/    #   GPU-free smoke-test overlay (TinyLlama)
├── scripts/                  # install / build / deploy / test helpers
├── Makefile                  # `make help` for all tasks
└── docs/                     # ARCHITECTURE.md, SETUP.md
```

---

## Quick start

> **Prerequisites:** a Linux host, `kubectl`, `docker`, and — for real Llama 3
> serving — an NVIDIA GPU with drivers + container toolkit installed. See
> [`docs/SETUP.md`](docs/SETUP.md) for the full GPU setup, and skip to
> [CPU smoke test](#no-gpu-cpu-smoke-test) if you just want to try the wiring.

### 1. Install k3s

```bash
make install-k3s
export KUBECONFIG="$(pwd)/kubeconfig"
```

Then install the NVIDIA device plugin so pods can request GPUs:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.1/deployments/static/nvidia-device-plugin.yml
```

### 2. Build the UI image

```bash
make build-ui   # builds and imports the image into k3s containerd
```

### 3. Deploy

Llama 3 is a **gated** model, so you need a
[HuggingFace access token](https://huggingface.co/settings/tokens) with access
to `meta-llama/Meta-Llama-3-8B-Instruct`:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx make deploy
```

The script creates the token secret, applies all manifests, and waits for vLLM
to finish loading the weights (first run downloads ~16&nbsp;GB and can take a few
minutes).

### 4. Chat

```bash
make port-forward     # forwards UI :8501 and vLLM API :8000
```

Open <http://localhost:8501> and start chatting. If you configured the ingress,
the UI is also at <http://llm.localhost>.

---

## No GPU? CPU smoke test

To exercise the full path (vLLM API → Streamlit UI) without a GPU, deploy the
CPU overlay, which serves the tiny, ungated **TinyLlama-1.1B** model:

```bash
make build-ui
make deploy-cpu
make port-forward
```

Generation is slow on CPU, but it proves the deployment end-to-end and is what
CI validates. See [`docs/SETUP.md`](docs/SETUP.md#cpu-only-testing) for notes on
CPU-capable vLLM images.

---

## Configuration

The model and serving parameters live in
[`k8s/vllm/configmap.yaml`](k8s/vllm/configmap.yaml):

| Key                      | Default                              | Description                                  |
| ------------------------ | ------------------------------------ | -------------------------------------------- |
| `MODEL_ID`               | `meta-llama/Meta-Llama-3-8B-Instruct`| HuggingFace model id to serve                |
| `SERVED_MODEL_NAME`      | `llama-3-8b-instruct`                | Alias clients use to select the model        |
| `GPU_MEMORY_UTILIZATION` | `0.90`                               | Fraction of GPU memory vLLM may use          |
| `MAX_MODEL_LEN`          | `8192`                               | Max context length                           |
| `TENSOR_PARALLEL_SIZE`   | `1`                                  | Number of GPUs to shard the model across     |

The UI is configured via [`k8s/streamlit/configmap.yaml`](k8s/streamlit/configmap.yaml)
(`VLLM_BASE_URL`, `VLLM_MODEL`, `APP_TITLE`). To swap models, edit the ConfigMap
values and re-apply — no image rebuild required.

---

## Using the raw API

vLLM speaks the OpenAI API, so any OpenAI client works:

```bash
make port-forward   # exposes the API on :8000
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or run the bundled smoke test: `make smoke-test`.

---

## Common tasks

```bash
make help          # list every target
make status        # pods, services, ingress
make logs          # tail vLLM logs
make lint          # client-side validate all manifests
make clean         # tear everything down
```

---

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the components fit together.
* [`docs/SETUP.md`](docs/SETUP.md) — full cluster + GPU prerequisites and troubleshooting.

## License

[MIT](LICENSE)
