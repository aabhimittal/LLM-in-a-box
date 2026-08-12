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
* **`llmbox`** is a dependency-free serving-edge library sitting between the two:
  context budgeting, token-priced rate limiting, deterministic response caching
  with stampede protection, PII redaction, prompt-injection screening, circuit
  breaking with jittered retries, and structured telemetry.
* Everything is packaged as plain Kubernetes manifests plus a Kustomize base, a
  CPU-only test overlay and a Prometheus overlay.

Every production behaviour above is covered by a **172-test suite that needs no
GPU and no cluster** and runs in ~2 seconds (`make test`). See
[`docs/EDGE-CASES.md`](docs/EDGE-CASES.md) for the catalogue of failure modes
handled and where each is tested.

---

## Repository layout

```
.
├── llmbox/                   # Serving-edge library (stdlib only, no deps)
│   ├── tokens.py             #   script-aware token estimation
│   ├── context.py            #   context-window budgeting & safe truncation
│   ├── ratelimit.py          #   token-bucket admission control
│   ├── cache.py              #   LRU+TTL cache and single-flight
│   ├── guardrails.py         #   PII redaction, prompt-injection screening
│   ├── resilience.py         #   circuit breaker, retry with full jitter
│   ├── observability.py      #   structured logs, Prometheus metrics
│   └── client.py             #   composes all of the above
├── tests/                    # 172 tests; no GPU or cluster required
│   ├── test_edge_cases.py    #   industrial failure scenarios
│   └── test_integration_openai.py  # real openai SDK vs a stub vLLM server
├── ui/                       # Streamlit chat application
│   ├── app.py                #   streaming chat UI built on llmbox
│   ├── requirements.txt
│   ├── Dockerfile            #   build from repo root: -f ui/Dockerfile .
│   └── .streamlit/config.toml
├── k8s/                      # Kubernetes / k3s manifests
│   ├── kustomization.yaml    #   wrapper: `kubectl apply -k k8s/`
│   ├── base/                 #   the full stack
│   │   ├── namespace.yaml
│   │   ├── vllm/             #     model server (Deployment, Service, PVC, Config, Secret)
│   │   ├── streamlit/        #     UI (Deployment, Service, Config)
│   │   ├── ingress.yaml      #     Traefik ingress for the UI
│   │   ├── pdb.yaml          #     PodDisruptionBudgets
│   │   ├── networkpolicy.yaml#     restrict who may call the inference API
│   │   └── kustomization.yaml
│   └── overlays/
│       ├── cpu-test/         #   GPU-free smoke-test overlay (TinyLlama)
│       └── observability/    #   ServiceMonitor + alerting rules
├── scripts/                  # install / build / deploy / test helpers
├── Makefile                  # `make help` for all tasks
└── docs/                     # ARCHITECTURE.md, SETUP.md, EDGE-CASES.md
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
[`k8s/base/vllm/configmap.yaml`](k8s/base/vllm/configmap.yaml):

| Key                      | Default                              | Description                                  |
| ------------------------ | ------------------------------------ | -------------------------------------------- |
| `MODEL_ID`               | `meta-llama/Meta-Llama-3-8B-Instruct`| HuggingFace model id to serve                |
| `SERVED_MODEL_NAME`      | `llama-3-8b-instruct`                | Alias clients use to select the model        |
| `GPU_MEMORY_UTILIZATION` | `0.90`                               | Fraction of GPU memory vLLM may use          |
| `MAX_MODEL_LEN`          | `8192`                               | Max context length                           |
| `TENSOR_PARALLEL_SIZE`   | `1`                                  | Number of GPUs to shard the model across     |

The UI is configured via [`k8s/base/streamlit/configmap.yaml`](k8s/base/streamlit/configmap.yaml)
(`VLLM_BASE_URL`, `VLLM_MODEL`, `APP_TITLE`). To swap models, edit the ConfigMap
values and re-apply — no image rebuild required.

### Serving edge

The same ConfigMap tunes the `llmbox` layer:

| Key                          | Default | Description                                              |
| ---------------------------- | ------- | -------------------------------------------------------- |
| `LLMBOX_MAX_MODEL_LEN`       | `8192`  | **Must match** vLLM's `MAX_MODEL_LEN`                     |
| `LLMBOX_MAX_OUTPUT_TOKENS`   | `512`   | Tokens held back for the reply when budgeting the prompt  |
| `LLMBOX_TOKENS_PER_MINUTE`   | `60000` | Sustained per-user token budget                           |
| `LLMBOX_BURST_TOKENS`        | `20000` | Per-user burst capacity                                   |
| `LLMBOX_CACHE_ENTRIES`       | `512`   | Max cached responses                                      |
| `LLMBOX_CACHE_TTL`           | `300`   | Cache TTL in seconds                                      |
| `LLMBOX_INJECTION_THRESHOLD` | `0.6`   | Prompt-injection block score in `[0,1]`                   |

Rate limiting is priced in **tokens, not requests**, so a handful of very large
prompts cannot monopolise the GPU while staying under a request-count cap. Only
`temperature=0` responses are cached — replaying a sampled completion would
silently destroy sampling diversity.

### Using `llmbox` on its own

The library has no dependencies and no HTTP coupling — the backend is any
callable — so it works in front of any OpenAI-compatible server:

```python
from llmbox import build_client

client = build_client(backend=my_backend, model="llama-3-8b-instruct",
                      max_model_len=8192)
result = client.chat([{"role": "user", "content": "Hello"}])
print(result.text, result.prompt_tokens, result.cached)
```

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
make test          # run the llmbox test suite (no GPU, no cluster, ~2s)
make status        # pods, services, ingress
make logs          # tail vLLM logs
make lint          # client-side validate all manifests
make clean         # tear everything down
```

---

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the components fit together.
* [`docs/SETUP.md`](docs/SETUP.md) — full cluster + GPU prerequisites and troubleshooting.
* [`docs/EDGE-CASES.md`](docs/EDGE-CASES.md) — the production failure modes handled, and where each is tested.

## License

[MIT](LICENSE)
