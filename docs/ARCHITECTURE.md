# Architecture

LLM-in-a-Box is intentionally small: two workloads in one namespace, wired
together by Kubernetes Services. This document explains each piece and the
request path.

## Components

### vLLM model server

* **Image:** `vllm/vllm-openai` — the upstream vLLM image that starts the
  OpenAI-compatible API server (`vllm.entrypoints.openai.api_server`).
* **What it does:** loads the configured HuggingFace model onto the GPU and
  exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and a
  `/health` endpoint on port `8000`.
* **State:** model weights are cached on a `PersistentVolumeClaim` mounted at
  `/models` (`HF_HOME`), so a pod restart doesn't re-download gigabytes of
  weights.
* **Config:** all serving parameters come from the `vllm-config` ConfigMap and
  are injected as CLI args via `$(VAR)` substitution, so the same manifest
  serves any model.
* **Secrets:** the HuggingFace token (for gated models like Llama 3) is read
  from the optional `vllm-secrets` Secret.

Because model loading is slow, the Deployment uses a `Recreate` strategy and a
generous `startupProbe` (up to ~15 minutes) before readiness/liveness kick in.
`/dev/shm` is backed by an in-memory `emptyDir` because vLLM/PyTorch need far
more shared memory than the container default.

### Streamlit UI

* **Image:** built from `ui/Dockerfile` (Python 3.11 slim + Streamlit +
  `openai`).
* **What it does:** renders a streaming chat interface. It uses the `openai`
  client pointed at the in-cluster vLLM Service, so it is backend-agnostic.
* **Model selection:** if `VLLM_MODEL` is set it uses that; otherwise it calls
  `/v1/models` and picks the first served model automatically.
* **Health:** a sidebar indicator calls `/v1/models` to show whether vLLM is
  reachable and which model is live.

### Networking

* Two `ClusterIP` Services: `vllm:8000` and `streamlit:8501`.
* An `Ingress` (k3s ships Traefik) exposes the UI at `http://llm.localhost`.
* For local access without ingress, `scripts/port-forward.sh` forwards both the
  UI and the raw API to `localhost`.

## Request path

```
Browser
  │  HTTP / websocket
  ▼
Ingress (Traefik)  ──►  Service streamlit:8501  ──►  Streamlit pod
                                                        │  openai client
                                                        │  POST /v1/chat/completions (stream=true)
                                                        ▼
                                              Service vllm:8000  ──►  vLLM pod  ──►  GPU
```

1. The user sends a message in the browser.
2. Streamlit builds the payload (system prompt + conversation history) and calls
   vLLM's `/v1/chat/completions` with `stream=true`.
3. vLLM generates tokens on the GPU and streams them back.
4. Streamlit renders tokens incrementally in the chat window.

## Design choices

* **OpenAI-compatible API** — decouples the UI from the serving engine. You
  could replace vLLM with any OpenAI-compatible server, or point the UI at a
  different `VLLM_BASE_URL`, with no code changes.
* **ConfigMap-driven serving** — swapping models is an edit-and-reapply, not an
  image rebuild.
* **Kustomize overlays** — the `cpu-test` overlay reuses the base and only
  patches the model + GPU request, keeping the GPU and CPU paths in sync.
* **Persistent model cache** — avoids repeated multi-gigabyte downloads and
  makes restarts fast.

## Scaling notes

* **Bigger models:** raise the PVC size, set `TENSOR_PARALLEL_SIZE` to the GPU
  count on the node, and request that many `nvidia.com/gpu`.
* **More throughput:** vLLM already batches requests continuously; a single
  replica serves many concurrent users. Horizontal scaling requires
  `ReadWriteMany` storage (or per-pod caches) and a load balancer in front.
* **Multiple models:** run a second vLLM Deployment/Service with its own
  ConfigMap and point additional UI instances (or a router) at them.
