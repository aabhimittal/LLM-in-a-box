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
  `openai`), with the build context at the repository root so the `llmbox`
  package ships alongside the app.
* **What it does:** renders a streaming chat interface. All production
  behaviour is delegated to `llmbox`; the module itself is presentation only.
* **Model selection:** if `VLLM_MODEL` is set it uses that; otherwise it calls
  `/v1/models` and picks the first served model automatically.
* **Health:** a sidebar indicator calls `/v1/models` to show whether vLLM is
  reachable and which model is live, alongside live cache hit-rate, circuit
  state and the raw Prometheus exposition.

Streamlit re-runs the whole script on every interaction, so the `llmbox` client
is built inside `@st.cache_resource`. That is not an optimisation: the cache,
rate-limiter buckets and circuit-breaker state are meaningless if they are
rebuilt on each keystroke.

### `llmbox` — the serving edge

A dependency-free library between the UI and vLLM. Each concern is independently
testable and injectable; the composed order in `client.py` is deliberate:

| Stage | Module | Why it sits here |
| --- | --- | --- |
| 1. Guardrails | `guardrails.py` | Cheapest check, and must run before the text is used for anything — including logging. |
| 2. Context budgeting | `context.py` | Determines the true token cost the limiter needs, and guarantees the request *can* succeed. |
| 3. Rate limiting | `ratelimit.py` | Priced in tokens, not requests. |
| 4. Cache | `cache.py` | Consulted only for deterministic requests. |
| 5. Single-flight | `cache.py` | Collapses a concurrent herd onto one generation. |
| 6. Queue-depth shed | `loadsignal.py` | Guards the *upstream call only*, so a cache hit is never rejected for load. Fails open. |
| 7. Bulkhead | `concurrency.py` | Bounds total in-flight work; held for a stream's whole lifetime. |
| 8. Breaker → retry | `resilience.py` | The breaker is *inside* the retry loop, so an open circuit aborts immediately rather than burning the retry budget. |

Stages 6 and 7 exist because a saturated vLLM does not fail, it **queues**:
requests keep succeeding, just slower, so nothing error-rate-based ever trips.
The bulkhead bounds concurrency locally; the shedder reads the server's own
`vllm:num_requests_waiting` gauge, which rises before users notice latency.

Supporting modules: `tokens.py` (script-aware estimation — the `len/4` rule
undercounts CJK by 3-5x), `streaming.py` (a watchdog on the gap *between*
tokens, because an open-but-silent socket never raises), `observability.py`
(JSON logs and a Prometheus registry), `metrics_server.py` (serves `/metrics`
and `/healthz` on port 9100 — the only place requests rejected before reaching
the model are visible), `errors.py` (a typed hierarchy so callers branch on
structure, not on error strings).

The backend is an injected callable `(payload, stream) -> response`, which is
why the library has no HTTP dependency and the whole suite runs without a
server. See [`EDGE-CASES.md`](EDGE-CASES.md) for the failure modes each layer
addresses.

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
                                                        │
                                                        ▼
                                            llmbox: guardrails → budget →
                                            rate limit → cache → single-flight
                                            → shed → bulkhead → breaker → retry
                                                        │  openai client
                                                        │  POST /v1/chat/completions (stream=true)
                                                        ▼
                                              Service vllm:8000  ──►  vLLM pod  ──►  GPU
```

1. The user sends a message in the browser.
2. `llmbox` screens the input, trims the conversation to fit the context window,
   charges the user's token budget, and checks the cache. A cache hit or a
   rejection returns here — the GPU is never touched.
3. Otherwise the request goes upstream under the bulkhead, circuit breaker and
   retry policy, with concurrent identical prompts collapsed into one call and
   load shed if vLLM's own queue is already too deep.
4. vLLM generates tokens on the GPU and streams them back.
5. Streamlit renders tokens incrementally, and surfaces any trimming, truncation
   or redaction that occurred as a caption under the reply.

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
* **Multiple UI replicas:** the cache, rate-limiter buckets and bulkhead are
  **per-pod, in-memory**. With N replicas the effective limits are multiplied by
  N and cache hit rate drops, so divide `LLMBOX_MAX_CONCURRENT` and the token
  budgets by the replica count, or move that state to a shared store. The
  queue-depth shedder is unaffected — it reads a single global signal from vLLM,
  which is precisely why it is the most reliable of the three under scale-out.
* **Multiple models:** run a second vLLM Deployment/Service with its own
  ConfigMap and point additional UI instances (or a router) at them.
