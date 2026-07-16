# LLM-in-a-Box — common tasks.
# Run `make help` for a list of targets.

IMAGE ?= llm-in-a-box/streamlit-ui:latest
NS    ?= llm-in-a-box

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help.
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install-k3s
install-k3s: ## Install a single-node k3s cluster.
	./scripts/install-k3s.sh

.PHONY: build-ui
build-ui: ## Build the Streamlit UI image (and import into k3s if present).
	IMAGE=$(IMAGE) ./scripts/build-ui.sh

.PHONY: deploy
deploy: ## Deploy the full GPU stack (pass HF_TOKEN=... for gated models).
	./scripts/deploy.sh

.PHONY: deploy-cpu
deploy-cpu: ## Deploy the CPU-only smoke-test overlay (TinyLlama).
	./scripts/deploy.sh cpu-test

.PHONY: port-forward
port-forward: ## Forward the UI (8501) and vLLM API (8000) to localhost.
	./scripts/port-forward.sh

.PHONY: smoke-test
smoke-test: ## Hit the vLLM API with a test chat request.
	./scripts/smoke-test.sh

.PHONY: ui-local
ui-local: ## Run the Streamlit UI locally against a port-forwarded vLLM.
	VLLM_BASE_URL=$${VLLM_BASE_URL:-http://localhost:8000/v1} \
		streamlit run ui/app.py

.PHONY: logs
logs: ## Tail vLLM server logs.
	kubectl -n $(NS) logs -f deploy/vllm

.PHONY: status
status: ## Show pods, services and ingress in the namespace.
	kubectl -n $(NS) get pods,svc,ingress

.PHONY: lint
lint: ## Validate all Kubernetes manifests (client-side dry run).
	kubectl apply -k k8s/ --dry-run=client >/dev/null && echo "manifests OK"

.PHONY: clean
clean: ## Delete the entire deployment and namespace.
	kubectl delete namespace $(NS) --ignore-not-found
