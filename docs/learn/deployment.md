# Deployment and integration

This page covers common ways to run LLMRouter in production-like settings.

## Two common deployment patterns

### Pattern A: route-only + your own model gateway (recommended)
Use LLMRouter to pick `model_name`, and then call your model gateway/service yourself. This keeps secrets, retries, logging, and observability in one place.

- Use `llmrouter infer --route-only ...` for a CLI workflow, or
- Load a router in Python and call `route_single` directly

### Pattern B: full inference through `llmrouter infer`
Let LLMRouter route and call the selected model via an OpenAI-compatible endpoint. This is the simplest end-to-end path, but you still need to manage `API_KEYS` and `api_endpoint`.

## CLI-based inference
For a single query:
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "Hello"
```

For batch inference:
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml   --input queries.jsonl --output results.jsonl --output-format jsonl
```

Use `--route-only` to skip API calls and return routing decisions only.

## Python integration (load once, reuse)
If you are embedding LLMRouter in a service, load the router once at startup and reuse it across requests:

```python
from llmrouter.cli.router_inference import load_router, route_query

router_name = "knnrouter"
router = load_router(router_name, "configs/model_config_test/knnrouter.yaml")

result = route_query("Explain transformers.", router, router_name)
print(result["model_name"])
```

To perform full inference inside Python, use `infer_query` (requires `API_KEYS` and `api_endpoint`):

```python
from llmrouter.cli.router_inference import infer_query

result = infer_query("Explain transformers.", router, router_name)
print(result["response"])
```

## Chat UI
LLMRouter includes a Gradio-based chat UI:
```bash
llmrouter chat --router knnrouter --config configs/model_config_test/knnrouter.yaml
```

## API credentials and endpoints
- Configure `api_endpoint` in your YAML config to point to your LLM endpoint.
- Set `API_KEYS` in the environment for `call_api`.
  - Single key: `API_KEYS=your-key`
  - Multiple keys: `API_KEYS='["key1", "key2"]'`

`call_api` rotates across keys in a round-robin fashion.

## Scaling notes
- Multi-process deployments: each process maintains its own API key rotation counters.
- Route-only is the safest way to scale quickly since it avoids external calls.

## Service integration
If you are embedding LLMRouter in a service:
- Load the router once and reuse it across requests.
- Consider using `llmrouter.cli.router_inference.load_router` to load from config.
- Keep config, data, and model artifacts versioned together.

## Operational tips
- Validate configs in CI by running `llmrouter list-routers` and a small `--route-only` test.
- Prefer batch inference for large offline jobs.
- Tune `--temp` and `--max-tokens` for your latency and cost targets.
