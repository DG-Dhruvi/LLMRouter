# Deployment and integration

This page covers common ways to run LLMRouter in production-like settings.

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

## Service integration
If you are embedding LLMRouter in a service:
- Load the router once and reuse it across requests.
- Consider using `llmrouter.cli.router_inference.load_router` to load from config.
- Keep config, data, and model artifacts versioned together.

## Operational tips
- Validate configs in CI by running `llmrouter list-routers` and a small `--route-only` test.
- Prefer batch inference for large offline jobs.
- Tune `--temp` and `--max-tokens` for your latency and cost targets.
