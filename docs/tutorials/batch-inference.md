# Batch inference

Batch inference is the easiest way to compare routers on many queries and to generate routing logs for analysis.

## Input and output schemas
See [Data formats](../getting-started/data-formats.md) for supported `--input` formats and output fields.

## Route-only batch run (recommended first)
`--route-only` is fast and requires no API keys.

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input queries.jsonl --output routed.jsonl --output-format jsonl --route-only
```

## Full inference batch run
This routes each query and then calls the selected model via an OpenAI-compatible endpoint.

Prereqs:
- `API_KEYS` is set (see [Installation](../getting-started/installation.md))
- each candidate model in `llm_data` has `api_endpoint` and `model`, or you set a global `api_endpoint` in the YAML (see [Config reference](../api/config.md))

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input queries.jsonl --output results.jsonl --output-format json
```

## Tips for large runs
- Prefer JSONL output: `--output-format jsonl` (stream-friendly, appendable).
- If you only need routing decisions, always include `--route-only`.
- Use `--verbose` if you are debugging errors (it prints more context).

## Troubleshooting
- `API endpoint not found`: add `api_endpoint` to `llm_data[model_name]` or set `api_endpoint` in the YAML.
- Many `success=false`: check your input file format and make sure `query` is present.
