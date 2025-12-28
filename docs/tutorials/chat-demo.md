# Chat demo

The chat UI is a Gradio app driven by the same router configs as `llmrouter infer`.

## Prereqs
- Install `gradio` (the CLI prints an error if it is missing).
- For full inference, set `API_KEYS` and make sure your `llm_data` has `api_endpoint` and `model`.

## Run
```bash
llmrouter chat --router knnrouter --config configs/model_config_test/knnrouter.yaml --host 0.0.0.0 --port 7860
```

## Key flags
- `--mode`: `full_context`, `current_only`, or `retrieval`
- `--top_k`: only used in retrieval mode
- `--temp`: generation temperature

See the full flag list in [CLI reference](../api/cli.md).
