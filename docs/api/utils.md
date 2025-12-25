# Utils

Helpers are exported from `llmrouter.utils`.

## Data helpers
- `load_csv`, `load_jsonl`, `jsonl_to_csv`, `load_pt`

## Embeddings
- `get_longformer_embedding`, `parallel_embedding_task`

## API calls
- `call_api(request, api_keys_env=None, max_tokens=512, temperature=0.01, top_p=0.9, timeout=30, max_retries=3)`
- Uses the `API_KEYS` environment variable for round-robin key selection

## Prompting and evaluation
- `format_*_prompt`, `generate_task_query`
- Evaluation helpers are available when optional dependencies are installed
