# Utils

Helpers are exported from `llmrouter.utils`.

## call_api
Send a query to an external LLM API using LiteLLM and round-robin API key selection.

Source: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/utils/api_calling.py

### Signature
```python
def call_api(
    request,
    api_keys_env=None,
    max_tokens=512,
    temperature=0.01,
    top_p=0.9,
    timeout=30,
    max_retries=3,
):
    ...
```

!!! note "Key responsibilities"
    - Parse and rotate API keys from `API_KEYS`
    - Send requests via LiteLLM (`openai/{api_name}` with `api_base`)
    - Return response text and token usage info

### Parameters
| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `request` | dict or list | API request(s) with `api_endpoint`, `query`, `model_name`, `api_name` | required |
| `api_keys_env` | str or None | Override for `API_KEYS` env var | none |
| `max_tokens` | int | Maximum tokens to generate | 512 |
| `temperature` | float | Sampling temperature | 0.01 |
| `top_p` | float | Top-p sampling | 0.9 |
| `timeout` | int | Request timeout (seconds) | 30 |
| `max_retries` | int | Retries for failed calls | 3 |

### Request schema
```json
{
  "api_endpoint": "https://integrate.api.nvidia.com/v1",
  "query": "Hello",
  "model_name": "my_model",
  "api_name": "qwen/qwen2.5-7b-instruct",
  "system_prompt": "You are a helpful assistant."
}
```

### Return schema
For a successful call, the function returns the original dict with added fields:

```json
{
  "response": "....",
  "token_num": 123,
  "prompt_tokens": 45,
  "completion_tokens": 78,
  "response_time": 1.23
}
```

### Notes
- `API_KEYS` can be a single key (`"key"`) or a JSON list of keys (`'["k1","k2"]'`).
- Key rotation is round-robin per `(api_endpoint, api_name)` for single calls; for batch calls, requests are distributed by index.
- If LiteLLM does not return usage, token counts fall back to GPT-2 tokenization (if `transformers` is installed) or whitespace splitting.
- `call_api` raises an `ImportError` if `litellm` is not installed, and `ValueError` if required request fields are missing.

## Other helpers
`llmrouter.utils` also exports helpers for:
- data loading and conversion
- embeddings and tensor utilities
- prompting templates and task helpers
- progress tracking
- evaluation (optional dependencies)
