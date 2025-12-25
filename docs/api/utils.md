# Utils

Helpers are exported from `llmrouter.utils`.

## call_api
Send a query to an external LLM API using LiteLLM and round-robin API key selection.

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
    - Send requests via LiteLLM
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
  "api_name": "qwen/qwen2.5-7b-instruct"
}
```

### Notes
- `API_KEYS` can be a single key or a JSON list of keys.
- `call_api` raises an ImportError if `litellm` is not installed.

## Other helpers
`llmrouter.utils` also exports helpers for:
- data loading and conversion
- embeddings and tensor utilities
- prompting templates and task helpers
- progress tracking
- evaluation (optional dependencies)
