# Config reference

LLMRouter uses YAML config files for training and inference. This page lists common keys and how they are used.

## Top-level keys
| Key | Description |
| --- | --- |
| `data_path` | Dataset and metadata paths |
| `model_path` | Model checkpoints and artifacts |
| `metric` | Objective weights for routing |
| `hparam` | Router-specific hyperparameters |
| `api_endpoint` | Optional inference endpoint |

## data_path
Supported keys (common across routers):
- `query_data_train` (jsonl)
- `query_data_test` (jsonl)
- `query_embedding_data` (pt)
- `routing_data_train` (jsonl)
- `routing_data_test` (jsonl)
- `llm_data` (json)
- `llm_embedding_data` (json)

Absolute paths are used as-is. Relative paths are resolved against the LLMRouter "project root" (the directory containing the `llmrouter/` package).

!!! note
    If you installed from PyPI (not a repo clone), relative paths often won't point to your local data. Prefer absolute paths.

## model_path
Common keys:
- `ini_model_path`
- `save_model_path`
- `load_model_path`

`llmrouter infer --load_model_path` overrides `model_path.load_model_path` in the config.

## metric
Weights for routing objectives. Example:
```yaml
metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0
```

## hparam
Router-specific hyperparameters. Refer to router-specific configs for exact fields.

## api_endpoint
Optional global fallback API base URL for inference.

Inference expects an endpoint from either:

1. `llm_data[model_name].api_endpoint` (per model), or
2. `api_endpoint` in the YAML config

If neither is set, `llmrouter infer` fails with an "API endpoint not found" error.

## RouterR1 special fields
`router_r1` expects `api_base` and `api_key` under `hparam`:
```yaml
hparam:
  api_base: "https://your-endpoint/v1"
  api_key: "..."
```
