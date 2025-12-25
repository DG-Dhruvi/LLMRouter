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

Paths are resolved relative to the repository root unless they are absolute.

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
Optional endpoint for inference. If omitted, the default in `call_api` is used.

## RouterR1 special fields
`router_r1` expects `api_base` and `api_key` under `hparam`:
```yaml
hparam:
  api_base: "https://your-endpoint/v1"
  api_key: "..."
```
