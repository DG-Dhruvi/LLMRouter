# Config reference

LLMRouter uses YAML config files. Common top-level keys are listed below.

## data_path
Dataset and metadata locations. Paths are resolved relative to the repo root.

Supported keys:
- `query_data_train` (jsonl)
- `query_data_test` (jsonl)
- `query_embedding_data` (pt)
- `routing_data_train` (jsonl)
- `routing_data_test` (jsonl)
- `llm_data` (json)
- `llm_embedding_data` (json)

## model_path
Model checkpoints and weights.

Common keys:
- `ini_model_path`
- `save_model_path`
- `load_model_path`

## metric
Routing objective weights. Example:

```yaml
metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0
```

## hparam
Router-specific hyperparameters.

## api_endpoint
Optional inference endpoint. If omitted, the default in the inference code is used.
