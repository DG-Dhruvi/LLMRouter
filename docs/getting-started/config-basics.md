# Config basics

LLMRouter uses YAML configs for both training and inference.

## How paths are resolved

Most configs reference local files under `data_path` (datasets, embeddings, candidate model metadata).

- Absolute paths are used as-is.
- Relative paths are resolved against the LLMRouter "project root" (the directory containing the `llmrouter/` package).
  - If you cloned the repo, this is the repo root.
  - If you installed from PyPI, this is inside `site-packages`, so relative paths usually do not point to your local data.

!!! tip
    If you are not running from a repo clone, use absolute paths in configs to avoid path confusion.

## Common top-level sections

- `data_path`: datasets and metadata files (JSON/JSONL/PT)
- `model_path`: checkpoints and artifacts to save/load
- `metric.weights`: optimization weights for performance/cost trade-offs
- `hparam`: router-specific hyperparameters
- `api_endpoint`: optional global fallback API base URL for inference

## `llm_data` (candidate model metadata)

Routers select a `model_name`, which is expected to match a key in `llm_data`.
During inference, LLMRouter maps that key to the provider-specific model id via the `model` field, and resolves `api_endpoint` either per-model or from the YAML config.

Example (minimal):

```json
{
  "small": {
    "size": "7B",
    "model": "provider/model-small",
    "api_endpoint": "https://your-openai-compatible-endpoint/v1"
  },
  "large": {
    "size": "70B",
    "model": "provider/model-large",
    "api_endpoint": "https://your-openai-compatible-endpoint/v1"
  }
}
```

See the full example used by the `main` branch configs:
[data/example_data/llm_candidates/default_llm.json](https://github.com/ulab-uiuc/LLMRouter/blob/main/data/example_data/llm_candidates/default_llm.json).

## Example A: minimal route-only config

This is enough for baseline routers like `smallest_llm` when running with `--route-only`.

```yaml
data_path:
  llm_data: "/absolute/path/to/llm_data.json"
```

## Example B: trainable router config (skeleton)

Trainable routers usually need query data, routing labels, embeddings, plus a place to save/load the trained model.
Start from the `main` branch example configs and edit paths/hparams:

- Train: [configs/model_config_train/knnrouter.yaml](https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/knnrouter.yaml)
- Test: [configs/model_config_test/knnrouter.yaml](https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/knnrouter.yaml)

```yaml
data_path:
  query_data_train: "/path/to/query_train.jsonl"
  query_data_test: "/path/to/query_test.jsonl"
  query_embedding_data: "/path/to/query_embeddings.pt"
  routing_data_train: "/path/to/routing_train.jsonl"
  routing_data_test: "/path/to/routing_test.jsonl"
  llm_data: "/path/to/llm_data.json"

model_path:
  save_model_path: "/path/to/saved_models/knnrouter.pkl"   # training
  load_model_path: "/path/to/saved_models/knnrouter.pkl"   # inference

metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0

hparam:
  # router-specific; see the example config for exact fields
  n_neighbors: 5
```

## Runtime overrides

- `llmrouter infer --load_model_path <path>` overrides `model_path.load_model_path` from the YAML.

## Next

- If you are new, start from [Quickstart](quickstart.md).
- For the full key list, see [Config reference](../api/config.md).
