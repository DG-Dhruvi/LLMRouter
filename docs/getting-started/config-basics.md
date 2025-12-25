# Config basics

LLMRouter uses YAML configs for training and inference. Paths under `data_path` are resolved relative to the repo root.

## Common sections
- data_path: datasets and metadata files
- model_path: checkpoints to save or load
- metric.weights: optimization weights for performance and cost
- hparam: router-specific hyperparameters
- api_endpoint: optional inference endpoint (default used if omitted)

## Example
```yaml
data_path:
  query_data_train: "data/example_data/query_data/default_query_train.jsonl"
  query_data_test: "data/example_data/query_data/default_query_test.jsonl"
  query_embedding_data: "data/example_data/routing_data/query_embeddings_longformer.pt"
  routing_data_train: "data/example_data/routing_data/default_routing_train_data.jsonl"
  routing_data_test: "data/example_data/routing_data/default_routing_test_data.jsonl"
  llm_data: "data/example_data/llm_candidates/default_llm.json"
  llm_embedding_data: "data/example_data/llm_candidates/default_llm_embeddings.json"

model_path:
  ini_model_path: ""
  save_model_path: "saved_models/knnrouter/knnrouter.pkl"

metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0

hparam:
  n_neighbors: 5
  weights: "uniform"

api_endpoint: "https://integrate.api.nvidia.com/v1"
```

## Runtime overrides
- `llmrouter infer --load_model_path <path>` overrides `model_path.load_model_path` in the config.
