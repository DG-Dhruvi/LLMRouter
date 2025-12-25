# Data and metrics

LLMRouter loads data based on `data_path` in your YAML config and attaches those datasets to the router instance.

## Loaded datasets
| Attribute on router | Config key | Typical format | Purpose |
| --- | --- | --- | --- |
| `query_data_train` | `query_data_train` | jsonl | training queries |
| `query_data_test` | `query_data_test` | jsonl | test queries |
| `query_embedding_data` | `query_embedding_data` | pt | query embeddings |
| `routing_data_train` | `routing_data_train` | jsonl | routing labels and metrics |
| `routing_data_test` | `routing_data_test` | jsonl | routing labels and metrics |
| `llm_data` | `llm_data` | json | model metadata |
| `llm_embedding_data` | `llm_embedding_data` | json | model embeddings |

Paths are resolved relative to the repository root unless they are already absolute.

## Query data
A minimal query JSONL line can be as simple as:
```json
{"query": "Explain transformers."}
```

Some routers or datasets include additional fields, but `query` is the common denominator.

## Routing data
`routing_data_*` is loaded from JSONL and converted to a table. Fields depend on the router, but commonly include:
- `query`
- `model_name`
- one or more metric columns such as `performance` or `cost`

Example line:
```json
{"query": "Explain transformers.", "model_name": "model_a", "performance": 0.82, "cost": 0.05}
```

When in doubt, inspect the example datasets under `data/example_data/`.

## LLM metadata
`llm_data` is a JSON file describing candidate models. During inference, if a model entry contains a `model` field, LLMRouter uses it as the API model name when calling external endpoints.

Example:
```json
{
  "model_a": {
    "model": "qwen/qwen2.5-7b-instruct",
    "cost": 0.5,
    "capability": 0.7
  }
}
```

## Metric weights
`metric.weights` specifies objective weights such as `performance`, `cost`, or `llm_judge`. How the weights are used depends on the router and trainer implementation.

Example:
```yaml
metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0
```

## Embeddings
Embedding-based routers rely on `query_embedding_data` and `llm_embedding_data`. Ensure the embedding formats match the router you are using.
