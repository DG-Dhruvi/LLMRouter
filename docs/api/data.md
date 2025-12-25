# Data utilities

## DataLoader
`llmrouter.data.DataLoader` attaches datasets to a router when a config is loaded.

### Signature
```python
class DataLoader:
    def __init__(self, project_root: str):
        ...

    def load_data(self, config, obj_ref):
        ...
```

!!! note "Key responsibilities"
    - Resolve relative paths to the project root
    - Load datasets from `data_path`
    - Attach datasets to the router instance

### Attributes loaded
| Attribute | Source key | Description |
| --- | --- | --- |
| `query_data_train` | `data_path.query_data_train` | training queries |
| `query_data_test` | `data_path.query_data_test` | test queries |
| `query_embedding_data` | `data_path.query_embedding_data` | query embeddings |
| `routing_data_train` | `data_path.routing_data_train` | routing labels |
| `routing_data_test` | `data_path.routing_data_test` | routing labels |
| `llm_data` | `data_path.llm_data` | model metadata |
| `llm_embedding_data` | `data_path.llm_embedding_data` | model embeddings |

## Input formats
`llmrouter infer --input` supports:
- `.txt` (one query per line)
- `.json` (list of strings or list of objects with `query`)
- `.jsonl` (one JSON object per line)

If you need to preprocess data, start with the example datasets under `data/example_data/`.
