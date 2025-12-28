# Data utilities

This page documents data-loading helpers and dataset schemas used by `llmrouter`.

Sources:
- https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/data/data_loader.py
- https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/data/data.py

## DataLoader
`llmrouter.data.DataLoader` attaches datasets to a router when a config is loaded.

### Signature
```python
class DataLoader:
    def __init__(self, project_root: str):
        ...

    def to_abs(self, path_str: str) -> str:
        ...

    def load_data(self, config, obj_ref):
        ...
```

!!! note "Key responsibilities"
    - Resolve relative paths to the project root
    - Load datasets from `data_path`
    - Attach datasets to the router instance

!!! note
    Missing files print warnings and the corresponding attributes are set to `None`.

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

## Data format models (validation)
The `llmrouter.data.data` module defines Pydantic models and validators for common dataset schemas.
This is mainly used for training/evaluation pipelines and for checking dataset correctness.

### DataFormatType
```python
from llmrouter.data import DataFormatType
```
Supported values:
- `DataFormatType.STANDARD`
- `DataFormatType.GMTROUTER`

### Standard format
Use `StandardQueryData` for query examples and `StandardRoutingData` for routing labels.

Minimum query example:
```json
{"query": "What is machine learning?"}
```

Minimum routing label example:
```json
{"query_id": "q_001", "best_model": "gpt-4"}
```

### GMTRouter format
GMTRouter uses a structured JSONL schema with embeddings and per-turn ratings.
See `GMTRouterInteraction` and `GMTRouterConversationTurn` in the source for full fields and validation rules.

### DataFormatDetector
```python
from llmrouter.data import DataFormatDetector

detector = DataFormatDetector()
is_valid, format_type, error_msg = detector.validate_and_detect(sample)
```

### Utilities
```python
from llmrouter.data import get_format_requirements, print_format_help, DataFormatType

print(get_format_requirements(DataFormatType.STANDARD))
print_format_help()  # prints all formats
```

## CLI query files
For `llmrouter infer --input` formats and output schemas, see [Data formats](../getting-started/data-formats.md).
