# Data utilities

## DataLoader
`llmrouter.data.DataLoader` attaches datasets to a router when a config is loaded.

Keys loaded from `data_path`:
- query_data_train
- query_data_test
- query_embedding_data
- routing_data_train
- routing_data_test
- llm_data
- llm_embedding_data

Paths are resolved relative to the repo root.

## Input formats
`llmrouter infer --input` supports:
- .txt (one query per line)
- .json (list of strings or list of objects with `query`)
- .jsonl (one JSON object per line)
