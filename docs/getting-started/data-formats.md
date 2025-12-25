# Data formats

`llmrouter infer --input` supports:

- `.txt`: one query per line
- `.json`: a list of strings, or a list of objects with a `query` field
- `.jsonl`: one JSON object per line, each with `query` (or a raw string)

## Examples

### Text
```
What is machine learning?
Explain transformers.
```

### JSON
```
[
  "What is machine learning?",
  {"query": "Explain transformers."}
]
```

### JSONL
```
{"query": "What is machine learning?"}
{"query": "Explain transformers."}
```
