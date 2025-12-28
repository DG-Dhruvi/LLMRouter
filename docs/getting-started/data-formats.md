# Data formats

This page documents the input and output formats used by `llmrouter infer`.

## Input formats (`--input`)

`llmrouter infer --input <file>` supports:

- `.txt`: one query per line
- `.json`: a list of strings, or a list of objects that contain a `query` field
- `.jsonl`: one JSON value per line - either an object with `query`, or a raw string

Extra fields in objects are allowed (they are ignored during loading; only `query` is used).

### Examples

**Text (`.txt`)**

```text
What is machine learning?
Explain transformers.
```

**JSON (`.json`)**

```json
[
  "What is machine learning?",
  {"query": "Explain transformers.", "id": "q2"}
]
```

**JSONL (`.jsonl`)**

```jsonl
{"query": "What is machine learning?"}
{"query": "Explain transformers.", "metadata": {"source": "faq"}}
```

## Output formats (`--output-format`)

For batch inference, `--output-format` controls how results are written to `--output`:

- `json` (default): a JSON array
- `jsonl`: one JSON object per line

!!! note
    When you pass `--output`, LLMRouter always writes a list of results (even for a single query).
    When you do not pass `--output`, a single query prints a single JSON object to stdout.

### Output schema

The exact content depends on flags and router type, but the common shapes are:

**Route-only (`--route-only`)**

```json
{
  "success": true,
  "query": "Explain transformers.",
  "model_name": "qwen2.5-7b-instruct",
  "routing_result": {"model_name": "qwen2.5-7b-instruct"}
}
```

**Full inference (default)**

```json
{
  "success": true,
  "query": "Explain transformers.",
  "model_name": "qwen2.5-7b-instruct",
  "api_model_name": "qwen/qwen2.5-7b-instruct",
  "response": "...",
  "routing_result": {"model_name": "qwen2.5-7b-instruct"}
}
```

On failures, results include `error` and may include `traceback`.

!!! tip
    If you only want routing decisions (no API calls, no API keys), always add `--route-only`.

## Next

- For a practical batch run, see [Batch inference](../tutorials/batch-inference.md).
- For all CLI flags, see [CLI reference](../api/cli.md).
