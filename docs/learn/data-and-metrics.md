# Data and metrics

LLMRouter is config-driven: your YAML points to data files, and the framework loads them and attaches the results to the router instance.

The loader implementation lives here (main branch):
[llmrouter/data/data_loader.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/data/data_loader.py).

## How paths are resolved
- Absolute paths are used as-is.
- Relative paths are resolved against the LLMRouter "project root" (the directory containing the `llmrouter/` package).

!!! tip
    If you installed from PyPI (not a repo clone), prefer absolute paths so your configs can find local data files.

## Loaded datasets (attributes on the router instance)

| Attribute on router | YAML key (`data_path`) | Typical format | Purpose |
| --- | --- | --- | --- |
| `query_data_train` | `query_data_train` | jsonl | training queries |
| `query_data_test` | `query_data_test` | jsonl | test queries |
| `query_embedding_data` | `query_embedding_data` | pt | query embeddings |
| `routing_data_train` | `routing_data_train` | jsonl | routing labels and metrics |
| `routing_data_test` | `routing_data_test` | jsonl | routing labels and metrics |
| `llm_data` | `llm_data` | json | candidate model metadata |
| `llm_embedding_data` | `llm_embedding_data` | json | model embeddings |

## Query data (`query_data_train` / `query_data_test`)

Query datasets are JSONL. The only universally required field is `query`.

Minimal example:

```jsonl
{"query": "Explain transformers."}
```

Many datasets include extra fields (for example `task_name`, `ground_truth`, `choices`, `metric`). Some routers and evaluators use these fields.

!!! note
    `llmrouter infer --input ...` only loads query strings (it ignores extra fields and keeps only `query`).
    Training pipelines may use the richer per-row objects loaded by `DataLoader`.

## Routing data (`routing_data_train` / `routing_data_test`)

Routing datasets are JSONL and are converted into a tabular form. The exact schema depends on the router/trainer, but common columns include:

- which model was used or is optimal (for example `model_name`, `best_model`)
- one or more numeric metrics (for example `performance`, `cost`, `llm_judge`)

Minimal example:

```jsonl
{"query": "Explain transformers.", "model_name": "small", "performance": 0.82, "cost": 0.05}
```

## From logs to supervision (how routing becomes learnable)

Most "trainable" routers need a supervision signal derived from routing logs.
LLMRouter supports multiple philosophies for extracting that signal:

### 1) Classification label: "best model"
Used by embedding classifiers (SVM/MLP) and sometimes as a target for other methods.

Given per-query measurements for each model, define a scalar objective and take the winner:

`best_model(q) = argmax_m Score(m, q)`

### 2) Pairwise preferences: "winner beats loser"
Used by ranking-style routers (Elo, matrix factorization) and preference learning setups.

For each query, build comparisons like:

- (winner, loser) for all other models, or
- a sampled subset of comparisons for efficiency

### 3) Top-k vs last-k sets
Used by contrastive approaches (for example RouterDC), where "positives" are top performers and "negatives" are bottom performers.

### 4) Cost-aware gating labels (2-model routing)
Used by small-vs-large routers (Hybrid LLM, Automix).
Instead of picking among many models, the signal is "is the small model good enough?"

!!! note
    Different routers consume different supervision formats. The safest workflow is: start from the router's README (linked from [API Reference - Routers](../api/routers.md)) and use the example config/data as a template.

## Candidate models (`llm_data`)

`llm_data` is a JSON object mapping `model_name` -> metadata.

For full inference (not `--route-only`), LLMRouter expects:

- `model`: provider-specific model id (used as `api_model_name`)
- `api_endpoint`: OpenAI-compatible base URL for the provider (per model), or `api_endpoint` in YAML config

Example:

```json
{
  "small": {
    "size": "7B",
    "model": "qwen/qwen2.5-7b-instruct",
    "api_endpoint": "https://integrate.api.nvidia.com/v1"
  }
}
```

See the full example used by the `main` branch configs:
[data/example_data/llm_candidates/default_llm.json](https://github.com/ulab-uiuc/LLMRouter/blob/main/data/example_data/llm_candidates/default_llm.json).

## Metric weights (`metric.weights`)

`metric.weights` is how you tell a trainer what to optimize (for example quality vs cost).
How weights are used depends on the router and trainer implementation.

Example:

```yaml
metric:
  weights:
    performance: 1
    cost: 0
    llm_judge: 0
```

## A simple objective you can reason about

Many workflows can be viewed as optimizing a weighted objective per (model, query):

`Score(m, q) = w_perf * Perf(m, q) - w_cost * Cost(m, q) + w_judge * Judge(m, q)`

Practical notes:

- Make sure your metrics are on comparable scales (normalization matters).
- Be explicit about the sign convention for cost: cost is usually "lower is better", so it often appears with a minus sign.
- If you add new metrics, document them and update your label-generation logic consistently.

## Embeddings

Embedding-based routers typically rely on:

- `query_embedding_data` (PyTorch `.pt` file)
- `llm_embedding_data` (JSON file)

Make sure the embedding formats match what the router expects.

## Next
- For config keys: [Config reference](../api/config.md)
- For training workflow: [Training and evaluation](training-and-evaluation.md)
