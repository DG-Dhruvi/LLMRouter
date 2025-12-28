# First routing run

This walkthrough gets you from "repo clone" to "routing decisions" in a few minutes.
It uses KNNRouter because it is fast and easy to debug.

## Related notebook tutorials
- [Data preparation](notebooks/data-preparation.md)
- [KNNRouter](notebooks/knnrouter.md)

## Prereqs
- Install from source so you have `configs/` and `data/` (see [Installation](../getting-started/installation.md)).
- If you want full inference (API calls), set `API_KEYS` (see [Installation](../getting-started/installation.md)).

## Step 1: (Optional) Prepare data
If you are fine with the example files in `data/example_data/`, you can skip this step.
Otherwise, run the data preparation notebook:
- https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/data_preparation/01_data_preparation.ipynb

## Step 2: Train a router (KNN)
```bash
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml
```

This should create a model artifact under `saved_models/knnrouter/` (as configured in the YAML).

## Step 3: Route a single query (no API calls)
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "Explain transformers." --route-only
```

## Step 4: (Optional) Full inference (routes + calls the selected model)
1. Ensure your `llm_data` has `api_endpoint` and `model` for each candidate model.
2. Set `API_KEYS`.

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "Explain transformers."
```

## Step 5: Batch routing/inference
Create `queries.jsonl`:

```jsonl
{"query":"What is machine learning?"}
{"query":"Explain transformers."}
```

Route-only batch run:

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input queries.jsonl --output results.jsonl --output-format jsonl --route-only
```

Input/output formats are documented in [Data formats](../getting-started/data-formats.md).

## Next
- Learn how to interpret the router output: [Data formats](../getting-started/data-formats.md)
- Explore other routers: [Routers](../api/routers.md#router-table)
