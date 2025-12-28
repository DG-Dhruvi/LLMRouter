# Evaluation workflow

This page is a practical checklist for comparing routers on the same query set.
For the underlying concepts (metrics, supervision, trade-offs), see [Training and evaluation](../learn/training-and-evaluation.md).

## 1) Prepare an evaluation file
Use JSONL and include stable IDs if you plan to join with labels later:

```jsonl
{"query_id":"q1","query":"What is machine learning?"}
{"query_id":"q2","query":"Explain transformers."}
```

## 2) Run routing-only for each router
Route-only runs are fast and avoid API costs:

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input eval.jsonl --output knn.jsonl --output-format jsonl --route-only
llmrouter infer --router svmrouter --config configs/model_config_test/svmrouter.yaml --input eval.jsonl --output svm.jsonl --output-format jsonl --route-only
```

## 3) Aggregate results
At minimum, you can compute:
- routing distribution (how often each model is selected)
- disagreement between routers

If you also have routing labels (for example, a `best_model` per query), you can compute simple accuracy by comparing `model_name` against that label.

## 4) (Optional) Full inference
Full inference can be useful for end-to-end validation, but it is slower and requires API credentials.

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input eval.jsonl --output knn_full.jsonl --output-format jsonl
```

## Next
- Notebook workflows live under `main/notebooks`: https://github.com/ulab-uiuc/LLMRouter/tree/main/notebooks
- Router capabilities and READMEs: [Routers](../api/routers.md#router-table)
