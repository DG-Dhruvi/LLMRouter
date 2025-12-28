# Training and evaluation

This page describes the end-to-end workflow for training and evaluating routers.
For the data formats and how supervision is derived, see [Data and metrics](data-and-metrics.md).

## Before you start
- Use `llmrouter list-routers` to see which routers are available for inference and training in your environment.
- Start from the example configs on `main` when possible:
  - Train: [configs/model_config_train](https://github.com/ulab-uiuc/LLMRouter/tree/main/configs/model_config_train)
  - Test: [configs/model_config_test](https://github.com/ulab-uiuc/LLMRouter/tree/main/configs/model_config_test)

## Training workflow
1. Choose a trainable router and a training config.
2. Run `llmrouter train --router <name> --config <path>`.
3. The router loads config and data via `DataLoader`.
4. The trainer runs `train()` and writes to `model_path.save_model_path`.

Example:

```bash
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml
```

### Device selection
- Default is auto-detect (uses GPU if available)
- Override with `--device cuda` or `--device cpu`
- Use `--quiet` to reduce logs

## Inference workflow (for evaluation)
1. Load a router with `llmrouter infer`.
2. Evaluate routing only with `--route-only`, or perform full inference with API calls.
3. Use `--output` and `--output-format` for batch runs.

Route-only example:

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is machine learning?" --route-only
```

Full inference example (requires `API_KEYS` and `api_endpoint`):

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is machine learning?"
```

## Evaluation options

### Offline routing evaluation
- Prefer `--route-only` to measure routing decisions without calling APIs.
- Use this to iterate quickly when tuning `metric.weights` or router hyperparameters.

#### What to measure (algorithmic view)
Offline evaluation is where you can compare routing algorithms fairly, because you can avoid provider noise and keep the decision problem fixed.
Common metrics:

- **Top-1 routing accuracy**: `chosen_model == best_model` (when you define `best_model` by an objective)
- **Objective regret**: `Score(best, q) - Score(chosen, q)` averaged over queries
- **Cost/quality trade-off curves**: average cost vs average performance under different weightings or thresholds
- **Escalation rate (2-model routers)**: fraction of queries routed to the large model (proxy for cost)

If you have per-query per-model metrics (not just a hard label), regret-based metrics are often more informative than accuracy.

### End-to-end evaluation
- Run full inference (no `--route-only`) to evaluate real outputs and costs.
- This path depends on your provider endpoint and `API_KEYS` setup.

#### What changes in end-to-end evaluation?
Once you start calling models, you introduce additional variance:

- stochastic decoding (temperature, sampling)
- provider-side changes and rate limits
- prompt formatting and parsing effects

For research comparisons, it helps to keep `--route-only` for algorithm iteration and reserve end-to-end evaluation for final validation.

### Evaluation utilities
The `main` branch includes an evaluation package with scripts/examples:

- [llmrouter/evaluation](https://github.com/ulab-uiuc/LLMRouter/tree/main/llmrouter/evaluation)
- [llmrouter/evaluation/README.md](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/evaluation/README.md)

## Artifacts and overrides
- Training checkpoints are written to `model_path.save_model_path`.
- For inference, `--load_model_path` overrides `model_path.load_model_path` in the config.

## Common pitfalls
- Missing data files in `data_path` will cause warnings or errors during router initialization.
- Some routers have optional dependencies; if a router is missing, verify with `llmrouter list-routers`.
- Full inference requires `API_KEYS` to be set for external API calls.
- `--route-only` returns routing decisions and never calls any external endpoint.

## Reproducibility tips
- Version your config files along with data snapshots.
- Keep router name, config, and the saved model artifact together for each experiment.
