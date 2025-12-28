# KNNMultiRoundRouter (training + inference)

KNNMultiRoundRouter extends KNN routing to multi-round workflows (decomposition, per-step execution, aggregation).

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/knnmultiroundrouter/01_knnmultiroundrouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/knnmultiroundrouter/02_knnmultiroundrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/knnmultiroundrouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/knnmultiroundrouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/knnmultiroundrouter.yaml

## Run (CLI)
Train:

```bash
llmrouter train --router knnmultiroundrouter --config configs/model_config_train/knnmultiroundrouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router knnmultiroundrouter --config configs/model_config_test/knnmultiroundrouter.yaml --query "Plan a 3-step study schedule for linear algebra." --route-only
```

Full inference:

```bash
llmrouter infer --router knnmultiroundrouter --config configs/model_config_test/knnmultiroundrouter.yaml --query "Plan a 3-step study schedule for linear algebra."
```

!!! note
    Full inference may call external LLM APIs during decomposition/execution. Set `API_KEYS` and ensure your `llm_data` has `api_endpoint` and `model` fields.

