# GraphRouter (training + inference)

GraphRouter is a graph neural network (GNN) router. It can capture relational structure beyond independent per-query decisions.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/graphrouter/01_graphrouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/graphrouter/02_graphrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/graphrouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/graphrouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/graphrouter.yaml

!!! note
    GraphRouter may require optional dependencies. If `llmrouter list-routers` does not show it, check the router README and install the extra requirements.

## Run (CLI)
Train:

```bash
llmrouter train --router graphrouter --config configs/model_config_train/graphrouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router graphrouter --config configs/model_config_test/graphrouter.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router graphrouter --config configs/model_config_test/graphrouter.yaml --query "Explain transformers."
```

