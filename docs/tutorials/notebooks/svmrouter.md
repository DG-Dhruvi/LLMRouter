# SVMRouter (training + inference)

SVMRouter is a classic classifier router that works well with high-dimensional embeddings.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/svmrouter/01_svmrouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/svmrouter/02_svmrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/svmrouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/svmrouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/svmrouter.yaml

!!! note
    The inference notebook writes a minimal `svmrouter_inference.yaml` for convenience. For the CLI, you can also use the repo test config above.

## Run (CLI)
Train:

```bash
llmrouter train --router svmrouter --config configs/model_config_train/svmrouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router svmrouter --config configs/model_config_test/svmrouter.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router svmrouter --config configs/model_config_test/svmrouter.yaml --query "Explain transformers."
```

## What to tweak
Tune the `hparam` section in the YAML (kernel/regularization) and the `metric.weights` trade-offs.

