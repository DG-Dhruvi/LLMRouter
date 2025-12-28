# MLPRouter (training + inference)

MLPRouter is a neural classifier router. It can learn non-linear routing boundaries on top of query embeddings.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/mlprouter/01_mlprouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/mlprouter/02_mlprouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/mlprouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/mlprouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/mlprouter.yaml

!!! note
    The inference notebook writes a minimal `mlprouter_inference.yaml` for convenience. For the CLI, you can also use the repo test config above.

## Run (CLI)
Train:

```bash
llmrouter train --router mlprouter --config configs/model_config_train/mlprouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router mlprouter --config configs/model_config_test/mlprouter.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router mlprouter --config configs/model_config_test/mlprouter.yaml --query "Explain transformers."
```

## Tips
- If you overfit quickly, reduce model size or add regularization (see `hparam`).
- If training is slow, try fewer epochs first.

