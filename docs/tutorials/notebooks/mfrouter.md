# MFRouter (training + inference)

MFRouter uses a matrix-factorization style objective for routing. It can be a good fit when routing supervision looks like preferences or scores.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/mfrouter/01_mfrouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/mfrouter/02_mfrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/mfrouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/mfrouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/mfrouter.yaml

!!! note
    The inference notebook writes a minimal `mfrouter_inference.yaml` for convenience. For the CLI, you can also use the repo test config above.

## Run (CLI)
Train:

```bash
llmrouter train --router mfrouter --config configs/model_config_train/mfrouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router mfrouter --config configs/model_config_test/mfrouter.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router mfrouter --config configs/model_config_test/mfrouter.yaml --query "Explain transformers."
```

