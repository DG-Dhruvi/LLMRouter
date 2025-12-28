# EloRouter

EloRouter is a lightweight ranking-style router. It can be used as a simple baseline or trained from routing supervision.

Notebook (inference): https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/elorouter/01_elorouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/elorouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/elorouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/elorouter.yaml

## Run (CLI)
Train:

```bash
llmrouter train --router elorouter --config configs/model_config_train/elorouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router elorouter --config configs/model_config_test/elorouter.yaml --query "What is retrieval augmented generation?" --route-only
```

Full inference:

```bash
llmrouter infer --router elorouter --config configs/model_config_test/elorouter.yaml --query "What is retrieval augmented generation?"
```

## Tips
- If training fails due to missing files, run [Data preparation](data-preparation.md) or point `data_path` to your own datasets.
- For output schema, see [Data formats](../../getting-started/data-formats.md).

