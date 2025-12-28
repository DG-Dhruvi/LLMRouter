# GMTRouter (training + inference)

GMTRouter is a specialized router for personalized and/or multi-turn settings, based on the GMTRouter project.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/gmtrouter/01_gmtrouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/gmtrouter/02_gmtrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/gmtrouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/gmtrouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/gmtrouter.yaml

## Notes on data format
GMTRouter uses a dedicated JSONL schema (with embeddings and ratings). See:
- [Data utilities](../../api/data.md)

## Run (CLI)
Train:

```bash
llmrouter train --router gmtrouter --config configs/model_config_train/gmtrouter.yaml
```

Inference:

```bash
llmrouter infer --router gmtrouter --config configs/model_config_test/gmtrouter.yaml --query "Explain transformers." --route-only
```

!!! note
    The GMTRouter configs expect an external dataset/checkpoint layout. Follow the notebook and the config comments to set paths correctly.

