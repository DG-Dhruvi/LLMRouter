# CausalLMRouter (training + inference)

CausalLMRouter is a heavier router that finetunes a causal language model for routing decisions.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/causallm_router/01_causallm_router_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/causallm_router/02_causallm_router_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/causallm_router/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/causallm_router.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/causallm_router.yaml

!!! warning
    Training and inference are GPU-heavy. The config includes vLLM-related settings and a merged checkpoint path.
    Follow the notebook for the most reliable end-to-end run.

## Run (CLI)
Train:

```bash
llmrouter train --router causallm_router --config configs/model_config_train/causallm_router.yaml
```

Inference:

```bash
llmrouter infer --router causallm_router --config configs/model_config_test/causallm_router.yaml --query "Explain transformers." --route-only
```

