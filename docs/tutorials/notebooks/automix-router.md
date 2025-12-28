# AutoMixRouter (training + inference)

AutoMixRouter targets cost-aware routing between small and large models. Training is parameter search (not gradient descent).

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/automix_router/01_automix_router_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/automix_router/02_automix_router_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/automix/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/automix.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/automix.yaml

## Router name in the CLI
- Training registry includes `automix` (and `automixrouter`).
- Inference registry uses `automixrouter`.

If you are unsure, run `llmrouter list-routers`.

## Run (CLI)
Train:

```bash
llmrouter train --router automix --config configs/model_config_train/automix.yaml
```

Route-only inference:

```bash
llmrouter infer --router automixrouter --config configs/model_config_test/automix.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router automixrouter --config configs/model_config_test/automix.yaml --query "Explain transformers."
```

## What to tweak
- `hparam.routing_method`: choose the routing method (for example, POMDP vs threshold-based).
- `hparam.small_model_cost` / `hparam.large_model_cost`: encode your cost assumptions.

