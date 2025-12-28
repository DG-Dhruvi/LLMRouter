# Routers

## Summary
Routers implement the routing interface and are addressed by name in the CLI. Some routers are trainable, while others are inference-only baselines or pretrained models.

Sources:
- Inference registry: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_inference.py
- Training registry: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_train.py

!!! note "Key responsibilities"
    - Provide `route_single` and `route_batch`
    - Return a model name in the routing output
    - Integrate with CLI registries for training and inference

## Router table
| Router | Train | Infer | Docs | Notes |
| --- | --- | --- | --- | --- |
| `knnrouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/knnrouter/README.md) | baseline, fast |
| `svmrouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/svmrouter/README.md) | linear classifier |
| `mlprouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/mlprouter/README.md) | MLP classifier |
| `mfrouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/mfrouter/README.md) | matrix factorization |
| `elorouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/elorouter/README.md) | pairwise ranking |
| `dcrouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/routerdc/README.md) | alias: `routerdc` |
| `automix` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/automix/README.md) | alias: `automixrouter` |
| `hybrid_llm` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/hybrid_llm/README.md) | alias: `hybridllm`, optional deps |
| `graphrouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/graphrouter/README.md) | alias: `graph_router`, optional deps |
| `causallm_router` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/causallm_router/README.md) | alias: `causallmrouter`, optional deps |
| `gmtrouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/gmtrouter/README.md) | alias: `gmt_router`, optional deps |
| `knnmultiroundrouter` | yes | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/knnmultiroundrouter/README.md) | multi-round |
| `llmmultiroundrouter` | no | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/llmmultiroundrouter/README.md) | LLM-based multi-round |
| `router_r1` | no | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/router_r1/README.md) | alias: `router-r1`, requires `api_base` and `api_key` |
| `smallest_llm` | no | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/smallest_llm/README.md) | baseline |
| `largest_llm` | no | yes | [README](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/largest_llm/README.md) | baseline |

Train/Infer reflect the `main` branch CLI registries (what `llmrouter train` / `llmrouter infer` support).
If you installed a released version from PyPI, run `llmrouter list-routers` to see what is available in your environment.

## Routing output contract
Inference extracts the routed model name from one of these keys:
- `model_name`
- `predicted_llm`
- `predicted_llm_name`

Custom routers should return one of these keys to ensure compatibility.

## Optional dependencies
Some routers are optional and may be unavailable if their dependencies are not installed. Use `llmrouter list-routers` to confirm availability.

## Example configs (on `main`)
The ready-to-run YAML configs live on the `main` branch:
- Train configs: https://github.com/ulab-uiuc/LLMRouter/tree/main/configs/model_config_train
- Test/inference configs: https://github.com/ulab-uiuc/LLMRouter/tree/main/configs/model_config_test
