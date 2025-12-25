# Routers

## Summary
Routers implement the routing interface and are addressed by name in the CLI. Some routers are trainable, while others are inference-only baselines or pretrained models.

!!! note "Key responsibilities"
    - Provide `route_single` and `route_batch`
    - Return a model name in the routing output
    - Integrate with CLI registries for training and inference

## Router table
| Router | Train | Infer | Notes |
| --- | --- | --- | --- |
| `knnrouter` | yes | yes | baseline, fast |
| `svmrouter` | yes | yes | linear classifier |
| `mlprouter` | yes | yes | MLP classifier |
| `mfrouter` | yes | yes | matrix factorization |
| `elorouter` | yes | yes | pairwise ranking |
| `dcrouter` | yes | yes | alias: `routerdc` |
| `automix` | yes | yes | infer name: `automixrouter` |
| `hybrid_llm` | yes | yes | alias: `hybridllm`, optional deps |
| `graphrouter` | yes | yes | alias: `graph_router`, optional deps |
| `causallm_router` | yes | yes | alias: `causallmrouter`, optional deps |
| `gmtrouter` | yes | yes | alias: `gmt_router`, optional deps |
| `knnmultiroundrouter` | yes | yes | multi-round |
| `llmmultiroundrouter` | no | yes | LLM-based multi-round |
| `router_r1` | no | yes | alias: `router-r1`, requires `api_base` and `api_key` |
| `smallest_llm` | no | yes | baseline |
| `largest_llm` | no | yes | baseline |

## Routing output contract
Inference extracts the routed model name from one of these keys:
- `model_name`
- `predicted_llm`
- `predicted_llm_name`

Custom routers should return one of these keys to ensure compatibility.

## Optional dependencies
Some routers are optional and may be unavailable if their dependencies are not installed. Use `llmrouter list-routers` to confirm availability.
