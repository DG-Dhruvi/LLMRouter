# Router families

Routers make model selection decisions with different assumptions and costs. This page groups routers by how they behave and how much data or training they require.

## Category overview
| Category | When to use | Examples |
| --- | --- | --- |
| Baselines | sanity checks and lower bounds | `smallest_llm`, `largest_llm` |
| Classic ML / embedding | fast, interpretable, strong baselines | `knnrouter`, `svmrouter`, `mlprouter`, `mfrouter` |
| Ranking and preference | preference-driven selection | `elorouter`, `dcrouter` |
| Hybrid or probabilistic | combine multiple signals | `hybrid_llm` |
| Graph-based | model relations or user preferences | `graphrouter`, `gmtrouter` |
| LLM-driven | use an LLM to decide or route | `causallm_router`, `llmmultiroundrouter`, `router_r1` |
| Mixing and ensembling | blend multiple models | `automix` / `automixrouter` |
| Multi-round and agentic | multi-turn or tool-like routing | `knnmultiroundrouter`, `llmmultiroundrouter`, `router_r1` |

## Data and compute needs
- Baselines require no training data and typically only use `llm_data`.
- Classic ML routers usually rely on embeddings and routing labels.
- Ranking routers depend on preference-like signals.
- LLM-driven routers may require API access at inference time.
- Multi-round routers incorporate conversation context; see router-specific configs.

## Router availability
| Router | Train | Infer | Notes |
| --- | --- | --- | --- |
| `knnrouter` | yes | yes | baseline, fast |
| `svmrouter` | yes | yes | linear classifier |
| `mlprouter` | yes | yes | MLP classifier |
| `mfrouter` | yes | yes | matrix factorization |
| `elorouter` | yes | yes | pairwise ranking |
| `dcrouter` (`routerdc`) | yes | yes | dual contrastive routing |
| `automix` (`automixrouter`) | yes | yes | auto mixing |
| `hybrid_llm` (`hybridllm`) | yes | yes | optional dependencies |
| `graphrouter` (`graph_router`) | yes | yes | optional dependencies |
| `causallm_router` (`causallmrouter`) | yes | yes | optional dependencies |
| `gmtrouter` (`gmt_router`) | yes | yes | optional dependencies, personalized |
| `knnmultiroundrouter` | yes | yes | multi-round |
| `llmmultiroundrouter` | no | yes | LLM-based multi-round |
| `router_r1` (`router-r1`) | no | yes | pretrained, needs `api_base` and `api_key` |
| `smallest_llm` | no | yes | always chooses smallest model |
| `largest_llm` | no | yes | always chooses largest model |

Use `llmrouter list-routers` to see what is available in your environment, especially when optional dependencies are involved.

## Trade-offs
- Speed vs accuracy: smaller models and simple routers are fast but can be less accurate.
- Cost vs quality: routing to smaller models reduces cost but may reduce quality.
- Interpretability: classical models are easier to inspect than neural or LLM-driven routers.
