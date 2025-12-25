# Routers

Use `llmrouter list-routers` to see which routers are available in your environment. Some routers require optional dependencies.

## Train + infer routers
- knnrouter
- svmrouter
- mlprouter
- mfrouter
- elorouter
- dcrouter (alias: routerdc)
- automixrouter (alias: automix)
- hybrid_llm (alias: hybridllm, optional)
- graphrouter (alias: graph_router, optional)
- causallm_router (alias: causallmrouter, optional)
- gmtrouter (alias: gmt_router, optional)
- knnmultiroundrouter

## Inference-only routers
- smallest_llm
- largest_llm
- llmmultiroundrouter

## Pretrained or external
- router_r1 (alias: router-r1). Requires api_base and api_key in config.
