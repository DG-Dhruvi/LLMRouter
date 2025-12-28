# Architecture

LLMRouter is organized around a small set of core modules: a unified CLI, router implementations, trainer implementations, and a config-driven data loader.

!!! note
    Source links below point to the `main` branch (the docs site is built from `website`).

## Key source files (main branch)
- CLI entrypoint: [llmrouter/cli/router_main.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_main.py)
- Inference registry + helpers: [llmrouter/cli/router_inference.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_inference.py)
- Training registry + helpers: [llmrouter/cli/router_train.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_train.py)
- Chat UI (Gradio): [llmrouter/cli/router_chat.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_chat.py)
- Router base class: [llmrouter/models/meta_router.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/meta_router.py)
- Data loader: [llmrouter/data/data_loader.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/data/data_loader.py)
- API calling (LiteLLM): [llmrouter/utils/api_calling.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/utils/api_calling.py)
- Plugin discovery: [llmrouter/plugin_system.py](https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/plugin_system.py)

## Design principles (why the code is shaped this way)

LLMRouter is designed to make it easy to compare routing algorithms under a shared interface and shared evaluation story.

1. **A stable routing contract**
   - Every router implements `route_single` / `route_batch` and returns a model choice via `model_name` (or an alias key).
   - This lets you swap from KNN to GNN to LLM-based routing without changing the surrounding CLI/workflow.

2. **Separation of decision vs execution**
   - Routing is "which model should answer?"
   - Execution is "call that model (or your gateway) and get a response"
   - `--route-only` exists so you can evaluate routing decisions without coupling to provider endpoints or API keys.

3. **Config-driven reproducibility**
   - Data paths, model paths, and hyperparameters live in YAML configs.
   - This makes experiments easy to version, share, and rerun.

4. **Model metadata as an interface boundary**
   - `llm_data` decouples routing decisions (`model_name`) from provider-specific identifiers (`model`) and endpoints (`api_endpoint`).
   - This is what allows the same router policy to be reused across different backends.

5. **Extensibility without forking**
   - The plugin system registers custom routers into the same registries used by built-in routers.
   - This is intentional: new algorithms should be first-class citizens, not a special case.

## High-level lifecycle
1. A YAML config defines data paths, model paths, and router hyperparameters.
2. The CLI resolves a router name into a concrete router class (and trainer class, if training).
3. The router loads config and data (if initialized with `yaml_path`) and exposes `route_single` / `route_batch`.
4. Trainers implement the training loop for trainable routers.
5. Inference either returns a routing decision (`--route-only`) or calls an external LLM endpoint.

## Configuration-driven router initialization
Most routers inherit from `MetaRouter`. When you construct a router with `yaml_path`:

- YAML is loaded into `self.cfg`
- `DataLoader` attaches datasets as attributes on the router instance (for example `query_data_train`, `llm_data`)
- Missing files are logged as warnings, so routers can still initialize even if some data is optional

See [Data and metrics](data-and-metrics.md) for what those datasets look like.

## Registries and naming
LLMRouter uses registries to map CLI router names to implementations:

- `ROUTER_REGISTRY` (inference): maps `--router` name to router class
- `ROUTER_TRAINER_REGISTRY` (training): maps `--router` name to `(router_class, trainer_class)`
- Aliases are handled by registering multiple keys pointing to the same implementation (for example `routerdc` and `dcrouter`)

Use `llmrouter list-routers` to see what your environment can actually load (especially when optional dependencies are involved).

## Inference flow

### Route-only (`--route-only`)
1. `route_single` runs and returns a routing result.
2. CLI extracts `model_name` from one of the expected keys.
3. Output includes a `routing_result` object and the selected `model_name`.

### Full inference (default)
LLMRouter routes and then calls the selected model via an OpenAI-compatible endpoint:

1. `route_single` runs and returns a `model_name`.
2. The API model id is resolved from `llm_data[model_name].model` (fallback: `model_name`).
3. The endpoint base URL is resolved in this order:
   - `llm_data[model_name].api_endpoint` (per model), else
   - `api_endpoint` in the YAML config
4. `call_api` sends the request via LiteLLM and returns `response`.

!!! tip
    If you only want routing decisions (no API calls, no secrets), always use `--route-only`.

### Multi-round routers and `answer_query`
Some routers expose `answer_query` for an end-to-end multi-round pipeline. In that case, inference uses `answer_query` instead of calling `route_single` + `call_api`.

### RouterR1 special case
`router_r1` requires credentials inside YAML (`hparam.api_base` and `hparam.api_key`) and does not support `--route-only`.

## Routing contract (what `route_single` should return)
Inference extracts the chosen model from one of these keys:

- `model_name`
- `predicted_llm`
- `predicted_llm_name`

If you write a custom router, make sure `route_single` returns one of the keys above.
