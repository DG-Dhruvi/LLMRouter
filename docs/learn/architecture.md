# Architecture

LLMRouter is organized around a small set of core modules: a CLI entry point, router classes, trainer classes, and a data loader that binds config data to routers.

## High-level flow
1. A YAML config defines data paths, model paths, and router hyperparameters.
2. The CLI resolves a router name into a concrete router class.
3. The router loads data (if a config is provided) and exposes `route_single` and `route_batch`.
4. Trainers implement the training loop for trainable routers.
5. Inference either returns a routing decision or calls an API using `llmrouter.utils.call_api`.

## Core components

### CLI layer
- Entry point: `llmrouter.cli.router_main`
- Subcommands: `train`, `infer`, `chat`, `list-routers`, `version`
- Registry wiring:
  - `llmrouter.cli.router_train` maps router names to trainer classes.
  - `llmrouter.cli.router_inference` maps router names to router classes.

### Router layer
- Base class: `llmrouter.models.meta_router.MetaRouter`
- Required methods: `route_single`, `route_batch`
- Optional helpers: `compute_metrics`, `save_router`, `load_router`
- If `yaml_path` is provided, the router loads config and data on init.

### Trainer layer
- Base class: `llmrouter.models.base_trainer.BaseTrainer`
- Trainers live next to router implementations and encapsulate training logic.

### Data layer
- `llmrouter.data.DataLoader` resolves paths and attaches datasets to the router instance.

### Plugin layer
- `llmrouter.plugin_system` discovers custom routers in `custom_routers/` or user plugin paths and registers them into CLI registries.

## Configuration-driven initialization
When `yaml_path` is provided, `MetaRouter` loads the YAML into `self.cfg` and calls `DataLoader` to attach datasets as attributes. Missing files are logged as warnings so a router can still initialize even if some data is optional.

## Registries and naming
- `ROUTER_REGISTRY` maps CLI router names to router classes for inference.
- `ROUTER_TRAINER_REGISTRY` maps names to `(router_class, trainer_class)` for training.
- Aliases (for example `routerdc` for `dcrouter`) are registered in these dicts.

Use `llmrouter list-routers` to see the resolved names in your environment.

## Execution paths

### Train
1. `llmrouter train` parses args.
2. Router and trainer classes are selected from the registry.
3. Router loads config and data.
4. Trainer runs `train()` and writes to `model_path.save_model_path`.

### Inference
1. `llmrouter infer` loads the router from config.
2. `route_single` runs and returns a model name.
3. If not `--route-only`, `call_api` sends the query to the selected model.

### Chat
1. `llmrouter chat` loads the router.
2. Gradio wraps the router to serve interactive queries.

## Routing contract
Routing outputs should include one of these keys so inference can extract the model name:
- `model_name`
- `predicted_llm`
- `predicted_llm_name`

Custom routers should return one of the keys above from `route_single` and `route_batch` results.
