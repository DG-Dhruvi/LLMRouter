# Plugin system

LLMRouter discovers and registers custom routers at runtime, so you can extend the framework without modifying core code.

## Discovery paths
- `./custom_routers/`
- `~/.llmrouter/plugins/`
- `$LLMROUTER_PLUGINS` (colon-separated list of directories)

!!! tip
    On Windows, prefer `./custom_routers/` or `~/.llmrouter/plugins/`. The environment variable path parsing is Unix-oriented.

## How discovery works
- Plugin directories are added to `sys.path`.
- The system looks for a Router class in `__init__.py`, `router.py`, or `model.py`.
- A class is accepted if its name ends with `Router` and it implements `route_single` and `route_batch`.
- If `trainer.py` defines a class ending with `Trainer`, it is registered for training.

## Naming
The CLI router name is the plugin folder name (lowercased), not the Python class name.
For example, `custom_routers/MyRouter/` becomes `--router myrouter`.

## Directory layout
```
custom_routers/
  my_router/
    __init__.py
    router.py
    trainer.py  # optional
```

## Router interface
Custom routers should implement `route_single` and `route_batch` and expose a class name ending with `Router`.

```python
from llmrouter.models.meta_router import MetaRouter

class MyRouter(MetaRouter):
    def route_single(self, query_input: dict) -> dict:
        return {"model_name": "model_a"}

    def route_batch(self, batch: list) -> list:
        return [self.route_single(q) for q in batch]
```

## Trainer interface (optional)
If you provide a trainer, implement a class ending with `Trainer` in `trainer.py` and follow the `BaseTrainer` interface.

## Using plugins
1. Place the plugin under a discovery path.
2. Run `llmrouter list-routers` to verify it was loaded.
3. Use `llmrouter train` or `llmrouter infer` with the custom router name.

## Debugging
Use `discover_and_register_plugins(verbose=True)` to see discovery details during development.
