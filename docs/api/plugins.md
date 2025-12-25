# Plugin system

LLMRouter discovers and registers custom routers at runtime.

!!! note "Key responsibilities"
    - Discover custom routers from plugin directories
    - Validate router interfaces (`route_single`, `route_batch`)
    - Register routers into CLI registries

## Discovery paths
- `./custom_routers/`
- `~/.llmrouter/plugins/`
- `$LLMROUTER_PLUGINS` (colon-separated)

## Discovery behavior
- Adds plugin directories to `sys.path`.
- Looks for Router classes in `__init__.py`, `router.py`, or `model.py`.
- Looks for Trainer classes in `trainer.py`.

## Core API
### discover_and_register_plugins
```python
from llmrouter.plugin_system import discover_and_register_plugins

registry = discover_and_register_plugins(plugin_dirs=None, verbose=False)
```

Parameters:
- `plugin_dirs`: optional list of directories to scan
- `verbose`: enable discovery logs

### PluginRegistry
Key methods:
| Method | Purpose |
| --- | --- |
| `discover_plugins(plugin_dir, verbose=False)` | scan a directory for plugins |
| `register_to_dict(target_dict)` | register routers into a registry |
| `get_router_names()` | list discovered router names |
| `get_router(name)` | get a router class by name |

## Router interface
- Class name ends with `Router`
- Implements `route_single` and `route_batch`
- Optional trainer class in `trainer.py`

## Example layout
```
custom_routers/
  my_router/
    __init__.py
    router.py
    trainer.py  # optional
```
