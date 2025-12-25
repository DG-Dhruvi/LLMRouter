# Plugin system

LLMRouter discovers and registers custom routers at runtime.

## Discovery paths
- `./custom_routers/`
- `~/.llmrouter/plugins/`
- `$LLMROUTER_PLUGINS` (colon-separated)

## API
- `llmrouter.plugin_system.discover_and_register_plugins(plugin_dirs=None, verbose=False)`
- `PluginRegistry.discover_plugins(plugin_dir, verbose=False)`

## Router interface
- Class name ends with `Router`
- Implements `route_single` and `route_batch`
- Optional `Trainer` class in `trainer.py` for training support
