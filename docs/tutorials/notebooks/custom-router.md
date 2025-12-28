# Custom routers (notebook + plugin)

This tutorial follows the custom router notebook and shows how to turn an experiment into a reusable plugin.

Notebook: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/custom_router/01_creating_custom_routers.ipynb

See also:
- [Plugin system](../../api/plugins.md)
- [Routing output contract](../../api/routers.md#routing-output-contract)

## What the notebook covers
- Simple rule-based router
- Keyword-based router
- Trainable custom router (example)
- Ensemble router pattern
- File-based inference

## Packaging as a plugin (recommended)
Create a folder under `custom_routers/` (relative to your working directory):

```
custom_routers/
  my_router/
    __init__.py
    router.py
    trainer.py  # optional
```

Minimal `router.py` example:

```python
from llmrouter.models.meta_router import MetaRouter

class MyRouter(MetaRouter):
    def route_single(self, query_input: dict) -> dict:
        q = str(query_input.get(\"query\", \"\"))
        model = \"small\" if len(q) < 80 else \"large\"
        return {\"model_name\": model}

    def route_batch(self, batch):
        return [self.route_single(x) for x in batch]
```

Then verify discovery:

```bash
llmrouter list-routers
llmrouter infer --router my_router --config configs/model_config_test/smallest_llm.yaml --query \"Hello\" --route-only
```

!!! note
    The CLI router name is the plugin folder name (lowercased). For `custom_routers/MyRouter/`, the name becomes `myrouter`.
    See [Plugin system](../../api/plugins.md) for details.

