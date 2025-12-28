# Build a custom router

This workflow is about turning an idea into a router that works with the CLI.

## Start from the notebook
The most complete, hands-on guide lives in the notebook:
- [Custom routers (notebook + plugin)](notebooks/custom-router.md)

## Minimal steps (plugin)
1. Create a plugin folder:

   ```text
   custom_routers/my_router/
     __init__.py
     router.py
   ```

2. Implement `route_single` and `route_batch` (see [Routing output contract](../api/routers.md#routing-output-contract)).

3. Verify discovery:

   ```bash
   llmrouter list-routers
   ```

4. Run routing:

   ```bash
   llmrouter infer --router my_router --config configs/model_config_test/smallest_llm.yaml --query "Hello" --route-only
   ```

## Next
- Plugin discovery rules: [Plugin system](../api/plugins.md)
