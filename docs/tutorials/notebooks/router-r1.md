# RouterR1 (inference)

RouterR1 is an agentic router with a special runtime (vLLM + external routing service).

Notebook: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/router_r1/01_router_r1_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/router_r1/README.md

## Config
- Test config: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/router_r1.yaml

## Important behavior
- `router_r1` does not support `--route-only` in the CLI.
- It requires `hparam.api_base` and `hparam.api_key` in the YAML config.

## Run (CLI)
```bash
llmrouter infer --router router_r1 --config configs/model_config_test/router_r1.yaml --query "Explain transformers."
```

!!! warning
    RouterR1 requires CUDA (vLLM GPU runtime) and optional dependencies. Follow the notebook and the router README for setup.

