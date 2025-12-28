# LLMMultiRoundRouter (inference)

LLMMultiRoundRouter is an inference-time, prompt-based multi-round router.
It does not require training artifacts.

Notebook: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/llmmultiroundrouter/01_llmmultiroundrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/llmmultiroundrouter/README.md

## Config
- Test config: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/llmmultiroundrouter.yaml

## Run (CLI)
Route-only routing:

```bash
llmrouter infer --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml --query "Summarize the pros and cons of RLHF." --route-only
```

Full inference:

```bash
llmrouter infer --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml --query "Summarize the pros and cons of RLHF."
```

