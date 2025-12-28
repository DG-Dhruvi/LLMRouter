# LargestLLM (baseline)

LargestLLM is a baseline router: it always routes to the largest candidate model (based on the `size` field in `llm_data`).

Notebook: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/largest_llm/01_largest_llm_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/largest_llm/README.md

## Config
- Test config: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/largest_llm.yaml

!!! warning
    The notebook currently references `configs/model_config_train/largest_llm.yaml`, but the repo ships the baseline config under `configs/model_config_test/`.
    Use the test config above (or change the notebook path).

## Run (CLI)
Route-only (no API calls):

```bash
llmrouter infer --router largest_llm --config configs/model_config_test/largest_llm.yaml --query "Explain transformers." --route-only
```

Full inference (routes + calls the selected model):
- Ensure `llm_data` contains `api_endpoint` and `model` for the routed model.
- Set `API_KEYS` (see [Installation](../../getting-started/installation.md)).

```bash
llmrouter infer --router largest_llm --config configs/model_config_test/largest_llm.yaml --query "Explain transformers."
```

## Next
- Compare with [SmallestLLM](smallest-llm.md).
- See the router table: [Routers](../../api/routers.md#router-table).

