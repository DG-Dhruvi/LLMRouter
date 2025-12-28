# Quickstart

This page gives you two fast paths:

- Route-only (no API calls): see which model a router selects
- Full inference: route + call the selected model via an OpenAI-compatible endpoint

!!! tip
    If you install from PyPI (not a repo clone), prefer absolute paths in your YAML configs. See [Config basics](config-basics.md).

## 0) Recommended: use the example configs on `main`

The ready-to-run configs and example data live on the `main` branch:

- `smallest_llm` (inference-only): [configs/model_config_test/smallest_llm.yaml](https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/smallest_llm.yaml)
- `knnrouter` (train): [configs/model_config_train/knnrouter.yaml](https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/knnrouter.yaml)
- `knnrouter` (test/infer): [configs/model_config_test/knnrouter.yaml](https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/knnrouter.yaml)

If you cloned the repo, you can use `configs/...` paths directly.

## Path A: Route-only in 30 seconds (`smallest_llm`)

`smallest_llm` is a baseline router that always picks the smallest candidate model (based on the `size` field in your `llm_data` JSON).

### 1) Prepare `llm_data` (candidate models)

Start from the example file (and edit it to your provider/models):

- [data/example_data/llm_candidates/default_llm.json](https://github.com/ulab-uiuc/LLMRouter/blob/main/data/example_data/llm_candidates/default_llm.json)

At minimum, each model entry should include `size` (e.g., `7B`, `70B`). For full inference, also include `model` and `api_endpoint`.

### 2) Create a minimal YAML config

Create `smallest_llm.yaml`:

```yaml
data_path:
  llm_data: "/absolute/path/to/llm_data.json"
```

### 3) Run route-only inference

```bash
llmrouter infer --router smallest_llm --config smallest_llm.yaml --query "Explain transformers." --route-only
```

The output includes `model_name` and `routing_result`.

## Path B: Train + infer baseline (`knnrouter`)

### 1) Train

```bash
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml
```

### 2) Sanity check with route-only inference (no API keys needed)

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is machine learning?" --route-only
```

### 3) Full inference (requires `API_KEYS` and `api_endpoint`)

**macOS/Linux**

```bash
export API_KEYS="YOUR_KEY"
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is machine learning?"
```

**Windows PowerShell**

```powershell
$env:API_KEYS="YOUR_KEY"
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is machine learning?"
```

## Batch inference

Input file formats are documented in [Data formats](data-formats.md).

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input queries.jsonl --output results.jsonl --output-format jsonl --route-only
```

## Chat UI

```bash
llmrouter chat --router knnrouter --config configs/model_config_test/knnrouter.yaml --host 0.0.0.0 --port 7860
```

## Troubleshooting

- Unknown router: run `llmrouter list-routers` and check spelling
- `API_KEYS environment variable is not set`: set `API_KEYS` (see [Installation](installation.md))
- `API endpoint not found`: set `api_endpoint` in `llm_data` (per model) or in your YAML config

## Next

- If you want to edit YAML configs, read [Config basics](config-basics.md).
- For a guided walkthrough, follow [First routing run](../tutorials/first-routing.md).
