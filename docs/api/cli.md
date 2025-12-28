# CLI

The `llmrouter` command is installed via the `llmrouter` package and is registered in `pyproject.toml`.

## Summary
The CLI is the primary interface for training, inference, and chat.
It is implemented in:

- `llmrouter/cli/router_main.py` (entrypoint): https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_main.py
- `llmrouter/cli/router_train.py` (training): https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_train.py
- `llmrouter/cli/router_inference.py` (inference): https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/cli/router_inference.py

!!! note "Dependencies"
    The `chat` command requires `gradio` to be installed.

## Global options
- `llmrouter --version` prints the CLI version.
- `llmrouter <subcommand> --help` prints flags for a subcommand.

## Environment variables
### `API_KEYS` (only for real API calls)
`llmrouter infer` (without `--route-only`) calls an OpenAI-compatible endpoint via LiteLLM and requires `API_KEYS`.
It supports either a single key or a JSON list of keys (used round-robin).

**Single key**

```bash
export API_KEYS="YOUR_KEY"
```

```powershell
$env:API_KEYS="YOUR_KEY"
```

**Multiple keys (JSON list)**

```bash
export API_KEYS='["key1","key2","key3"]'
```

```powershell
$env:API_KEYS='["key1","key2","key3"]'
```

!!! tip
    If you only want routing decisions (no API calls, no API keys), always add `--route-only`.

!!! note "`router_r1` is different"
    `router_r1` does not use `API_KEYS`. It requires `hparam.api_base` and `hparam.api_key` in its YAML config, and it does not support `--route-only`.

### `LLMROUTER_PLUGINS` (optional)
If you use custom routers, set `LLMROUTER_PLUGINS` to point to plugin directories (colon-separated). See [Plugin system](plugins.md).

## train
Train a router model.

### Signature
```bash
llmrouter train --router <name> --config <path> [--device cuda|cpu|auto] [--quiet]
```

### Parameters
| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--router` | str | Router method name | required |
| `--config` | path | YAML config file | required |
| `--device` | str | Training device (`cuda`, `cpu`, or `auto`) | auto |
| `--quiet` | flag | Suppress verbose output | false |

### Example
```bash
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml
```

## infer
Run inference or routing.

### Signature
```bash
llmrouter infer --router <name> --config <path> (--query <text> | --input <file>)   [--load_model_path <path>] [--route-only] [--output <file>] [--output-format json|jsonl]   [--temp <float>] [--max-tokens <int>] [--verbose]
```

!!! note "Input/output schemas"
    For `--input` formats and output fields, see [Data formats](../getting-started/data-formats.md).

!!! note "API endpoint resolution"
    Inference expects an endpoint from either:

    1. `llm_data[model_name].api_endpoint` (per model), or
    2. `api_endpoint` in the YAML config

    See [Config reference](config.md).

### Parameters
| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--router` | str | Router method name | required |
| `--config` | path | YAML config file | required |
| `--query` | str | Single query string | required (mutually exclusive) |
| `--input` | path | Input file (`.txt`, `.json`, `.jsonl`) | required (mutually exclusive) |
| `--load_model_path` | path | Override `model_path.load_model_path` | none |
| `--route-only` | flag | Skip API call and only return routing | false |
| `--output` | path | Output file for batch inference | none |
| `--output-format` | str | Output format (`json` or `jsonl`) | json |
| `--temp` | float | Generation temperature | 0.8 |
| `--max-tokens` | int | Max tokens for generation | 1024 |
| `--verbose` | flag | Print verbose output | false |

### Example
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is AI?"
```

### Batch example (JSONL)
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input queries.jsonl --output results.jsonl --output-format jsonl --route-only
```

## chat
Launch the interactive chat UI.

### Signature
```bash
llmrouter chat --router <name> --config <path> [--load_model_path <path>] [--temp <float>]   [--host <host>] [--port <port>] [--mode full_context|current_only|retrieval] [--top_k <int>] [--share]
```

### Parameters
| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `--router` | str | Router method name | required |
| `--config` | path | YAML config file | required |
| `--load_model_path` | path | Override `model_path.load_model_path` | none |
| `--temp` | float | Generation temperature | 0.8 |
| `--host` | str | Host to bind the server | all interfaces |
| `--port` | int | Server port | 8001 |
| `--mode` | str | Query mode (`full_context`, `current_only`, `retrieval`) | current_only |
| `--top_k` | int | Retrieval count when `mode=retrieval` | 3 |
| `--share` | flag | Create a public share link | false |

### Example
```bash
llmrouter chat --router knnrouter --config configs/model_config_test/knnrouter.yaml
```

## list-routers
List all available routers (including discovered plugins).

### Signature
```bash
llmrouter list-routers
```

### Notes
- This prints routers available for inference and routers available for training (separately).
- If a router is missing, check optional dependencies and plugin discovery paths.

## version
Show CLI version information.

### Signature
```bash
llmrouter version
```
