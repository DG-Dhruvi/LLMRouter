# CLI

The `llmrouter` command is installed via the `llmrouter` package and is registered in `pyproject.toml`.

## Summary
The CLI is the primary interface for training, inference, and chat. It dispatches to `llmrouter.cli.router_train`, `llmrouter.cli.router_inference`, and `llmrouter.cli.router_chat`.

!!! note "Dependencies"
    The `chat` command requires `gradio` to be installed.

## Global options
- `llmrouter --version` prints the CLI version.

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

## version
Show CLI version information.

### Signature
```bash
llmrouter version
```
