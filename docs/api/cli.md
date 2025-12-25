# CLI

The `llmrouter` command is installed via the `llmrouter` package.

## train
Train a router model.

Arguments:
- `--router` (required)
- `--config` (required)
- `--device` (cuda|cpu|auto, default: auto)
- `--quiet`

Example:
```bash
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml
```

## infer
Run inference or routing.

Arguments:
- `--router` (required)
- `--config` (required)
- `--query` or `--input` (mutually exclusive)
- `--load_model_path`
- `--route-only`
- `--output`
- `--output-format` (json|jsonl, default: json)
- `--temp` (default: 0.8)
- `--max-tokens` (default: 1024)
- `--verbose`

Example:
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is AI?"
```

## chat
Launch the interactive chat UI.

Arguments:
- `--router` (required)
- `--config` (required)
- `--load_model_path`
- `--temp` (default: 0.8)
- `--host` (default: all interfaces)
- `--port` (default: 8001)
- `--mode` (full_context|current_only|retrieval, default: current_only)
- `--top_k` (default: 3)
- `--share`

Example:
```bash
llmrouter chat --router knnrouter --config configs/model_config_test/knnrouter.yaml
```

## list-routers
List all available routers (including discovered plugins).

```bash
llmrouter list-routers
```

## version
Show CLI version information.

```bash
llmrouter version
```
