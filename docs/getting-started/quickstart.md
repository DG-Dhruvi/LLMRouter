# Quickstart

This page walks through a minimal training and inference run.

## 1) Choose a router and config
Configs live in `configs/model_config_train/` and `configs/model_config_test/`.
Start with `knnrouter` for a baseline.

## 2) Train (for trainable routers)
```bash
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml
```

## 3) Run inference
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is machine learning?"
```

## 4) Batch inference
```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --input queries.jsonl --output results.jsonl --output-format jsonl
```

## 5) Chat UI
```bash
llmrouter chat --router knnrouter --config configs/model_config_test/knnrouter.yaml --host 0.0.0.0 --port 7860
```

## Tips
- Use `--route-only` to skip API calls and only compute routing decisions.
- Use `llmrouter list-routers` to see what is available in your environment.
