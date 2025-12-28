# DCRouter (training + inference)

DCRouter is a stronger, heavier router that uses a transformer backbone. It is usually more compute-intensive than embedding-only baselines.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/dcrouter/01_dcrouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/dcrouter/02_dcrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/routerdc/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/dcrouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/dcrouter.yaml

!!! note "Checkpoint loading"
    DCRouter loads checkpoints from the directory containing `model_path.save_model_path` (it looks for `best_model.pth` or `best_training_model.pth`).
    Make sure you ran training at least once before expecting good results.

## Run (CLI)
Train:

```bash
llmrouter train --router dcrouter --config configs/model_config_train/dcrouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router dcrouter --config configs/model_config_test/dcrouter.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router dcrouter --config configs/model_config_test/dcrouter.yaml --query "Explain transformers."
```

## Tips
- `model_path.backbone_model` controls the transformer backbone.
- For full inference, set `API_KEYS` and ensure `llm_data` has `api_endpoint` and `model`.

