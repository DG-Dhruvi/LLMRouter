# KNNRouter (training + inference)

KNNRouter is a strong baseline when you already have query embeddings. It learns by nearest-neighbor lookup in embedding space.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/knnrouter/01_knnrouter_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/knnrouter/02_knnrouter_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/knnrouter/README.md

## Configs
- Train: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/knnrouter.yaml
- Test: https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/knnrouter.yaml

!!! note
    The inference notebook writes a minimal `knnrouter_inference.yaml` for convenience. For the CLI, you can also use the repo test config above.

## Run (CLI)
Train:

```bash
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml
```

Route-only inference:

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "Explain transformers."
```

## What to tweak
Start with the `hparam` section in the YAML:
- `n_neighbors`: the most important knob
- `metric`: distance metric (for example, cosine vs euclidean)
- `weights`: uniform vs distance weighting

## Troubleshooting
- Missing embedding file: run [Data preparation](data-preparation.md) or update `data_path.query_embedding_data`.
- Unknown router: run `llmrouter list-routers`.

## Next
- Compare with [SVMRouter](svmrouter.md) and [MLPRouter](mlprouter.md).
- Batch runs: [Data formats](../../getting-started/data-formats.md).

