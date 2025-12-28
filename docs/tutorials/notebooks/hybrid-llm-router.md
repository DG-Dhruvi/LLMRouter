# HybridLLMRouter (training + inference)

HybridLLMRouter trains an MLP regressor to decide whether to route to the smallest or largest candidate model.

Notebooks:
- Training: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/hybrid_llm_router/01_hybrid_llm_router_training.ipynb
- Inference: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/hybrid_llm_router/02_hybrid_llm_router_inference.ipynb

Router docs: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/hybrid_llm/README.md

## Configs
- Train config (as shipped): https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_train/hybrid_llm.yaml
- Test config (as shipped): https://github.com/ulab-uiuc/LLMRouter/blob/main/configs/model_config_test/hybrid_llm.yaml

!!! warning
    HybridLLMRouter requires `llm_data`, `routing_data_train`, and `query_embedding_data` to build its training set.
    If your config does not include `data_path`, add it (see the minimal example below).

### Minimal config example
Use this as a starting point (adjust paths to your environment):

```yaml
router_mode: probabilistic
router_tau: 0.1
router_threshold: 0.5

data_path:
  llm_data: data/example_data/llm_candidates/default_llm.json
  llm_embedding_data: data/example_data/llm_candidates/default_llm_embeddings.json
  query_embedding_data: data/example_data/routing_data/query_embeddings_longformer.pt
  routing_data_train: data/example_data/routing_data/default_routing_train_data.jsonl
  routing_data_test: data/example_data/routing_data/default_routing_test_data.jsonl

hparam:
  hidden_layer_sizes: [128, 64]
  activation: relu
  solver: adam
  max_iter: 300

model_path:
  save_model_path: saved_models/hybrid_llm/hybrid_trained.pkl
  load_model_path: saved_models/hybrid_llm/hybrid_trained.pkl
```

## Run (CLI)
Train:

```bash
llmrouter train --router hybrid_llm --config configs/model_config_train/hybrid_llm.yaml
```

Route-only inference:

```bash
llmrouter infer --router hybrid_llm --config configs/model_config_test/hybrid_llm.yaml --query "Explain transformers." --route-only
```

Full inference:

```bash
llmrouter infer --router hybrid_llm --config configs/model_config_test/hybrid_llm.yaml --query "Explain transformers."
```

## Next
- If you want more than two models (not just small vs large), start from [KNNRouter](knnrouter.md).

