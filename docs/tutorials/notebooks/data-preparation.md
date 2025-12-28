# Data preparation (notebook)

This tutorial follows the data preparation notebook and produces the files used by most training and inference configs.

Notebook: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/data_preparation/01_data_preparation.ipynb

## When you need this
- You want to train routers on your own dataset (not just the shipped example files).
- You want to generate fresh query embeddings or routing labels.

If you just want to try the framework end-to-end, you can skip this and use the example assets on `main`:
- Example data: https://github.com/ulab-uiuc/LLMRouter/tree/main/data/example_data
- Example configs: https://github.com/ulab-uiuc/LLMRouter/tree/main/configs

## Outputs (what you should end up with)
These map directly to the `data_path` keys in YAML configs (see [Config reference](../../api/config.md) and [Data utilities](../../api/data.md)).

| YAML key | Typical example path (main) | What it contains |
| --- | --- | --- |
| `query_data_train` | `data/example_data/query_data/default_query_train.jsonl` | training queries (JSONL) |
| `query_data_test` | `data/example_data/query_data/default_query_test.jsonl` | test queries (JSONL) |
| `query_embedding_data` | `data/example_data/routing_data/query_embeddings_longformer.pt` | query embeddings tensor |
| `routing_data_train` | `data/example_data/routing_data/default_routing_train_data.jsonl` | routing supervision (labels/scores) |
| `routing_data_test` | `data/example_data/routing_data/default_routing_test_data.jsonl` | routing supervision (labels/scores) |
| `llm_data` | `data/example_data/llm_candidates/default_llm.json` | candidate models + metadata |
| `llm_embedding_data` | `data/example_data/llm_candidates/default_llm_embeddings.json` | model embeddings / metadata |

!!! note
    The notebook can optionally call external LLM APIs to produce routing labels. That step requires network access and API credentials.

## Run the notebook
1. Clone + install from source (recommended so `configs/` and `data/` are available):

   ```bash
   git clone https://github.com/ulab-uiuc/LLMRouter.git
   cd LLMRouter
   python -m pip install -U pip
   python -m pip install -e .
   python -m pip install jupyter
   ```

2. Open the notebook:

   ```bash
   jupyter notebook notebooks/data_preparation/01_data_preparation.ipynb
   ```

3. Follow the sections in order:
   - dataset download
   - query generation
   - candidate model list (`llm_data`)
   - query embeddings
   - optional API calling and evaluation
   - verification

## Next
Pick a router tutorial, for example:
- [KNNRouter](knnrouter.md)
- [SVMRouter](svmrouter.md)
- [MLPRouter](mlprouter.md)

