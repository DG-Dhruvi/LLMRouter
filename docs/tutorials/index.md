# Tutorials

This section is a hands-on companion to the Jupyter notebooks in the `main` branch:

- Notebooks folder: https://github.com/ulab-uiuc/LLMRouter/tree/main/notebooks
- Notebooks README: https://github.com/ulab-uiuc/LLMRouter/blob/main/notebooks/README.md

The goal is to help you reproduce the notebook workflows and translate them into repeatable CLI runs.

## Quick links
- Workflows: [First routing run](first-routing.md), [Batch inference](batch-inference.md), [Chat demo](chat-demo.md), [Evaluation workflow](evaluation.md), [Custom router](custom-router.md)
- [Data preparation](notebooks/data-preparation.md)
- Single-Round Routers: [KNNRouter](notebooks/knnrouter.md), [SVMRouter](notebooks/svmrouter.md), [MLPRouter](notebooks/mlprouter.md), [MFRouter](notebooks/mfrouter.md), [EloRouter](notebooks/elorouter.md), [DCRouter](notebooks/dcrouter.md), [AutoMixRouter](notebooks/automix-router.md), [HybridLLMRouter](notebooks/hybrid-llm-router.md), [GraphRouter](notebooks/graphrouter.md), [CausalLMRouter](notebooks/causallm-router.md), [SmallestLLM](notebooks/smallest-llm.md), [LargestLLM](notebooks/largest-llm.md)
- Multi-Round Routers: [RouterR1](notebooks/router-r1.md)
- Personalized Routers: [GMTRouter](notebooks/gmtrouter.md)
- Agentic Routers: [KNNMultiRoundRouter](notebooks/knnmultiroundrouter.md), [LLMMultiRoundRouter](notebooks/llmmultiroundrouter.md)
- Extending the framework: [Custom routers](notebooks/custom-router.md)

## Recommended path
1. Run (or reuse) data preparation outputs: [Data preparation](notebooks/data-preparation.md)
2. Pick a router family and train it (if needed): start with [KNNRouter](notebooks/knnrouter.md)
3. Do route-only inference first, then full inference if you have API access
4. Batch runs and output schemas: [Data formats](../getting-started/data-formats.md)

## Running notebooks locally
Most notebooks assume you cloned the repo (so `configs/` and `data/` exist).

```bash
git clone https://github.com/ulab-uiuc/LLMRouter.git
cd LLMRouter
python -m pip install -U pip
python -m pip install -e .
python -m pip install jupyter
```

Then open a notebook, for example:

```bash
jupyter notebook notebooks/data_preparation/01_data_preparation.ipynb
```

!!! tip
    If you only want to validate routing decisions (no API calls), use `--route-only` in the CLI and skip API key setup.

## Router comparison (high-level)
This table is a rough guide for choosing a starting point. For exact availability in your environment, run `llmrouter list-routers`.

| Router | Type | Training | GPU | Best for |
| --- | --- | --- | --- | --- |
| KNNRouter | classification | yes | no | simple baseline, fast iteration |
| SVMRouter | classification | yes | no | high-dimensional embeddings |
| MLPRouter | neural network | yes | no | non-linear decision boundary |
| MFRouter | matrix factorization | yes | no | preference-style routing signals |
| DCRouter | transformer router | yes | recommended | stronger accuracy (more compute) |
| GraphRouter | GNN router | yes | recommended | relational patterns / graph structure |
| CausalLMRouter | finetuned LLM | yes | required | complex queries (heavier training) |
| AutoMixRouter | mixing / policy | yes | no | cost-aware routing |
| HybridLLMRouter | hybrid classifier | yes | no | two-stage / binary routing settings |
| GMTRouter | multi-turn / personalized | yes | recommended | personalized multi-turn routing |
| EloRouter | rating-based | optional | no | simple rating baseline |
| SmallestLLM | baseline | no | no | cost-efficiency baseline |
| LargestLLM | baseline | no | no | upper bound baseline |
| RouterR1 | agentic | no | required | complex reasoning (special runtime) |
| KNNMultiRoundRouter | multi-round KNN | yes | no | multi-step queries with training |
| LLMMultiRoundRouter | multi-round LLM | no | optional | zero-shot multi-round (API-driven) |

## Next
Start with [Data preparation](notebooks/data-preparation.md).
