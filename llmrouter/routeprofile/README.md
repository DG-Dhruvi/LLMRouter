# RouteProfile — Graph-based LLM Profiling for LLMRouter

RouteProfile enriches the **model embeddings** used by routers like GraphRouter, MFRouter, and PersonalizedRouter. Instead of encoding only a model's text description with Longformer, it builds a heterogeneous graph that captures relationships between models, benchmark datasets, architectural families, and query domains — then propagates information across this graph to generate richer, context-aware model profiles.

## Why Use RouteProfile?

Routers that use model embeddings (GraphRouter, MFRouter, PersonalizedRouter) currently rely on Longformer embeddings of model descriptions. These embeddings capture *what a model says about itself* but not *how it actually performs* across tasks. RouteProfile's graph-based profiles encode actual benchmark performance, architectural similarity, and domain coverage — giving routers a richer basis for selection.

## Pipeline Overview

```
Step 1: Profile Data (JSON files)
        model_feature_standard.json — model metadata + benchmark scores
        model_family_feature.json   — architecture descriptions
        task_feature.json           — benchmark descriptions
        domain_task_map.json        — domain → benchmark mapping
              │
              ▼
Step 2: Build Heterogeneous Graph (llmrouter profile build-graph)
        HeteroData .pt file with node types: model, architecture, dataset, [domain, query]
        Edge weights on model↔dataset edges = benchmark scores
              │
              ▼
Step 3: Generate Model Profile (llmrouter profile build-profile)
        Per-model 768-dim embeddings saved as .npz
        Methods: flat | emb_gnn | index | text_gnn | trainable_gnn
              │
              ▼
Step 4: Apply to LLMRouter (llmrouter profile apply)
        Overwrites the "embedding" field in llm_data JSON
        Routers (GraphRouter, MFRouter, etc.) pick up the new embeddings automatically
```

## Profile Methods

| Method | Type | GPU for Training | Quality | Speed | Description |
|--------|------|:---:|------|------|-------------|
| `index` | Training-free | — | Baseline | Fast | Random 768-dim vectors (ablation baseline) |
| `flat` | Training-free | — | Low | Fast | Random-sample neighbour texts, encode with Longformer |
| `emb_gnn` | Training-free | — | **Medium-High** | Fast | K-hop degree-normalised graph propagation |
| `text_gnn` | Training-free | Required (vLLM) | High | Slow | LLM-based text summarisation per hop |
| `trainable_gnn` | Trainable | Required | **Highest** | Medium | Self-supervised HANConv (masked feature reconstruction) |

**Recommendation**: Start with `emb_gnn` (best cost/quality tradeoff, no GPU needed).

## Installation

```bash
# Standard (emb_gnn, flat, index, trainable_gnn — no GPU needed for emb_gnn/flat/index)
pip install -e ".[routeprofile]"

# With text_gnn support (requires GPU + vLLM)
pip install -e ".[routeprofile-text-gnn]"
```

## Quick Start (3 CLI Commands)

```bash
# 1. Build graph (uses bundled default data, ~2 min with Longformer)
llmrouter profile build-graph \
    --graph-type task_domain \
    --mode standard \
    --output-dir data/my_graphs/

# 2. Generate model profiles
llmrouter profile build-profile \
    --method emb_gnn \
    --graph data/my_graphs/task_domain_graph_full.pt \
    --output data/my_profiles/emb_gnn.npz

# 3. Apply to llm_data JSON (in-place update or new file)
llmrouter profile apply \
    --profile data/my_profiles/emb_gnn.npz \
    --llm-data data/example_data/llm_candidates/default_llm.json \
    --output data/example_data/llm_candidates/default_llm_rp.json
```

Then update your router YAML to point at the enriched file:

```yaml
# configs/model_config_train/graphrouter.yaml
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm_rp.json'
  # everything else stays the same
```

That's it — train and infer as normal:

```bash
llmrouter train --router graphrouter --config configs/model_config_train/graphrouter.yaml
llmrouter infer --router graphrouter --config configs/model_config_test/graphrouter.yaml --query "..."
```

## Detailed CLI Reference

### `llmrouter profile build-graph`

Builds a PyTorch Geometric `HeteroData` graph from JSON metadata files.

```
llmrouter profile build-graph [OPTIONS]

Options:
  --graph-type  {task,query,query_task,task_domain,query_task_domain,all}
                Graph topology (default: task_domain)
  --mode        {standard,newllm}   Routing setting (default: standard)
  --profile-data-dir DIR            Override bundled JSON data directory
  --output-dir  DIR                 Output directory for .pt files
                                    (default: ./results/result_data_graph/{mode}/)
```

**Graph types** (richer = more node types = more context, but slower to build):

| Type | Node types | Best for |
|------|-----------|----------|
| `task` | arch · model · dataset | Fastest; captures benchmark performance |
| `task_domain` | arch · model · dataset · domain | Adds domain hierarchy |
| `query_task_domain` | arch · model · dataset · domain · query | Richest; adds per-query context |
| `query` / `query_task` | model · dataset · query | Query-focused variants |

### `llmrouter profile build-profile`

Generates one 768-dim embedding per model from a graph.

```
llmrouter profile build-profile --method METHOD --graph GRAPH_PT --output OUTPUT_NPZ [OPTIONS]

Options:
  --method      {flat,emb_gnn,index,text_gnn,trainable_gnn}  Required
  --graph       PATH    Input .pt graph file (not needed for 'index')
  --output      PATH    Output .npz file (keys = model names)
  --K           INT     Propagation hops for emb_gnn (default: 2)
  --norm        {sym,right,left,none}  Normalisation for emb_gnn (default: sym)
  --normalize           L2-normalise output embeddings (emb_gnn)
  --top-k       INT     Neighbours per node type for flat (default: 5)
  --epochs      INT     Training epochs for trainable_gnn (default: 100)
  --seed        INT     Random seed (default: 42)
```

### `llmrouter profile apply`

Merges .npz embeddings into llm_data JSON, ready for router consumption.

```
llmrouter profile apply --profile NPZ --llm-data JSON [OPTIONS]

Options:
  --profile    PATH              RouteProfile .npz file
  --llm-data   PATH              LLMRouter llm_data JSON (source of API metadata)
  --output     PATH              Output path (default: overwrite --llm-data in-place)
  --format     {json,pkl}        json for GraphRouter/MFRouter; pkl for PersonalizedRouter
  --missing    {warn,skip,error} Behaviour for models in llm_data not in .npz (default: warn)
```

## Python API

```python
from llmrouter.routeprofile import (
    build_task_domain_graph,
    build_emb_gnn_profile,
)
from llmrouter.routeprofile.utils import npz_to_llm_embeddings_json

# Step 2: Build graph
build_task_domain_graph(
    mode="standard",
    save="data/my_graphs/task_domain_graph_full.pt",
)

# Step 3: Generate profile
build_emb_gnn_profile(
    graph="data/my_graphs/task_domain_graph_full.pt",
    K=2,
    norm="sym",
    save="data/my_profiles/emb_gnn.npz",
    keep=[],  # [] → save all models; None → use default model list
)

# Step 4: Apply
npz_to_llm_embeddings_json(
    npz_path="data/my_profiles/emb_gnn.npz",
    llm_data_path="data/example_data/llm_candidates/default_llm.json",
    output_path="data/example_data/llm_candidates/default_llm_rp.json",
)
```

For PersonalizedRouter (expects `.pkl`):

```python
from llmrouter.routeprofile.utils import npz_to_pkl

npz_to_pkl(
    npz_path="data/my_profiles/emb_gnn.npz",
    output_path="data/example_data/llm_embeddings/routeprofile_emb_gnn.pkl",
)
```

Then set in your YAML:

```yaml
# configs/model_config_train/personalizedrouter.yaml
data_path:
  llm_embedding_data: 'data/example_data/llm_embeddings/routeprofile_emb_gnn.pkl'
```

## Routers That Benefit

| Router | Embedding field used | How to update |
|--------|---------------------|---------------|
| `graphrouter` | `llm_data[model]["embedding"]` | Use `apply --format json` |
| `mfrouter` | `llm_data[model]["embedding"]` | Use `apply --format json` |
| `personalizedrouter` | `llm_embedding_data` (.pkl) | Use `apply --format pkl` |
| `gmtrouter` | Inline in interaction JSONL | Not applicable (uses interaction data) |

## Using Custom Profile Data

By default, `build-graph` uses bundled JSON files covering 8 LLM candidates and standard benchmarks (IFEval, BBH, MATH, GSM8K, HumanEval, MBPP, etc.).

To use your own models, create a directory with:
```
my_profile_data/
├── model_feature_standard.json   # model metadata + benchmark scores
├── model_family_feature.json     # architecture family descriptions
├── task_feature.json             # benchmark descriptions
├── domain_task_map.json          # domain → benchmark mapping
└── domain_feature.json           # domain descriptions
```

Then pass `--profile-data-dir my_profile_data/` to `build-graph`.

**`model_feature_standard.json` format:**

```json
{
  "your-model-name": {
    "size": "7B",
    "feature": "Natural language description of the model...",
    "architecture": "LlamaForCausalLM",
    "detailed_scores": {
      "ifeval": 75.85,
      "bbh": 53.94,
      "math": 50.0
    },
    "parameters": 7.6,
    "model": "provider/model-id",
    "service": "NVIDIA",
    "api_endpoint": "https://integrate.api.nvidia.com/v1"
  }
}
```

The model name (key) must match the model name in `llm_data.json` for the `apply` step to work.

## Bundled Default Data

The package includes data for 8 candidate models across 11+ benchmarks:

**Models**: qwen2.5-7b-instruct · gemma-2-9b-it · llama-3.1-8b-instruct · mixtral-8x7b-instruct · mixtral-8x22b-instruct · llama-3.2-3b-instruct · mistral-small-24b-instruct · llama-3.3-70b-instruct

**Benchmarks**: IFEval · BBH · MATH · GSM8K · HumanEval · MBPP · MMLU · MMLU-Pro · TheoremQA · C-Eval · ARC

**Domains**: knowledge · reasoning · math · coding

## Reference

RouteProfile is based on the paper:
> *RouteProfile: A General Framework for Designing LLM Profiles for Routing*  
> [https://github.com/ulab-uiuc/RouteProfile](https://github.com/ulab-uiuc/RouteProfile)
