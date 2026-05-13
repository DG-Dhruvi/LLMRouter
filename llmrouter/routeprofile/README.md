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

## Customising the Graph — Adding Your Own Models, Benchmarks & Queries

The bundled defaults cover 8 LLMs and ~17–29 benchmarks. Four data-management subcommands let you extend the graph with custom data into a **user-owned directory** (auto-initialised from the bundled files on first run). The installed package is never modified.

### Command Overview

| Command | Modifies | When to use |
|---------|----------|-------------|
| `add-domain` | `domain_feature.json` | Before adding a benchmark that belongs to a new domain |
| `add-task` | `task_feature.json`, `domain_task_map.json`, optionally `task_queries_{mode}.json` | Add a new benchmark (dataset node) and optionally seed its query samples |
| `add-model` | `model_feature_{mode}.json` | Add or update a model; use `--mode` to target standard, newllm, or both |
| `add-query` | `task_queries_{mode}.json` | Append representative query strings to an existing benchmark |

`--mode` controls which mode-specific file is updated:

| `--mode` | Model file | Query file |
|----------|-----------|-----------|
| `standard` | `model_feature_standard.json` | `task_queries_standard.json` |
| `newllm` | `model_feature_newllm.json` | `task_queries_newllm.json` |
| `both` | both | both |

### Scenario A — Add a new model to the newllm evaluation set

```bash
llmrouter profile add-model \
  --name "my-new-llm" \
  --feature "A 13B instruction-tuned LLaMA model." \
  --architecture LlamaForCausalLM \
  --scores "ifeval:72.5,bbh:48.3,gsm8k:61.2" \
  --mode newllm \
  --output-dir data/my_profile_data/
```

Omit `--mode` (or pass `--mode both`) to add the model to both the standard and newllm files.

### Scenario B — Add a new benchmark + model (full pipeline with query_task_domain graph)

```bash
# Step 1: new domain (skip if domain already exists)
llmrouter profile add-domain \
  --name "multimodal" \
  --feature "Tasks requiring joint understanding of text and image." \
  --output-dir data/my_profile_data/

# Step 2: new benchmark + seed queries for the standard mode
llmrouter profile add-task \
  --name "my-mm-bench" \
  --feature "A multimodal reasoning benchmark evaluating scene understanding." \
  --domain "multimodal" \
  --query "Describe what is happening in this image." \
  --query "Which object appears larger in the scene?" \
  --mode standard \
  --output-dir data/my_profile_data/

# Step 3: add the model referencing the new benchmark
llmrouter profile add-model \
  --name "my-model" \
  --feature "A 13B multimodal LLM capable of reasoning over text and images." \
  --architecture LlamaForCausalLM \
  --scores "my-mm-bench:72.5,ifeval:80.0" \
  --mode both \
  --output-dir data/my_profile_data/

# Step 4: rebuild the graph (use query_task_domain to include query nodes)
llmrouter profile build-graph \
  --graph-type query_task_domain \
  --mode standard \
  --profile-data-dir data/my_profile_data/ \
  --output-dir data/my_graphs/
```

### Scenario C — Patch one score into an existing model

```bash
llmrouter profile add-model \
  --name "qwen2.5-7b-instruct" \
  --scores "my-mm-bench:68.0" \
  --mode both \
  --output-dir data/my_profile_data/
# → Only detailed_scores["my-mm-bench"] is updated; all other fields are preserved.
```

### Scenario D — Append query samples to an existing benchmark

```bash
# Add to standard file only
llmrouter profile add-query \
  --task "ifeval" \
  --query "Explain quantum entanglement in simple terms." \
  --mode standard \
  --output-dir data/my_profile_data/

# Add from a text file (one query per line) to both files
llmrouter profile add-query \
  --task "bbh" \
  --queries-file data/bbh_extra_queries.txt \
  --mode both \
  --output-dir data/my_profile_data/
```

Duplicate queries are silently skipped (idempotent). If the benchmark key does not yet appear in the query file, it is initialised to an empty list before appending.

## Detailed CLI Reference

### `llmrouter profile build-graph`

Builds a PyTorch Geometric `HeteroData` graph from JSON metadata files and saves it as a `.pt` file.

```
llmrouter profile build-graph [OPTIONS]

Options:
  --graph-type  {task,query,query_task,task_domain,query_task_domain,all}
                Graph topology to build (default: task_domain)
  --mode        {standard,newllm}
                Routing setting — controls which model feature file is loaded
                  standard : uses model_feature_standard.json (8 known candidate LLMs)
                  newllm   : uses model_feature_newllm.json   (unseen/held-out LLMs)
                (default: standard)
  --profile-data-dir DIR
                Directory containing the 5–7 input JSON files.
                If omitted, the bundled default data is used automatically.
  --output-dir  DIR
                Output directory for .pt graph files.
                (default: ./results/result_data_graph/{mode}/)
```

#### `--graph-type` — choosing a graph topology

Each graph type corresponds to a different set of node and edge types. Richer graphs encode more context for model embeddings but take longer to build (mainly due to Longformer encoding of text features).

| `--graph-type` | Node types | Edge types | Build time | When to use |
|----------------|-----------|-----------|:----------:|-------------|
| `task` | `architecture` · `model` · `dataset` | arch↔model (family), model↔dataset (benchmark scores) | ~2 min | Good default; captures benchmark performance without domain overhead |
| `task_domain` | `architecture` · `model` · `dataset` · `domain` | above + dataset↔domain (task-domain grouping) | ~2 min | **Recommended**: adds domain hierarchy (knowledge / reasoning / math / coding) |
| `query_task_domain` | `architecture` · `model` · `dataset` · `domain` · `query` | above + query↔dataset (per-query alignment) | ~10 min | Richest context; useful when query-level routing diversity matters |
| `query` | `model` · `dataset` · `query` | model↔dataset, query↔dataset | ~5 min | Query-focused; no architecture nodes |
| `query_task` | `model` · `dataset` · `query` | model↔dataset, query↔dataset | ~5 min | Query + task; no architecture or domain nodes |
| `all` | — | — | ~20 min | Builds all five graph types in one run |

**Node feature encoding**: text fields (model description, benchmark description, etc.) are encoded with `get_longformer_embedding` (768-dim). Numerical benchmark scores are stored as edge weights on model↔dataset edges.

**Output filename** for each graph type:

| `--graph-type` | Output file |
|----------------|------------|
| `task` | `task_graph_full.pt` |
| `task_domain` | `task_domain_graph_full.pt` |
| `query_task_domain` | `query_task_domain_graph_full.pt` |
| `query` | `query_graph_full.pt` |
| `query_task` | `query_task_graph_full.pt` |

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

### `llmrouter profile add-domain`

Adds or modifies a domain entry. Run this before `add-task` if your benchmark belongs to a new domain.

```
llmrouter profile add-domain --name NAME --feature TEXT --output-dir DIR

Options:
  --name        Domain key (e.g. "multimodal")
  --feature     Text description for the domain node's Longformer feature
  --output-dir  Directory containing the profile JSON files (auto-initialised from bundled data)
```

If the domain already exists, only the feature text is updated; the `domain_task_map.json` entry is left untouched.

### `llmrouter profile add-task`

Adds or modifies a benchmark (dataset node) entry.

```
llmrouter profile add-task --name NAME --feature TEXT --output-dir DIR [OPTIONS]

Options:
  --name          Benchmark key matching keys used in model detailed_scores
  --feature       Text description for the dataset node's Longformer feature
  --domain        Domain(s) this benchmark belongs to — may be repeated for multiple domains
                  The domain must already exist (run add-domain first)
  --query         Representative query string (repeat for multiple). Requires --mode.
  --queries-file  Text file with one query per line. Requires --mode.
  --mode          {standard,newllm,both} — which task_queries file to update.
                  Required when --query or --queries-file is provided.
  --output-dir    Directory containing the profile JSON files
```

If `--domain` specifies a domain not yet in `domain_feature.json`, the command errors with instructions to run `add-domain` first. No files are modified on error.

### `llmrouter profile add-model`

Adds or modifies a model entry in the chosen `model_feature_{mode}.json` file(s).

```
llmrouter profile add-model --name NAME --output-dir DIR [OPTIONS]

Options:
  --name          Model key (must match the key in llm_data.json)
  --feature       Text description for the model node (required when adding a new model)
  --architecture  Architecture class name, e.g. LlamaForCausalLM (required for new models)
  --arch-feature  Description for a new architecture node (required if --architecture is unknown)
  --scores        Benchmark scores: "bench1:val1,bench2:val2,..." format
  --from-json     Path to a JSON file containing all fields (supports new_tasks array)
  --output-dir    Directory containing the profile JSON files
  --replace       Replace the entire existing record instead of merging (default: merge)
  --mode          {standard,newllm,both} — which model_feature file(s) to update (default: both)
  --size          Human-readable size string (e.g. "13B")
  --parameters    Parameter count in billions
  --model-id      Provider model identifier
  --service       API service provider name
  --api-endpoint  API endpoint URL
```

**Add mode** (model not present): `--feature` and `--architecture` are required.

**Modify mode** (model already present): only the provided arguments are merged; unspecified fields are preserved. Pass `--replace` to overwrite the entire record.

Scores for benchmarks not in `task_feature.json` trigger a warning — run `add-task` first to create graph edges for those benchmarks.

#### Using `--from-json`

```json
{
  "name": "my-model",
  "feature": "A 13B instruction-tuned model.",
  "architecture": "LlamaForCausalLM",
  "detailed_scores": { "my-new-bench": 72.5, "ifeval": 80.0 },
  "new_tasks": [
    {
      "name": "my-new-bench",
      "feature": "A new reasoning benchmark.",
      "domain": "reasoning"
    }
  ]
}
```

`new_tasks` entries are written to `task_feature.json` before the model entry, so their benchmark keys are valid when score validation runs.

### `llmrouter profile add-query`

Appends representative query strings to `task_queries_{mode}.json` for an existing benchmark.

```
llmrouter profile add-query --task TASK --mode MODE --output-dir DIR [OPTIONS]

Options:
  --task          Benchmark key (must exist in task_feature.json)
  --query         Query string to append (repeat for multiple)
  --queries-file  Text file with one query per line
  --mode          {standard,newllm,both} — which task_queries file to update. Required.
  --output-dir    Directory containing the profile JSON files
```

Duplicate queries (exact string match) are silently skipped. If the benchmark key does not exist in the target query file, it is initialised to an empty list before appending.

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

## Bundled Default Data Location

The default input JSON files are packaged inside the library at:

```
llmrouter/routeprofile/data/profile_data/
├── model_feature_standard.json   ← model metadata + benchmark scores (standard setting)
├── model_feature_newllm.json     ← same schema, for new/unseen LLMs (--mode newllm)
├── model_family_feature.json     ← architecture family → text description
├── task_feature.json             ← benchmark name → text description
├── task_queries_standard.json    ← benchmark name → list of representative queries
├── domain_feature.json           ← domain name → text description
└── domain_task_map.json          ← domain name → list of benchmark names
```

When you run `llmrouter profile build-graph` **without** `--profile-data-dir`, these files are loaded automatically from the installed package. You do not need to download or copy them.

To inspect the bundled data from Python:
```python
from llmrouter.routeprofile.data import get_bundled_path
import json

path = get_bundled_path("model_feature_standard.json")
data = json.load(open(path))
print(list(data.keys()))
# ['qwen2.5-7b-instruct', 'gemma-2-9b-it', 'llama-3.1-8b-instruct', ...]
```

## Using Custom Profile Data

To profile models not in the bundled data, create a directory containing the required JSON files and pass it via `--profile-data-dir`:

```bash
llmrouter profile build-graph \
    --graph-type task_domain \
    --mode standard \
    --profile-data-dir my_profile_data/ \
    --output-dir data/my_graphs/
```

### Required files and their formats

#### `model_feature_standard.json` (and `model_feature_newllm.json`)

One entry per candidate LLM. The **key** (model name) must match the model name in your `llm_data.json` so that the `apply` step can merge embeddings correctly.

```json
{
  "your-model-name": {
    "size": "7B",
    "feature": "Natural language description used as the model node's text feature.",
    "architecture": "LlamaForCausalLM",
    "parameters": 7.6,
    "model": "provider/model-id",
    "service": "NVIDIA",
    "api_endpoint": "https://integrate.api.nvidia.com/v1",
    "average_score": 45.2,
    "detailed_scores": {
      "ifeval": 75.85,
      "bbh":   53.94,
      "math":  50.0,
      "gsm8k": 86.58,
      "human_eval": 11.33,
      "mbpp":  1.33
    }
  }
}
```

Fields used by the graph builders:

| Field | Used for | Required |
|-------|---------|:--------:|
| `feature` | Longformer node embedding (model node text) | Yes |
| `architecture` | Links model node to an architecture node | Yes |
| `detailed_scores` | Edge weights on model↔dataset edges | Yes |
| `size`, `parameters`, etc. | Informational only | No |

`detailed_scores` keys must match keys in `task_feature.json`. Scores can be `null` (edge is omitted for that benchmark).

#### `model_family_feature.json`

Architecture class name → natural language description. Used as the text feature for `architecture` nodes.

```json
{
  "LlamaForCausalLM": "A family of decoder-only Transformer LLMs from Meta...",
  "Qwen2ForCausalLM": "Decoder-only Transformers from Alibaba Cloud...",
  "MistralForCausalLM": "Efficient sliding-window attention models from Mistral AI..."
}
```

Every `architecture` value that appears in `model_feature_standard.json` must have an entry here.

#### `task_feature.json`

Benchmark name → natural language description. Used as the text feature for `dataset` nodes.

```json
{
  "ifeval":   "IFEval evaluates instruction-following...",
  "bbh":      "BIG-Bench Hard covers 23 challenging reasoning tasks...",
  "math":     "MATH covers competition-level mathematics...",
  "gsm8k":    "GSM8K is a dataset of grade-school math word problems...",
  "human_eval": "HumanEval tests functional correctness of generated Python code..."
}
```

Keys must match the keys used in `detailed_scores` inside `model_feature_standard.json`.

#### `domain_feature.json`

Domain name → natural language description. Used as the text feature for `domain` nodes. Only needed for `--graph-type task_domain` or `query_task_domain`.

```json
{
  "knowledge": "Tasks requiring broad factual knowledge...",
  "reasoning": "Tasks requiring multi-step logical inference...",
  "math":      "Tasks involving numerical computation and proof...",
  "coding":    "Tasks requiring generating or debugging code..."
}
```

#### `domain_task_map.json`

Domain name → list of benchmark names. Defines which benchmarks belong to each domain. Creates `dataset↔domain` edges in the graph. Only needed for `--graph-type task_domain` or `query_task_domain`.

```json
{
  "knowledge": ["mmlu", "gpqa", "natural_qa", "trivia_qa"],
  "reasoning": ["bbh", "arc_challenge", "commonsense_qa", "musr"],
  "math":      ["math", "gsm8k", "TheoremQA"],
  "coding":    ["human_eval", "mbpp"]
}
```

Benchmark names must be a subset of keys in `task_feature.json`.

#### `task_queries_standard.json`

Benchmark name → list of representative query strings. Used to create `query` nodes in `--graph-type query`, `query_task`, or `query_task_domain`. Not needed for `task` or `task_domain`.

```json
{
  "ifeval":  ["Write an extremely short essay on...", "Translate the following sentence..."],
  "gsm8k":   ["Janet has 4 apples. She gives 2 to Bob...", "..."],
  "human_eval": ["def fibonacci(n):\n    # complete this function", "..."]
}
```

Each query is Longformer-encoded to create the text feature of a `query` node. More queries = more `query` nodes = slower graph construction but richer representation.

## Reference

RouteProfile is based on the paper:
> *RouteProfile: A General Framework for Designing LLM Profiles for Routing*  
> [https://github.com/ulab-uiuc/RouteProfile](https://github.com/ulab-uiuc/RouteProfile)
