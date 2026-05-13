# RouteProfile Integration — Change Summary

This document records all changes made to integrate RouteProfile into LLMRouter.

## Background

Several LLMRouter routers (GraphRouter, MFRouter, PersonalizedRouter) represent each candidate LLM as a 768-dim embedding vector. Before this integration, those embeddings were always produced by running Longformer over each model's text description — capturing what the model says about itself but nothing about how it actually performs.

RouteProfile generates richer embeddings by building a **heterogeneous graph** that encodes relationships between models, benchmark scores, architectural families, and query domains, then propagating information across the graph. The goal of this integration is to plug RouteProfile's graph-building and profile-generation pipeline into LLMRouter so that any router that consumes model embeddings can switch to RouteProfile-generated ones — **without modifying the router code at all**.

RouteProfile's own routing models (SimRouter, MLPRouter, GraphRouter variants) were deliberately **not** imported; LLMRouter already has equivalent implementations.

---

## New Files

### Core package: `llmrouter/routeprofile/`

| File | Description |
|------|-------------|
| `__init__.py` | Package entry-point; re-exports the 5 graph builders and 5 profile functions |
| `utils.py` | Format-conversion helpers: `npz_to_llm_embeddings_json`, `npz_to_pkl` |
| `README.md` | Full user-facing documentation for the routeprofile sub-package |

#### `build_data_graph/` — 5 heterogeneous graph builders

| File | Graph type | Node types |
|------|-----------|------------|
| `build_task_graph.py` | `task` | arch · model · dataset |
| `build_query_graph.py` | `query` | model · dataset · query |
| `build_query_task_graph.py` | `query_task` | model · dataset · query |
| `build_task_domain_graph.py` | `task_domain` | arch · model · dataset · domain |
| `build_query_task_domain_graph.py` | `query_task_domain` | arch · model · dataset · domain · query |
| `print_graph.py` | Utility to print graph statistics | — |

All builders emit a PyTorch Geometric `HeteroData` `.pt` file. Edge weights on model↔dataset edges encode benchmark scores.

#### `get_model_profile/training_free/` — 4 training-free profile methods

| File | Method | GPU needed | Description |
|------|--------|:---:|-------------|
| `flat_profile.py` | `flat` | No | Randomly samples neighbour node texts, concatenates, encodes with Longformer |
| `emb_gnn_profile.py` | `emb_gnn` | No | K-hop degree-normalised graph propagation over pre-computed `.x` tensors |
| `index_profile.py` | `index` | No | Generates random orthogonal unit vectors (ablation baseline) |
| `text_gnn_profile.py` | `text_gnn` | Yes (vLLM) | Summarises K-hop neighbour texts with a local LLM, then Longformer-encodes the summary |

#### `get_model_profile/trainable/` — 1 trainable profile method

| File | Method | GPU needed | Description |
|------|--------|:---:|-------------|
| `trainable_gnn_profile.py` | `trainable_gnn` | Yes | Self-supervised HANConv GNN trained via masked feature reconstruction |

#### `data/profile_data/` — bundled default input data

Eight JSON files covering 8 LLM candidates across 11+ benchmarks, packaged into the library so the CLI works out of the box without any external data download.

| File | Content |
|------|---------|
| `model_feature_standard.json` | Model metadata + per-benchmark scores (standard routing setting) |
| `model_feature_newllm.json` | Same for the new-LLM routing setting |
| `model_family_feature.json` | Architecture family text descriptions |
| `task_feature.json` | Benchmark dataset descriptions |
| `task_queries_standard.json` | Representative queries per benchmark (17 benchmarks) |
| `task_queries_newllm.json` | Representative queries for the newllm setting (29 benchmarks) |
| `domain_feature.json` | Domain text descriptions |
| `domain_task_map.json` | Domain → benchmark mapping |

#### `data_management.py` — profile data extension API

New module providing four functions for extending the bundled data with custom models, benchmarks, and domains:

| Function | Description |
|----------|-------------|
| `init_profile_data_dir(output_dir)` | Copies bundled files to a user-owned directory; skips existing files |
| `add_domain(name, feature, output_dir)` | Adds/modifies a domain in `domain_feature.json`; initialises its `domain_task_map.json` entry |
| `add_task(name, feature, output_dir, *, domains)` | Adds/modifies a benchmark; validates all domains exist before writing |
| `add_model(name, output_dir, *, ...)` | Adds/modifies a model in both `model_feature_standard.json` and `model_feature_newllm.json` |

---

### CLI: `llmrouter/cli/router_profile.py`

New file implementing the `llmrouter profile` sub-command group with six sub-commands:

```
llmrouter profile build-graph   --graph-type ... --mode ... --output-dir ...
llmrouter profile build-profile --method ...    --graph ... --output ...
llmrouter profile apply          --profile ...  --llm-data ... --output ...
llmrouter profile add-domain    --name ... --feature ... --output-dir ...
llmrouter profile add-task      --name ... --feature ... --output-dir ... [--domain ...]
llmrouter profile add-model     --name ... --output-dir ... [--feature ...] [--scores ...]
```

`add-domain`, `add-task`, and `add-model` delegate to `data_management.py` and support adding new entries to a user-owned profile data directory before (re)building the graph.

---

### Notebook: `notebooks/routeprofile/01_routeprofile_tutorial.ipynb`

End-to-end Jupyter tutorial covering all four pipeline steps. Demonstrates all five profile methods, including a guard for `text_gnn` (skipped when vLLM is not installed, with a K=0 CPU-only alternative shown in comments).

---

### Tests: `tests/routeprofile/`

| File | What it covers | Tests |
|------|----------------|------:|
| `conftest.py` | Shared fixtures: tiny synthetic `HeteroData` graph saved to `.pt` | — |
| `test_data.py` | Bundled JSON files are importable, well-formed, and contain expected keys | 5 |
| `test_build_graph.py` | All 5 graph builders produce valid `HeteroData` with correct node/edge types | 10 |
| `test_build_profile.py` | All 5 profile methods produce valid float32 `[768]` `.npz`; K=0 vs K=2 text_gnn | 24 |
| `test_utils.py` | `npz_to_llm_embeddings_json` and `npz_to_pkl` format correctness | 8 |
| `test_cli.py` | Parser registration and `--help` smoke tests for all 6 sub-commands | 19 |
| `test_data_management.py` | `init_profile_data_dir`, `add_domain`, `add_task`, `add_model` — add/modify/replace/warn behaviour | 22 |
| `test_router_integration.py` | Level-1: synthetic `.npz` → GraphRouter / PersonalizedRouter loading; Level-2: real end-to-end profile → router for all 5 methods | 20 |

Total: **96 tests**.

---

## Modified Files

### `llmrouter/cli/router_main.py`

Added 3 lines to register the `profile` sub-command:

```python
try:
    from llmrouter.cli.router_profile import add_profile_parser
    add_profile_parser(subparsers)
except ImportError:
    pass
```

### `llmrouter/models/graphrouter/router.py` — bug fix

`_prepare_llm_embeddings` was reading from `self.cfg` (the raw YAML dict, which contains path strings, not the loaded JSON data), so model embeddings were always randomly initialised regardless of what was in `llm_data.json`.

**Before:**
```python
def _prepare_llm_embeddings(self):
    llm_data = self.cfg.get("llm_data", {})
```

**After:**
```python
def _prepare_llm_embeddings(self):
    llm_data = getattr(self, "llm_data", None) or self.cfg.get("llm_data", {}) or {}
```

This bug existed before this integration; it was discovered while writing the router integration tests.

### `pyproject.toml`

Two new optional-dependency extras and package-data declaration:

```toml
[project.optional-dependencies]
routeprofile = [
    "torch-geometric>=2.4.0",
    "scipy>=1.11.0",
]
routeprofile-text-gnn = [
    "torch-geometric>=2.4.0",
    "scipy>=1.11.0",
    "vllm>=0.3.0",
]

[tool.setuptools.package-data]
"llmrouter.routeprofile.data.profile_data" = ["*.json"]
```

---

## Design Decisions

**Zero router-code changes.** The integration works entirely at the data level: RouteProfile produces a `.npz` file, `npz_to_llm_embeddings_json` merges the embeddings into the existing `llm_data.json` format, and existing routers read that file as before. No router class needed to be modified (other than the pre-existing GraphRouter bug fix).

**text_gnn K>0 requires `VLLM_WORKER_MULTIPROC_METHOD=spawn`.** vLLM v1 (tested with 0.18.1) spawns its EngineCore worker via `multiprocessing.fork`. When any `torch` import has already initialised CUDA in the parent process (which always happens inside pytest), the forked child crashes with `CUDA error: initialization error`. The fix is to set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before the vLLM call. The `TestTextGnnProfileKN` test class handles this automatically via an `autouse` fixture.

**Format compatibility.** RouteProfile outputs `.npz` (keys = model names, values = `float32 [768]`). LLMRouter's GraphRouter and MFRouter read `llm_data["model_name"]["embedding"]` (list/array); PersonalizedRouter reads a `.pkl` dict. The two conversion functions in `utils.py` cover both cases, so no router code needed changing.
