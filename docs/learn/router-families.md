# Router families

LLMRouter includes routers ranging from simple baselines to trainable multi-round and personalized routers.
This page goes deeper into the algorithmic design space and the routing philosophies that motivated the framework.

!!! note
    Router availability depends on your installed dependencies. Always verify with `llmrouter list-routers`.

## Routing as a learning problem

At a high level, routing is a decision problem:

- Input: a user query `q` and a candidate set of models `M = {m_1, ..., m_k}`
- Output: a choice `m*` (and optionally a response) that optimizes some objective

Most routers in LLMRouter can be understood as different ways to approximate:

`m* = argmax_m Score(m, q)`

The main design choice is: what is `Score`, what supervision signal do we learn from, and what features do we condition on?

### Why so many router types?
Different deployment contexts have very different constraints:

- Do you have routing logs (query, model, quality/cost signals) or not?
- Do you need a single-pass decision, or do you allow multi-round decomposition/aggregation?
- Is personalization required (user-specific routing), or is one policy enough?
- Do you optimize cost-quality trade-offs explicitly, or treat routing as pure "best model" prediction?

LLMRouter exposes these choices through a unified contract (`route_single` / `route_batch`) so you can swap algorithms without rewriting the surrounding workflow.

## Design axes (a mental model)

You can place most routers on a few axes:

- **Decision granularity**: global (same model for all queries) vs per-query vs per-turn (multi-round)
- **Features**: query embeddings vs raw text (LLM-based) vs graph structure vs user/session history
- **Learning signal**: classification labels vs pairwise preferences vs contrastive objectives vs self-verification
- **Model set size**: 2-model gating vs multi-model selection

## Taxonomy (high-level)

This taxonomy follows the categories used in the `main` branch README, plus a few practical subfamilies.
For the full router list and per-router links, see [API Reference - Routers](../api/routers.md).

### Single-round routers (one query -> one model)
- Baselines: `smallest_llm`, `largest_llm`
- Classic ML / embedding-based: `knnrouter`, `svmrouter`, `mlprouter`, `mfrouter`
- Ranking and preference-style: `elorouter`, `dcrouter` (alias: `routerdc`)
- Hybrid or probabilistic: `hybrid_llm`
- Mixing and ensembling: `automix` (alias: `automixrouter`)

### Multi-round routers (conversation-aware)
- LLM-driven decomposition/aggregation: `llmmultiroundrouter`
- Retrieval-style multi-round: `knnmultiroundrouter`
- Pretrained multi-round policy: `router_r1`

### Personalized routers (user preference-aware)
- `gmtrouter` (alias: `gmt_router`)

### Graph-based and structured routers
- `graphrouter` (alias: `graph_router`)

### LLM-driven single-round routers
- `causallm_router` (alias: `causallmrouter`)

## Data and compute needs (what you pay for)
- Baselines: no training, usually only require `llm_data` (model metadata)
- Embedding-based supervised routers: need training data + embeddings + routing labels; fast at inference
- Graph/personalized: need additional structure (and often extra dependencies)
- LLM-driven and multi-round: may require API calls at inference time (and additional configs)

## Quick chooser
| If you want | Start with |
| --- | --- |
| Validate configs/endpoints quickly | `smallest_llm` + `--route-only` |
| Strong trainable baseline | `knnrouter` |
| Multi-turn chat routing | `llmmultiroundrouter` or `knnmultiroundrouter` |
| Pretrained multi-round routing policy | `router_r1` |
| Personalization | `gmtrouter` |

## How to choose (practical rules of thumb)
- If you are wiring up endpoints for the first time, start with `smallest_llm` + `--route-only` to validate `llm_data`.
- If you want a strong, fast trainable baseline, start with `knnrouter`.
- If you care about cost/quality trade-offs, ensure your data contains cost/performance signals and tune `metric.weights`.
- If your workload is multi-turn chat, look at multi-round routers (they may use `answer_query` internally).
- If you need personalization, use `gmtrouter` and make sure your data matches its expected format.

## Family deep dives (what each one is actually doing)

This section summarizes the algorithmic idea behind each family. For full details and citations, follow the per-router READMEs linked from [API Reference - Routers](../api/routers.md).

### Baselines and global ranking

**`smallest_llm` / `largest_llm`**
- Routing philosophy: "one model fits all", optimized for cost (smallest) or quality (largest)
- Signal: none (heuristic based on `llm_data[*].size`)
- Failure mode: ignores query difficulty and domain

**`elorouter`**
- Formulation: learn a global ranking from pairwise "battles" between models, then always pick the top model
- Signal: preference-style comparisons derived from routing data
- Strength: extremely simple and stable; useful as a global baseline
- Failure mode: still query-independent, so it cannot specialize by query type

### Embedding-based supervised selection
These routers treat routing as supervised learning on query embeddings. You can think of them as learning a decision boundary in embedding space.

**`knnrouter`**
- Formulation: instance-based nearest-neighbor vote in embedding space
- Signal: "best model" label from historical data
- Strength: interpretable; no parametric training (stores examples)
- Failure mode: memory and latency scale with dataset size; sensitive to embedding quality

**`svmrouter`**
- Formulation: multi-class classification with margin maximization (kernel SVM)
- Signal: "best model" label
- Strength: strong baseline in high-dimensional spaces; convex optimization
- Failure mode: kernel choice and hyperparameters matter; limited expressivity without enough support vectors

**`mlprouter`**
- Formulation: multi-class classification with a small neural network (MLP) on embeddings
- Signal: "best model" label
- Strength: flexible non-linear boundaries; easy to extend
- Failure mode: can overfit; needs careful regularization and data coverage

### Preference learning and latent factor models
These routers model routing as ranking rather than classification.

**`mfrouter` (matrix factorization)**
- Formulation: learn a shared latent space for queries and models and score interactions with a bilinear function
- Signal: pairwise ranking comparisons (winner vs losers per query)
- Strength: captures structured query-model affinities; scalable at inference
- Failure mode: cold-start for new models; latent factors are less interpretable

**`routerdc` / `dcrouter` (dual-contrastive)**
- Formulation: represent queries with a pretrained encoder + learn model embeddings; train with multiple contrastive objectives
- Signal: contrast query-to-model alignment (top-k vs last-k), plus optional query-to-query clustering/task contrast
- Strength: learns a geometry aligned to routing behavior, not just labels; can generalize beyond hard class boundaries
- Failure mode: heavier compute and more moving parts (encoder, clustering, temperature, etc.)

### Cost-aware 2-model gating
These routers explicitly model the cost-quality trade-off between a small and a large model.

**`hybrid_llm`**
- Formulation: predict a "quality gap" signal and route to the small model when it is likely good enough
- Signal: supervision derived from small-vs-large performance on training data
- Strength: simple and effective when your deployment is dominated by a clear small/large pair
- Failure mode: only 2 models; depends on good "quality" signals in data

**`automix` / `automixrouter`**
- Formulation: self-verification; call a small model, estimate confidence, escalate to a large model when uncertain (threshold or POMDP-style policy)
- Signal: verification consistency + training data to fit the decision policy
- Strength: does not require learning a multi-model classifier; aligns with "use expensive model only when needed"
- Failure mode: verification adds extra calls; still only routes between 2 models

### Graph-based routing
These routers use message passing to exploit relationships between queries, models, and their interactions.

**`graphrouter`**
- Formulation: build a bipartite (or heterogeneous) graph and learn routing as edge scoring / link prediction
- Signal: observed query-model performance edges; training via edge masking
- Strength: captures higher-order structure (similar queries, correlated models)
- Failure mode: graph construction and training cost; cold-start can be tricky

### LLM-driven routing and multi-round reasoning
These routers use LLMs directly in the routing loop.

**`causallm_router`**
- Formulation: finetune a causal LM to generate the best model name given the query (routing as generation)
- Signal: supervised finetuning on (query -> best model name) pairs
- Strength: can capture routing heuristics that are hard to encode as embeddings alone
- Failure mode: finetuning cost; parsing errors; GPU requirements

**`llmmultiroundrouter`**
- Formulation: zero-shot decomposition and routing with an LLM; execute sub-queries and aggregate responses
- Signal: none (prompted reasoning), relies on model descriptions in `llm_data`
- Strength: works when you have no training data; handles multi-faceted queries
- Failure mode: high latency/cost; prompt sensitivity

**`knnmultiroundrouter`**
- Formulation: decompose -> route each sub-query with KNN -> execute -> aggregate
- Signal: KNN routing data + an LLM for decomposition/aggregation
- Strength: combines data-driven routing with multi-step reasoning
- Failure mode: multi-call cost; relies on both good decomposition and good embeddings

**`router_r1`**
- Formulation: agentic reasoning loop; iteratively calls an external "routing pool" API via `<search>` tags and stops at `<answer>`
- Signal: pretrained policy (not trained via the standard CLI)
- Strength: exposes a more general "routing as search" interface
- Failure mode: requires GPU + external routing pool; higher variance and cost

### Personalized routing
**`gmtrouter`**
- Formulation: heterogeneous graph neural network with user/session/query/model/response nodes to learn user preferences
- Signal: multi-turn interaction logs with embeddings and ratings (special JSONL format)
- Strength: captures personalization and session dynamics
- Failure mode: requires special data format and additional compute; harder to debug than single-round routers

## Full router table
For the complete, up-to-date router list (including train/infer support and per-router docs links), see:
[API Reference - Routers](../api/routers.md#router-table)

## Avoiding drift
This page is intentionally high-level. The canonical "what exists" view is:

- `llmrouter list-routers` (runtime truth)
- [API Reference - Routers](../api/routers.md) (docs index with per-router links)
