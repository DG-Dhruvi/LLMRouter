"""
RouteProfile integration for LLMRouter.

Provides graph-based LLM profiling as an alternative to simple Longformer-encoded
model descriptions. Generated profiles can be used as drop-in replacements for
the 'embedding' field in llm_data JSON, improving routers that rely on model
embeddings (GraphRouter, MFRouter, PersonalizedRouter, etc.).

Pipeline:
    1. build-graph  : build a heterogeneous graph from model/task/domain metadata
    2. build-profile: generate per-model embeddings from the graph
    3. apply        : merge generated embeddings into llm_data JSON for use by routers

CLI:
    llmrouter profile build-graph   --graph-type task_domain --mode standard --output-dir ...
    llmrouter profile build-profile --method emb_gnn --graph graph.pt --output profile.npz
    llmrouter profile apply         --profile profile.npz --llm-data llm.json --output llm_rp.json

Note: graph building and profile generation require torch and torch_geometric.
      The 'apply' utility only requires numpy and the standard library.
"""


def __getattr__(name: str):
    """Lazy-load graph/profile functions on first access to avoid hard torch dependency."""
    _graph_fns = {
        "build_task_graph",
        "build_query_graph",
        "build_query_task_graph",
        "build_task_domain_graph",
        "build_query_task_domain_graph",
    }
    _profile_fns = {
        "build_flat_profile",
        "build_emb_gnn_profile",
        "build_index_profile",
        "build_trainable_gnn_profile",
        "build_text_gnn_profile",
    }

    if name in _graph_fns:
        from llmrouter.routeprofile import build_data_graph as _bdg
        return getattr(_bdg, name)

    if name in _profile_fns:
        from llmrouter.routeprofile import get_model_profile as _gmp
        return getattr(_gmp, name)

    raise AttributeError(f"module 'llmrouter.routeprofile' has no attribute '{name!r}'")


__all__ = [
    "build_task_graph",
    "build_query_graph",
    "build_query_task_graph",
    "build_task_domain_graph",
    "build_query_task_domain_graph",
    "build_flat_profile",
    "build_emb_gnn_profile",
    "build_index_profile",
    "build_trainable_gnn_profile",
    "build_text_gnn_profile",
]
