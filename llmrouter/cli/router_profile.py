"""
llmrouter profile — RouteProfile pipeline CLI subcommand.

Provides three sub-subcommands:
  build-graph   : build a heterogeneous PyG graph from model/task/domain metadata
  build-profile : generate per-model embedding profiles from a graph
  apply         : merge RouteProfile .npz embeddings into LLMRouter llm_data JSON

Full workflow example:
  llmrouter profile build-graph --graph-type task_domain --mode standard --output-dir /tmp/graphs
  llmrouter profile build-profile --method emb_gnn --graph /tmp/graphs/task_domain_graph_full.pt --output /tmp/emb_gnn.npz
  llmrouter profile apply --profile /tmp/emb_gnn.npz --llm-data data/llm_candidates.json --output data/llm_candidates_rp.json
"""

import argparse
import sys


# ── Graph type → function name mapping ────────────────────────────────────────

_GRAPH_BUILDERS = {
    "task":                "build_task_graph",
    "query":               "build_query_graph",
    "query_task":          "build_query_task_graph",
    "task_domain":         "build_task_domain_graph",
    "query_task_domain":   "build_query_task_domain_graph",
}

_GRAPH_OUTPUT_NAMES = {
    "task":                "task_graph_full.pt",
    "query":               "query_graph_full.pt",
    "query_task":          "query_task_graph_full.pt",
    "task_domain":         "task_domain_graph_full.pt",
    "query_task_domain":   "query_task_domain_graph_full.pt",
}


# ── Sub-subcommand handlers ────────────────────────────────────────────────────

def _build_graph_command(args) -> None:
    import os
    try:
        import llmrouter.routeprofile.build_data_graph as bdg
    except ImportError as e:
        print(f"Error: RouteProfile graph-building dependencies not installed.\n{e}", file=sys.stderr)
        print("Install with: pip install 'llmrouter-lib[routeprofile]'", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(os.getcwd(), "results", "result_data_graph", args.mode)
    os.makedirs(output_dir, exist_ok=True)

    graph_types = (
        list(_GRAPH_BUILDERS.keys()) if args.graph_type == "all" else [args.graph_type]
    )

    for gtype in graph_types:
        fn = getattr(bdg, _GRAPH_BUILDERS[gtype])
        save_path = os.path.join(output_dir, _GRAPH_OUTPUT_NAMES[gtype])
        print(f"[profile] Building {gtype} graph → {save_path}")

        kwargs = dict(mode=args.mode, save=save_path)
        if args.profile_data_dir:
            pd = args.profile_data_dir
            # Pass overrides for commonly customized paths
            if gtype in ("task_domain", "query_task_domain"):
                import os as _os
                kwargs.update(
                    json=_os.path.join(pd, f"model_feature_{args.mode}.json"),
                    arch=_os.path.join(pd, "model_family_feature.json"),
                    dataset=_os.path.join(pd, "task_feature.json"),
                    domain_map=_os.path.join(pd, "domain_task_map.json"),
                    domain_feat=_os.path.join(pd, "domain_feature.json"),
                )
            else:
                import os as _os
                kwargs.update(
                    json=_os.path.join(pd, f"model_feature_{args.mode}.json"),
                    arch=_os.path.join(pd, "model_family_feature.json"),
                    dataset=_os.path.join(pd, "task_feature.json"),
                )
            if gtype in ("query", "query_task", "query_task_domain"):
                import os as _os
                kwargs["query"] = _os.path.join(pd, f"task_queries_{args.mode}.json")

        fn(**kwargs)
        print(f"[profile] Saved: {save_path}")


def _build_profile_command(args) -> None:
    import os
    try:
        import llmrouter.routeprofile.get_model_profile as gmp
    except ImportError as e:
        print(f"Error: RouteProfile profiling dependencies not installed.\n{e}", file=sys.stderr)
        print("Install with: pip install 'llmrouter-lib[routeprofile]'", file=sys.stderr)
        sys.exit(1)

    if not args.graph:
        print("Error: --graph is required for build-profile.", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required for build-profile.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    method = args.method
    print(f"[profile] Building {method} profile from {args.graph} → {args.output}")

    if method == "flat":
        gmp.build_flat_profile(
            graph=args.graph,
            save=args.output,
            top_k=args.top_k,
            seed=args.seed,
            keep=[],  # save all models
        )
    elif method == "emb_gnn":
        gmp.build_emb_gnn_profile(
            graph=args.graph,
            K=args.K,
            norm=args.norm,
            normalize=args.normalize,
            save=args.output,
            keep=[],
        )
    elif method == "index":
        gmp.build_index_profile(
            save=args.output,
            seed=args.seed,
        )
    elif method == "text_gnn":
        try:
            text_fn = getattr(gmp, "build_text_gnn_profile")
        except AttributeError:
            print("Error: text_gnn requires vllm. Install with: pip install 'llmrouter-lib[routeprofile-text-gnn]'",
                  file=sys.stderr)
            sys.exit(1)
        text_fn(graph=args.graph, save=args.output, keep=[])
    elif method == "trainable_gnn":
        gmp.build_trainable_gnn_profile(
            graph=args.graph,
            save_emb=args.output,
            epochs=args.epochs,
            seed=args.seed,
            keep=[],
        )
    else:
        print(f"Error: Unknown method '{method}'.", file=sys.stderr)
        sys.exit(1)

    print(f"[profile] Profile saved: {args.output}")


def _apply_command(args) -> None:
    from llmrouter.routeprofile.utils import npz_to_llm_embeddings_json, npz_to_pkl

    output = args.output or args.llm_data

    if args.format == "pkl":
        npz_to_pkl(args.profile, output)
    else:
        npz_to_llm_embeddings_json(
            npz_path=args.profile,
            llm_data_path=args.llm_data,
            output_path=output,
            missing=args.missing,
        )
    print(f"[profile] Done. Output: {output}")


# ── Parser registration ────────────────────────────────────────────────────────

def add_profile_parser(subparsers) -> argparse.ArgumentParser:
    """Attach the 'profile' subcommand (with sub-subcommands) to the main parser."""

    profile_parser = subparsers.add_parser(
        "profile",
        help="RouteProfile pipeline: build graphs, model profiles, and apply to llm_data",
        description=(
            "RouteProfile integration: generate richer LLM model embeddings using "
            "graph-based profiling methods, and apply them to LLMRouter's llm_data JSON.\n\n"
            "Sub-subcommands:\n"
            "  build-graph   Build a heterogeneous graph from model/task/domain metadata\n"
            "  build-profile Generate per-model embeddings from a graph (.npz output)\n"
            "  apply         Merge .npz embeddings into llm_data JSON (for router use)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    profile_sub = profile_parser.add_subparsers(
        dest="profile_command",
        metavar="SUBCOMMAND",
    )
    profile_sub.required = True

    # ── build-graph ────────────────────────────────────────────────────────────
    bg = profile_sub.add_parser(
        "build-graph",
        help="Build heterogeneous PyG graph from model/task/domain metadata",
        description=(
            "Constructs a PyTorch Geometric HeteroData graph from model metadata JSON files.\n"
            "Uses bundled default data; override with --profile-data-dir.\n\n"
            "Graph types:\n"
            "  task              nodes: architecture, model, dataset\n"
            "  query             nodes: architecture, model, dataset, query\n"
            "  query_task        nodes: model, dataset, query\n"
            "  task_domain       nodes: architecture, model, dataset, domain\n"
            "  query_task_domain nodes: architecture, model, dataset, domain, query (richest)\n"
            "  all               build all five graph types\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bg.add_argument(
        "--graph-type",
        choices=["task", "query", "query_task", "task_domain", "query_task_domain", "all"],
        default="task_domain",
        help="Graph topology to build (default: task_domain)",
    )
    bg.add_argument(
        "--mode",
        choices=["standard", "newllm"],
        default="standard",
        help="Routing setting: 'standard' (known models) or 'newllm' (unseen models) (default: standard)",
    )
    bg.add_argument(
        "--profile-data-dir",
        default=None,
        metavar="DIR",
        help="Directory containing profile JSON files (default: bundled data)",
    )
    bg.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for output .pt graph files (default: ./results/result_data_graph/{mode}/)",
    )
    bg.set_defaults(func=_build_graph_command)

    # ── build-profile ──────────────────────────────────────────────────────────
    bp = profile_sub.add_parser(
        "build-profile",
        help="Generate per-model embedding profiles from a graph",
        description=(
            "Generates one 768-dim embedding per model using the chosen profiling method.\n"
            "Output is a .npz file (keys = model names, values = float32 arrays).\n\n"
            "Methods:\n"
            "  flat          random-sample neighbour texts, encode with Longformer\n"
            "  emb_gnn       K-hop degree-normalised message passing (training-free GNN)\n"
            "  index         random baseline (useful for ablations)\n"
            "  text_gnn      LLM-based text summarisation per hop (requires vllm)\n"
            "  trainable_gnn self-supervised HANConv with masked feature reconstruction\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bp.add_argument(
        "--method",
        choices=["flat", "emb_gnn", "index", "text_gnn", "trainable_gnn"],
        required=True,
        help="Profile generation method",
    )
    bp.add_argument(
        "--graph",
        default=None,
        metavar="PATH",
        help="Input .pt HeteroData graph file (required for all methods except 'index')",
    )
    bp.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output .npz file path",
    )
    bp.add_argument(
        "--K",
        type=int,
        default=2,
        help="Number of propagation hops for emb_gnn (default: 2)",
    )
    bp.add_argument(
        "--norm",
        choices=["sym", "right", "left", "none"],
        default="sym",
        help="Degree normalisation for emb_gnn (default: sym)",
    )
    bp.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalise output embeddings (emb_gnn only)",
    )
    bp.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Neighbours to sample per node type for flat profile (default: 5)",
    )
    bp.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for trainable_gnn (default: 100)",
    )
    bp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    bp.set_defaults(func=_build_profile_command)

    # ── apply ──────────────────────────────────────────────────────────────────
    ap = profile_sub.add_parser(
        "apply",
        help="Merge RouteProfile .npz embeddings into LLMRouter llm_data JSON",
        description=(
            "Reads a RouteProfile-generated .npz file and overwrites the 'embedding' field\n"
            "for each model in the llm_data JSON. The output JSON is a drop-in replacement\n"
            "for the original llm_data file — just update the YAML config path.\n\n"
            "For PersonalizedRouter, use --format pkl to produce a .pkl file instead.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--profile",
        required=True,
        metavar="PATH",
        help="RouteProfile .npz file (model_name → embedding)",
    )
    ap.add_argument(
        "--llm-data",
        required=True,
        metavar="PATH",
        help="LLMRouter llm_data JSON file (source of model metadata)",
    )
    ap.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output path (default: overwrite --llm-data in-place)",
    )
    ap.add_argument(
        "--format",
        choices=["json", "pkl"],
        default="json",
        help="Output format: 'json' for GraphRouter/MFRouter, 'pkl' for PersonalizedRouter (default: json)",
    )
    ap.add_argument(
        "--missing",
        choices=["warn", "skip", "error"],
        default="warn",
        help="Behaviour when a model in llm_data has no profile entry (default: warn)",
    )
    ap.set_defaults(func=_apply_command)

    return profile_parser
