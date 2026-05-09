"""Tests for all 5 build_*_graph functions.

get_longformer_embedding is mocked so no GPU or network access is required.
Each builder is tested with the bundled default data files (mode='standard').
"""
import importlib
import os
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import HeteroData

def mock_longformer(texts, batch_size=32, **kwargs):
    """Fake get_longformer_embedding — deterministic random [768] tensors, no GPU."""
    if isinstance(texts, str):
        torch.manual_seed(abs(hash(texts)) % (2**32))
        return torch.randn(768)
    n = len(texts)
    if n == 1:
        torch.manual_seed(abs(hash(texts[0])) % (2**32))
        return torch.randn(768)
    result = []
    for t in texts:
        torch.manual_seed(abs(hash(t)) % (2**32))
        result.append(torch.randn(768))
    return torch.stack(result)

# Import modules directly (bypasses __init__.py aliasing)
_task_mod  = importlib.import_module("llmrouter.routeprofile.build_data_graph.build_task_graph")
_query_mod = importlib.import_module("llmrouter.routeprofile.build_data_graph.build_query_graph")
_qt_mod    = importlib.import_module("llmrouter.routeprofile.build_data_graph.build_query_task_graph")
_td_mod    = importlib.import_module("llmrouter.routeprofile.build_data_graph.build_task_domain_graph")
_qtd_mod   = importlib.import_module("llmrouter.routeprofile.build_data_graph.build_query_task_domain_graph")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_graph(path: str) -> HeteroData:
    return torch.load(path, weights_only=False)


# ── build_task_graph ──────────────────────────────────────────────────────────

class TestBuildTaskGraph:
    def test_creates_pt_file(self, tmp_path):
        save = str(tmp_path / "task_graph.pt")
        with patch.object(_task_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _task_mod.build_task_graph(mode="standard", save=save)
        assert os.path.isfile(save)

    def test_node_types(self, tmp_path):
        save = str(tmp_path / "task_graph.pt")
        with patch.object(_task_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _task_mod.build_task_graph(mode="standard", save=save)
        g = _load_graph(save)
        assert "model"        in g.node_types
        assert "dataset"      in g.node_types
        assert "architecture" in g.node_types

    def test_model_embeddings_shape(self, tmp_path):
        save = str(tmp_path / "task_graph.pt")
        with patch.object(_task_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _task_mod.build_task_graph(mode="standard", save=save)
        g = _load_graph(save)
        x = g["model"].x
        assert x.ndim == 2
        assert x.shape[1] == 768
        assert x.shape[0] >= 1

    def test_model_dataset_edges_have_attr(self, tmp_path):
        """model↔dataset edges must carry benchmark-score edge_attr."""
        save = str(tmp_path / "task_graph.pt")
        with patch.object(_task_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _task_mod.build_task_graph(mode="standard", save=save)
        g = _load_graph(save)
        # At least one direction of model↔dataset edges must have edge_attr
        edge_types_with_attr = [
            et for et in g.edge_types
            if "model" in et and "dataset" in et and hasattr(g[et], "edge_attr")
        ]
        assert edge_types_with_attr, "No model↔dataset edges with edge_attr found"

    def test_node_names_stored(self, tmp_path):
        save = str(tmp_path / "task_graph.pt")
        with patch.object(_task_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _task_mod.build_task_graph(mode="standard", save=save)
        g = _load_graph(save)
        assert hasattr(g["model"], "node_names"), "model nodes must have .node_names"
        assert len(g["model"].node_names) == g["model"].x.shape[0]


# ── build_query_graph ─────────────────────────────────────────────────────────

class TestBuildQueryGraph:
    def test_creates_pt_file(self, tmp_path):
        save = str(tmp_path / "query_graph.pt")
        with patch.object(_query_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _query_mod.build_query_graph(mode="standard", save=save)
        assert os.path.isfile(save)

    def test_has_query_nodes(self, tmp_path):
        save = str(tmp_path / "query_graph.pt")
        with patch.object(_query_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _query_mod.build_query_graph(mode="standard", save=save)
        g = _load_graph(save)
        assert "query" in g.node_types
        assert g["query"].x.shape[1] == 768


# ── build_query_task_graph ────────────────────────────────────────────────────

class TestBuildQueryTaskGraph:
    def test_node_types(self, tmp_path):
        save = str(tmp_path / "qt_graph.pt")
        with patch.object(_qt_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _qt_mod.build_query_task_graph(mode="standard", save=save)
        g = _load_graph(save)
        for nt in ("model", "dataset", "query"):
            assert nt in g.node_types, f"Missing node type: {nt}"


# ── build_task_domain_graph ───────────────────────────────────────────────────

class TestBuildTaskDomainGraph:
    def test_has_domain_nodes(self, tmp_path):
        save = str(tmp_path / "td_graph.pt")
        with patch.object(_td_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _td_mod.build_task_domain_graph(mode="standard", save=save)
        g = _load_graph(save)
        assert "domain" in g.node_types
        assert g["domain"].x.shape[1] == 768

    def test_domain_dataset_edges_exist(self, tmp_path):
        save = str(tmp_path / "td_graph.pt")
        with patch.object(_td_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _td_mod.build_task_domain_graph(mode="standard", save=save)
        g = _load_graph(save)
        domain_edges = [et for et in g.edge_types if "domain" in et]
        assert domain_edges, "No edges involving domain nodes"


# ── build_query_task_domain_graph ─────────────────────────────────────────────

class TestBuildQueryTaskDomainGraph:
    def test_all_node_types(self, tmp_path):
        """The richest graph type must contain all 5 node types."""
        save = str(tmp_path / "qtd_graph.pt")
        with patch.object(_qtd_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _qtd_mod.build_query_task_domain_graph(mode="standard", save=save)
        g = _load_graph(save)
        for nt in ("architecture", "model", "dataset", "domain", "query"):
            assert nt in g.node_types, f"Missing node type: {nt}"

    def test_all_embeddings_dim_768(self, tmp_path):
        save = str(tmp_path / "qtd_graph.pt")
        with patch.object(_qtd_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _qtd_mod.build_query_task_domain_graph(mode="standard", save=save)
        g = _load_graph(save)
        for nt in g.node_types:
            assert g[nt].x.shape[1] == 768, f"Node type '{nt}' has wrong embedding dim"
