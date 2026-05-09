"""Shared pytest fixtures for routeprofile tests.

All fixtures that require torch/torch_geometric are marked so tests can be
collected even when those packages are unavailable.
"""
import json

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData


# ── Synthetic graph ────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_task_graph():
    """Minimal HeteroData with arch(2) + model(3) + dataset(4) nodes.

    model→dataset edges carry benchmark scores (edge_attr).
    Both directions stored so HANConv and propagation can find neighbours.
    """
    data = HeteroData()

    data["architecture"].x               = torch.randn(2, 768)
    data["architecture"].node_names      = ["arch-0", "arch-1"]
    data["architecture"].node_feature_text = ["Architecture A", "Architecture B"]

    data["model"].x              = torch.randn(3, 768)
    data["model"].node_names     = ["model-a", "model-b", "model-c"]
    data["model"].node_feature_text = ["Model A description", "Model B description", "Model C description"]

    data["dataset"].x            = torch.randn(4, 768)
    data["dataset"].node_names   = ["ds-0", "ds-1", "ds-2", "ds-3"]
    data["dataset"].node_feature_text = ["Dataset 0", "Dataset 1", "Dataset 2", "Dataset 3"]

    # architecture → model and reverse
    data["architecture", "arch_to_model", "model"].edge_index    = torch.tensor([[0, 1, 0], [0, 1, 2]])
    data["model", "model_to_arch", "architecture"].edge_index    = torch.tensor([[0, 1, 2], [0, 1, 0]])

    # model → dataset with benchmark scores
    ei_md = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])
    data["model", "model_to_dataset", "dataset"].edge_index = ei_md
    data["model", "model_to_dataset", "dataset"].edge_attr  = torch.rand(5, 1)

    # dataset → model reverse
    data["dataset", "dataset_to_model", "model"].edge_index = ei_md.flip(0)
    data["dataset", "dataset_to_model", "model"].edge_attr  = torch.rand(5, 1)

    return data


@pytest.fixture
def tiny_task_graph_with_domain(tiny_task_graph):
    """Extends tiny_task_graph with domain nodes."""
    data = tiny_task_graph

    data["domain"].x              = torch.randn(2, 768)
    data["domain"].node_names     = ["domain-0", "domain-1"]
    data["domain"].node_feature_text = ["Domain 0 description", "Domain 1 description"]

    # dataset → domain
    data["dataset", "dataset_to_domain", "domain"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    data["domain", "domain_to_dataset", "dataset"].edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]])

    return data


# ── Synthetic embeddings ───────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    """Three L2-normalised 768-dim float32 vectors keyed by model name."""
    rng = np.random.default_rng(0)
    result = {}
    for name in ("model-a", "model-b", "model-c"):
        v = rng.standard_normal(768).astype("float32")
        result[name] = v / (np.linalg.norm(v) + 1e-8)
    return result


# ── File fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def graph_pt_path(tiny_task_graph, tmp_path):
    """Save tiny_task_graph to a temp .pt file and return the path string."""
    path = str(tmp_path / "graph.pt")
    torch.save(tiny_task_graph, path)
    return path


@pytest.fixture
def npz_profile_path(mock_embeddings, tmp_path):
    """Save mock_embeddings to a temp .npz file and return the path string."""
    path = str(tmp_path / "profiles.npz")
    np.savez(path, **mock_embeddings)
    return path


@pytest.fixture
def llm_data_json_path(tmp_path):
    """Minimal llm_data JSON matching mock_embeddings model names."""
    data = {
        "model-a": {
            "size": "7B", "feature": "Model A description", "architecture": "arch-0",
            "model": "provider/model-a", "service": "TestService",
            "api_endpoint": "https://test.api/v1",
            "input_price": 0.1, "output_price": 0.1, "average_score": 0.7,
        },
        "model-b": {
            "size": "9B", "feature": "Model B description", "architecture": "arch-1",
            "model": "provider/model-b", "service": "TestService",
            "api_endpoint": "https://test.api/v1",
            "input_price": 0.2, "output_price": 0.2, "average_score": 0.6,
        },
        "model-c": {
            "size": "70B", "feature": "Model C description", "architecture": "arch-0",
            "model": "provider/model-c", "service": "TestService",
            "api_endpoint": "https://test.api/v1",
            "input_price": 0.5, "output_price": 0.5, "average_score": 0.9,
        },
    }
    path = str(tmp_path / "llm_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ── Mock Longformer ────────────────────────────────────────────────────────────

def mock_longformer(texts, batch_size=32, **kwargs):
    """Fake get_longformer_embedding: returns deterministic random tensors of shape [768]."""
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
