"""Integration tests: RouteProfile-generated embeddings consumed by LLM routers.

Verifies two levels of integration:

Level 1 — format compatibility (profile_npz fixture: synthetic random .npz)
  npz profile → apply (json / pkl) → router loading method
  Tests that the router can consume any correctly-formatted .npz, regardless
  of how the embeddings were generated.

Level 2 — end-to-end pipeline (all_method_profile parametrized fixture)
  graph + build_model_profile(method) → .npz → apply → router loading
  Covers all 5 RouteProfile methods:
    emb_gnn      — K-hop propagation on graph .x tensors (no Longformer)
    flat         — average Longformer embeddings (Longformer mocked)
    index        — random orthogonal baseline (no graph, uses TARGET_MODELS)
    trainable_gnn — self-supervised HANConv (no Longformer)
    text_gnn_k0  — Longformer on raw node_feature_text, K=0 (Longformer mocked)

Routers tested:
  - GraphRouter  (reads llm_data["model"]["embedding"])
  - PersonalizedRouter  (reads llm_embedding_data .pkl)

MFRouter is excluded: it learns model embeddings from scratch during training
and does not consume pre-computed profiles.

Router instantiation is deliberately avoided — initialising GraphRouter or
PersonalizedRouter requires real training data files and GPU setup.
The embedding-loading methods are invoked directly on a lightweight stub.
"""
import importlib
import json
import os
import pickle
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from llmrouter.routeprofile.utils import npz_to_llm_embeddings_json, npz_to_pkl


MODEL_NAMES = ["model-a", "model-b", "model-c"]
EMB_DIM = 768

_emb_mod       = importlib.import_module("llmrouter.routeprofile.get_model_profile.training_free.emb_gnn_profile")
_flat_mod      = importlib.import_module("llmrouter.routeprofile.get_model_profile.training_free.flat_profile")
_index_mod     = importlib.import_module("llmrouter.routeprofile.get_model_profile.training_free.index_profile")
_trainable_mod = importlib.import_module("llmrouter.routeprofile.get_model_profile.trainable.trainable_gnn_profile")
_text_gnn_mod  = importlib.import_module("llmrouter.routeprofile.get_model_profile.training_free.text_gnn_profile")


def _mock_longformer(texts, batch_size=32, **kwargs):
    if isinstance(texts, str):
        torch.manual_seed(abs(hash(texts)) % (2**32))
        return torch.randn(768)
    result = []
    for t in texts:
        torch.manual_seed(abs(hash(t)) % (2**32))
        result.append(torch.randn(768))
    return torch.stack(result) if len(result) > 1 else result[0]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def profile_npz(tmp_path):
    """Synthetic .npz in the correct RouteProfile format (random arrays, no generation)."""
    rng = np.random.default_rng(42)
    data = {name: rng.standard_normal(EMB_DIM).astype("float32") for name in MODEL_NAMES}
    path = str(tmp_path / "profile.npz")
    np.savez(path, **data)
    return path, data  # (path, raw_arrays)


def _make_llm_data_json(model_names, path):
    """Write a minimal llm_data JSON for the given model names."""
    data = {
        name: {
            "size": "7B", "feature": f"Description of {name}",
            "architecture": "transformer-decoder", "model": f"provider/{name}",
            "service": "TestService", "api_endpoint": "https://test.api/v1",
            "input_price": 0.1, "output_price": 0.1, "average_score": 0.7,
        }
        for name in model_names
    }
    with open(path, "w") as f:
        json.dump(data, f)


@pytest.fixture(
    params=["emb_gnn", "flat", "index", "trainable_gnn", "text_gnn_k0", "text_gnn_k2"],
)
def all_method_profile(request, graph_pt_path, tmp_path):
    """
    Parametrized fixture that generates a real RouteProfile .npz using each of
    the 6 profile configurations, then returns (npz_path, used_model_names, raw_arrays).

    Notes:
      - emb_gnn / trainable_gnn: operate on graph .x tensors — no Longformer needed.
      - flat / text_gnn_k0: call get_longformer_embedding — mocked here.
      - index: ignores the graph and generates profiles for its own TARGET_MODELS
        list (real LLM names), so used_model_names differs from MODEL_NAMES.
      - text_gnn_k2: requires vLLM + free GPU (run standalone with CUDA_VISIBLE_DEVICES=9).
        Skipped automatically if vllm is not installed.
    """
    method = request.param
    out = str(tmp_path / f"profile_{method}.npz")

    if method == "emb_gnn":
        _emb_mod.build_model_profile(graph=graph_pt_path, K=1, save=out, keep=MODEL_NAMES)
        used_models = MODEL_NAMES

    elif method == "flat":
        with patch.object(_flat_mod, "get_longformer_embedding", side_effect=_mock_longformer):
            _flat_mod.build_model_profile(graph=graph_pt_path, seed=42, save=out, keep=MODEL_NAMES)
        used_models = MODEL_NAMES

    elif method == "index":
        _index_mod.build_model_profile(save=out, seed=42)
        used_models = list(np.load(out).files)   # TARGET_MODELS names

    elif method == "trainable_gnn":
        _trainable_mod.build_model_profile(
            graph=graph_pt_path, save_emb=out, epochs=2, seed=42, keep=MODEL_NAMES,
        )
        used_models = MODEL_NAMES

    elif method == "text_gnn_k0":
        with patch.object(_text_gnn_mod, "get_longformer_embedding", side_effect=_mock_longformer):
            _text_gnn_mod.build_model_profile(
                graph=graph_pt_path, K=0, emb_save=out, keep=MODEL_NAMES,
            )
        used_models = MODEL_NAMES

    elif method == "text_gnn_k2":
        if importlib.util.find_spec("vllm") is None:
            pytest.skip("text_gnn_k2 requires vllm")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        _text_gnn_mod.build_model_profile(
            graph=graph_pt_path, K=2,
            model="Qwen/Qwen2.5-7B-Instruct",
            emb_save=out, keep=MODEL_NAMES,
        )
        used_models = MODEL_NAMES

    raw = {name: np.load(out)[name] for name in used_models}
    return out, used_models, raw


@pytest.fixture
def llm_data_json(tmp_path):
    """Minimal llm_data.json compatible with LLMRouter routers."""
    data = {
        name: {
            "size": "7B",
            "feature": f"Description of {name}",
            "architecture": "transformer-decoder",
            "model": f"provider/{name}",
            "service": "TestService",
            "api_endpoint": "https://test.api/v1",
            "input_price": 0.1,
            "output_price": 0.1,
            "average_score": 0.7,
        }
        for name in MODEL_NAMES
    }
    path = str(tmp_path / "llm_data.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── GraphRouter integration ───────────────────────────────────────────────────

class TestGraphRouterEmbeddingIntegration:
    """
    Tests that GraphRouter._prepare_llm_embeddings correctly consumes
    embeddings produced by npz_to_llm_embeddings_json.

    We call the method directly on a stub object to avoid needing real
    training data files.
    """

    def _make_stub(self, llm_data: dict) -> types.SimpleNamespace:
        """Minimal object with the attributes _prepare_llm_embeddings needs."""
        stub = types.SimpleNamespace()
        stub.llm_data = llm_data          # set by DataLoader normally
        stub.cfg = {}                     # empty cfg — method must use llm_data
        stub.model_names = MODEL_NAMES
        stub.query_dim = EMB_DIM
        return stub

    def test_embeddings_loaded_not_random(self, profile_npz, llm_data_json, tmp_path):
        """_prepare_llm_embeddings must use profile embeddings, not random init."""
        npz_path, _ = profile_npz
        out_json = str(tmp_path / "llm_with_emb.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_json, out_json)

        with open(out_json) as f:
            llm_data = json.load(f)

        from llmrouter.models.graphrouter.router import GraphRouter
        stub = self._make_stub(llm_data)
        GraphRouter._prepare_llm_embeddings(stub)

        # Run twice — deterministic profile → identical scaled embeddings
        stub2 = self._make_stub(llm_data)
        GraphRouter._prepare_llm_embeddings(stub2)

        np.testing.assert_allclose(
            stub.llm_embedding, stub2.llm_embedding, rtol=1e-5,
            err_msg="_prepare_llm_embeddings is non-deterministic; likely using random init",
        )

    def test_embedding_shape(self, profile_npz, llm_data_json, tmp_path):
        """Resulting llm_embedding must be (num_models, EMB_DIM) after MinMaxScaler."""
        npz_path, _ = profile_npz
        out_json = str(tmp_path / "llm_with_emb.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_json, out_json)

        with open(out_json) as f:
            llm_data = json.load(f)

        from llmrouter.models.graphrouter.router import GraphRouter
        stub = self._make_stub(llm_data)
        GraphRouter._prepare_llm_embeddings(stub)

        assert stub.llm_embedding.shape == (len(MODEL_NAMES), EMB_DIM)
        assert stub.llm_dim == EMB_DIM

    def test_embedding_values_match_profile(self, profile_npz, llm_data_json, tmp_path):
        """Relative ordering of embeddings must come from the profile, not random."""
        npz_path, raw = profile_npz
        out_json = str(tmp_path / "llm_with_emb.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_json, out_json)

        with open(out_json) as f:
            llm_data = json.load(f)

        from llmrouter.models.graphrouter.router import GraphRouter
        stub = self._make_stub(llm_data)
        GraphRouter._prepare_llm_embeddings(stub)

        # MinMaxScaler preserves rank order per dimension.
        # Check that the pre-scale embedding order matches the raw profile order.
        raw_matrix = np.stack([raw[m] for m in MODEL_NAMES])
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(raw_matrix)
        # JSON round-trip (float32 → Python float → float32) introduces ~1e-7 error;
        # use atol instead of tight rtol.
        np.testing.assert_allclose(stub.llm_embedding, expected, atol=1e-5,
                                   err_msg="Scaled embeddings don't match profile after MinMaxScaler")

    def test_missing_model_falls_back_to_random(self, tmp_path):
        """Models not in the profile must fall back to random init, not error."""
        rng = np.random.default_rng(0)
        partial_npz = str(tmp_path / "partial.npz")
        np.savez(partial_npz, **{"model-a": rng.standard_normal(EMB_DIM).astype("float32")})

        llm_data = {
            "model-a": {"feature": "A"},
            "model-b": {"feature": "B"},  # not in .npz
        }
        partial_json = str(tmp_path / "partial_llm.json")
        with open(partial_json, "w") as f:
            json.dump(llm_data, f)

        out_json = str(tmp_path / "out.json")
        npz_to_llm_embeddings_json(partial_npz, partial_json, out_json, missing="warn")

        with open(out_json) as f:
            merged = json.load(f)

        from llmrouter.models.graphrouter.router import GraphRouter
        stub = types.SimpleNamespace(
            llm_data=merged,
            cfg={},
            model_names=["model-a", "model-b"],
            query_dim=EMB_DIM,
        )
        # Must not raise; model-b uses random init
        GraphRouter._prepare_llm_embeddings(stub)
        assert stub.llm_embedding.shape == (2, EMB_DIM)

    def test_metadata_preserved_after_apply(self, profile_npz, llm_data_json, tmp_path):
        """Non-embedding fields (api_endpoint, size, …) must survive the apply step."""
        npz_path, _ = profile_npz
        out_json = str(tmp_path / "llm_with_emb.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_json, out_json)

        with open(llm_data_json) as f:
            original = json.load(f)
        with open(out_json) as f:
            result = json.load(f)

        for name in MODEL_NAMES:
            for field in ("size", "feature", "api_endpoint", "service"):
                assert result[name].get(field) == original[name].get(field), (
                    f"Field '{field}' changed for model '{name}'"
                )
            assert "embedding" in result[name], f"No embedding field for '{name}'"


# ── PersonalizedRouter integration ────────────────────────────────────────────

class TestPersonalizedRouterEmbeddingIntegration:
    """
    Tests that PersonalizedRouter._normalize_llm_embeddings correctly consumes
    the .pkl produced by npz_to_pkl.

    Instantiating the full router requires real training CSVs and GPU setup,
    so we call the method on a stub instead.
    """

    def test_normalize_from_pkl_shape(self, profile_npz, tmp_path):
        """_normalize_llm_embeddings must return (num_models, EMB_DIM) float32 array."""
        npz_path, _ = profile_npz
        pkl_path = str(tmp_path / "embeddings.pkl")
        npz_to_pkl(npz_path, pkl_path)

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        from llmrouter.models.personalizedrouter.router import PersonalizedRouter
        result = PersonalizedRouter._normalize_llm_embeddings(None, pkl_data, MODEL_NAMES)

        assert result.shape == (len(MODEL_NAMES), EMB_DIM)
        assert result.dtype == np.float32

    def test_normalize_values_match_profile(self, profile_npz, tmp_path):
        """Values from _normalize_llm_embeddings must exactly match the raw profile."""
        npz_path, raw = profile_npz
        pkl_path = str(tmp_path / "embeddings.pkl")
        npz_to_pkl(npz_path, pkl_path)

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        from llmrouter.models.personalizedrouter.router import PersonalizedRouter
        result = PersonalizedRouter._normalize_llm_embeddings(None, pkl_data, MODEL_NAMES)

        for i, name in enumerate(MODEL_NAMES):
            np.testing.assert_allclose(
                result[i], raw[name], rtol=1e-5,
                err_msg=f"Embedding mismatch for '{name}'",
            )

    def test_pkl_format_accepted(self, profile_npz, tmp_path):
        """_normalize_llm_embeddings must not raise for the dict format npz_to_pkl produces."""
        npz_path, _ = profile_npz
        pkl_path = str(tmp_path / "embeddings.pkl")
        npz_to_pkl(npz_path, pkl_path)

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        assert isinstance(pkl_data, dict), "npz_to_pkl must produce a dict"
        for name in MODEL_NAMES:
            assert name in pkl_data
            assert isinstance(pkl_data[name], np.ndarray)
            assert pkl_data[name].dtype == np.float32

        from llmrouter.models.personalizedrouter.router import PersonalizedRouter
        # Must not raise
        result = PersonalizedRouter._normalize_llm_embeddings(None, pkl_data, MODEL_NAMES)
        assert result is not None

    def test_model_names_filter_applied(self, profile_npz, tmp_path):
        """npz_to_pkl with model_names kwarg → only those models in pkl."""
        npz_path, _ = profile_npz
        pkl_path = str(tmp_path / "filtered.pkl")
        npz_to_pkl(npz_path, pkl_path, model_names=["model-a", "model-c"])

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        assert set(pkl_data.keys()) == {"model-a", "model-c"}

        from llmrouter.models.personalizedrouter.router import PersonalizedRouter
        result = PersonalizedRouter._normalize_llm_embeddings(
            None, pkl_data, ["model-a", "model-c"]
        )
        assert result.shape == (2, EMB_DIM)

    def test_full_pipeline_graphrouter_to_personalizedrouter(self, profile_npz, llm_data_json, tmp_path):
        """
        Smoke test: the same profile can be applied in both formats
        (JSON for GraphRouter, pkl for PersonalizedRouter) and both produce
        embeddings of the right shape.
        """
        npz_path, raw = profile_npz

        # JSON branch (GraphRouter)
        json_out = str(tmp_path / "llm_with_emb.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_json, json_out)
        with open(json_out) as f:
            llm_data = json.load(f)

        from llmrouter.models.graphrouter.router import GraphRouter
        stub_gr = types.SimpleNamespace(
            llm_data=llm_data, cfg={}, model_names=MODEL_NAMES, query_dim=EMB_DIM,
        )
        GraphRouter._prepare_llm_embeddings(stub_gr)
        assert stub_gr.llm_embedding.shape == (len(MODEL_NAMES), EMB_DIM)

        # pkl branch (PersonalizedRouter)
        pkl_out = str(tmp_path / "embeddings.pkl")
        npz_to_pkl(npz_path, pkl_out)
        with open(pkl_out, "rb") as f:
            pkl_data = pickle.load(f)

        from llmrouter.models.personalizedrouter.router import PersonalizedRouter
        emb_pr = PersonalizedRouter._normalize_llm_embeddings(None, pkl_data, MODEL_NAMES)
        assert emb_pr.shape == (len(MODEL_NAMES), EMB_DIM)


# ── Level-2: end-to-end with real RouteProfile output (all 5 methods) ─────────

class TestEndToEndAllMethodsToRouter:
    """
    Level-2 integration: profiles are generated by RouteProfile's actual pipeline
    for all 5 methods, then consumed by the router loading methods.

    The all_method_profile fixture is parametrized over:
      emb_gnn, flat, index, trainable_gnn, text_gnn_k0

    Each method × 2 routers = 10 parametrized test cases.
    """

    def test_graphrouter_consumes_profile(self, all_method_profile, tmp_path):
        """GraphRouter._prepare_llm_embeddings must accept output of every profile method."""
        npz_path, used_models, raw = all_method_profile

        llm_data_path = str(tmp_path / "llm_data.json")
        _make_llm_data_json(used_models, llm_data_path)

        out_json = str(tmp_path / "llm_with_emb.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_path, out_json)

        with open(out_json) as f:
            llm_data = json.load(f)

        from llmrouter.models.graphrouter.router import GraphRouter
        stub = types.SimpleNamespace(
            llm_data=llm_data, cfg={}, model_names=used_models, query_dim=EMB_DIM,
        )
        GraphRouter._prepare_llm_embeddings(stub)

        assert stub.llm_embedding.shape == (len(used_models), EMB_DIM)
        assert np.isfinite(stub.llm_embedding).all(), "llm_embedding contains NaN/Inf"

    def test_graphrouter_profile_is_deterministic(self, all_method_profile, tmp_path):
        """Same profile applied twice must give identical GraphRouter embeddings (not random)."""
        npz_path, used_models, _ = all_method_profile

        llm_data_path = str(tmp_path / "llm_data.json")
        _make_llm_data_json(used_models, llm_data_path)

        out_json = str(tmp_path / "llm_with_emb.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_path, out_json)

        with open(out_json) as f:
            llm_data = json.load(f)

        from llmrouter.models.graphrouter.router import GraphRouter
        stub1 = types.SimpleNamespace(llm_data=llm_data, cfg={}, model_names=used_models, query_dim=EMB_DIM)
        stub2 = types.SimpleNamespace(llm_data=llm_data, cfg={}, model_names=used_models, query_dim=EMB_DIM)
        GraphRouter._prepare_llm_embeddings(stub1)
        GraphRouter._prepare_llm_embeddings(stub2)

        np.testing.assert_allclose(
            stub1.llm_embedding, stub2.llm_embedding, rtol=1e-5,
            err_msg="GraphRouter embedding is non-deterministic; likely falling back to random init",
        )

    def test_personalizedrouter_consumes_profile(self, all_method_profile, tmp_path):
        """PersonalizedRouter._normalize_llm_embeddings must accept pkl from every method."""
        npz_path, used_models, raw = all_method_profile

        pkl_path = str(tmp_path / "embeddings.pkl")
        npz_to_pkl(npz_path, pkl_path)

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        from llmrouter.models.personalizedrouter.router import PersonalizedRouter
        result = PersonalizedRouter._normalize_llm_embeddings(None, pkl_data, used_models)

        assert result.shape == (len(used_models), EMB_DIM)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_personalizedrouter_pkl_values_match_profile(self, all_method_profile, tmp_path):
        """pkl round-trip must preserve embedding values for every profile method."""
        npz_path, used_models, raw = all_method_profile

        pkl_path = str(tmp_path / "embeddings.pkl")
        npz_to_pkl(npz_path, pkl_path)

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        from llmrouter.models.personalizedrouter.router import PersonalizedRouter
        result = PersonalizedRouter._normalize_llm_embeddings(None, pkl_data, used_models)

        for i, name in enumerate(used_models):
            np.testing.assert_allclose(
                result[i], raw[name], rtol=1e-5,
                err_msg=f"Value mismatch for '{name}' after pkl round-trip",
            )
