"""Tests for profile generation methods (flat, emb_gnn, index, trainable_gnn).

get_longformer_embedding is mocked so no GPU or network access is required.
All tests use the tiny_task_graph / graph_pt_path fixtures from conftest.py.
text_gnn_profile is skipped unless vllm is installed.
"""
import importlib
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

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

_flat_mod      = importlib.import_module("llmrouter.routeprofile.get_model_profile.training_free.flat_profile")
_emb_mod       = importlib.import_module("llmrouter.routeprofile.get_model_profile.training_free.emb_gnn_profile")
_index_mod     = importlib.import_module("llmrouter.routeprofile.get_model_profile.training_free.index_profile")
_trainable_mod = importlib.import_module("llmrouter.routeprofile.get_model_profile.trainable.trainable_gnn_profile")

MODEL_NAMES = ["model-a", "model-b", "model-c"]


# ── Shared assertions ──────────────────────────────────────────────────────────

def _assert_valid_npz(path: str, expected_models: list):
    """Check .npz: exists, contains expected keys, each value is float32 [768]."""
    assert os.path.isfile(path), f".npz file not created: {path}"
    data = np.load(path)
    for name in expected_models:
        assert name in data.files, f"Model '{name}' missing from .npz"
        emb = data[name]
        assert emb.dtype == np.float32, f"'{name}': expected float32, got {emb.dtype}"
        assert emb.shape == (768,), f"'{name}': expected shape (768,), got {emb.shape}"
        assert np.isfinite(emb).all(), f"'{name}': contains NaN or Inf"


# ── emb_gnn_profile ───────────────────────────────────────────────────────────

class TestEmbGnnProfile:
    def test_creates_npz(self, graph_pt_path, tmp_path):
        out = str(tmp_path / "emb_gnn.npz")
        _emb_mod.build_model_profile(graph=graph_pt_path, K=1, save=out, keep=MODEL_NAMES)
        _assert_valid_npz(out, MODEL_NAMES)

    def test_k2_hops(self, graph_pt_path, tmp_path):
        out = str(tmp_path / "emb_gnn_k2.npz")
        _emb_mod.build_model_profile(graph=graph_pt_path, K=2, save=out, keep=MODEL_NAMES)
        _assert_valid_npz(out, MODEL_NAMES)

    def test_k1_k2_different(self, graph_pt_path, tmp_path):
        """K=1 and K=2 propagation should produce different embeddings."""
        out1 = str(tmp_path / "k1.npz")
        out2 = str(tmp_path / "k2.npz")
        _emb_mod.build_model_profile(graph=graph_pt_path, K=1, save=out1, keep=MODEL_NAMES)
        _emb_mod.build_model_profile(graph=graph_pt_path, K=2, save=out2, keep=MODEL_NAMES)
        d1 = np.load(out1)["model-a"]
        d2 = np.load(out2)["model-a"]
        assert not np.allclose(d1, d2, atol=1e-4), "K=1 and K=2 produced identical embeddings"

    @pytest.mark.parametrize("norm", ["sym", "right", "left", "none"])
    def test_normalisation_modes(self, graph_pt_path, tmp_path, norm):
        out = str(tmp_path / f"emb_gnn_{norm}.npz")
        _emb_mod.build_model_profile(graph=graph_pt_path, K=1, norm=norm, save=out, keep=MODEL_NAMES)
        _assert_valid_npz(out, MODEL_NAMES)

    def test_l2_normalize_flag(self, graph_pt_path, tmp_path):
        """With normalize=True each embedding should have unit L2 norm."""
        out = str(tmp_path / "emb_gnn_norm.npz")
        _emb_mod.build_model_profile(graph=graph_pt_path, K=1, normalize=True,
                                     save=out, keep=MODEL_NAMES)
        data = np.load(out)
        for name in MODEL_NAMES:
            norm = np.linalg.norm(data[name])
            assert abs(norm - 1.0) < 1e-3, f"'{name}': L2 norm is {norm:.4f}, expected ~1.0"


# ── flat_profile ──────────────────────────────────────────────────────────────

class TestFlatProfile:
    def test_creates_npz(self, graph_pt_path, tmp_path):
        out = str(tmp_path / "flat.npz")
        with patch.object(_flat_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _flat_mod.build_model_profile(graph=graph_pt_path, save=out, keep=MODEL_NAMES)
        _assert_valid_npz(out, MODEL_NAMES)

    def test_different_seeds_produce_different_results(self, graph_pt_path, tmp_path):
        """Flat profile samples neighbours randomly; different seeds → different embeddings."""
        out1 = str(tmp_path / "flat_s1.npz")
        out2 = str(tmp_path / "flat_s2.npz")
        with patch.object(_flat_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _flat_mod.build_model_profile(graph=graph_pt_path, save=out1, seed=1, keep=MODEL_NAMES)
            _flat_mod.build_model_profile(graph=graph_pt_path, save=out2, seed=999, keep=MODEL_NAMES)
        d1 = np.load(out1)["model-a"]
        d2 = np.load(out2)["model-a"]
        assert not np.allclose(d1, d2), "Different seeds produced identical flat profiles"

    def test_same_seed_is_reproducible(self, graph_pt_path, tmp_path):
        out1 = str(tmp_path / "flat_r1.npz")
        out2 = str(tmp_path / "flat_r2.npz")
        with patch.object(_flat_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _flat_mod.build_model_profile(graph=graph_pt_path, save=out1, seed=42, keep=MODEL_NAMES)
            _flat_mod.build_model_profile(graph=graph_pt_path, save=out2, seed=42, keep=MODEL_NAMES)
        d1 = np.load(out1)["model-a"]
        d2 = np.load(out2)["model-a"]
        np.testing.assert_allclose(d1, d2, rtol=1e-5)


# ── index_profile ─────────────────────────────────────────────────────────────

class TestIndexProfile:
    def test_creates_npz(self, tmp_path):
        out = str(tmp_path / "index.npz")
        _index_mod.build_model_profile(save=out, seed=0)
        # index profile uses a fixed model list; just check the file exists and is valid
        assert os.path.isfile(out)
        data = np.load(out)
        for name in data.files:
            emb = data[name]
            assert emb.shape == (768,)
            assert emb.dtype == np.float32
            assert np.isfinite(emb).all()

    def test_reproducible_with_seed(self, tmp_path):
        out1 = str(tmp_path / "idx1.npz")
        out2 = str(tmp_path / "idx2.npz")
        _index_mod.build_model_profile(save=out1, seed=7)
        _index_mod.build_model_profile(save=out2, seed=7)
        d1 = np.load(out1)
        d2 = np.load(out2)
        for name in d1.files:
            np.testing.assert_allclose(d1[name], d2[name], rtol=1e-5)

    def test_different_seeds_differ(self, tmp_path):
        out1 = str(tmp_path / "idx_s1.npz")
        out2 = str(tmp_path / "idx_s2.npz")
        _index_mod.build_model_profile(save=out1, seed=1)
        _index_mod.build_model_profile(save=out2, seed=2)
        d1 = np.load(out1)
        d2 = np.load(out2)
        name = d1.files[0]
        assert not np.allclose(d1[name], d2[name])


# ── trainable_gnn_profile ─────────────────────────────────────────────────────

class TestTrainableGnnProfile:
    # trainable_gnn uses HANConv on pre-computed graph node features (.x tensors),
    # so no get_longformer_embedding call occurs — no mocking needed.

    def test_creates_npz(self, graph_pt_path, tmp_path):
        out_emb  = str(tmp_path / "trainable.npz")
        out_ckpt = str(tmp_path / "trainable_ckpt.pt")
        _trainable_mod.build_model_profile(
            graph=graph_pt_path,
            save_emb=out_emb,
            save_ckpt=out_ckpt,
            epochs=2,       # minimal epochs for a fast smoke test
            seed=42,
            keep=MODEL_NAMES,
        )
        _assert_valid_npz(out_emb, MODEL_NAMES)

    def test_checkpoint_saved(self, graph_pt_path, tmp_path):
        out_emb  = str(tmp_path / "t.npz")
        out_ckpt = str(tmp_path / "t_ckpt.pt")
        _trainable_mod.build_model_profile(
            graph=graph_pt_path,
            save_emb=out_emb,
            save_ckpt=out_ckpt,
            epochs=2,
            seed=42,
            keep=MODEL_NAMES,
        )
        assert os.path.isfile(out_ckpt), "Checkpoint file not saved"

    def test_different_epochs_differ(self, graph_pt_path, tmp_path):
        """More training epochs should change the output embeddings."""
        out1 = str(tmp_path / "t1.npz")
        out2 = str(tmp_path / "t2.npz")
        _trainable_mod.build_model_profile(
            graph=graph_pt_path, save_emb=out1, epochs=1, seed=0, keep=MODEL_NAMES)
        _trainable_mod.build_model_profile(
            graph=graph_pt_path, save_emb=out2, epochs=5, seed=0, keep=MODEL_NAMES)
        d1 = np.load(out1)["model-a"]
        d2 = np.load(out2)["model-a"]
        assert not np.allclose(d1, d2, atol=1e-4), \
            "1 epoch and 5 epochs produced identical trainable_gnn embeddings"


# ── text_gnn_profile ──────────────────────────────────────────────────────────
# K=0 path skips vLLM entirely (Longformer-encodes raw node_feature_text),
# so those tests run regardless of whether vllm is installed.

_text_gnn_mod = importlib.import_module(
    "llmrouter.routeprofile.get_model_profile.training_free.text_gnn_profile"
)


class TestTextGnnProfileK0:
    """Tests for the K=0 (no-LLM) path — no vllm required."""

    def test_creates_npz(self, graph_pt_path, tmp_path):
        out = str(tmp_path / "text_gnn_k0.npz")
        with patch.object(_text_gnn_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _text_gnn_mod.build_model_profile(graph=graph_pt_path, K=0, emb_save=out, keep=MODEL_NAMES)
        _assert_valid_npz(out, MODEL_NAMES)

    def test_text_save_written(self, graph_pt_path, tmp_path):
        """text_save must produce a JSON with final_texts mapping model_name → str."""
        import json as _json
        out_emb  = str(tmp_path / "text_gnn_k0.npz")
        out_text = str(tmp_path / "text_gnn_k0.json")
        with patch.object(_text_gnn_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _text_gnn_mod.build_model_profile(
                graph=graph_pt_path, K=0,
                emb_save=out_emb, text_save=out_text, keep=MODEL_NAMES,
            )
        assert os.path.isfile(out_text)
        saved = _json.load(open(out_text))
        assert "final_texts" in saved, "text_save JSON missing 'final_texts' key"
        final = saved["final_texts"]
        for name in MODEL_NAMES:
            assert name in final, f"'{name}' missing from final_texts"
            assert isinstance(final[name], str) and final[name].strip()

    def test_k0_uses_node_feature_text(self, graph_pt_path, tmp_path):
        """K=0 embeddings must be determined by node_feature_text, not random."""
        out1 = str(tmp_path / "k0_a.npz")
        out2 = str(tmp_path / "k0_b.npz")
        with patch.object(_text_gnn_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _text_gnn_mod.build_model_profile(graph=graph_pt_path, K=0, emb_save=out1, keep=MODEL_NAMES)
            _text_gnn_mod.build_model_profile(graph=graph_pt_path, K=0, emb_save=out2, keep=MODEL_NAMES)
        d1 = np.load(out1)
        d2 = np.load(out2)
        for name in MODEL_NAMES:
            np.testing.assert_allclose(d1[name], d2[name], rtol=1e-5,
                                       err_msg=f"K=0 is non-deterministic for '{name}'")

    def test_different_node_texts_differ(self, tmp_path):
        """Graphs with different node_feature_text must produce different K=0 embeddings."""
        import torch
        from torch_geometric.data import HeteroData

        def _make_graph(suffix):
            g = HeteroData()
            g["model"].x = torch.randn(3, 768)
            g["model"].node_names = MODEL_NAMES
            g["model"].node_feature_text = [f"Model {n} {suffix}" for n in MODEL_NAMES]
            return g

        p1 = str(tmp_path / "g1.pt")
        p2 = str(tmp_path / "g2.pt")
        torch.save(_make_graph("alpha"), p1)
        torch.save(_make_graph("beta"),  p2)

        out1 = str(tmp_path / "e1.npz")
        out2 = str(tmp_path / "e2.npz")
        with patch.object(_text_gnn_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _text_gnn_mod.build_model_profile(graph=p1, K=0, emb_save=out1, keep=MODEL_NAMES)
            _text_gnn_mod.build_model_profile(graph=p2, K=0, emb_save=out2, keep=MODEL_NAMES)
        d1 = np.load(out1)["model-a"]
        d2 = np.load(out2)["model-a"]
        assert not np.allclose(d1, d2), "Different node_feature_text produced identical K=0 embeddings"


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="text_gnn K>0 requires vllm",
)
class TestTextGnnProfileKN:
    """Tests for K>0 (LLM aggregation) path.

    Requires a GPU. Run with:
        CUDA_VISIBLE_DEVICES=<id> pytest tests/routeprofile/test_build_profile.py::TestTextGnnProfileKN -v
    """

    @pytest.fixture(autouse=True)
    def _force_vllm_spawn(self):
        """Force vLLM to use spawn instead of fork to avoid CUDA re-init errors."""
        old = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        yield
        if old is None:
            os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        else:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = old

    def test_k2_creates_npz(self, graph_pt_path, tmp_path):
        """K=2 LLM aggregation must produce a valid float32 [768] .npz."""
        out = str(tmp_path / "text_gnn_k2.npz")
        _text_gnn_mod.build_model_profile(
            graph=graph_pt_path, K=2,
            model="Qwen/Qwen2.5-7B-Instruct",
            emb_save=out, keep=MODEL_NAMES,
        )
        _assert_valid_npz(out, MODEL_NAMES)

    def test_k2_differs_from_k0(self, graph_pt_path, tmp_path):
        """K=2 LLM aggregation must change embeddings relative to K=0 baseline."""
        out0 = str(tmp_path / "k0.npz")
        out2 = str(tmp_path / "k2.npz")
        with patch.object(_text_gnn_mod, "get_longformer_embedding", side_effect=mock_longformer):
            _text_gnn_mod.build_model_profile(graph=graph_pt_path, K=0, emb_save=out0, keep=MODEL_NAMES)
        _text_gnn_mod.build_model_profile(
            graph=graph_pt_path, K=2,
            model="Qwen/Qwen2.5-7B-Instruct",
            emb_save=out2, keep=MODEL_NAMES,
        )
        d0 = np.load(out0)["model-a"]
        d2 = np.load(out2)["model-a"]
        assert not np.allclose(d0, d2, atol=1e-4), "K=2 LLM aggregation produced same embeddings as K=0"

    def test_text_save_written_k2(self, graph_pt_path, tmp_path):
        """text_save must contain final_texts and hop_texts for K=2."""
        import json as _json
        out_emb  = str(tmp_path / "k2.npz")
        out_text = str(tmp_path / "k2_texts.json")
        _text_gnn_mod.build_model_profile(
            graph=graph_pt_path, K=2,
            model="Qwen/Qwen2.5-7B-Instruct",
            emb_save=out_emb, text_save=out_text, keep=MODEL_NAMES,
        )
        assert os.path.isfile(out_text)
        saved = _json.load(open(out_text))
        assert "final_texts" in saved
        assert "hop_texts" in saved
        # K=2 → hop_texts should have entries for hop 1 and hop 2
        hop_keys = {int(k) for k in saved["hop_texts"].keys()}
        assert {1, 2}.issubset(hop_keys), f"Expected hops 1,2 in hop_texts, got {hop_keys}"
        for name in MODEL_NAMES:
            assert name in saved["final_texts"]
            assert isinstance(saved["final_texts"][name], str) and saved["final_texts"][name].strip()
