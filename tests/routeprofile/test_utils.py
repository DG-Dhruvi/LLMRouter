"""Unit tests for llmrouter.routeprofile.utils format-conversion functions.

Tests npz_to_llm_embeddings_json and npz_to_pkl without requiring torch or
torch_geometric.
"""
import json
import os
import pickle

import numpy as np
import pytest

from llmrouter.routeprofile.utils import npz_to_llm_embeddings_json, npz_to_pkl


# ── npz_to_llm_embeddings_json ─────────────────────────────────────────────────

class TestNpzToLlmEmbeddingsJson:
    def test_overwrites_embedding_field(self, npz_profile_path, llm_data_json_path, tmp_path):
        out = str(tmp_path / "result.json")
        npz_to_llm_embeddings_json(npz_profile_path, llm_data_json_path, out)

        with open(out) as f:
            result = json.load(f)

        profiles = np.load(npz_profile_path)
        for model_name in profiles.files:
            assert model_name in result, f"Model '{model_name}' missing from output"
            assert "embedding" in result[model_name], f"No embedding field for '{model_name}'"
            saved_emb = np.array(result[model_name]["embedding"], dtype="float32")
            np.testing.assert_allclose(saved_emb, profiles[model_name], rtol=1e-5,
                                       err_msg=f"Embedding mismatch for '{model_name}'")

    def test_preserves_non_embedding_metadata(self, npz_profile_path, llm_data_json_path, tmp_path):
        """Existing metadata fields (api_endpoint, size, etc.) must not be lost."""
        out = str(tmp_path / "result.json")
        npz_to_llm_embeddings_json(npz_profile_path, llm_data_json_path, out)

        with open(llm_data_json_path) as f:
            original = json.load(f)
        with open(out) as f:
            result = json.load(f)

        for model_name, meta in original.items():
            for field in ("size", "feature", "api_endpoint", "service"):
                assert result[model_name].get(field) == meta.get(field), (
                    f"Field '{field}' changed for model '{model_name}'"
                )

    def test_in_place_update(self, npz_profile_path, llm_data_json_path):
        """output_path == llm_data_path should update the file in place."""
        npz_to_llm_embeddings_json(npz_profile_path, llm_data_json_path, llm_data_json_path)

        with open(llm_data_json_path) as f:
            result = json.load(f)

        profiles = np.load(npz_profile_path)
        for model_name in profiles.files:
            assert "embedding" in result[model_name]

    def test_missing_model_warn(self, tmp_path, llm_data_json_path):
        """When .npz covers only a subset of models, missing='warn' should not raise."""
        # .npz with only model-a
        rng = np.random.default_rng(1)
        npz_path = str(tmp_path / "partial.npz")
        np.savez(npz_path, **{"model-a": rng.standard_normal(768).astype("float32")})

        out = str(tmp_path / "result.json")
        npz_to_llm_embeddings_json(npz_path, llm_data_json_path, out, missing="warn")

        with open(out) as f:
            result = json.load(f)
        # model-a should have the new embedding
        assert "embedding" in result["model-a"]
        # model-b and model-c should retain their original state (no embedding key in fixture data)
        assert "embedding" not in result["model-b"]

    def test_missing_model_error(self, tmp_path, llm_data_json_path):
        """missing='error' should raise KeyError when a model in llm_data is absent from .npz."""
        rng = np.random.default_rng(2)
        npz_path = str(tmp_path / "partial.npz")
        np.savez(npz_path, **{"model-a": rng.standard_normal(768).astype("float32")})

        out = str(tmp_path / "result.json")
        with pytest.raises(KeyError, match="model-b"):
            npz_to_llm_embeddings_json(npz_path, llm_data_json_path, out, missing="error")

    def test_output_json_is_valid(self, npz_profile_path, llm_data_json_path, tmp_path):
        """Output file must be parseable as JSON."""
        out = str(tmp_path / "result.json")
        npz_to_llm_embeddings_json(npz_profile_path, llm_data_json_path, out)
        with open(out) as f:
            json.load(f)  # should not raise


# ── npz_to_pkl ─────────────────────────────────────────────────────────────────

class TestNpzToPkl:
    def test_produces_pkl_with_correct_embeddings(self, npz_profile_path, tmp_path):
        out = str(tmp_path / "result.pkl")
        npz_to_pkl(npz_profile_path, out)

        assert os.path.isfile(out)
        with open(out, "rb") as f:
            data = pickle.load(f)

        profiles = np.load(npz_profile_path)
        for model_name in profiles.files:
            assert model_name in data, f"Model '{model_name}' missing from pkl"
            np.testing.assert_allclose(data[model_name], profiles[model_name], rtol=1e-5)

    def test_model_names_filter(self, npz_profile_path, tmp_path):
        """model_names kwarg should restrict which models are saved."""
        out = str(tmp_path / "result.pkl")
        npz_to_pkl(npz_profile_path, out, model_names=["model-a", "model-c"])

        with open(out, "rb") as f:
            data = pickle.load(f)

        assert set(data.keys()) == {"model-a", "model-c"}
        assert "model-b" not in data

    def test_pkl_values_are_numpy_arrays(self, npz_profile_path, tmp_path):
        """PersonalizedRouter expects numpy arrays, not plain lists."""
        out = str(tmp_path / "result.pkl")
        npz_to_pkl(npz_profile_path, out)

        with open(out, "rb") as f:
            data = pickle.load(f)

        for model_name, emb in data.items():
            assert isinstance(emb, np.ndarray), (
                f"Embedding for '{model_name}' is {type(emb)}, expected np.ndarray"
            )
            assert emb.shape == (768,), f"Expected shape (768,), got {emb.shape}"
