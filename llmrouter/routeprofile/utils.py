"""
Format conversion utilities: RouteProfile .npz output → LLMRouter-compatible formats.

RouteProfile generates per-model embeddings as .npz files (model_name → np.ndarray[768]).
LLMRouter routers consume model embeddings in two formats:
  - JSON (llm_data with "embedding" field): used by GraphRouter, MFRouter, etc.
  - Pickle (.pkl): used by PersonalizedRouter

These utilities bridge the two without modifying any existing router code.
"""

import json
import pickle
from typing import Optional

import numpy as np


def npz_to_llm_embeddings_json(
    npz_path: str,
    llm_data_path: str,
    output_path: str,
    missing: str = "warn",
) -> None:
    """Merge RouteProfile .npz embeddings into a LLMRouter llm_data JSON file.

    Reads the existing llm_data JSON (which contains model metadata like API
    endpoints, prices, feature text) and overwrites the "embedding" field for
    each model found in the .npz file. Models present in llm_data but absent
    from the .npz keep their existing embedding (or have none).

    Args:
        npz_path: Path to RouteProfile-generated .npz file.
        llm_data_path: Path to LLMRouter llm_data JSON (source of model metadata).
        output_path: Destination path for the merged JSON. May equal llm_data_path
            to update in-place.
        missing: What to do when a model in llm_data has no entry in the .npz.
            "warn"  – print a warning and leave the model's existing embedding.
            "skip"  – silently leave unchanged.
            "error" – raise KeyError.
    """
    profiles = np.load(npz_path, allow_pickle=False)
    with open(llm_data_path, "r", encoding="utf-8") as f:
        llm_data = json.load(f)

    updated = 0
    for model_name, meta in llm_data.items():
        if model_name in profiles:
            meta["embedding"] = profiles[model_name].tolist()
            updated += 1
        elif missing == "error":
            raise KeyError(
                f"Model '{model_name}' is in llm_data but not in the .npz profile. "
                f"Available models in .npz: {list(profiles.keys())}"
            )
        elif missing == "warn":
            print(
                f"[routeprofile] Warning: '{model_name}' not found in profile .npz; "
                "keeping existing embedding."
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(llm_data, f, indent=2, ensure_ascii=False)

    print(f"[routeprofile] Updated {updated}/{len(llm_data)} model embeddings → {output_path}")


def npz_to_pkl(
    npz_path: str,
    output_path: str,
    model_names: Optional[list] = None,
) -> None:
    """Convert RouteProfile .npz embeddings to a .pkl file for PersonalizedRouter.

    PersonalizedRouter loads llm_embedding_data from a pickle file containing
    a dict mapping model names to embedding arrays.

    Args:
        npz_path: Path to RouteProfile-generated .npz file.
        output_path: Destination .pkl path.
        model_names: Optional list of model names to include (preserves order).
            If None, all models in the .npz are included.
    """
    profiles = np.load(npz_path, allow_pickle=False)

    if model_names is None:
        model_names = list(profiles.keys())

    result = {name: profiles[name] for name in model_names if name in profiles}

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[routeprofile] Saved {len(result)} model embeddings to {output_path}")
