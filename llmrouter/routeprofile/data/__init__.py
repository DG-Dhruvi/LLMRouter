"""Utilities for locating bundled RouteProfile data files."""

import os

_DATA_DIR = os.path.dirname(__file__)


def get_bundled_path(filename: str, subdir: str = "profile_data") -> str:
    """Return the absolute path to a bundled data file.

    Args:
        filename: File name (e.g. 'model_feature_standard.json').
        subdir: Sub-directory under data/ (default: 'profile_data').

    Returns:
        Absolute path string. Raises FileNotFoundError if not found.
    """
    path = os.path.join(_DATA_DIR, subdir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Bundled data file not found: {path}. "
            f"Available files: {os.listdir(os.path.join(_DATA_DIR, subdir))}"
        )
    return path


def get_profile_data_dir() -> str:
    """Return the directory containing bundled profile_data JSON files."""
    return os.path.join(_DATA_DIR, "profile_data")
