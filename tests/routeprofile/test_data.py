"""Tests for bundled profile data integrity.

Verifies that bundled JSON files exist, are valid JSON, and contain the
required fields so that graph-building functions can use them as defaults.
"""
import json
import os

import pytest

from llmrouter.routeprofile.data import get_bundled_path, get_profile_data_dir


# ── Directory-level checks ─────────────────────────────────────────────────────

def test_profile_data_dir_exists():
    pd = get_profile_data_dir()
    assert os.path.isdir(pd), f"profile_data directory not found: {pd}"


@pytest.mark.parametrize("filename", [
    "model_feature_standard.json",
    "model_feature_newllm.json",
    "model_family_feature.json",
    "task_feature.json",
    "task_queries_standard.json",
    "domain_feature.json",
    "domain_task_map.json",
])
def test_bundled_file_exists(filename):
    path = get_bundled_path(filename)
    assert os.path.isfile(path), f"Bundled file missing: {filename}"


# ── JSON validity ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", [
    "model_feature_standard.json",
    "model_feature_newllm.json",
    "model_family_feature.json",
    "task_feature.json",
    "task_queries_standard.json",
    "domain_feature.json",
    "domain_task_map.json",
])
def test_bundled_file_is_valid_json(filename):
    path = get_bundled_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data, f"File is empty or null: {filename}"


# ── Schema checks ──────────────────────────────────────────────────────────────

def test_model_feature_standard_schema():
    """Each model entry must have the fields needed by graph builders."""
    required_fields = {"feature", "architecture", "detailed_scores"}
    path = get_bundled_path("model_feature_standard.json")
    with open(path) as f:
        data = json.load(f)

    assert len(data) >= 1, "model_feature_standard.json is empty"
    for model_name, meta in data.items():
        missing = required_fields - set(meta.keys())
        assert not missing, f"Model '{model_name}' missing fields: {missing}"
        assert isinstance(meta["detailed_scores"], dict), (
            f"'{model_name}': detailed_scores must be a dict"
        )
        assert len(meta["detailed_scores"]) >= 1, (
            f"'{model_name}': detailed_scores is empty"
        )


def test_model_family_feature_schema():
    """Architecture names must map to non-empty description strings."""
    path = get_bundled_path("model_family_feature.json")
    with open(path) as f:
        data = json.load(f)

    assert len(data) >= 1, "model_family_feature.json is empty"
    for arch_name, desc in data.items():
        assert isinstance(desc, str) and desc.strip(), (
            f"Architecture '{arch_name}' has empty description"
        )


def test_task_feature_schema():
    """Task names must map to non-empty description strings."""
    path = get_bundled_path("task_feature.json")
    with open(path) as f:
        data = json.load(f)

    assert len(data) >= 1, "task_feature.json is empty"
    for task_name, desc in data.items():
        assert isinstance(desc, str) and desc.strip(), (
            f"Task '{task_name}' has empty description"
        )


def test_domain_task_map_schema():
    """Domain map must map domain names to non-empty lists of task names."""
    path = get_bundled_path("domain_task_map.json")
    with open(path) as f:
        data = json.load(f)

    assert len(data) >= 1, "domain_task_map.json is empty"
    for domain, tasks in data.items():
        assert isinstance(tasks, list) and len(tasks) >= 1, (
            f"Domain '{domain}' has empty task list"
        )


def test_model_architectures_in_family_feature():
    """Every architecture referenced in model_feature_standard.json must appear
    in model_family_feature.json so graph builders can embed it."""
    with open(get_bundled_path("model_feature_standard.json")) as f:
        models = json.load(f)
    with open(get_bundled_path("model_family_feature.json")) as f:
        families = json.load(f)

    missing = {
        m["architecture"] for m in models.values()
        if "architecture" in m and m["architecture"] not in families
    }
    assert not missing, (
        f"Architectures in model_feature but missing from model_family_feature: {missing}"
    )


def test_task_queries_standard_schema():
    """task_queries_standard.json must be a dict of task → list of query strings."""
    path = get_bundled_path("task_queries_standard.json")
    with open(path) as f:
        data = json.load(f)

    assert len(data) >= 1
    for task_name, queries in data.items():
        assert isinstance(queries, list) and len(queries) >= 1, (
            f"Task '{task_name}' has no queries"
        )
        assert all(isinstance(q, str) for q in queries), (
            f"Task '{task_name}': all queries must be strings"
        )
