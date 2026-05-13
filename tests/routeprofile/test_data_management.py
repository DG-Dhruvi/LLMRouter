"""Tests for llmrouter.routeprofile.data_management."""
import json
import os
import warnings

import pytest

from llmrouter.routeprofile.data_management import (
    add_domain,
    add_model,
    add_query,
    add_task,
    init_profile_data_dir,
)
from llmrouter.routeprofile.data import get_profile_data_dir

_BUNDLED_FILES = [
    "model_feature_standard.json",
    "model_feature_newllm.json",
    "model_family_feature.json",
    "task_feature.json",
    "domain_feature.json",
    "domain_task_map.json",
    "task_queries_standard.json",
    "task_queries_newllm.json",
]


# ── init_profile_data_dir ─────────────────────────────────────────────────────

class TestInitProfileDataDir:
    def test_copies_all_bundled_files(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        for fname in _BUNDLED_FILES:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_skips_existing_files(self, tmp_path):
        sentinel = "SENTINEL"
        target = tmp_path / "model_feature_standard.json"
        target.write_text(sentinel)
        init_profile_data_dir(str(tmp_path))
        assert target.read_text() == sentinel, "Existing file was overwritten"

    def test_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "new_subdir"
        assert not new_dir.exists()
        init_profile_data_dir(str(new_dir))
        assert new_dir.exists()


# ── add_domain ────────────────────────────────────────────────────────────────

class TestAddDomain:
    def test_add_new_domain(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_domain("multimodal", "Tasks requiring text + image understanding.", str(tmp_path))

        feat = json.loads((tmp_path / "domain_feature.json").read_text())
        assert "multimodal" in feat
        assert feat["multimodal"] == "Tasks requiring text + image understanding."

        dmap = json.loads((tmp_path / "domain_task_map.json").read_text())
        assert "multimodal" in dmap
        assert dmap["multimodal"] == []

    def test_add_domain_does_not_overwrite_task_map_if_exists(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_domain("reasoning", "New description.", str(tmp_path))
        # existing domain already has tasks — should NOT be reset to []
        dmap = json.loads((tmp_path / "domain_task_map.json").read_text())
        assert len(dmap["reasoning"]) > 0, "Existing domain task list was reset"

    def test_modify_existing_domain(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        original = json.loads((tmp_path / "domain_feature.json").read_text())["knowledge"]
        add_domain("knowledge", "Updated knowledge description.", str(tmp_path))
        feat = json.loads((tmp_path / "domain_feature.json").read_text())
        assert feat["knowledge"] == "Updated knowledge description."
        assert feat["knowledge"] != original


# ── add_task ──────────────────────────────────────────────────────────────────

class TestAddTask:
    def test_add_new_task(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_task("my-bench", "A new benchmark.", str(tmp_path))
        tasks = json.loads((tmp_path / "task_feature.json").read_text())
        assert "my-bench" in tasks
        assert tasks["my-bench"] == "A new benchmark."

    def test_add_task_with_existing_domain(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_task("my-bench", "A bench.", str(tmp_path), domains=["reasoning"])
        dmap = json.loads((tmp_path / "domain_task_map.json").read_text())
        assert "my-bench" in dmap["reasoning"]

    def test_add_task_with_new_domain_that_exists(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_domain("newdomain", "A brand new domain.", str(tmp_path))
        add_task("my-bench", "A bench.", str(tmp_path), domains=["newdomain"])
        dmap = json.loads((tmp_path / "domain_task_map.json").read_text())
        assert "my-bench" in dmap["newdomain"]

    def test_add_task_unknown_domain_errors(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            add_task("my-bench", "A bench.", str(tmp_path), domains=["nonexistent-domain"])
        # task must NOT have been written
        tasks = json.loads((tmp_path / "task_feature.json").read_text())
        assert "my-bench" not in tasks

    def test_add_task_modify_updates_feature(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_task("ifeval", "Updated description.", str(tmp_path))
        tasks = json.loads((tmp_path / "task_feature.json").read_text())
        assert tasks["ifeval"] == "Updated description."

    def test_add_task_idempotent_domain_append(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_task("my-bench", "A bench.", str(tmp_path), domains=["reasoning"])
        add_task("my-bench", "A bench.", str(tmp_path), domains=["reasoning"])
        dmap = json.loads((tmp_path / "domain_task_map.json").read_text())
        assert dmap["reasoning"].count("my-bench") == 1, "Task appended twice"


# ── add_model ─────────────────────────────────────────────────────────────────

class TestAddModel:
    def test_add_new_model(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_model(
            "test-llama",
            str(tmp_path),
            feature="A test LLaMA model.",
            architecture="LlamaForCausalLM",
            detailed_scores={"ifeval": 72.5, "bbh": 48.3},
        )
        for fname in ("model_feature_standard.json", "model_feature_newllm.json"):
            d = json.loads((tmp_path / fname).read_text())
            assert "test-llama" in d
            assert d["test-llama"]["feature"] == "A test LLaMA model."
            assert d["test-llama"]["detailed_scores"]["ifeval"] == 72.5

    def test_add_model_requires_feature_for_new(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        with pytest.raises(ValueError, match="feature"):
            add_model("new-model", str(tmp_path), architecture="LlamaForCausalLM")

    def test_add_model_requires_architecture_for_new(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        with pytest.raises(ValueError, match="architecture"):
            add_model("new-model", str(tmp_path), feature="A model.")

    def test_modify_merges_fields_only(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        # Seed with a real model entry
        existing_name = list(
            json.loads((tmp_path / "model_feature_standard.json").read_text()).keys()
        )[0]
        add_model(existing_name, str(tmp_path), detailed_scores={"my-new-bench": 99.0})
        d = json.loads((tmp_path / "model_feature_standard.json").read_text())
        # New score added
        assert d[existing_name]["detailed_scores"]["my-new-bench"] == 99.0
        # Existing feature text preserved
        assert "feature" in d[existing_name]
        assert d[existing_name]["feature"]  # non-empty

    def test_replace_overwrites_entry(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        existing_name = list(
            json.loads((tmp_path / "model_feature_standard.json").read_text()).keys()
        )[0]
        add_model(
            existing_name, str(tmp_path),
            feature="Replaced feature.",
            architecture="LlamaForCausalLM",
            replace=True,
        )
        d = json.loads((tmp_path / "model_feature_standard.json").read_text())
        assert d[existing_name]["feature"] == "Replaced feature."
        # detailed_scores should be absent (not carried over from old entry)
        assert "detailed_scores" not in d[existing_name]

    def test_no_scores_warns(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            add_model("no-score-model", str(tmp_path),
                      feature="A model.", architecture="LlamaForCausalLM")
        msgs = [str(w.message) for w in caught]
        assert any("no model↔dataset edges" in m or "No benchmark scores" in m for m in msgs)

    def test_unknown_benchmark_warns(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            add_model("warn-model", str(tmp_path),
                      feature="A model.", architecture="LlamaForCausalLM",
                      detailed_scores={"nonexistent-bench": 50.0})
        msgs = [str(w.message) for w in caught]
        assert any("nonexistent-bench" in m for m in msgs)

    def test_new_arch_requires_arch_feature(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        with pytest.raises(ValueError, match="arch_feature"):
            add_model("arch-model", str(tmp_path),
                      feature="A model.", architecture="BrandNewArch")

    def test_new_arch_with_feature_updates_family_file(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_model("arch-model", str(tmp_path),
                  feature="A model.", architecture="BrandNewArch",
                  arch_feature="A brand new architecture.")
        arch_data = json.loads((tmp_path / "model_family_feature.json").read_text())
        assert "BrandNewArch" in arch_data
        assert arch_data["BrandNewArch"] == "A brand new architecture."

    def test_add_model_with_new_tasks_via_api(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_domain("mydom", "My domain.", str(tmp_path))
        add_model(
            "task-model", str(tmp_path),
            feature="A model.", architecture="LlamaForCausalLM",
            detailed_scores={"my-new-bench": 77.0},
            new_tasks=[{"name": "my-new-bench", "feature": "A new bench.", "domain": "mydom"}],
        )
        tasks = json.loads((tmp_path / "task_feature.json").read_text())
        assert "my-new-bench" in tasks
        d = json.loads((tmp_path / "model_feature_standard.json").read_text())
        assert d["task-model"]["detailed_scores"]["my-new-bench"] == 77.0

    def test_add_model_mode_standard_only(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_model("std-only-model", str(tmp_path),
                  feature="A model.", architecture="LlamaForCausalLM",
                  mode="standard")
        std = json.loads((tmp_path / "model_feature_standard.json").read_text())
        newllm = json.loads((tmp_path / "model_feature_newllm.json").read_text())
        assert "std-only-model" in std
        assert "std-only-model" not in newllm

    def test_add_model_mode_newllm_only(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_model("newllm-only-model", str(tmp_path),
                  feature="A model.", architecture="LlamaForCausalLM",
                  mode="newllm")
        std = json.loads((tmp_path / "model_feature_standard.json").read_text())
        newllm = json.loads((tmp_path / "model_feature_newllm.json").read_text())
        assert "newllm-only-model" not in std
        assert "newllm-only-model" in newllm


# ── add_task with queries ─────────────────────────────────────────────────────

class TestAddTaskWithQueries:
    def test_add_task_with_queries(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_task("my-bench", "A bench.", str(tmp_path),
                 queries=["Q1", "Q2"], queries_mode="standard")
        tasks = json.loads((tmp_path / "task_feature.json").read_text())
        assert "my-bench" in tasks
        qdata = json.loads((tmp_path / "task_queries_standard.json").read_text())
        assert "Q1" in qdata["my-bench"]
        assert "Q2" in qdata["my-bench"]
        # newllm file should be untouched for this task
        newllm = json.loads((tmp_path / "task_queries_newllm.json").read_text())
        assert "my-bench" not in newllm

    def test_add_task_queries_without_mode_errors(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        with pytest.raises(ValueError, match="queries_mode"):
            add_task("my-bench", "A bench.", str(tmp_path), queries=["Q1"])


# ── add_query ─────────────────────────────────────────────────────────────────

class TestAddQuery:
    def test_add_query_mode_both(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_query("ifeval", ["Q1", "Q2"], str(tmp_path), mode="both")
        for fname in ("task_queries_standard.json", "task_queries_newllm.json"):
            qdata = json.loads((tmp_path / fname).read_text())
            assert "Q1" in qdata["ifeval"]
            assert "Q2" in qdata["ifeval"]

    def test_add_query_mode_standard_only(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        # Get original newllm ifeval queries to compare
        orig_newllm = json.loads((tmp_path / "task_queries_newllm.json").read_text())
        orig_ifeval = list(orig_newllm.get("ifeval", []))
        add_query("ifeval", ["UNIQUE_STANDARD_Q"], str(tmp_path), mode="standard")
        std = json.loads((tmp_path / "task_queries_standard.json").read_text())
        newllm = json.loads((tmp_path / "task_queries_newllm.json").read_text())
        assert "UNIQUE_STANDARD_Q" in std["ifeval"]
        assert "UNIQUE_STANDARD_Q" not in newllm.get("ifeval", [])

    def test_add_query_mode_newllm_only(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_query("ifeval", ["UNIQUE_NEWLLM_Q"], str(tmp_path), mode="newllm")
        std = json.loads((tmp_path / "task_queries_standard.json").read_text())
        newllm = json.loads((tmp_path / "task_queries_newllm.json").read_text())
        assert "UNIQUE_NEWLLM_Q" not in std.get("ifeval", [])
        assert "UNIQUE_NEWLLM_Q" in newllm["ifeval"]

    def test_add_query_unknown_task_errors(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        orig_std = (tmp_path / "task_queries_standard.json").read_text()
        with pytest.raises(ValueError, match="not found"):
            add_query("nonexistent-bench", ["Q1"], str(tmp_path), mode="standard")
        # File must be unmodified
        assert (tmp_path / "task_queries_standard.json").read_text() == orig_std

    def test_add_query_deduplication(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        add_query("ifeval", ["DEDUP_Q"], str(tmp_path), mode="standard")
        add_query("ifeval", ["DEDUP_Q"], str(tmp_path), mode="standard")
        qdata = json.loads((tmp_path / "task_queries_standard.json").read_text())
        assert qdata["ifeval"].count("DEDUP_Q") == 1

    def test_add_query_task_key_not_in_queries_file(self, tmp_path):
        init_profile_data_dir(str(tmp_path))
        # Add a task that exists in task_feature but not in any queries file
        add_task("brand-new-bench", "A new bench.", str(tmp_path))
        add_query("brand-new-bench", ["First Q"], str(tmp_path), mode="standard")
        qdata = json.loads((tmp_path / "task_queries_standard.json").read_text())
        assert "brand-new-bench" in qdata
        assert qdata["brand-new-bench"] == ["First Q"]
