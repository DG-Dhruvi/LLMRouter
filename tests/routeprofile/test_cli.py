"""Tests for the 'llmrouter profile' CLI subcommand.

Tests argument parsing, help output, and end-to-end subcommand dispatch
using subprocess so the installed entry point is exercised.
"""
import argparse
import json
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest


PYTHON = sys.executable
LLMROUTER_MODULE = "llmrouter.cli.router_main"


# ── Parser registration tests ─────────────────────────────────────────────────

class TestParserRegistration:
    def test_profile_subcommand_exists(self):
        """'profile' must appear as a known subcommand in the main parser."""
        from llmrouter.cli.router_main import create_parser
        parser = create_parser()
        # argparse stores subparsers choices in _subparsers/_group_actions
        subparser_choices = {}
        for action in parser._subparsers._group_actions:
            if hasattr(action, "_name_parser_map"):
                subparser_choices.update(action._name_parser_map)
        assert "profile" in subparser_choices, (
            "'profile' not registered in main CLI subcommands"
        )

    def test_build_graph_parser(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        # Should not raise and must parse known args
        args = p.parse_args(["profile", "build-graph", "--graph-type", "task", "--mode", "standard"])
        assert args.graph_type == "task"
        assert args.mode == "standard"

    def test_build_profile_parser(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        args = p.parse_args([
            "profile", "build-profile",
            "--method", "emb_gnn",
            "--graph", "/tmp/g.pt",
            "--output", "/tmp/out.npz",
            "--K", "3",
        ])
        assert args.method == "emb_gnn"
        assert args.K == 3

    def test_apply_parser(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        args = p.parse_args([
            "profile", "apply",
            "--profile", "/tmp/p.npz",
            "--llm-data", "/tmp/llm.json",
            "--format", "pkl",
        ])
        assert args.format == "pkl"
        assert args.missing == "warn"  # default

    def test_invalid_method_rejected(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        with pytest.raises(SystemExit):
            p.parse_args(["profile", "build-profile", "--method", "nonexistent_method",
                          "--graph", "/x.pt", "--output", "/y.npz"])

    def test_invalid_norm_rejected(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        with pytest.raises(SystemExit):
            p.parse_args(["profile", "build-profile", "--method", "emb_gnn",
                          "--graph", "/x.pt", "--output", "/y.npz", "--norm", "bad_norm"])

    def test_add_domain_parser(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        args = p.parse_args([
            "profile", "add-domain",
            "--name", "multimodal",
            "--feature", "Multimodal tasks.",
            "--output-dir", "/tmp/rp",
        ])
        assert args.name == "multimodal"
        assert args.output_dir == "/tmp/rp"

    def test_add_task_parser(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        args = p.parse_args([
            "profile", "add-task",
            "--name", "my-bench",
            "--feature", "A benchmark.",
            "--domain", "reasoning",
            "--output-dir", "/tmp/rp",
        ])
        assert args.name == "my-bench"
        assert args.domain == ["reasoning"]

    def test_add_model_parser(self):
        from llmrouter.cli.router_profile import add_profile_parser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_profile_parser(sub)
        args = p.parse_args([
            "profile", "add-model",
            "--name", "my-llm",
            "--feature", "A model.",
            "--architecture", "LlamaForCausalLM",
            "--scores", "ifeval:72.5,bbh:48.3",
            "--output-dir", "/tmp/rp",
        ])
        assert args.name == "my-llm"
        assert args.scores == "ifeval:72.5,bbh:48.3"
        assert not args.replace


# ── Help output tests ─────────────────────────────────────────────────────────

class TestHelpOutput:
    def _run_help(self, *args):
        result = subprocess.run(
            [PYTHON, "-m", LLMROUTER_MODULE, *args, "--help"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )},
        )
        return result.stdout + result.stderr

    def test_profile_help_shows_subcommands(self):
        out = self._run_help("profile")
        assert "build-graph"   in out
        assert "build-profile" in out
        assert "apply"         in out

    def test_build_graph_help(self):
        out = self._run_help("profile", "build-graph")
        assert "--graph-type" in out
        assert "--mode"       in out

    def test_build_profile_help(self):
        out = self._run_help("profile", "build-profile")
        assert "--method"  in out
        assert "--graph"   in out
        assert "--output"  in out

    def test_apply_help(self):
        out = self._run_help("profile", "apply")
        assert "--profile"  in out
        assert "--llm-data" in out
        assert "--format"   in out

    def test_add_domain_help(self):
        out = self._run_help("profile", "add-domain")
        assert "--name"       in out
        assert "--feature"    in out
        assert "--output-dir" in out

    def test_add_task_help(self):
        out = self._run_help("profile", "add-task")
        assert "--name"       in out
        assert "--feature"    in out
        assert "--output-dir" in out
        assert "--domain"     in out

    def test_add_model_help(self):
        out = self._run_help("profile", "add-model")
        assert "--name"         in out
        assert "--output-dir"   in out
        assert "--feature"      in out
        assert "--architecture" in out
        assert "--scores"       in out


# ── End-to-end apply dispatch ─────────────────────────────────────────────────

class TestApplyDispatch:
    """Verify that 'llmrouter profile apply' correctly calls npz_to_llm_embeddings_json."""

    def test_apply_json_end_to_end(self, npz_profile_path, llm_data_json_path, tmp_path):
        from llmrouter.cli.router_profile import _apply_command

        out = str(tmp_path / "result.json")

        class FakeArgs:
            profile  = npz_profile_path
            llm_data = llm_data_json_path
            output   = out
            format   = "json"
            missing  = "warn"

        _apply_command(FakeArgs())

        assert os.path.isfile(out)
        with open(out) as f:
            data = json.load(f)
        profiles = np.load(npz_profile_path)
        for name in profiles.files:
            assert "embedding" in data[name], f"No embedding for '{name}'"

    def test_apply_pkl_end_to_end(self, npz_profile_path, llm_data_json_path, tmp_path):
        import pickle
        from llmrouter.cli.router_profile import _apply_command

        out = str(tmp_path / "result.pkl")

        class FakeArgs:
            profile  = npz_profile_path
            llm_data = llm_data_json_path
            output   = out
            format   = "pkl"
            missing  = "warn"

        _apply_command(FakeArgs())

        assert os.path.isfile(out)
        with open(out, "rb") as f:
            data = pickle.load(f)
        profiles = np.load(npz_profile_path)
        for name in profiles.files:
            assert name in data

    def test_apply_missing_error_raises(self, tmp_path, llm_data_json_path):
        """apply with --missing error and a partial .npz must exit non-zero."""
        from llmrouter.cli.router_profile import _apply_command

        rng = np.random.default_rng(0)
        partial_npz = str(tmp_path / "partial.npz")
        np.savez(partial_npz, **{"model-a": rng.standard_normal(768).astype("float32")})

        out = str(tmp_path / "result.json")

        class FakeArgs:
            profile  = partial_npz
            llm_data = llm_data_json_path
            output   = out
            format   = "json"
            missing  = "error"

        with pytest.raises(KeyError):
            _apply_command(FakeArgs())
