"""
Integration tests for the refactored configuration and experiment framework.

Tests:
- YAML config loading, validation, and round-trips
- Pydantic validation catches bad inputs
- Experiment registry discovery
- BaseExperiment lifecycle (create → execute → save → load)
- CLI argument parsing (no actual experiment execution)
- Legacy config bridge compatibility
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# =====================================================================
# Config tests
# =====================================================================


class TestPipelineConfig:
    """Tests for src.config.schema.PipelineConfig."""

    def test_load_default_yaml(self):
        from src.config import load_config

        cfg = load_config("configs/default.yaml")
        assert cfg.name == "default"
        assert cfg.system.N == 16
        assert cfg.system.K == 4
        assert cfg.gan.learning_rate == 1e-4
        assert cfg.output_dir == "outputs"

    def test_load_quick_yaml(self):
        from src.config import load_config

        cfg = load_config("configs/quick.yaml")
        assert cfg.name == "quick_test"
        assert cfg.system.N == 8
        assert cfg.gan.n_epochs == 10

    def test_load_paper_yaml(self):
        from src.config import load_config

        cfg = load_config("configs/paper.yaml")
        assert cfg.name == "paper"
        assert cfg.system.N == 16
        assert cfg.gan.n_epochs == 500
        assert "eps" in cfg.plot.formats

    def test_preset_quick(self):
        from src.config.loader import load_preset

        cfg = load_preset("quick")
        assert cfg.name == "quick_test"
        assert cfg.system.N == 8

    def test_preset_paper(self):
        from src.config.loader import load_preset

        cfg = load_preset("paper")
        assert cfg.name == "paper"
        assert cfg.system.N == 16

    def test_preset_unknown_raises(self):
        from src.config.loader import load_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent")

    def test_with_overrides_flat(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig()
        cfg2 = cfg.with_overrides(seed=123)
        assert cfg2.seed == 123
        assert cfg.seed == 42  # original unchanged

    def test_with_overrides_dotted(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig()
        cfg2 = cfg.with_overrides(**{"system.N": 32, "gan.learning_rate": 0.01})
        assert cfg2.system.N == 32
        assert cfg2.gan.learning_rate == 0.01

    def test_with_overrides_none_ignored(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig()
        cfg2 = cfg.with_overrides(seed=None)
        assert cfg2.seed == cfg.seed

    def test_validation_bad_stage(self):
        from src.config import PipelineConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PipelineConfig(stages=["nonexistent_stage"])

    def test_validation_negative_antennas(self):
        from src.config import PipelineConfig, SystemConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SystemConfig(N=-1)

    def test_validation_extra_field_rejected(self):
        from src.config import PipelineConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PipelineConfig(bogus_field=42)

    def test_yaml_roundtrip(self, tmp_path):
        from src.config import PipelineConfig, load_config

        cfg = PipelineConfig(name="roundtrip_test", seed=99)
        yaml_path = tmp_path / "test.yaml"
        cfg.to_yaml(yaml_path)

        loaded = load_config(yaml_path)
        assert loaded.name == "roundtrip_test"
        assert loaded.seed == 99
        assert loaded.system.N == cfg.system.N

    def test_json_roundtrip(self, tmp_path):
        from src.config import PipelineConfig

        cfg = PipelineConfig(name="json_test")
        json_path = tmp_path / "test.json"
        cfg.to_json(json_path)

        loaded = PipelineConfig.from_json(json_path)
        assert loaded.name == "json_test"

    def test_legacy_bridge_sys_config(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig()
        legacy = cfg.sys_config
        assert legacy.N == cfg.system.N
        assert legacy.K == cfg.system.K
        assert abs(legacy.N0 - cfg.system.N0) < 1e-12

    def test_legacy_bridge_bnb_config(self):
        from src.config import PipelineConfig

        cfg = PipelineConfig()
        legacy = cfg.bnb_legacy
        assert legacy.rule == cfg.bnb.rule
        assert legacy.tol == cfg.bnb.tol

    def test_load_with_overrides(self):
        from src.config import load_config

        cfg = load_config(
            "configs/default.yaml",
            overrides={"seed": 777, "gan": {"learning_rate": 0.01}},
        )
        assert cfg.seed == 777
        assert cfg.gan.learning_rate == 0.01

    def test_system_n0_property(self):
        from src.config import SystemConfig

        sys = SystemConfig(PT=1.0, SNR_dB=10.0)
        expected = 1.0 / 10.0  # 10^(10/10) = 10
        assert abs(sys.N0 - expected) < 1e-12


# =====================================================================
# Experiment registry tests
# =====================================================================


class TestExperimentRegistry:
    """Tests for the experiment registry and BaseExperiment."""

    def test_builtin_experiments_registered(self):
        from src.experiments import ExperimentRegistry

        names = ExperimentRegistry.list_experiments()
        assert "convergence" in names
        assert "rate_sweep" in names
        assert "dataset" in names
        assert "gan_train" in names
        assert "waveform_eval" in names

    def test_list_detailed(self):
        from src.experiments import ExperimentRegistry

        detailed = ExperimentRegistry.list_detailed()
        assert len(detailed) >= 5
        for item in detailed:
            assert "name" in item
            assert "description" in item
            assert len(item["description"]) > 0

    def test_get_unknown_raises(self):
        from src.experiments import ExperimentRegistry

        with pytest.raises(KeyError, match="Unknown experiment"):
            ExperimentRegistry.get("no_such_experiment")

    def test_create_experiment(self):
        from src.config import PipelineConfig
        from src.experiments import ExperimentRegistry, BaseExperiment

        cfg = PipelineConfig(name="test_create", output_dir="outputs")
        exp = ExperimentRegistry.create("convergence", cfg)
        assert isinstance(exp, BaseExperiment)
        assert exp.name == "convergence"

    def test_custom_experiment_autoregisters(self):
        from src.config import PipelineConfig
        from src.experiments import BaseExperiment, ExperimentRegistry

        class _TestAutoReg(BaseExperiment):
            name = "test_autoreg"
            description = "Auto-registered test experiment"

            def run(self, verbose=True):
                return {"status": "ok"}

        assert "test_autoreg" in ExperimentRegistry.list_experiments()

        # Cleanup
        del ExperimentRegistry._registry["test_autoreg"]

    def test_experiment_save_load_results(self, tmp_path):
        from src.config import PipelineConfig
        from src.experiments import BaseExperiment, ExperimentRegistry

        class _TestPersist(BaseExperiment):
            name = "test_persist"
            description = "Persistence test"

            def run(self, verbose=True):
                data = {"metric_a": 3.14, "tags": ["foo", "bar"]}
                arrays = {"weights": np.array([1.0, 2.0, 3.0])}
                self.save_results(scalars=data, arrays=arrays)
                return data

        cfg = PipelineConfig(name="persist_test", output_dir=str(tmp_path))
        exp = ExperimentRegistry.create("test_persist", cfg)
        exp.execute(verbose=False)

        # Reload from disk
        loaded = exp.load_results()
        assert loaded is not None
        assert loaded.scalars["metric_a"] == 3.14
        assert loaded.scalars["tags"] == ["foo", "bar"]
        assert np.allclose(loaded.arrays["weights"], [1.0, 2.0, 3.0])

        # Cleanup
        del ExperimentRegistry._registry["test_persist"]

    def test_experiment_result_standalone_load(self, tmp_path):
        from src.experiments.base import ExperimentResult

        er = ExperimentResult("test_standalone", tmp_path / "test_exp")
        er.add_scalar("answer", 42)
        er.add_array("data", np.zeros((3, 3)))
        er.save()

        loaded = ExperimentResult.load(tmp_path / "test_exp")
        assert loaded.scalars["answer"] == 42
        assert loaded.arrays["data"].shape == (3, 3)


# =====================================================================
# CLI parsing tests (no actual execution)
# =====================================================================


class TestCLIParsing:
    """Test that CLI argument parsing works without running experiments."""

    def test_mode_generate(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "generate", "--config", "configs/quick.yaml",
        ])
        assert args.mode == "generate"
        assert args.config == "configs/quick.yaml"

    def test_mode_train(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "train", "--preset", "quick", "--quiet",
        ])
        assert args.mode == "train"
        assert args.preset == "quick"
        assert args.quiet is True

    def test_mode_evaluate(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "evaluate", "--config", "configs/paper.yaml",
        ])
        assert args.mode == "evaluate"

    def test_experiment_subcommand(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "experiment", "--name", "convergence",
            "--config", "configs/quick.yaml",
        ])
        assert args.command == "experiment"
        assert args.exp_name == "convergence"

    def test_experiment_list(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args(["experiment", "--list"])
        assert args.command == "experiment"
        assert args.list is True

    def test_pipeline_subcommand(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "pipeline", "--preset", "quick",
            "--stages", "convergence", "rate_sweep",
        ])
        assert args.command == "pipeline"
        assert args.stages == ["convergence", "rate_sweep"]

    def test_mode_with_overrides(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "train",
            "--config", "configs/quick.yaml",
            "--N", "32",
            "--gan-epochs", "100",
            "--gan-lr", "0.001",
        ])
        assert args.mode == "train"
        assert args.N == 32
        assert args.gan_epochs == 100
        assert args.gan_lr == 0.001

    def test_resolve_config_from_yaml(self):
        from main import build_parser, _resolve_config

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "generate", "--config", "configs/quick.yaml",
        ])
        cfg = _resolve_config(args)
        assert cfg.name == "quick_test"
        assert cfg.system.N == 8

    def test_resolve_config_from_preset(self):
        from main import build_parser, _resolve_config

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "train", "--preset", "paper",
        ])
        cfg = _resolve_config(args)
        assert cfg.name == "paper"
        assert cfg.system.N == 16

    def test_resolve_config_with_cli_overrides(self):
        from main import build_parser, _resolve_config

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "train",
            "--config", "configs/quick.yaml",
            "--N", "32",
            "--seed", "999",
            "--gan-lr", "0.01",
        ])
        cfg = _resolve_config(args)
        assert cfg.system.N == 32
        assert cfg.seed == 999
        assert cfg.gan.learning_rate == 0.01

    def test_legacy_run_subcommand(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run", "--preset", "quick", "--stages", "convergence",
        ])
        assert args.command == "run"
        assert args.preset == "quick"
        assert args.stages == ["convergence"]
