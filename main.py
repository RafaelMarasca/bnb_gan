#!/usr/bin/env python
"""
RadCom Waveform Design — Unified CLI
=====================================

Single entry point for the entire pipeline.

Modes (run one stage)
---------------------
::

    python main.py --mode generate   --config configs/quick.yaml
    python main.py --mode train      --config configs/quick.yaml
    python main.py --mode evaluate   --config configs/quick.yaml

Run a single experiment by name
-------------------------------
::

    python main.py experiment --name convergence --config configs/quick.yaml
    python main.py experiment --name rate_sweep  --config configs/paper.yaml
    python main.py experiment --list

Run the full pipeline (all stages)
----------------------------------
::

    python main.py pipeline --config configs/quick.yaml
    python main.py pipeline --preset quick
    python main.py pipeline --preset paper --stages convergence rate_sweep plots

Legacy subcommands (backward-compatible)
-----------------------------------------
::

    python main.py run    --preset quick
    python main.py sweep  --preset quick --axis N=8,16,32
    python main.py report results/quick_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


# =====================================================================
# Config loading helpers
# =====================================================================

def _resolve_config(args: argparse.Namespace) -> Any:
    """Build a PipelineConfig from CLI args (--config, --preset, overrides)."""
    from src.config import PipelineConfig, load_config
    from src.config.loader import load_preset

    # Start from file, preset, or defaults
    if getattr(args, "config", None):
        cfg = load_config(args.config)
    elif getattr(args, "preset", None):
        cfg = load_preset(args.preset)
    else:
        cfg = PipelineConfig()

    # Apply CLI overrides
    overrides: dict[str, Any] = {}
    for attr, key in _CLI_OVERRIDE_MAP.items():
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val

    if getattr(args, "stages", None):
        overrides["stages"] = args.stages
    # --run-name or --name (for legacy) map to config name
    run_name = getattr(args, "run_name", None) or getattr(args, "name", None)
    if run_name:
        overrides["name"] = run_name
    if getattr(args, "output_dir", None):
        overrides["output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed

    return cfg.with_overrides(**overrides) if overrides else cfg


# Map CLI argument names → dotted config keys
_CLI_OVERRIDE_MAP: dict[str, str] = {
    "N":             "system.N",
    "K":             "system.K",
    "L":             "system.L",
    "PT":            "system.PT",
    "snr_db":        "system.SNR_dB",
    "bnb_tol":       "bnb.tol",
    "bnb_max_iter":  "bnb.max_iter",
    "ds_n_samples":  "dataset.n_samples",
    "ds_n_workers":  "dataset.n_workers",
    "gan_epochs":    "gan.n_epochs",
    "gan_batch_size": "gan.batch_size",
    "gan_lr":        "gan.learning_rate",
    "eval_n_samples": "eval.n_samples",
}


# =====================================================================
# Argument builders
# =====================================================================

def _add_config_args(p: argparse.ArgumentParser, *, include_name: bool = True) -> None:
    """Add --config, --preset, --run-name, --output-dir, --seed, --quiet."""
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML or JSON config file")
    p.add_argument("--preset", choices=["quick", "paper"], default=None,
                   help="Use a built-in preset (overridden by --config)")
    if include_name:
        p.add_argument("--run-name", type=str, default=None, dest="run_name",
                       help="Experiment run name (overrides config 'name')")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (overrides config)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (overrides config)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")


def _add_system_overrides(p: argparse.ArgumentParser) -> None:
    """Optional CLI overrides for system/physics parameters."""
    g = p.add_argument_group("System overrides")
    g.add_argument("--N", type=int, default=None, help="Antennas")
    g.add_argument("--K", type=int, default=None, help="Users")
    g.add_argument("--L", type=int, default=None, help="Frame length")
    g.add_argument("--PT", type=float, default=None, help="Total power")
    g.add_argument("--snr-db", type=float, default=None, dest="snr_db",
                   help="SNR (dB)")


def _add_stage_overrides(p: argparse.ArgumentParser) -> None:
    """Optional CLI overrides for stage-specific parameters."""
    g = p.add_argument_group("Stage overrides")
    g.add_argument("--bnb-tol", type=float, default=None)
    g.add_argument("--bnb-max-iter", type=int, default=None)
    g.add_argument("--ds-n-samples", type=int, default=None)
    g.add_argument("--ds-n-workers", type=int, default=None,
                   help="Parallel workers (0=all cores, 1=sequential)")
    g.add_argument("--gan-epochs", type=int, default=None)
    g.add_argument("--gan-batch-size", type=int, default=None)
    g.add_argument("--gan-lr", type=float, default=None)
    g.add_argument("--eval-n-samples", type=int, default=None)


# =====================================================================
# Mode handlers (--mode generate / train / evaluate)
# =====================================================================

_MODE_TO_EXPERIMENTS: dict[str, list[str]] = {
    "generate":  ["dataset"],
    "train":     ["gan_train"],
    "evaluate":  ["waveform_eval"],
}


def _cmd_mode(args: argparse.Namespace) -> None:
    """Run one or more experiments for a given --mode."""
    from src.experiments import ExperimentRegistry

    cfg = _resolve_config(args)
    mode = args.mode
    verbose = not args.quiet

    if mode not in _MODE_TO_EXPERIMENTS:
        print(f"Error: unknown mode '{mode}'. "
              f"Choose from: {list(_MODE_TO_EXPERIMENTS.keys())}")
        sys.exit(1)

    for exp_name in _MODE_TO_EXPERIMENTS[mode]:
        exp = ExperimentRegistry.create(exp_name, cfg)
        exp.execute(verbose=verbose)


# =====================================================================
# Subcommand: experiment
# =====================================================================

def _cmd_experiment(args: argparse.Namespace) -> None:
    """Run a single registered experiment by name."""
    from src.experiments import ExperimentRegistry

    if args.list:
        print("\nAvailable experiments:")
        print("-" * 60)
        for info in ExperimentRegistry.list_detailed():
            print(f"  {info['name']:20s}  {info['description']}")
        print()
        return

    if not args.exp_name:
        print("Error: --name is required (or use --list to see available).")
        sys.exit(1)

    cfg = _resolve_config(args)
    verbose = not args.quiet
    exp = ExperimentRegistry.create(args.exp_name, cfg)
    exp.execute(verbose=verbose)


# =====================================================================
# Subcommand: pipeline
# =====================================================================

def _cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline (all stages or selected via --stages)."""
    from src.experiments import ExperimentRegistry

    cfg = _resolve_config(args)
    verbose = not args.quiet

    stages = cfg.stages
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  PIPELINE: {cfg.name}")
        print(f"  Stages: {stages}")
        print(f"  Output: {cfg.output_dir}/{cfg.name}")
        print(f"{'=' * 60}\n")

    for stage_name in stages:
        try:
            exp = ExperimentRegistry.create(stage_name, cfg)
            exp.execute(verbose=verbose)
        except KeyError:
            # Stage not in the experiment registry (e.g. "plots", "report")
            # Fall back to legacy runner for those stages
            _run_legacy_stage(stage_name, cfg, verbose)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  PIPELINE COMPLETE: {cfg.name}")
        print(f"{'=' * 60}\n")


def _run_legacy_stage(stage_name: str, cfg: Any, verbose: bool) -> None:
    """Fall back to legacy ExperimentRunner for stages like plots/report."""
    from src.experiments.config import ExperimentConfig
    from src.experiments.runner import ExperimentRunner

    # Build a legacy config from the new PipelineConfig
    legacy_cfg = ExperimentConfig(
        name=cfg.name,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        N=cfg.system.N,
        K=cfg.system.K,
        L=cfg.system.L,
        PT=cfg.system.PT,
        SNR_dB=cfg.system.SNR_dB,
        bnb_rule=cfg.bnb.rule,
        bnb_lb=cfg.bnb.lb_solver,
        bnb_ub=cfg.bnb.ub_solver,
        bnb_tol=cfg.bnb.tol,
        bnb_max_iter=cfg.bnb.max_iter,
        bnb_gp_iters=cfg.bnb.gp_max_iter,
        conv_epsilon=cfg.convergence.epsilon,
        rate_epsilons=cfg.rate_sweep.epsilons,
        rate_n_trials=cfg.rate_sweep.n_trials,
        rate_tol=cfg.rate_sweep.tol,
        rate_max_iter=cfg.rate_sweep.max_iter,
        ds_n_samples=cfg.dataset.n_samples,
        ds_epsilons=cfg.dataset.epsilons,
        ds_chunk_size=cfg.dataset.chunk_size,
        gan_latent_dim=cfg.gan.latent_dim,
        gan_hidden_g=cfg.gan.hidden_g,
        gan_hidden_c=cfg.gan.hidden_c,
        gan_n_epochs=cfg.gan.n_epochs,
        gan_batch_size=cfg.gan.batch_size,
        gan_lr=cfg.gan.learning_rate,
        gan_n_critic=cfg.gan.n_critic,
        gan_lambda_gp=cfg.gan.lambda_gp,
        gan_eval_every=cfg.gan.eval_every,
        gan_save_every=cfg.gan.save_every,
        eval_n_samples=cfg.eval.n_samples,
        eval_epsilons=cfg.eval.epsilons,
        plot_formats=cfg.plot.formats,
        plot_dpi=cfg.plot.dpi,
        stages=[stage_name],
    )

    runner = ExperimentRunner(legacy_cfg)
    method = getattr(runner, f"run_{stage_name}", None)
    if method is None:
        if verbose:
            print(f"  Warning: Unknown stage '{stage_name}', skipping.")
        return
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  STAGE (legacy): {stage_name}")
        print(f"{'=' * 60}")
    method(verbose=verbose)


# =====================================================================
# Legacy subcommands (backward-compatible with run.py)
# =====================================================================

def _cmd_legacy_run(args: argparse.Namespace) -> None:
    """Legacy: run full experiment via ExperimentRunner."""
    from src.experiments.config import ExperimentConfig
    from src.experiments.runner import ExperimentRunner

    cfg = _build_legacy_config(args)
    runner = ExperimentRunner(cfg)
    runner.run(verbose=not args.quiet)


def _cmd_legacy_sweep(args: argparse.Namespace) -> None:
    """Legacy: parameter sweep."""
    from src.experiments.config import ParameterGrid
    from src.experiments.sweep import run_sweep

    cfg = _build_legacy_config(args)

    axes: dict[str, list] = {}
    if args.grid:
        import json
        with open(args.grid) as f:
            axes = json.load(f)
    for ax in (args.axis or []):
        name, vals = ax.split("=", 1)
        axes[name] = [_infer_type(v.strip()) for v in vals.split(",")]

    if not axes:
        print("Error: specify at least one sweep axis.")
        sys.exit(1)

    grid = ParameterGrid(cfg, axes)
    run_sweep(grid, verbose=not args.quiet)


def _cmd_legacy_report(args: argparse.Namespace) -> None:
    """Legacy: generate / compare reports."""
    from src.experiments.results import ExperimentResult, ResultsAggregator
    from src.experiments.report import ReportGenerator

    p = Path(args.path)

    if args.compare:
        agg = ResultsAggregator()
        n = agg.add_dir(p)
        if n == 0:
            print(f"No experiments found in {p}")
            sys.exit(1)
        report = agg.generate_comparison_report()
        out = p / "comparison_report.md"
        out.write_text(report, encoding="utf-8")
        print(f"Comparison report ({n} experiments): {out}")
    else:
        result = ExperimentResult(p)
        gen = ReportGenerator(result)
        path = gen.save()
        print(f"Report saved: {path}")


def _build_legacy_config(args: argparse.Namespace):
    """Build a legacy ExperimentConfig from CLI args."""
    from src.experiments.config import ExperimentConfig

    if getattr(args, "config", None):
        cfg = ExperimentConfig.from_json(args.config)
    elif getattr(args, "preset", None) == "quick":
        cfg = ExperimentConfig.quick_test()
    elif getattr(args, "preset", None) == "paper":
        cfg = ExperimentConfig.paper()
    else:
        cfg = ExperimentConfig()

    overrides: dict = {}
    mapping = {
        "name": "name", "output_dir": "output_dir", "seed": "seed",
        "N": "N", "K": "K", "L": "L", "PT": "PT",
        "snr_db": "SNR_dB",
        "bnb_tol": "bnb_tol", "bnb_max_iter": "bnb_max_iter",
        "ds_n_samples": "ds_n_samples",
        "gan_epochs": "gan_n_epochs", "gan_batch_size": "gan_batch_size",
        "eval_n_samples": "eval_n_samples",
    }
    for cli_key, cfg_key in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            overrides[cfg_key] = val
    if getattr(args, "stages", None):
        overrides["stages"] = args.stages

    return cfg.with_overrides(**overrides) if overrides else cfg


# =====================================================================
# Helpers
# =====================================================================

def _infer_type(s: str) -> int | float | str:
    """Convert a CLI string to int → float → str (first that works)."""
    for cast in (int, float):
        try:
            return cast(s)
        except (ValueError, TypeError):
            pass
    return s


# =====================================================================
# Main
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RadCom Waveform Design — Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Top-level --mode (shortcut) ─────────────────────────
    parser.add_argument(
        "--mode",
        choices=["generate", "train", "evaluate"],
        default=None,
        help="Quick mode: generate data, train GAN, or evaluate",
    )
    _add_config_args(parser)
    _add_system_overrides(parser)
    _add_stage_overrides(parser)

    # ── Subcommands ─────────────────────────────────────────
    sub = parser.add_subparsers(dest="command")

    # experiment
    p_exp = sub.add_parser(
        "experiment",
        help="Run a single registered experiment by name",
    )
    p_exp.add_argument("--name", type=str, default=None, dest="exp_name",
                       help="Experiment name (see --list)")
    p_exp.add_argument("--list", action="store_true",
                       help="List all registered experiments")
    _add_config_args(p_exp, include_name=False)
    _add_system_overrides(p_exp)
    _add_stage_overrides(p_exp)

    # pipeline
    p_pipe = sub.add_parser(
        "pipeline",
        help="Run the full pipeline (all stages or --stages subset)",
    )
    p_pipe.add_argument("--stages", nargs="+", default=None,
                        help="Stages to run (space-separated)")
    _add_config_args(p_pipe)
    _add_system_overrides(p_pipe)
    _add_stage_overrides(p_pipe)

    # --- Legacy: run ---
    p_run = sub.add_parser("run", help="[Legacy] Run a single experiment")
    _add_config_args(p_run)
    _add_system_overrides(p_run)
    _add_stage_overrides(p_run)
    p_run.add_argument("--stages", nargs="+", default=None)

    # --- Legacy: sweep ---
    p_sweep = sub.add_parser("sweep", help="[Legacy] Run parameter sweep")
    _add_config_args(p_sweep)
    _add_system_overrides(p_sweep)
    _add_stage_overrides(p_sweep)
    p_sweep.add_argument("--stages", nargs="+", default=None)
    p_sweep.add_argument("--axis", action="append", default=[],
                         help="Sweep axis, e.g. N=8,16,32")
    p_sweep.add_argument("--grid", type=str, default=None,
                         help="JSON file with sweep axes")

    # --- Legacy: report ---
    p_report = sub.add_parser("report", help="[Legacy] Generate/compare reports")
    p_report.add_argument("path", type=str,
                          help="Experiment dir or parent for --compare")
    p_report.add_argument("--compare", action="store_true")

    return parser


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Dispatch: --mode takes priority over subcommands
    if args.mode:
        _cmd_mode(args)
        return

    dispatch = {
        "experiment": _cmd_experiment,
        "pipeline":   _cmd_pipeline,
        "run":        _cmd_legacy_run,
        "sweep":      _cmd_legacy_sweep,
        "report":     _cmd_legacy_report,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
