#!/usr/bin/env python
"""
RadCom Waveform Design — Experiment CLI
========================================

Unified command-line interface for running experiments, parameter sweeps,
and generating reports.

Examples
--------
Run a single experiment::

    python run.py run --preset quick
    python run.py run --preset paper --stages convergence rate_sweep plots
    python run.py run --N 8 --K 2 --gan-epochs 50

Parameter sweep::

    python run.py sweep --preset quick --axis N=8,16,32 --axis K=2,4
    python run.py sweep --preset quick --grid sweep_grid.json

Generate / compare reports::

    python run.py report results/quick_test
    python run.py report results/ --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# =====================================================================
# Argument helpers
# =====================================================================

def _add_common(p: argparse.ArgumentParser) -> None:
    """Arguments shared by 'run' and 'sweep'."""
    p.add_argument("--preset", choices=["quick", "paper"], default=None,
                   help="Start from a named preset config")
    p.add_argument("--config", type=str, default=None,
                   help="Load config from a JSON file")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stages", nargs="+", default=None,
                   help="Stages to run (space-separated)")
    p.add_argument("--quiet", action="store_true")


def _add_system(p: argparse.ArgumentParser) -> None:
    """System-physics overrides."""
    g = p.add_argument_group("System parameters")
    g.add_argument("--N", type=int, default=None, help="Antennas")
    g.add_argument("--K", type=int, default=None, help="Users")
    g.add_argument("--L", type=int, default=None, help="Frame length")
    g.add_argument("--PT", type=float, default=None, help="Total power")
    g.add_argument("--snr-db", type=float, default=None, dest="snr_db",
                   help="SNR (dB)")


def _add_stage_overrides(p: argparse.ArgumentParser) -> None:
    """Stage-specific overrides."""
    g = p.add_argument_group("Stage overrides")
    g.add_argument("--bnb-tol", type=float, default=None)
    g.add_argument("--bnb-max-iter", type=int, default=None)
    g.add_argument("--ds-n-samples", type=int, default=None)
    g.add_argument("--gan-epochs", type=int, default=None)
    g.add_argument("--gan-batch-size", type=int, default=None)
    g.add_argument("--eval-n-samples", type=int, default=None)
    g.add_argument("--plot-formats", nargs="+", default=None)


# =====================================================================
# Build config from CLI args
# =====================================================================

def _build_config(args: argparse.Namespace):
    """Create an ExperimentConfig from parsed CLI arguments."""
    from src.experiments.config import ExperimentConfig

    if args.config:
        cfg = ExperimentConfig.from_json(args.config)
    elif args.preset == "quick":
        cfg = ExperimentConfig.quick_test()
    elif args.preset == "paper":
        cfg = ExperimentConfig.paper()
    else:
        cfg = ExperimentConfig()

    # Map CLI names → config field names
    overrides: dict = {}
    _mapping = {
        "name":          "name",
        "output_dir":    "output_dir",
        "seed":          "seed",
        "N":             "N",
        "K":             "K",
        "L":             "L",
        "PT":            "PT",
        "snr_db":        "SNR_dB",
        "bnb_tol":       "bnb_tol",
        "bnb_max_iter":  "bnb_max_iter",
        "ds_n_samples":  "ds_n_samples",
        "gan_epochs":    "gan_n_epochs",
        "gan_batch_size": "gan_batch_size",
        "eval_n_samples": "eval_n_samples",
        "plot_formats":  "plot_formats",
    }
    for cli_key, cfg_key in _mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            overrides[cfg_key] = val

    if args.stages:
        overrides["stages"] = args.stages

    return cfg.with_overrides(**overrides) if overrides else cfg


# =====================================================================
# Subcommands
# =====================================================================

def _cmd_run(args: argparse.Namespace) -> None:
    from src.experiments.runner import ExperimentRunner

    cfg = _build_config(args)
    runner = ExperimentRunner(cfg)
    runner.run(verbose=not args.quiet)


def _cmd_sweep(args: argparse.Namespace) -> None:
    from src.experiments.config import ParameterGrid
    from src.experiments.sweep import run_sweep

    cfg = _build_config(args)

    # Collect axes
    axes: dict[str, list] = {}
    if args.grid:
        import json
        with open(args.grid) as f:
            axes = json.load(f)
    for ax in (args.axis or []):
        name, vals = ax.split("=", 1)
        axes[name] = [_infer_type(v.strip()) for v in vals.split(",")]

    if not axes:
        print("Error: specify at least one sweep axis.\n"
              "  --axis N=8,16,32 --axis K=2,4\n"
              "  --grid sweep_grid.json")
        sys.exit(1)

    grid = ParameterGrid(cfg, axes)
    run_sweep(grid, verbose=not args.quiet)


def _cmd_report(args: argparse.Namespace) -> None:
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


# =====================================================================
# Helpers
# =====================================================================

def _infer_type(s: str):
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

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="RadCom Waveform Design — Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a single experiment")
    _add_common(p_run)
    _add_system(p_run)
    _add_stage_overrides(p_run)

    # --- sweep ---
    p_sweep = sub.add_parser("sweep", help="Run parameter sweep")
    _add_common(p_sweep)
    _add_system(p_sweep)
    _add_stage_overrides(p_sweep)
    p_sweep.add_argument("--axis", action="append", default=[],
                         help="Sweep axis, e.g. N=8,16,32")
    p_sweep.add_argument("--grid", type=str, default=None,
                         help="JSON file with sweep axes")

    # --- report ---
    p_report = sub.add_parser("report", help="Generate / compare reports")
    p_report.add_argument("path", type=str,
                          help="Experiment directory or parent for --compare")
    p_report.add_argument("--compare", action="store_true",
                          help="Compare all experiments in directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "run": _cmd_run,
        "sweep": _cmd_sweep,
        "report": _cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
