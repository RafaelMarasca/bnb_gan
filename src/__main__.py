"""
Entry point for ``python -m src``.

Quick shortcut — runs the pipeline with a preset config.

Usage
-----
::

    python -m src --preset quick
    python -m src --preset paper --stages convergence rate_sweep plots

For the full CLI with subcommands (run, sweep, report)::

    python run.py --help
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m src",
        description="RadCom Pipeline (shortcut). Use run.py for full CLI.",
    )
    p.add_argument("--preset", choices=["quick", "paper"], default="quick",
                   help="Config preset (default: quick)")
    p.add_argument("--stages", nargs="+", default=None,
                   help="Stages to run (space-separated)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    from .experiments.config import ExperimentConfig
    from .experiments.runner import ExperimentRunner

    cfg = (ExperimentConfig.quick_test() if args.preset == "quick"
           else ExperimentConfig.paper())
    if args.stages:
        cfg = cfg.with_overrides(stages=args.stages)

    runner = ExperimentRunner(cfg)
    runner.run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
