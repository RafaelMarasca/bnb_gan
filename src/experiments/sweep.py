"""
Parameter Sweep
===============
Run a grid of experiments with different parameter combinations.

Usage
-----
>>> from src.experiments import ExperimentConfig, ParameterGrid, run_sweep
>>> base = ExperimentConfig.quick_test()
>>> grid = ParameterGrid(base, {"N": [8, 16], "K": [2, 4]})
>>> results = run_sweep(grid)
"""

from __future__ import annotations

import time
from typing import Any

from .config import ExperimentConfig, ParameterGrid
from .runner import ExperimentRunner


def run_sweep(
    grid: ParameterGrid,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run one experiment for every config in the parameter grid.

    Parameters
    ----------
    grid : ParameterGrid
        Generated from ``ParameterGrid(base_config, axes_dict)``.
    verbose : bool
        Print progress banners.

    Returns
    -------
    list of dict
        Each dict has keys ``config``, ``results``, ``status``.
    """
    configs = grid.configs()
    n = len(configs)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  PARAMETER SWEEP: {n} experiments")
        print(f"  Axes: {list(grid.axes.keys())}")
        for k, v in grid.axes.items():
            print(f"    {k}: {v}")
        print(f"{'=' * 60}\n")

    all_results: list[dict[str, Any]] = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  [{i + 1}/{n}] {cfg.name}")
            print(f"{'─' * 60}")

        runner = ExperimentRunner(cfg)
        try:
            results = runner.run(verbose=verbose)
            all_results.append({
                "config": cfg,
                "results": results,
                "status": "ok",
            })
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            all_results.append({
                "config": cfg,
                "results": None,
                "status": f"error: {e}",
            })

    elapsed = time.time() - t0

    if verbose:
        ok = sum(1 for r in all_results if r["status"] == "ok")
        print(f"\n{'=' * 60}")
        print(f"  SWEEP COMPLETE — {n} experiments in {elapsed:.1f}s")
        print(f"  Success: {ok}/{n}")
        print(f"{'=' * 60}\n")

    return all_results
