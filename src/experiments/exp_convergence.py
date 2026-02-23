"""
Built-in Experiments — Convergence Analysis
============================================
BnB convergence behavior across 4 solver combinations.
Reproduces Fig.7 of the reference paper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseExperiment


class ConvergenceExperiment(BaseExperiment):
    """BnB convergence analysis (4 solver combos).

    Runs a single-column BnB optimization with ARS/BRS rules and
    CVXPY/SLSQP vs GP/GP solvers, tracking upper/lower bound histories.

    Config keys used
    ----------------
    - ``system.N``, ``system.K``, ``system.PT``
    - ``convergence.epsilon``
    - ``bnb.max_iter``, ``bnb.tol``, ``bnb.gp_max_iter``
    - ``seed``
    """

    name = "convergence"
    description = "BnB convergence analysis (4 solver combinations)"

    def run(self, verbose: bool = True) -> dict[str, Any]:
        from ..data.experiments import run_convergence_experiment

        cfg = self.config
        result = run_convergence_experiment(
            N=cfg.system.N,
            K=cfg.system.K,
            epsilon=cfg.convergence.epsilon,
            PT=cfg.system.PT,
            max_iter=cfg.bnb.max_iter,
            tol=cfg.bnb.tol,
            gp_iters=cfg.bnb.gp_max_iter,
            seed=cfg.seed,
            verbose=verbose,
        )

        # ── Persist ─────────────────────────────────────
        arrays: dict[str, np.ndarray] = {
            "H": result["H"],
            "s": result["s"],
            "x0": result["x0"],
        }
        summary = []
        for i, r in enumerate(result["results"]):
            arrays[f"ub_history_{i}"] = np.array(r.ub_history)
            arrays[f"lb_history_{i}"] = np.array(r.lb_history)
            summary.append({
                "label": r.label,
                "rule": r.rule,
                "lb_solver": r.lb_solver,
                "ub_solver": r.ub_solver,
                "objective": r.objective,
                "n_iterations": r.n_iterations,
                "elapsed_s": r.elapsed_s,
                "gap": (
                    r.ub_history[-1] - r.lb_history[-1]
                    if r.ub_history
                    else None
                ),
            })

        self.save_results(scalars={"summary": summary}, arrays=arrays)
        return result
