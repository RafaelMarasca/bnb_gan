"""
Built-in Experiments — Rate vs Epsilon Sweep
=============================================
Sum-rate vs similarity tolerance sweep.
Reproduces Fig.8 of the reference paper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseExperiment


class RateSweepExperiment(BaseExperiment):
    """Sum-rate vs epsilon sweep across multiple channel realizations.

    Config keys used
    ----------------
    - ``system.*`` (N, K, L, PT, SNR_dB)
    - ``rate_sweep.epsilons``, ``rate_sweep.n_trials``
    - ``rate_sweep.tol``, ``rate_sweep.max_iter``
    - ``bnb.gp_max_iter``
    - ``seed``
    """

    name = "rate_sweep"
    description = "Sum-rate vs similarity tolerance (epsilon) sweep"

    def run(self, verbose: bool = True) -> dict[str, Any]:
        from ..data.experiments import run_rate_vs_epsilon_experiment

        cfg = self.config
        result = run_rate_vs_epsilon_experiment(
            N=cfg.system.N,
            K=cfg.system.K,
            L=cfg.system.L,
            PT=cfg.system.PT,
            SNR_dB=cfg.system.SNR_dB,
            epsilons=np.array(cfg.rate_sweep.epsilons),
            n_trials=cfg.rate_sweep.n_trials,
            bnb_tol=cfg.rate_sweep.tol,
            bnb_max_iter=cfg.rate_sweep.max_iter,
            gp_iters=cfg.bnb.gp_max_iter,
            seed=cfg.seed,
            verbose=verbose,
        )

        # ── Persist ─────────────────────────────────────
        self.save_results(
            scalars={
                "epsilons": result.epsilons.tolist(),
                "rate_bnb": result.rate_bnb.tolist(),
                "rate_relaxed": result.rate_relaxed.tolist(),
                "awgn_capacity": result.awgn_capacity,
                "n_trials": result.n_trials,
                "elapsed_s": result.elapsed_s,
            },
            arrays={
                "epsilons": result.epsilons,
                "rate_bnb": result.rate_bnb,
                "rate_relaxed": result.rate_relaxed,
                "awgn_capacity": np.array([result.awgn_capacity]),
            },
        )
        return {
            "epsilons": result.epsilons,
            "rate_bnb": result.rate_bnb,
            "rate_relaxed": result.rate_relaxed,
            "awgn_capacity": result.awgn_capacity,
        }
