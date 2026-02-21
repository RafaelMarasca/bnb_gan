"""
Experiment Runner
=================
Orchestrates the paper's key experiments using all modular components.

Provides structured data output suitable for plotting in Step 5.
Reproduces:
  - Fig.7: BnB convergence behavior (single-column, 4 solver combos)
  - Fig.8: Sum-rate vs epsilon (multi-column, multiple trials)

Each experiment returns a dict with all data needed for visualization,
keeping the experiment logic cleanly separated from plotting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.config import SystemConfig, BnBConfig
from ..signal_proc.waveform import generate_channel, generate_chirp, generate_symbols
from ..optimizer.bnb import BranchAndBoundSolver, bnb_solve
from ..optimizer.solvers.lb_gp import LBSolverGP
from ..metrics.rate import sum_rate


# =========================================================================
# Fig.7: Convergence Experiment
# =========================================================================

@dataclass
class ConvergenceResult:
    """Result of a single solver-combo convergence run.

    Attributes
    ----------
    label : str
        Human-readable label (e.g., "ARS + CVXPY/SLSQP").
    rule : str
        Branching rule used ('ARS' or 'BRS').
    lb_solver : str
        Lower-bound solver name.
    ub_solver : str
        Upper-bound solver name.
    lb_history : list[float]
        Lower bound at each BnB iteration.
    ub_history : list[float]
        Upper bound at each BnB iteration.
    objective : float
        Best objective found.
    elapsed_s : float
        Wall-clock time in seconds.
    n_iterations : int
        Number of BnB iterations.
    """

    label: str
    rule: str
    lb_solver: str
    ub_solver: str
    lb_history: list[float]
    ub_history: list[float]
    objective: float
    elapsed_s: float
    n_iterations: int


def run_convergence_experiment(
    N: int = 16,
    K: int = 4,
    epsilon: float = 1.0,
    PT: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    gp_iters: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Reproduce Fig.7: BnB convergence for 4 solver combinations.

    Runs a single-column BnB optimization with ARS and BRS rules,
    each with CVXPY/SLSQP and GP/GP solvers.

    Parameters
    ----------
    N, K : int
        System dimensions (antennas, users).
    epsilon : float
        Similarity tolerance.
    PT : float
        Total transmit power.
    max_iter : int
        Maximum BnB iterations.
    tol : float
        Convergence tolerance.
    gp_iters : int
        GP solver max iterations per bound evaluation.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        'results': list[ConvergenceResult] — one per solver combo
        'H': ndarray — channel used
        's': ndarray — symbol vector used
        'x0': ndarray — reference waveform column
        'params': dict — experiment parameters
    """
    np.random.seed(seed)
    H = generate_channel(K, N)
    Ht = np.sqrt(PT / N) * H
    s = generate_symbols(K, 1).ravel()
    x0 = np.exp(1j * np.pi * np.arange(N) ** 2 / N)  # single chirp column

    combos = [
        ("ARS", "cvxpy", "slsqp", "ARS + CVXPY/SLSQP"),
        ("BRS", "cvxpy", "slsqp", "BRS + CVXPY/SLSQP"),
        ("ARS", "gp", "gp", "ARS + GP"),
        ("BRS", "gp", "gp", "BRS + GP"),
    ]

    results: list[ConvergenceResult] = []

    for rule, lb_sol, ub_sol, label in combos:
        if verbose:
            print(f"\n  {label} ...")
        t0 = time.time()

        x_opt, obj, lb_hist, ub_hist = bnb_solve(
            Ht, s, x0, epsilon,
            rule=rule, lb_solver=lb_sol, ub_solver=ub_sol,
            tol=tol, max_iter=max_iter, gp_iters=gp_iters,
            verbose=verbose,
        )
        dt = time.time() - t0

        cr = ConvergenceResult(
            label=label,
            rule=rule,
            lb_solver=lb_sol,
            ub_solver=ub_sol,
            lb_history=list(lb_hist),
            ub_history=list(ub_hist),
            objective=float(obj),
            elapsed_s=dt,
            n_iterations=len(ub_hist),
        )
        results.append(cr)

        if verbose:
            gap = ub_hist[-1] - lb_hist[-1] if ub_hist else float("inf")
            print(
                f"  {len(ub_hist)} iters, {dt:.1f}s, "
                f"obj={obj:.6f}, gap={gap:.6f}"
            )

    return {
        "results": results,
        "H": H,
        "s": s,
        "x0": x0,
        "params": {
            "N": N, "K": K, "epsilon": epsilon, "PT": PT,
            "max_iter": max_iter, "tol": tol, "seed": seed,
        },
    }


# =========================================================================
# Fig.8: Sum-Rate vs Epsilon Experiment
# =========================================================================

@dataclass
class RateVsEpsilonResult:
    """Result of a rate-vs-epsilon sweep.

    Attributes
    ----------
    epsilons : ndarray
        Array of epsilon values tested.
    rate_bnb : ndarray
        Average BnB sum-rate at each epsilon.
    rate_relaxed : ndarray
        Average convex-relaxation sum-rate at each epsilon.
    awgn_capacity : float
        AWGN capacity (MUI=0) upper bound.
    n_trials : int
        Number of channel realizations averaged.
    elapsed_s : float
        Total wall-clock time.
    """

    epsilons: NDArray
    rate_bnb: NDArray
    rate_relaxed: NDArray
    awgn_capacity: float
    n_trials: int
    elapsed_s: float


def run_rate_vs_epsilon_experiment(
    N: int = 16,
    K: int = 4,
    L: int = 20,
    PT: float = 1.0,
    SNR_dB: float = 10.0,
    epsilons: NDArray | None = None,
    n_trials: int = 5,
    bnb_tol: float = 5e-3,
    bnb_max_iter: int = 40,
    gp_iters: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> RateVsEpsilonResult:
    """Reproduce Fig.8: sum-rate vs similarity tolerance.

    For each epsilon value and each trial (channel realization), runs
    the full multi-column BnB optimization and a convex-relaxation
    baseline. Returns averaged rates.

    Parameters
    ----------
    N, K, L : int
        System dimensions.
    PT : float
        Transmit power.
    SNR_dB : float
        SNR in dB.
    epsilons : ndarray, optional
        Epsilon values to sweep (default: paper set).
    n_trials : int
        Number of independent channel realizations.
    bnb_tol : float
        BnB convergence tolerance (coarser for speed).
    bnb_max_iter : int
        Max BnB iterations per column.
    gp_iters : int
        GP iterations per bound evaluation.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    RateVsEpsilonResult
        Averaged rate curves and metadata.
    """
    if epsilons is None:
        epsilons = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0])

    N0 = PT / 10.0 ** (SNR_dB / 10.0)
    sc = np.sqrt(PT / N)
    awgn_cap = K * np.log2(1.0 + 1.0 / N0)

    r_bnb = np.zeros(len(epsilons))
    r_relax = np.zeros(len(epsilons))

    t0 = time.time()

    for trial in range(n_trials):
        np.random.seed(seed + trial)
        H = generate_channel(K, N)
        S = generate_symbols(K, L)
        X0 = generate_chirp(N, L, PT)
        Ht = sc * H
        X0n = X0 / sc

        # Precompute GP step size
        eigs = np.linalg.eigvalsh(Ht.conj().T @ Ht)
        step = 1.0 / max(2.0 * eigs.max(), 1e-12)

        if verbose:
            print(f"\n  Trial {trial + 1}/{n_trials}")

        for ei, eps in enumerate(epsilons):
            hw0 = np.arccos(np.clip(1 - eps**2 / 2, -1, 1))

            X_bnb_mat = np.zeros((N, L), dtype=complex)
            X_rel_mat = np.zeros((N, L), dtype=complex)

            for t in range(L):
                x0c = X0n[:, t]
                sc_col = S[:, t]
                lb = np.angle(x0c) - hw0
                ub = np.angle(x0c) + hw0

                # BnB (GP solver for speed — matches bnb_comprehensive)
                xb, _, _, _ = bnb_solve(
                    Ht, sc_col, x0c, eps,
                    rule="ARS", lb_solver="gp", ub_solver="gp",
                    tol=bnb_tol, max_iter=bnb_max_iter, gp_iters=gp_iters,
                )
                X_bnb_mat[:, t] = (xb if xb is not None else x0c) * sc

                # Convex relaxation (GP-LB only, not constant-modulus feasible)
                relax_solver = LBSolverGP(max_iter=gp_iters * 2, step=step)
                _, xl = relax_solver.solve(Ht, sc_col, lb, ub)
                X_rel_mat[:, t] = (xl if xl is not None else x0c) * sc

            r_bnb[ei] += sum_rate(H, X_bnb_mat, S, N0)
            r_relax[ei] += sum_rate(H, X_rel_mat, S, N0)

            if verbose:
                print(
                    f"    eps={eps:.2f}  "
                    f"BnB={r_bnb[ei] / (trial + 1):.2f}  "
                    f"Relax={r_relax[ei] / (trial + 1):.2f} bps/Hz"
                )

    r_bnb /= n_trials
    r_relax /= n_trials
    elapsed = time.time() - t0

    return RateVsEpsilonResult(
        epsilons=epsilons,
        rate_bnb=r_bnb,
        rate_relaxed=r_relax,
        awgn_capacity=awgn_cap,
        n_trials=n_trials,
        elapsed_s=elapsed,
    )
