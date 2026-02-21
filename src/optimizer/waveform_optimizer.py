"""
Multi-Column Waveform Optimizer
================================
Manages the full waveform matrix X by solving for each column x_t
independently (exploiting eq.27 separability).

Matches ``bnb_comprehensive.optimize_waveform`` exactly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.config import SystemConfig, BnBConfig
from .bnb import BranchAndBoundSolver, BnBResult


class WaveformMatrixOptimizer:
    """Optimize the full N x L waveform matrix column-by-column.

    The full-matrix objective is separable:
        ||HX - S||_F^2 = sum_t ||H x_t - s_t||^2
    so each column is an independent BnB sub-problem.

    Parameters
    ----------
    sys_config : SystemConfig
        Physical system parameters.
    bnb_config : BnBConfig
        BnB algorithm settings.

    Example
    -------
    >>> optimizer = WaveformMatrixOptimizer(SystemConfig(), BnBConfig())
    >>> X_opt, results = optimizer.optimize(H, S, X0, epsilon=1.0)
    """

    def __init__(
        self,
        sys_config: SystemConfig | None = None,
        bnb_config: BnBConfig | None = None,
    ) -> None:
        self.sys_config = sys_config or SystemConfig()
        self.bnb_config = bnb_config or BnBConfig()

    def optimize(
        self,
        H: NDArray,
        S: NDArray,
        X0: NDArray,
        epsilon: float,
        verbose_col: bool = False,
    ) -> tuple[NDArray, list[BnBResult]]:
        """Optimize waveform matrix column-by-column.

        Parameters
        ----------
        H : ndarray, shape (K, N)
            Channel matrix.
        S : ndarray, shape (K, L)
            Symbol matrix.
        X0 : ndarray, shape (N, L)
            Reference waveform matrix (constant modulus, |X0_{n,l}| = sqrt(PT/N)).
        epsilon : float
            Similarity tolerance.
        verbose_col : bool
            Print progress for each column.

        Returns
        -------
        X_opt : ndarray, shape (N, L)
            Optimized waveform matrix.
        results : list[BnBResult]
            Per-column optimization results.
        """
        N, L = X0.shape
        PT = self.sys_config.PT
        sc = np.sqrt(PT / N)

        # Scaled channel Ht = sqrt(PT/N) * H
        Ht = sc * H
        # Normalize reference to unit modulus
        X0n = X0 / sc

        solver = BranchAndBoundSolver(self.bnb_config)

        X_opt = np.zeros((N, L), dtype=complex)
        results: list[BnBResult] = []

        for t in range(L):
            result = solver.solve(Ht, S[:, t], X0n[:, t], epsilon)
            results.append(result)

            if result.x_opt is not None:
                X_opt[:, t] = result.x_opt * sc
            else:
                X_opt[:, t] = X0n[:, t] * sc

            if verbose_col:
                print(f"    col {t + 1}/{L} done  "
                      f"(obj={result.objective:.6f}, "
                      f"iters={result.n_iterations})")

        return X_opt, results


# =========================================================================
# Convenience functional API (mirrors bnb_comprehensive.optimize_waveform)
# =========================================================================

def optimize_waveform(
    H: NDArray,
    S: NDArray,
    X0: NDArray,
    epsilon: float,
    PT: float = 1.0,
    verbose_col: bool = False,
    **bnb_kw,
) -> NDArray:
    """Functional wrapper matching ``bnb_comprehensive.optimize_waveform``.

    Parameters
    ----------
    H  : (K, N) channel
    S  : (K, L) symbols
    X0 : (N, L) reference waveform with |X0_{n,l}| = sqrt(PT/N)
    epsilon : similarity tolerance
    PT : total transmit power
    **bnb_kw : forwarded to BnBConfig (rule, lb_solver, ub_solver, tol, etc.)

    Returns
    -------
    X_opt : (N, L) optimized waveform
    """
    # Map bnb_kw to BnBConfig fields
    config_fields = {
        "rule": bnb_kw.pop("rule", "ARS"),
        "lb_solver": bnb_kw.pop("lb_solver", "cvxpy"),
        "ub_solver": bnb_kw.pop("ub_solver", "slsqp"),
        "tol": bnb_kw.pop("tol", 1e-3),
        "max_iter": bnb_kw.pop("max_iter", 200),
        "gp_max_iter": bnb_kw.pop("gp_iters", 100),
        "verbose": bnb_kw.pop("verbose", False),
    }
    sys_cfg = SystemConfig(
        N=X0.shape[0],
        K=H.shape[0],
        L=X0.shape[1],
        PT=PT,
    )
    bnb_cfg = BnBConfig(**config_fields)  # type: ignore[arg-type]
    optimizer = WaveformMatrixOptimizer(sys_cfg, bnb_cfg)
    X_opt, _ = optimizer.optimize(H, S, X0, epsilon, verbose_col=verbose_col)
    return X_opt
