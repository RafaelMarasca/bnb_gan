"""
LB Solver — Gradient Projection (FISTA/Nesterov + PR2)
======================================================
Solves QP-LB via accelerated gradient projection (eqs.43-44).

    v       = x^k + (k-1)/(k+2) * (x^k - x^{k-1})     [momentum]
    x^{k+1} = PR2(v - step * grad_f(v))                 [project]

step = 1 / (2 * lambda_max(Ht^H Ht))  (inverse Lipschitz constant).

Matches ``bnb_comprehensive.solve_lb_gp`` exactly.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ..projections import PR2
from .base import LBSolverBase, default_registry


class LBSolverGP(LBSolverBase):
    """Lower-bound solver using accelerated gradient projection with PR2.

    Parameters
    ----------
    step : float or None
        Gradient step size. If None, must be set before calling ``solve``
        (typically computed as ``compute_lipschitz_step(Ht)``).
    max_iter : int
        Maximum GP iterations.
    tol : float
        Convergence tolerance on ||x^{k+1} - x^k||.
    """

    name: ClassVar[str] = "gp"

    def __init__(
        self,
        step: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        self.step = step
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        Ht: NDArray,
        s: NDArray,
        l: NDArray,
        u: NDArray,
    ) -> tuple[float, NDArray | None]:
        """Solve QP-LB via FISTA + PR2 (Paper eqs.43-44).

        Uses Nesterov acceleration (momentum term) and projects onto
        the convex hull of the feasible arc via PR2 at each step.
        Initializes at the arc midpoint: x^(0) = exp(j*(l+u)/2).
        """
        s = np.asarray(s).ravel()
        l = np.asarray(l, dtype=float).ravel()
        u = np.asarray(u, dtype=float).ravel()

        step = self.step
        if step is None:
            eigs = np.linalg.eigvalsh(Ht.conj().T @ Ht)
            step = 1.0 / max(2.0 * eigs.max(), 1e-12)

        HtH = Ht.conj().T @ Ht
        Hts = Ht.conj().T @ s

        # Initialize at arc midpoint (paper: x^(0) = x^(1) = exp(j(l+u)/2))
        xk = np.exp(1j * (l + u) / 2.0)
        xkm1 = xk.copy()

        for k in range(1, self.max_iter + 1):
            # Nesterov momentum (eq.43)
            v = xk + (k - 1) / (k + 2) * (xk - xkm1)
            # Gradient of ||Ht x - s||^2 (eq.44)
            grad = 2.0 * (HtH @ v - Hts)
            # Project onto convex hull via PR2
            xnew = PR2(v - step * grad, l, u)
            if np.linalg.norm(xnew - xk) < self.tol:
                break
            xkm1 = xk
            xk = xnew

        val = float(np.linalg.norm(Ht @ xk - s) ** 2)
        return val, xk


# Register with default registry
default_registry.register_lb("gp", LBSolverGP)
