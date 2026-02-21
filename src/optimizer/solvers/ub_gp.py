"""
UB Solver — Gradient Projection (PR1)
======================================
Solves QP-UB via gradient projection with PR1 (no momentum).

    x^{k+1} = PR1(x^k - step * grad_f(x^k))

Paper note (after eq.44): "we use x^(k) instead of the interpolated
point v, and replace PR2 with PR1".

Matches ``bnb_comprehensive.solve_ub_gp`` exactly.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ..projections import PR1
from .base import UBSolverBase, default_registry


class UBSolverGP(UBSolverBase):
    """Upper-bound solver using gradient projection with PR1.

    Unlike the LB-GP solver, this uses no Nesterov momentum and projects
    onto the arc (PR1) instead of the convex hull (PR2), maintaining
    constant-modulus feasibility at every iterate.

    Parameters
    ----------
    step : float or None
        Gradient step size. If None, computed from eigenvalues of Ht^H Ht.
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
        x_init: NDArray,
    ) -> tuple[float, NDArray]:
        """Solve QP-UB via gradient projection with PR1.

        Tracks the best feasible objective seen across all iterates
        (non-monotone GP can occasionally increase due to PR1 projection).
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
        xk = np.asarray(x_init, dtype=complex).ravel().copy()

        best_val = float(np.linalg.norm(Ht @ xk - s) ** 2)
        best_x = xk.copy()

        for _ in range(self.max_iter):
            grad = 2.0 * (HtH @ xk - Hts)
            xnew = PR1(xk - step * grad, l, u)
            v = float(np.linalg.norm(Ht @ xnew - s) ** 2)
            if v < best_val:
                best_val = v
                best_x = xnew.copy()
            if np.linalg.norm(xnew - xk) < self.tol:
                break
            xk = xnew

        return best_val, best_x


# Register with default registry
default_registry.register_ub("gp", UBSolverGP)
