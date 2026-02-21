"""
LB Solver — CVXPY (Interior Point)
===================================
Solves QP-LB (eq.40) via CVXPY with the SCS backend.

    min  ||Ht x - s||^2
    s.t. |x_n| <= 1                                (relaxed modulus)
         Re(x_n * e^{-j*mid_n}) >= cos(hw_n)       (convex hull)

Matches ``bnb_comprehensive.solve_lb_cvxpy`` exactly.
"""

from __future__ import annotations

from typing import ClassVar

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .base import LBSolverBase, default_registry


class LBSolverCVXPY(LBSolverBase):
    """Lower-bound solver using CVXPY interior-point (SCS).

    Parameters
    ----------
    solver_eps : float
        SCS solver tolerance (``eps`` parameter).
    """

    name: ClassVar[str] = "cvxpy"

    def __init__(self, solver_eps: float = 1e-5) -> None:
        self.solver_eps = solver_eps

    def solve(
        self,
        Ht: NDArray,
        s: NDArray,
        l: NDArray,
        u: NDArray,
    ) -> tuple[float, NDArray | None]:
        """Solve QP-LB via CVXPY/SCS (Paper eq.40).

        The convex relaxation replaces |x_n| = 1 with |x_n| <= 1 and
        adds a linear inner-product constraint that restricts x_n to
        lie inside the convex hull of the feasible arc.
        """
        N = Ht.shape[1]
        s = np.asarray(s).ravel()
        l = np.asarray(l, dtype=float).ravel()
        u = np.asarray(u, dtype=float).ravel()

        x = cp.Variable(N, complex=True)
        mid = (l + u) / 2.0
        hw = (u - l) / 2.0
        ph = np.exp(-1j * mid)

        # Constraints: |x_n| <= 1, Re(x_n * e^{-j*mid_n}) >= cos(hw_n)
        cons = [cp.abs(x) <= 1]
        for n in range(N):
            cons.append(cp.real(x[n] * ph[n]) >= np.cos(hw[n]))

        prob = cp.Problem(cp.Minimize(cp.sum_squares(Ht @ x - s)), cons)
        try:
            prob.solve(solver=cp.SCS, eps=self.solver_eps)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return float(prob.value), np.array(x.value).ravel()
        except Exception:
            pass
        return np.inf, None


# Register with default registry
default_registry.register_lb("cvxpy", LBSolverCVXPY)
