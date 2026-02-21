"""
UB Solver — SLSQP (Local Non-convex)
=====================================
Solves QP-UB (eq.42) via scipy SLSQP, initialized with PR1(x_l).

    min  ||Ht x - s||^2
    s.t. |x_n|^2 = 1   (constant modulus — equality)
         Re(x_n * e^{-j*mid_n}) >= cos(hw_n)

Matches ``bnb_comprehensive.solve_ub_slsqp`` exactly.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as scipy_minimize

from ..projections import PR1
from .base import UBSolverBase, default_registry


class UBSolverSLSQP(UBSolverBase):
    """Upper-bound solver using scipy SLSQP (analogous to MATLAB fmincon).

    Parameters
    ----------
    max_iter : int
        Maximum SLSQP iterations.
    ftol : float
        Function tolerance for SLSQP convergence.
    """

    name: ClassVar[str] = "slsqp"

    def __init__(self, max_iter: int = 100, ftol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.ftol = ftol

    def solve(
        self,
        Ht: NDArray,
        s: NDArray,
        l: NDArray,
        u: NDArray,
        x_init: NDArray,
    ) -> tuple[float, NDArray]:
        """Solve QP-UB via scipy SLSQP (Paper eq.42).

        Optimizes over the real/imaginary split of x, with:
          - Equality constraint: |x_n|^2 = 1 for all n.
          - Inequality constraint: Re(x_n e^{-j mid_n}) >= cos(hw_n).

        After SLSQP, the solution is projected via PR1 to enforce
        strict feasibility (constant modulus + arc bounds).
        """
        N = Ht.shape[1]
        s = np.asarray(s).ravel()
        l = np.asarray(l, dtype=float).ravel()
        u = np.asarray(u, dtype=float).ravel()
        x_init = np.asarray(x_init, dtype=complex).ravel()

        mid = (l + u) / 2.0
        hw = (u - l) / 2.0
        ph = np.exp(-1j * mid)

        def obj(z: NDArray) -> float:
            xc = z[:N] + 1j * z[N:]
            return float(np.linalg.norm(Ht @ xc - s) ** 2)

        def ceq(z: NDArray) -> NDArray:
            """Equality: |x_n|^2 - 1 = 0."""
            xc = z[:N] + 1j * z[N:]
            return np.abs(xc) ** 2 - 1.0

        def cineq(z: NDArray) -> NDArray:
            """Inequality: Re(x_n * e^{-j mid_n}) - cos(hw_n) >= 0."""
            xc = z[:N] + 1j * z[N:]
            return np.real(xc * ph) - np.cos(hw)

        z0 = np.concatenate([x_init.real, x_init.imag])
        res = scipy_minimize(
            obj,
            z0,
            method="SLSQP",
            constraints=[
                {"type": "eq", "fun": ceq},
                {"type": "ineq", "fun": cineq},
            ],
            options={"maxiter": self.max_iter, "ftol": self.ftol},
        )

        # Enforce strict feasibility via PR1
        xopt = PR1(res.x[:N] + 1j * res.x[N:], l, u)
        val = float(np.linalg.norm(Ht @ xopt - s) ** 2)
        return val, xopt


# Register with default registry
default_registry.register_ub("slsqp", UBSolverSLSQP)
