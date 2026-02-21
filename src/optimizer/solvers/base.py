"""
Abstract Solver Base Classes
============================
Defines the interfaces that all lower-bound and upper-bound solvers
must implement. Also provides a registry for solver discovery.

The optimizer module depends ONLY on numpy/scipy/cvxpy — never on
matplotlib or signal_proc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


class LBSolverBase(ABC):
    """Abstract base class for QP-LB (lower-bound) solvers.

    A lower-bound solver computes f_L(Theta) by solving the convex
    relaxation of the constant-modulus problem (Paper eq.40):

        min  ||Ht x - s||^2
        s.t. |x_n| <= 1   (relaxed from |x_n| = 1)
             Re(x_n * e^{-j*mid_n}) >= cos(half_width_n)

    Implementations
    ---------------
    - ``LBSolverCVXPY``: Interior-point via CVXPY/SCS.
    - ``LBSolverGP``: Accelerated gradient projection with PR2 (FISTA).
    """

    name: ClassVar[str] = "base_lb"

    @abstractmethod
    def solve(
        self,
        Ht: NDArray,
        s: NDArray,
        l: NDArray,
        u: NDArray,
    ) -> tuple[float, NDArray | None]:
        """Solve the QP-LB convex relaxation.

        Parameters
        ----------
        Ht : ndarray, shape (K, N)
            Scaled channel matrix  Ht = sqrt(PT/N) * H.
        s : ndarray, shape (K,)
            Symbol vector for one time slot.
        l, u : ndarray, shape (N,)
            Lower/upper angle bounds defining the feasible arc.

        Returns
        -------
        lb_value : float
            Lower bound on the objective. ``np.inf`` if infeasible.
        x_opt : ndarray or None
            Optimal (relaxed) solution, or None if infeasible.
        """
        ...


class UBSolverBase(ABC):
    """Abstract base class for QP-UB (upper-bound) solvers.

    An upper-bound solver computes f_U(Theta) by solving the non-convex
    constant-modulus problem locally (Paper eq.42):

        min  ||Ht x - s||^2
        s.t. |x_n| = 1   (constant modulus, exact)
             Re(x_n * e^{-j*mid_n}) >= cos(half_width_n)

    The solver is initialized with PR1(x_l), the projection of the
    lower-bound solution onto the feasible arc.

    Implementations
    ---------------
    - ``UBSolverSLSQP``: Local optimization via scipy SLSQP.
    - ``UBSolverGP``: Gradient projection with PR1.
    """

    name: ClassVar[str] = "base_ub"

    @abstractmethod
    def solve(
        self,
        Ht: NDArray,
        s: NDArray,
        l: NDArray,
        u: NDArray,
        x_init: NDArray,
    ) -> tuple[float, NDArray]:
        """Solve the QP-UB non-convex problem locally.

        Parameters
        ----------
        Ht : ndarray, shape (K, N)
            Scaled channel matrix.
        s : ndarray, shape (K,)
            Symbol vector for one time slot.
        l, u : ndarray, shape (N,)
            Angle bounds.
        x_init : ndarray, shape (N,)
            Initial feasible point (typically PR1(x_l)).

        Returns
        -------
        ub_value : float
            Upper bound on the objective.
        x_opt : ndarray, shape (N,)
            Feasible constant-modulus solution.
        """
        ...


class SolverRegistry:
    """Registry mapping solver names to classes for dynamic dispatch.

    Usage
    -----
    >>> reg = SolverRegistry()
    >>> reg.register_lb("cvxpy", LBSolverCVXPY)
    >>> lb_solver = reg.get_lb("cvxpy")
    """

    def __init__(self) -> None:
        self._lb_solvers: dict[str, type[LBSolverBase]] = {}
        self._ub_solvers: dict[str, type[UBSolverBase]] = {}

    # -- LB registration --
    def register_lb(self, name: str, cls: type[LBSolverBase]) -> None:
        """Register a lower-bound solver class by name."""
        self._lb_solvers[name.lower()] = cls

    def get_lb(self, name: str, **kwargs) -> LBSolverBase:
        """Instantiate a registered lower-bound solver."""
        key = name.lower()
        if key not in self._lb_solvers:
            available = ", ".join(sorted(self._lb_solvers))
            raise KeyError(
                f"Unknown LB solver '{name}'. Available: {available}"
            )
        return self._lb_solvers[key](**kwargs)

    # -- UB registration --
    def register_ub(self, name: str, cls: type[UBSolverBase]) -> None:
        """Register an upper-bound solver class by name."""
        self._ub_solvers[name.lower()] = cls

    def get_ub(self, name: str, **kwargs) -> UBSolverBase:
        """Instantiate a registered upper-bound solver."""
        key = name.lower()
        if key not in self._ub_solvers:
            available = ", ".join(sorted(self._ub_solvers))
            raise KeyError(
                f"Unknown UB solver '{name}'. Available: {available}"
            )
        return self._ub_solvers[key](**kwargs)

    @property
    def available_lb(self) -> list[str]:
        return sorted(self._lb_solvers)

    @property
    def available_ub(self) -> list[str]:
        return sorted(self._ub_solvers)


# Global default registry — populated when solver sub-packages are imported.
default_registry = SolverRegistry()
