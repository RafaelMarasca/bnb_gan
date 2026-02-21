"""
Branch-and-Bound Solver
========================
Generic BnB framework implementing Algorithm 2 from the paper.

Supports pluggable strategies for:
  - Branching: ARS (eq.36) vs BRS (eq.35)
  - Lower bounding: CVXPY interior-point or GP with PR2
  - Upper bounding: SLSQP or GP with PR1

Matches ``bnb_comprehensive.bnb_solve`` exactly in logic and solver
integration, but structured as a class with pluggable solver objects.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.config import BnBConfig
from ..utils.math_helpers import compute_lipschitz_step, initial_angle_bounds
from .node import BnBNode
from .projections import PR1
from .solvers.base import LBSolverBase, UBSolverBase, default_registry


@dataclass
class BnBResult:
    """Result container for a single-column BnB optimization.

    Attributes
    ----------
    x_opt : ndarray or None
        Optimal feasible (constant-modulus) solution vector.
    objective : float
        Best objective value achieved (global UB at termination).
    lb_history : list[float]
        Global lower bound at each iteration.
    ub_history : list[float]
        Global upper bound at each iteration.
    n_iterations : int
        Total BnB iterations performed.
    """

    x_opt: NDArray | None
    objective: float
    lb_history: list[float]
    ub_history: list[float]
    n_iterations: int


class BranchAndBoundSolver:
    """Branch-and-Bound solver for one column of the constant-modulus problem.

    This class orchestrates Algorithm 2: it manages the priority queue of
    BnBNode objects, dispatches to pluggable LB/UB solvers, and applies
    the chosen subdivision rule (ARS or BRS).

    Parameters
    ----------
    config : BnBConfig
        Algorithm configuration (rule, solver names, tolerances).
    lb_solver : LBSolverBase, optional
        Explicit lower-bound solver instance. If None, resolved from
        ``config.lb_solver`` via the default registry.
    ub_solver : UBSolverBase, optional
        Explicit upper-bound solver instance. If None, resolved from
        ``config.ub_solver`` via the default registry.

    Example
    -------
    >>> from src.utils.config import BnBConfig
    >>> solver = BranchAndBoundSolver(BnBConfig(rule="ARS"))
    >>> result = solver.solve(Ht, s, x0, epsilon=1.0)
    """

    def __init__(
        self,
        config: BnBConfig | None = None,
        lb_solver: LBSolverBase | None = None,
        ub_solver: UBSolverBase | None = None,
    ) -> None:
        self.config = config or BnBConfig()
        self._lb_solver = lb_solver
        self._ub_solver = ub_solver

    # ------------------------------------------------------------------
    # Lazy solver initialisation (allows step-size injection after init)
    # ------------------------------------------------------------------

    def _get_lb_solver(self, step: float) -> LBSolverBase:
        """Return LB solver, creating from registry if needed."""
        if self._lb_solver is not None:
            # Inject step if it's a GP solver and step was not set
            if hasattr(self._lb_solver, "step") and self._lb_solver.step is None:
                self._lb_solver.step = step
            return self._lb_solver
        cfg = self.config
        kwargs: dict[str, Any] = {}
        if cfg.lb_solver == "gp":
            kwargs = dict(step=step, max_iter=cfg.gp_max_iter, tol=cfg.gp_tol)
        return default_registry.get_lb(cfg.lb_solver, **kwargs)

    def _get_ub_solver(self, step: float) -> UBSolverBase:
        """Return UB solver, creating from registry if needed."""
        if self._ub_solver is not None:
            if hasattr(self._ub_solver, "step") and self._ub_solver.step is None:
                self._ub_solver.step = step
            return self._ub_solver
        cfg = self.config
        kwargs: dict[str, Any] = {}
        if cfg.ub_solver == "gp":
            kwargs = dict(step=step, max_iter=cfg.gp_max_iter, tol=cfg.gp_tol)
        return default_registry.get_ub(cfg.ub_solver, **kwargs)

    # ------------------------------------------------------------------
    # Main solve method — Algorithm 2
    # ------------------------------------------------------------------

    def solve(
        self,
        Ht: NDArray,
        s: NDArray,
        x0: NDArray,
        epsilon: float,
    ) -> BnBResult:
        """Run the BnB algorithm for a single waveform column.

        Parameters
        ----------
        Ht : ndarray, shape (K, N)
            Scaled channel matrix: sqrt(PT/N) * H.
        s : ndarray, shape (K,)
            Symbol vector for one time slot.
        x0 : ndarray, shape (N,)
            Unit-modulus reference waveform column.
        epsilon : float
            Similarity tolerance (0 <= epsilon <= 2).

        Returns
        -------
        BnBResult
            Contains optimal solution, objective, and convergence history.
        """
        cfg = self.config
        eps = min(float(epsilon), 2.0)
        s = np.asarray(s).ravel()
        x0 = np.asarray(x0).ravel()

        # GP step size: 1 / (2 * lambda_max(Ht^H Ht))
        step = compute_lipschitz_step(Ht)

        # Resolve solvers (may inject step into GP solvers)
        lb_solver = self._get_lb_solver(step)
        ub_solver = self._get_ub_solver(step)

        # Initial angle bounds from similarity constraint (eq.30)
        l0, u0 = initial_angle_bounds(x0, eps)

        # =============== Root node ===============
        root = BnBNode(l0, u0)
        root.LB, root.x_l = lb_solver.solve(Ht, s, root.l, root.u)

        if root.x_l is None:
            return BnBResult(
                x_opt=None,
                objective=np.inf,
                lb_history=[],
                ub_history=[],
                n_iterations=0,
            )

        root.UB, root.x_u = ub_solver.solve(
            Ht, s, root.l, root.u, PR1(root.x_l, root.l, root.u)
        )

        gUB = root.UB
        gLB = root.LB
        best_x = root.x_u.copy()

        pq: queue.PriorityQueue[BnBNode] = queue.PriorityQueue()
        pq.put(root)
        lb_hist = [gLB]
        ub_hist = [gUB]

        # =============== Main BnB loop ===============
        n_iter = 0
        for it in range(cfg.max_iter):
            if pq.empty() or gUB - gLB <= cfg.tol:
                break

            node = pq.get()
            if node.LB > gUB:  # pruning
                continue

            # --- Subdivision rule ---
            if cfg.rule == "ARS":
                # eq.36: split dimension with max |x_u(n) - x_l(n)|
                idx = int(np.argmax(np.abs(node.x_u - node.x_l)))
            else:
                # BRS eq.35: split dimension with max arc width
                idx = int(np.argmax(node.u - node.l))

            mid_angle = (node.l[idx] + node.u[idx]) / 2.0

            # --- Create two children ---
            for c in range(2):
                cl = node.l.copy()
                cu = node.u.copy()
                if c == 0:
                    cu[idx] = mid_angle
                else:
                    cl[idx] = mid_angle

                child = BnBNode(cl, cu)
                child.LB, child.x_l = lb_solver.solve(Ht, s, cl, cu)

                if child.x_l is not None and child.LB <= gUB:
                    child.UB, child.x_u = ub_solver.solve(
                        Ht, s, cl, cu, PR1(child.x_l, cl, cu)
                    )
                    if child.UB < gUB:
                        gUB = child.UB
                        best_x = child.x_u.copy()
                    pq.put(child)

            # --- Update global LB (peek at top of PQ) ---
            if not pq.empty():
                top = pq.get()
                gLB = top.LB
                pq.put(top)

            lb_hist.append(gLB)
            ub_hist.append(gUB)
            n_iter = it + 1

            if cfg.verbose and n_iter % cfg.verbose_interval == 0:
                print(
                    f"      iter {n_iter:4d}: LB={gLB:.6f}  UB={gUB:.6f}  "
                    f"gap={gUB - gLB:.6f}"
                )

        return BnBResult(
            x_opt=best_x,
            objective=gUB,
            lb_history=lb_hist,
            ub_history=ub_hist,
            n_iterations=n_iter,
        )


# =========================================================================
# Convenience functional API (mirrors bnb_comprehensive.bnb_solve)
# =========================================================================

def bnb_solve(
    Ht: NDArray,
    s: NDArray,
    x0: NDArray,
    epsilon: float,
    rule: str = "ARS",
    lb_solver: str = "cvxpy",
    ub_solver: str = "slsqp",
    tol: float = 1e-3,
    max_iter: int = 200,
    gp_iters: int = 100,
    verbose: bool = False,
) -> tuple[NDArray | None, float, list[float], list[float]]:
    """Functional wrapper around BranchAndBoundSolver.

    Signature and return format match ``bnb_comprehensive.bnb_solve`` exactly:
        (best_x, best_obj, lb_hist, ub_hist)
    """
    config = BnBConfig(
        rule=rule,  # type: ignore[arg-type]
        lb_solver=lb_solver,  # type: ignore[arg-type]
        ub_solver=ub_solver,  # type: ignore[arg-type]
        tol=tol,
        max_iter=max_iter,
        gp_max_iter=gp_iters,
        verbose=verbose,
    )
    solver = BranchAndBoundSolver(config)
    result = solver.solve(Ht, s, x0, epsilon)
    return result.x_opt, result.objective, result.lb_history, result.ub_history
