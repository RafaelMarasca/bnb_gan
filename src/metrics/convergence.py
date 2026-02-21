"""
Convergence Metric
==================
Tracks BnB convergence: LB/UB gap vs. iteration number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import MetricBase, MetricResult


class ConvergenceMetric(MetricBase):
    """Convergence analysis from BnB iteration histories.

    Computes:
      - Gap trajectory (UB - LB per iteration)
      - Final gap
      - Number of iterations to reach a given tolerance
    """

    @property
    def name(self) -> str:
        return "convergence"

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute convergence metrics.

        Parameters (via kwargs)
        -----------------------
        lb_history : list[float]
            Lower bound at each BnB iteration.
        ub_history : list[float]
            Upper bound at each BnB iteration.
        tol : float, optional (default 1e-3)
            Tolerance threshold for counting iterations-to-convergence.

        Returns
        -------
        MetricResult with values:
            lb_history, ub_history, gap_history, final_gap, iters_to_tol
        """
        lb = np.asarray(kwargs["lb_history"], dtype=float)
        ub = np.asarray(kwargs["ub_history"], dtype=float)
        tol = float(kwargs.get("tol", 1e-3))

        gap = ub - lb
        final_gap = float(gap[-1]) if len(gap) > 0 else np.inf

        # Iterations to reach tolerance
        reached = np.where(gap <= tol)[0]
        iters_to_tol = int(reached[0]) + 1 if len(reached) > 0 else len(gap)

        return MetricResult(
            name=self.name,
            values={
                "lb_history": lb.tolist(),
                "ub_history": ub.tolist(),
                "gap_history": gap.tolist(),
                "final_gap": final_gap,
                "iters_to_tol": iters_to_tol,
            },
            metadata={"tol": tol, "n_iterations": len(gap)},
        )
