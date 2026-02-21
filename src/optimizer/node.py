"""
BnB Node
=========
Data structure for Branch-and-Bound tree nodes, storing angle bounds
and cached solutions exactly as in Algorithm 2 of the paper.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BnBNode:
    """A node in the Branch-and-Bound search tree.

    Each node represents a rectangular sub-region Theta of the feasible
    angle space, defined by element-wise bounds [l_n, u_n] for n=1..N.

    Attributes
    ----------
    l : ndarray, shape (N,)
        Lower angle bounds (radians).
    u : ndarray, shape (N,)
        Upper angle bounds (radians).
    LB : float
        Lower bound f_L(Theta) from the convex relaxation (QP-LB).
    UB : float
        Upper bound f_U(Theta) from the feasible projection (QP-UB).
    x_l : ndarray or None
        Solution of the lower-bound sub-problem (relaxed, possibly infeasible).
    x_u : ndarray or None
        Solution of the upper-bound sub-problem (feasible, constant modulus).
    """

    __slots__ = ("l", "u", "LB", "UB", "x_l", "x_u")

    def __init__(self, l: NDArray, u: NDArray) -> None:
        self.l: NDArray = l.copy()
        self.u: NDArray = u.copy()
        self.LB: float = -np.inf
        self.UB: float = np.inf
        self.x_l: NDArray | None = None
        self.x_u: NDArray | None = None

    # PriorityQueue ordering: smallest LB is popped first (best-first search)
    def __lt__(self, other: BnBNode) -> bool:
        return self.LB < other.LB

    @property
    def gap(self) -> float:
        """Current bound gap: UB - LB."""
        return self.UB - self.LB

    @property
    def arc_widths(self) -> NDArray:
        """Per-element arc widths (u - l)."""
        return self.u - self.l

    def __repr__(self) -> str:
        return (
            f"BnBNode(LB={self.LB:.6f}, UB={self.UB:.6f}, "
            f"gap={self.gap:.6f}, N={len(self.l)})"
        )
