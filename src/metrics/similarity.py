"""
Waveform Similarity Metric
==========================
Measures how far an optimized waveform deviates from the reference
waveform, and whether the similarity constraint is satisfied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import MetricBase, MetricResult


class WaveformSimilarityMetric(MetricBase):
    r"""Waveform similarity metric.

    Computes element-wise and aggregate distance between the optimized
    waveform *x* and the reference waveform *x0*:

    * :math:`\|x - x_0\|_2` — Euclidean (L2) distance
    * :math:`\|x - x_0\|_\infty` — Chebyshev (L-inf) distance
    * :math:`\|x - x_0\|_2 / \sqrt{N}` — per-element RMS distance
    * Feasibility flag: whether :math:`\|x - x_0\|_2 \le \varepsilon`

    Works for both single-column vectors (N,) and full waveform matrices
    (N, L).  When matrices are supplied, metrics are reported per-column
    (averaged) **and** for the worst-case column.
    """

    @property
    def name(self) -> str:
        return "WaveformSimilarity"

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute waveform similarity.

        Parameters (via kwargs)
        -----------------------
        x : ndarray, shape (N,) or (N, L)
            Optimized waveform (unit-modulus or power-normalized).
        x0 : ndarray, shape (N,) or (N, L)
            Reference waveform (same normalization as *x*).
        epsilon : float
            Similarity tolerance bound.

        Returns
        -------
        MetricResult
            ``values`` dict contains:

            - ``l2_dist`` : float — ||x - x0||_2 (or mean over columns)
            - ``linf_dist`` : float — ||x - x0||_inf (or max over columns)
            - ``rms_dist`` : float — ||x - x0||_2 / sqrt(N)
            - ``epsilon`` : float — the ε bound supplied
            - ``feasible`` : bool — True if every column satisfies the bound
            - ``margin`` : float — ε − max column ||·||_2  (positive = feasible)
        """
        x = np.asarray(kwargs["x"])
        x0 = np.asarray(kwargs["x0"])
        epsilon: float = float(kwargs["epsilon"])

        if x.shape != x0.shape:
            raise ValueError(
                f"Shape mismatch: x {x.shape} vs x0 {x0.shape}"
            )

        # Ensure 2-D  (N, L)
        if x.ndim == 1:
            x = x[:, None]
            x0 = x0[:, None]

        N, L = x.shape
        diff = x - x0

        # Per-column L2 norms  (L,)
        col_l2 = np.sqrt(np.sum(np.abs(diff) ** 2, axis=0))
        # Per-column L-inf norms  (L,)
        col_linf = np.max(np.abs(diff), axis=0)

        mean_l2 = float(np.mean(col_l2))
        max_l2 = float(np.max(col_l2))
        mean_linf = float(np.mean(col_linf))
        max_linf = float(np.max(col_linf))
        rms = float(mean_l2 / np.sqrt(N))

        feasible = bool(max_l2 <= epsilon + 1e-10)  # small tolerance
        margin = float(epsilon - max_l2)

        return MetricResult(
            name=self.name,
            values={
                "l2_dist": mean_l2,
                "l2_dist_max": max_l2,
                "linf_dist": mean_linf,
                "linf_dist_max": max_linf,
                "rms_dist": rms,
                "epsilon": epsilon,
                "feasible": feasible,
                "margin": margin,
            },
            metadata={"N": N, "L": L},
        )
