"""
Rate Metric
============
Communication sum-rate: R = sum_i log2(1 + gamma_i).
Paper eqs.4-5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import MetricBase, MetricResult


class RateMetric(MetricBase):
    """Sum-rate calculator for the joint Radar-Communication system."""

    @property
    def name(self) -> str:
        return "sum_rate"

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute sum-rate R = sum_i log2(1 + gamma_i).

        Parameters (via kwargs)
        -----------------------
        H : ndarray (K, N) — channel matrix
        X : ndarray (N, L) — waveform matrix
        S : ndarray (K, L) — symbol matrix
        N0 : float — noise power

        Returns
        -------
        MetricResult with values: sum_rate (float), per_user_rate (list)
        """
        H = np.asarray(kwargs["H"])
        X = np.asarray(kwargs["X"])
        S = np.asarray(kwargs["S"])
        N0 = float(kwargs["N0"])

        K = S.shape[0]
        HX = H @ X  # (K, L)

        per_user = []
        total = 0.0
        for i in range(K):
            sig = np.mean(np.abs(S[i]) ** 2)  # = 1 for unit-power QPSK
            mui = np.mean(np.abs(HX[i] - S[i]) ** 2)
            r_i = np.log2(1.0 + sig / (mui + N0))
            per_user.append(float(r_i))
            total += r_i

        return MetricResult(
            name=self.name,
            values={"sum_rate": float(total), "per_user_rate": per_user},
            metadata={"K": K, "N0": N0},
        )


def sum_rate(H: NDArray, X: NDArray, S: NDArray, N0: float) -> float:
    """Functional API matching ``bnb_comprehensive.sum_rate``.

    R = sum_i log2(1 + gamma_i)
    gamma_i = E|s_{i,j}|^2 / (E|h_i^T x_j - s_{i,j}|^2 + N0)
    """
    metric = RateMetric()
    result = metric.compute(H=H, X=X, S=S, N0=N0)
    return result.values["sum_rate"]
