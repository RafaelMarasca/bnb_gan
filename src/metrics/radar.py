"""
Radar Metrics
=============
ISL (Integrated Sidelobe Level) and PSL (Peak Sidelobe Level)
computed from autocorrelation / pulse compression profiles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import MetricBase, MetricResult


def _autocorrelation(waveform: NDArray) -> NDArray:
    """Aperiodic autocorrelation (full, length 2N-1)."""
    w = np.asarray(waveform).ravel()
    return np.correlate(w, w, mode="full")


class ISLMetric(MetricBase):
    """Integrated Sidelobe Level metric.

    ISL = sum of |r(k)|^2 for all k != 0, where r is the autocorrelation.
    Normalized: ISL_dB = 10*log10(ISL / |r(0)|^2).
    """

    @property
    def name(self) -> str:
        return "ISL"

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute Integrated Sidelobe Level.

        Parameters (via kwargs)
        -----------------------
        waveform : ndarray (N,), complex — transmit waveform

        Returns
        -------
        MetricResult with values: ISL_dB (float), ISL_linear (float)
        """
        waveform = np.asarray(kwargs["waveform"]).ravel()
        r = _autocorrelation(waveform)
        N = len(waveform)
        center = N - 1  # index of r(0)

        mainlobe_power = float(np.abs(r[center]) ** 2)
        sidelobe_power = float(
            np.sum(np.abs(r[:center]) ** 2) + np.sum(np.abs(r[center + 1:]) ** 2)
        )

        isl_linear = sidelobe_power / mainlobe_power if mainlobe_power > 0 else np.inf
        isl_dB = 10.0 * np.log10(isl_linear + 1e-30)

        return MetricResult(
            name=self.name,
            values={"ISL_dB": float(isl_dB), "ISL_linear": float(isl_linear)},
            metadata={"N": N},
        )


class PSLMetric(MetricBase):
    """Peak Sidelobe Level metric.

    PSL = max |r(k)| for k != 0, where r is the autocorrelation.
    Normalized: PSL_dB = 20*log10(PSL / |r(0)|).
    """

    @property
    def name(self) -> str:
        return "PSL"

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute Peak Sidelobe Level.

        Parameters (via kwargs)
        -----------------------
        waveform : ndarray (N,), complex — transmit waveform

        Returns
        -------
        MetricResult with values: PSL_dB (float), PSL_linear (float)
        """
        waveform = np.asarray(kwargs["waveform"]).ravel()
        r = _autocorrelation(waveform)
        N = len(waveform)
        center = N - 1

        mainlobe = float(np.abs(r[center]))
        sidelobes = np.concatenate([np.abs(r[:center]), np.abs(r[center + 1:])])
        peak_sidelobe = float(np.max(sidelobes)) if len(sidelobes) > 0 else 0.0

        psl_linear = peak_sidelobe / mainlobe if mainlobe > 0 else np.inf
        psl_dB = 20.0 * np.log10(psl_linear + 1e-30)

        return MetricResult(
            name=self.name,
            values={"PSL_dB": float(psl_dB), "PSL_linear": float(psl_linear)},
            metadata={"N": N},
        )
