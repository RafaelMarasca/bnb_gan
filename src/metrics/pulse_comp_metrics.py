"""
Pulse-Compression Radar Metrics
================================
Mainlobe-to-Sidelobe Ratio (MSR) and 3 dB Mainlobe Width computed
from pulse compression range profiles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft, fftshift
from scipy.signal.windows import taylor

from .base import MetricBase, MetricResult


def _pulse_compress(
    waveform: NDArray,
    n_fft: int = 160,
    taylor_nbar: int = 4,
    taylor_sll: float = 35.0,
) -> tuple[NDArray, NDArray, int]:
    """Compute pulse-compression range profile.

    Returns
    -------
    bins : ndarray (n_fft,)
        IFFT bin indices centered at 0.
    mag_dB : ndarray (n_fft,)
        Magnitude in dB, peak-normalized to 0 dB.
    center : int
        Index of the mainlobe peak in *mag_dB*.
    """
    wf = np.asarray(waveform).ravel()
    Nw = len(wf)

    F = fft(wf)
    tw = taylor(Nw, nbar=taylor_nbar, sll=taylor_sll, norm=False)
    Fp = F * np.conj(F) * tw
    rp = fftshift(ifft(Fp, n_fft))

    mag = np.abs(rp)
    peak = np.max(mag)
    mag_dB = 20.0 * np.log10(mag / peak + 1e-30)

    center = int(np.argmax(mag))
    bins = np.arange(-n_fft // 2, n_fft // 2)

    return bins, mag_dB, center


class MainlobeToSidelobeRatio(MetricBase):
    """Mainlobe-to-Sidelobe Ratio (MSR) from pulse compression.

    MSR = peak mainlobe magnitude / peak sidelobe magnitude,
    reported in dB (always ≥ 0 dB).

    The mainlobe region is defined as the contiguous set of bins
    around the peak that stay above ``mainlobe_threshold_dB``
    (default −3 dB).  Everything outside is sidelobe.
    """

    def __init__(
        self,
        n_fft: int = 160,
        taylor_nbar: int = 4,
        taylor_sll: float = 35.0,
        mainlobe_threshold_dB: float = -3.0,
    ) -> None:
        self._n_fft = n_fft
        self._taylor_nbar = taylor_nbar
        self._taylor_sll = taylor_sll
        self._ml_thresh = mainlobe_threshold_dB

    @property
    def name(self) -> str:
        return "MSR"

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute mainlobe-to-sidelobe ratio.

        Parameters (via kwargs)
        -----------------------
        waveform : ndarray (N,), complex
            Transmit waveform (one row/column of the waveform matrix).

        Returns
        -------
        MetricResult
            ``values``: MSR_dB, peak_sidelobe_dB
        """
        waveform = kwargs["waveform"]
        bins, mag_dB, center = _pulse_compress(
            waveform, self._n_fft, self._taylor_nbar, self._taylor_sll,
        )

        # Identify mainlobe region (contiguous bins above threshold)
        above = mag_dB >= self._ml_thresh
        ml_left = center
        while ml_left > 0 and above[ml_left - 1]:
            ml_left -= 1
        ml_right = center
        while ml_right < len(mag_dB) - 1 and above[ml_right + 1]:
            ml_right += 1

        # Sidelobe region = everything outside [ml_left, ml_right]
        sl_mask = np.ones(len(mag_dB), dtype=bool)
        sl_mask[ml_left: ml_right + 1] = False

        if not np.any(sl_mask):
            # Entire profile is within the mainlobe — perfect
            msr_dB = float("inf")
            peak_sl_dB = float("-inf")
        else:
            peak_sl_dB = float(np.max(mag_dB[sl_mask]))
            msr_dB = float(-peak_sl_dB)  # since mainlobe peak is 0 dB

        return MetricResult(
            name=self.name,
            values={
                "MSR_dB": msr_dB,
                "peak_sidelobe_dB": peak_sl_dB,
            },
            metadata={
                "n_fft": self._n_fft,
                "mainlobe_bins": (int(ml_left), int(ml_right)),
                "N_waveform": len(np.asarray(waveform).ravel()),
            },
        )


class MainlobeWidthMetric(MetricBase):
    """3 dB mainlobe width from pulse compression.

    Measures the width (in IFFT bins) of the mainlobe at −3 dB below
    the peak.  Narrower mainlobe → better radar range resolution.

    Optionally reports the −6 dB and −10 dB widths as well.
    """

    def __init__(
        self,
        n_fft: int = 160,
        taylor_nbar: int = 4,
        taylor_sll: float = 35.0,
    ) -> None:
        self._n_fft = n_fft
        self._taylor_nbar = taylor_nbar
        self._taylor_sll = taylor_sll

    @property
    def name(self) -> str:
        return "MainlobeWidth"

    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute mainlobe width at multiple thresholds.

        Parameters (via kwargs)
        -----------------------
        waveform : ndarray (N,), complex
            Transmit waveform.

        Returns
        -------
        MetricResult
            ``values``: width_3dB, width_6dB, width_10dB (in bins)
        """
        waveform = kwargs["waveform"]
        bins, mag_dB, center = _pulse_compress(
            waveform, self._n_fft, self._taylor_nbar, self._taylor_sll,
        )

        widths: dict[str, float] = {}
        for label, thresh in [("3dB", -3.0), ("6dB", -6.0), ("10dB", -10.0)]:
            above = mag_dB >= thresh
            # Walk left from peak
            left = center
            while left > 0 and above[left - 1]:
                left -= 1
            # Walk right from peak
            right = center
            while right < len(mag_dB) - 1 and above[right + 1]:
                right += 1
            widths[f"width_{label}"] = float(right - left + 1)

        return MetricResult(
            name=self.name,
            values=widths,
            metadata={
                "n_fft": self._n_fft,
                "N_waveform": len(np.asarray(waveform).ravel()),
            },
        )
