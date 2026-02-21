"""
Pulse Compression
=================
FFT-based pulse compression with Taylor windowing, matching the
paper's 3-step frequency-domain method (Fig.9 logic from experiments.py).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def autocorrelation(waveform: NDArray) -> NDArray:
    """Compute the aperiodic autocorrelation of a waveform.

    Parameters
    ----------
    waveform : ndarray, shape (N,), complex

    Returns
    -------
    ndarray, shape (2*N - 1,), complex
        Full autocorrelation r[k] for k = -(N-1) .. (N-1).
    """
    waveform = np.asarray(waveform).ravel()
    N = len(waveform)
    r = np.correlate(waveform, waveform, mode="full")
    return r


def pulse_compress(
    waveform: NDArray,
    n_fft: int = 1024,
    taylor_nbar: int = 4,
    taylor_sll: float = 40.0,
) -> tuple[NDArray, NDArray]:
    """Compute the pulse compression profile of a waveform.

    Implements the paper's 3-step frequency-domain windowing method:
      1. FFT of the waveform.
      2. Frequency-domain matched filter (|F|^2) with Taylor window.
      3. IFFT to get the high-resolution range profile.

    This matches the logic in ``experiments.py:plot_pulse_compression``
    exactly, extracted as a reusable function.

    Parameters
    ----------
    waveform : ndarray, shape (N,), complex
        Single transmit element's waveform vector.
    n_fft : int
        Number of IFFT points for high-resolution range profile.
    taylor_nbar : int
        Taylor window nbar parameter.
    taylor_sll : float
        Taylor window sidelobe level (dB).

    Returns
    -------
    bin_index : ndarray, shape (n_fft,)
        IFFT bin indices centered at zero.
    magnitude_dB : ndarray, shape (n_fft,)
        Normalized magnitude in dB (peak = 0 dB).
    """
    from scipy.fft import fft, ifft, fftshift
    import scipy.signal as signal

    waveform = np.asarray(waveform).ravel()
    N = len(waveform)

    # Step 1: FFT of the waveform
    f = fft(waveform)

    # Step 2: Matched filter (|F|^2) weighted by Taylor window
    tw = signal.windows.taylor(N, nbar=taylor_nbar, sll=taylor_sll, norm=False)
    fp = f * np.conj(f) * tw

    # Step 3: IFFT to time-domain range profile (zero-padded to n_fft)
    rp = ifft(fp, n_fft)

    # Shift zero-frequency to center and normalize
    rp_shifted = fftshift(rp)
    rp_abs = np.abs(rp_shifted)
    peak = np.max(rp_abs)
    magnitude_dB = 20.0 * np.log10(rp_abs / peak + 1e-12)

    bin_index = np.arange(-n_fft / 2, n_fft / 2)

    return bin_index, magnitude_dB
