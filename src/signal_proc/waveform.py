"""
Waveform Generation
===================
Generate reference waveforms, channel matrices, and symbol matrices
matching the paper's simulation setup.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_channel(K: int, N: int) -> NDArray:
    """Generate i.i.d. CN(0,1) Rayleigh fading channel matrix.

    Parameters
    ----------
    K : int
        Number of users (rows).
    N : int
        Number of transmit antennas (columns).

    Returns
    -------
    ndarray, shape (K, N), complex
        Channel matrix H.
    """
    return (np.random.randn(K, N) + 1j * np.random.randn(K, N)) / np.sqrt(2)


def generate_symbols(K: int, L: int) -> NDArray:
    """Generate unit-power QPSK symbol matrix.

    Parameters
    ----------
    K : int
        Number of users.
    L : int
        Frame length.

    Returns
    -------
    ndarray, shape (K, L), complex
        Symbol matrix S with QPSK symbols.
    """
    alphabet = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    return np.random.choice(alphabet, (K, L))


def generate_chirp(
    N: int,
    L: int,
    PT: float,
    norm: str = "power",
) -> NDArray:
    """Generate orthogonal chirp (LFM) reference waveform matrix (eq. 33).

    Phase: phi(k, t) = 2*pi*(k-1)*(t-1)/L + pi*(t-1)^2/L

    Two normalization options:

    * ``"power"`` (default) — each entry has modulus ``sqrt(PT/N)``,
      so total power per snapshot equals *PT*.  This is the normalization
      expected by the optimizer (``optimize_waveform``).

    * ``"unitary"`` — each entry has modulus ``1/sqrt(N*L)``, matching
      the exact form of eq. 33 in the paper.  Useful for stand-alone
      pulse-compression plots where absolute power doesn't matter.

    Parameters
    ----------
    N : int
        Number of transmit antennas (rows).
    L : int
        Number of time samples / frame length (columns).
    PT : float
        Total transmit power (used only when *norm* = ``"power"``).
    norm : {"power", "unitary"}, optional
        Normalization convention.  Default ``"power"``.

    Returns
    -------
    ndarray, shape (N, L), complex
        Reference waveform matrix X0.
    """
    if norm not in ("power", "unitary"):
        raise ValueError(f"norm must be 'power' or 'unitary', got {norm!r}")

    k = np.arange(N)[:, None]  # antenna index (N, 1)
    t = np.arange(L)[None, :]  # time index    (1, L)
    phase = 2.0 * np.pi * k * t / L + np.pi * t**2 / L

    if norm == "power":
        sc = np.sqrt(PT / N)
    else:  # unitary
        sc = 1.0 / np.sqrt(N * L)

    return sc * np.exp(1j * phase)
