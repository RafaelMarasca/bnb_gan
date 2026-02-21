"""
Math Helpers
============
Low-level mathematical utilities used across the package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def angle_diff(a1: float | NDArray, a2: float | NDArray) -> float | NDArray:
    """Signed angular difference (a1 - a2), wrapped to (-pi, pi].

    Parameters
    ----------
    a1, a2 : float or ndarray
        Angles in radians.

    Returns
    -------
    float or ndarray
        The wrapped difference in (-pi, pi].
    """
    d = (a1 - a2) % (2.0 * np.pi)
    if np.isscalar(d):
        return d - 2.0 * np.pi if d > np.pi else d
    return np.where(d > np.pi, d - 2.0 * np.pi, d)


def compute_lipschitz_step(Ht: NDArray) -> float:
    """Compute the gradient step size 1/L where L = 2 * lambda_max(Ht^H Ht).

    This is the reciprocal of the Lipschitz constant of grad ||Ht x - s||^2.

    Parameters
    ----------
    Ht : ndarray, shape (K, N)
        Scaled channel matrix.

    Returns
    -------
    float
        Step size = 1 / (2 * lambda_max).
    """
    eigs = np.linalg.eigvalsh(Ht.conj().T @ Ht)
    return 1.0 / max(2.0 * eigs.max(), 1e-12)


def initial_angle_bounds(
    x0: NDArray, epsilon: float
) -> tuple[NDArray, NDArray]:
    """Compute initial angle bounds from similarity constraint (eq.30).

    Parameters
    ----------
    x0 : ndarray, shape (N,)
        Unit-modulus reference waveform column.
    epsilon : float
        Similarity tolerance, 0 <= epsilon <= 2.

    Returns
    -------
    l, u : ndarray, shape (N,)
        Lower and upper angle bounds in radians.
    """
    eps = min(float(epsilon), 2.0)
    hw0 = np.arccos(np.clip(1.0 - eps**2 / 2.0, -1.0, 1.0))
    ang0 = np.angle(x0)
    return ang0 - hw0, ang0 + hw0
