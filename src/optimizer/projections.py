"""
Projection Operators
====================
PR1 (projection onto arc, eq.41) and PR2 (projection onto convex hull,
Appendix eq.62) — critical building blocks for the BnB algorithm.

These follow ``bnb_comprehensive.py`` exactly, with the corrected
denominator in eq.62 (|T|^2, not |T|).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.math_helpers import angle_diff


# =========================================================================
# PR1 — Projection onto Arc  (Paper eq.41)
# =========================================================================

def PR1(x: NDArray, l: NDArray, u: NDArray) -> NDArray:
    """Element-wise projection onto unit-circle arcs [l_n, u_n].

    For each element n:
      - If arg(x_n) in [l_n, u_n]: normalize to unit modulus.
      - Otherwise: snap to whichever endpoint (l_n or u_n) is closer.

    Parameters
    ----------
    x : ndarray, shape (N,), complex
        Input vector.
    l, u : ndarray, shape (N,), float
        Lower and upper angle bounds (radians).

    Returns
    -------
    ndarray, shape (N,), complex
        Projected constant-modulus vector within angle bounds.
    """
    x = np.asarray(x, dtype=complex).ravel()
    l = np.asarray(l, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    N = len(x)
    out = np.empty(N, dtype=complex)

    mid = (l + u) * 0.5
    hw = (u - l) * 0.5
    ang = np.angle(x)

    for i in range(N):
        rot = angle_diff(ang[i], mid[i])
        if abs(rot) <= hw[i] + 1e-10:
            # Inside arc → normalize to unit modulus
            out[i] = (
                x[i] / abs(x[i]) if abs(x[i]) > 1e-12
                else np.exp(1j * mid[i])
            )
        else:
            # Outside arc → snap to nearer boundary
            dl = abs(angle_diff(ang[i], l[i]))
            du = abs(angle_diff(ang[i], u[i]))
            out[i] = np.exp(1j * l[i]) if dl <= du else np.exp(1j * u[i])
    return out


# =========================================================================
# PR2 — Projection onto Convex Hull  (Paper Appendix, eq.62)
#
# The convex hull Q(theta_n) of arc [l, u] on the unit circle is the
# circular segment bounded by the chord and the arc.
#
# Five regions M1-M5 are handled (see Fig.10 in paper).
# NOTE: Paper eq.62 has a typo — correct denominator is |T|^2, not |T|.
# =========================================================================

def _pr2_scalar(X: complex, l: float, u: float) -> complex:
    """Project a single complex number onto Q([l, u])."""
    phi = u - l
    if phi >= 2.0 * np.pi - 1e-10:
        # Full circle → project onto unit disk
        return X if abs(X) <= 1.0 + 1e-10 else X / abs(X)

    A = np.exp(1j * l)
    B = np.exp(1j * u)
    T = (A + B) * 0.5
    T2 = abs(T) ** 2  # |T|^2  (corrected from paper)

    # f1: positive => X is on the arc side of chord AB
    f1_raw = np.real(np.conj(T) * (X - T))
    f1 = f1_raw if phi <= np.pi else -f1_raw

    f2 = -np.real(1j * np.conj(A) * X)  # line OA test
    f3 = np.real(1j * np.conj(B) * X)   # line OB test
    f4 = np.real(np.conj(A - B) * (X - A))  # perpendicular at A
    f5 = np.real(np.conj(B - A) * (X - B))  # perpendicular at B

    EPS = 1e-10

    # M1: inside convex hull
    if f1 >= -EPS and abs(X) <= 1.0 + EPS:
        return X
    # M2: nearest vertex is A
    if f2 <= EPS and f4 >= -EPS:
        return A
    # M3: nearest vertex is B
    if f3 <= EPS and f5 >= -EPS:
        return B
    # M4: project onto chord AB (foot of perpendicular)
    if f1 <= EPS and f4 <= EPS and f5 <= EPS:
        if T2 > 1e-15:
            return X - np.real(np.conj(T) * (X - T)) * T / T2
        return T  # degenerate (phi ~ pi)
    # M5: normalize to unit circle
    return X / abs(X) if abs(X) > 1e-12 else np.exp(1j * (l + u) / 2.0)


def PR2(x: NDArray, l: NDArray, u: NDArray) -> NDArray:
    """Element-wise PR2 projection onto convex hulls Q(theta_n).

    Parameters
    ----------
    x : ndarray, shape (N,), complex
        Input vector.
    l, u : ndarray, shape (N,), float
        Lower and upper angle bounds (radians).

    Returns
    -------
    ndarray, shape (N,), complex
        Projected vector, each element inside its convex hull.
    """
    x = np.asarray(x, dtype=complex).ravel()
    l = np.asarray(l, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    return np.array([_pr2_scalar(x[i], l[i], u[i]) for i in range(len(x))])
