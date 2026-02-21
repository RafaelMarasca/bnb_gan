"""
Configuration Dataclasses
=========================
Immutable configuration objects for system parameters and BnB settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SystemConfig:
    """Physical system parameters matching the paper's simulation setup.

    Attributes
    ----------
    N : int
        Number of transmit antennas (paper default: 16).
    K : int
        Number of single-antenna communication users (paper default: 4).
    L : int
        Frame length / number of radar pulses (paper default: 20).
    PT : float
        Total transmit power (paper default: 1.0).
    SNR_dB : float
        Signal-to-noise ratio in dB (paper default: 10.0).
    """

    N: int = 16
    K: int = 4
    L: int = 20
    PT: float = 1.0
    SNR_dB: float = 10.0

    @property
    def N0(self) -> float:
        """Noise power derived from SNR: N0 = PT / 10^(SNR_dB/10)."""
        return self.PT / (10.0 ** (self.SNR_dB / 10.0))

    @property
    def scale(self) -> float:
        """Per-element amplitude: sqrt(PT / N)."""
        import numpy as np

        return float(np.sqrt(self.PT / self.N))


@dataclass(frozen=True)
class BnBConfig:
    """Branch-and-Bound algorithm configuration.

    Attributes
    ----------
    rule : {'ARS', 'BRS'}
        Subdivision rule. ARS = Adaptive Rectangular Subdivision (eq.36),
        BRS = Basic Rectangular Subdivision (eq.35).
    lb_solver : {'cvxpy', 'gp'}
        Lower-bound solver. 'cvxpy' = interior-point via SCS (eq.40),
        'gp' = accelerated gradient projection with PR2 (eqs.43-44).
    ub_solver : {'slsqp', 'gp'}
        Upper-bound solver. 'slsqp' = scipy SLSQP (eq.42),
        'gp' = gradient projection with PR1.
    tol : float
        Convergence tolerance: stop when UB - LB <= tol.
    max_iter : int
        Maximum BnB iterations.
    gp_max_iter : int
        Maximum gradient-projection iterations per bound solve.
    gp_tol : float
        Gradient-projection convergence tolerance.
    verbose : bool
        Print progress during optimization.
    verbose_interval : int
        Print every N iterations when verbose is True.
    """

    rule: Literal["ARS", "BRS"] = "ARS"
    lb_solver: Literal["cvxpy", "gp"] = "cvxpy"
    ub_solver: Literal["slsqp", "gp"] = "slsqp"
    tol: float = 1e-3
    max_iter: int = 200
    gp_max_iter: int = 100
    gp_tol: float = 1e-6
    verbose: bool = False
    verbose_interval: int = 20
