"""
Optimizer Package
=================
Core Branch-and-Bound optimization engine with pluggable solvers
and projection operators for constant-modulus waveform design.
"""

from .bnb import BranchAndBoundSolver, BnBResult, bnb_solve
from .node import BnBNode
from .projections import PR1, PR2
from .waveform_optimizer import WaveformMatrixOptimizer, optimize_waveform

__all__ = [
    "BranchAndBoundSolver",
    "BnBResult",
    "BnBNode",
    "PR1",
    "PR2",
    "WaveformMatrixOptimizer",
    "bnb_solve",
    "optimize_waveform",
]
