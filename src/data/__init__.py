"""
Data Package
============
Dataset generation, serialization, and experiment orchestration.
"""

from .generator import DatasetGenerator, DatasetSample
from .experiments import (
    ConvergenceResult,
    RateVsEpsilonResult,
    run_convergence_experiment,
    run_rate_vs_epsilon_experiment,
)

__all__ = [
    "DatasetGenerator",
    "DatasetSample",
    "ConvergenceResult",
    "RateVsEpsilonResult",
    "run_convergence_experiment",
    "run_rate_vs_epsilon_experiment",
]
