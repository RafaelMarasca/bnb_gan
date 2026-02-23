"""
Data Package
============
Dataset generation, serialization, and experiment orchestration.
"""

from .generator import DatasetGenerator, DatasetSample
from .nn_dataset import NNDatasetGenerator, RadComHDF5Dataset, EpsilonFilteredDataset
from .experiments import (
    ConvergenceResult,
    RateVsEpsilonResult,
    run_convergence_experiment,
    run_rate_vs_epsilon_experiment,
)

__all__ = [
    "DatasetGenerator",
    "DatasetSample",
    "NNDatasetGenerator",
    "RadComHDF5Dataset",
    "EpsilonFilteredDataset",
    "ConvergenceResult",
    "RateVsEpsilonResult",
    "run_convergence_experiment",
    "run_rate_vs_epsilon_experiment",
]
