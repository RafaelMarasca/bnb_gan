"""
RadCom Experiments Package
==========================
Configurable, step-wise experiment pipeline.

Quick Start
-----------
>>> from src.experiments import ExperimentConfig, ExperimentRunner
>>> cfg = ExperimentConfig.quick_test()
>>> runner = ExperimentRunner(cfg)
>>> runner.run()                                   # all stages
>>> runner.run(stages=["convergence", "plots"])     # selective
"""

from .config import ExperimentConfig, ParameterGrid, ALL_STAGES
from .runner import ExperimentRunner
from .sweep import run_sweep
from .results import ExperimentResult, ResultsAggregator
from .report import ReportGenerator

__all__ = [
    "ExperimentConfig",
    "ParameterGrid",
    "ALL_STAGES",
    "ExperimentRunner",
    "run_sweep",
    "ExperimentResult",
    "ResultsAggregator",
    "ReportGenerator",
]
