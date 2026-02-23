"""
RadCom Experiments Package
==========================
Configurable, step-wise experiment pipeline.

Quick Start (new — experiment framework)
-----------------------------------------
>>> from src.config import load_config
>>> from src.experiments import ExperimentRegistry
>>> cfg = load_config("configs/quick.yaml")
>>> exp = ExperimentRegistry.create("convergence", cfg)
>>> exp.execute()

Legacy (still works)
--------------------
>>> from src.experiments import ExperimentConfig, ExperimentRunner
>>> cfg = ExperimentConfig.quick_test()
>>> runner = ExperimentRunner(cfg)
>>> runner.run()
"""

# ── Legacy API (unchanged) ──────────────────────────────────
from .config import ExperimentConfig, ParameterGrid, ALL_STAGES
from .runner import ExperimentRunner
from .sweep import run_sweep
from .results import ExperimentResult as LegacyExperimentResult
from .results import ResultsAggregator
from .report import ReportGenerator

# ── New abstract experiment framework ───────────────────────
from .base import BaseExperiment, ExperimentRegistry, ExperimentResult

# Import concrete experiments to trigger auto-registration
from . import exp_convergence   # noqa: F401
from . import exp_rate_sweep    # noqa: F401
from . import exp_dataset       # noqa: F401
from . import exp_gan_train     # noqa: F401
from . import exp_waveform_eval # noqa: F401

__all__ = [
    # New framework
    "BaseExperiment",
    "ExperimentRegistry",
    "ExperimentResult",
    # Legacy
    "ExperimentConfig",
    "ParameterGrid",
    "ALL_STAGES",
    "ExperimentRunner",
    "run_sweep",
    "LegacyExperimentResult",
    "ResultsAggregator",
    "ReportGenerator",
]
