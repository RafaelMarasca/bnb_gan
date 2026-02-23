"""
Centralized Configuration
=========================
Pydantic-validated, YAML-driven configuration for the entire pipeline.

Usage
-----
>>> from src.config import load_config, PipelineConfig
>>> cfg = load_config("configs/default.yaml")
>>> cfg.system.N
16
>>> cfg.gan.learning_rate
0.0001
"""

from .schema import (
    PipelineConfig,
    SystemConfig,
    BnBConfig,
    DatasetConfig,
    GANConfig,
    EvalConfig,
    PlotConfig,
    ConvergenceStageConfig,
    RateSweepStageConfig,
)
from .loader import load_config

__all__ = [
    "PipelineConfig",
    "SystemConfig",
    "BnBConfig",
    "DatasetConfig",
    "GANConfig",
    "EvalConfig",
    "PlotConfig",
    "ConvergenceStageConfig",
    "RateSweepStageConfig",
    "load_config",
]
