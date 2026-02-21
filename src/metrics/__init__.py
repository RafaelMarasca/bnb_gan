"""
Metrics Package
===============
Modular metric calculators for convergence, communication rate,
radar performance, waveform similarity, and pulse-compression analysis.
"""

from .base import MetricBase
from .convergence import ConvergenceMetric
from .rate import RateMetric, sum_rate
from .radar import ISLMetric, PSLMetric
from .similarity import WaveformSimilarityMetric
from .pulse_comp_metrics import MainlobeToSidelobeRatio, MainlobeWidthMetric

__all__ = [
    "MetricBase",
    "ConvergenceMetric",
    "RateMetric",
    "sum_rate",
    "ISLMetric",
    "PSLMetric",
    "WaveformSimilarityMetric",
    "MainlobeToSidelobeRatio",
    "MainlobeWidthMetric",
]
