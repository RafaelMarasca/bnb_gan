"""
Abstract Metric Base Class
==========================
Defines the interface for all metric calculators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MetricResult:
    """Container for metric computation results.

    Attributes
    ----------
    name : str
        Metric name (e.g., 'convergence', 'sum_rate', 'ISL').
    values : dict[str, Any]
        Computed metric values (scalars, arrays, etc.).
    metadata : dict[str, Any]
        Additional context (parameters used, units, etc.).
    """

    name: str
    values: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricBase(ABC):
    """Abstract base class for all metrics.

    Every metric must implement ``compute()`` which returns a
    ``MetricResult`` containing named values and metadata.

    Subclasses
    ----------
    - ``ConvergenceMetric``: Tracks LB/UB gap vs. iteration.
    - ``RateMetric``: Communication sum-rate R = sum_i log2(1 + gamma_i).
    - ``ISLMetric``: Integrated Sidelobe Level.
    - ``PSLMetric``: Peak Sidelobe Level.
    - ``WaveformSimilarityMetric``: L2/L-inf distance from reference.
    - ``MainlobeToSidelobeRatio``: MSR from pulse compression.
    - ``MainlobeWidthMetric``: 3/6/10 dB mainlobe width (range precision).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the metric."""
        ...

    @abstractmethod
    def compute(self, **kwargs: Any) -> MetricResult:
        """Compute the metric from the given inputs.

        Parameters will vary by metric type — see subclass docstrings.

        Returns
        -------
        MetricResult
            Container with computed values and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
