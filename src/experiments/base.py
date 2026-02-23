"""
Abstract Experiment Framework
=============================
Provides ``BaseExperiment`` — the contract every experiment must implement —
and ``ExperimentRegistry`` for dynamic discovery and CLI integration.

How to add a new experiment
---------------------------
1.  Create a file under ``src/experiments/`` (e.g. ``my_exp.py``).
2.  Subclass ``BaseExperiment`` and set ``name`` / ``description``.
3.  Implement ``run()``.
4.  Decorate with ``@ExperimentRegistry.register`` (or it auto-registers
    via ``__init_subclass__``).

That's it — the experiment is automatically available via the CLI::

    python main.py experiment --name my_exp --config configs/quick.yaml

Example
-------
>>> from src.experiments.base import BaseExperiment, ExperimentRegistry
>>>
>>> class MyExperiment(BaseExperiment):
...     name = "my_analysis"
...     description = "Custom analysis of something cool"
...
...     def run(self, verbose: bool = True) -> dict:
...         result = {"answer": 42}
...         self.save_results(result)
...         return result
>>>
>>> ExperimentRegistry.list_experiments()
['convergence', 'rate_sweep', 'dataset', 'gan_train', 'waveform_eval', 'my_analysis']
"""

from __future__ import annotations

import abc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


class ExperimentResult:
    """Serializable envelope wrapping the output of any experiment.

    Persists to disk as a directory containing:
    - ``meta.json``   — timing, experiment name, config snapshot
    - ``result.json``  — JSON-safe scalars / small arrays
    - ``data.npz``     — large numpy arrays

    Parameters
    ----------
    name : str
        Experiment name.
    output_dir : Path
        Directory to save into.
    """

    def __init__(self, name: str, output_dir: Path) -> None:
        self.name = name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.meta: dict[str, Any] = {
            "experiment": name,
            "created_at": datetime.now().isoformat(),
        }
        self.scalars: dict[str, Any] = {}
        self.arrays: dict[str, np.ndarray] = {}

    # ── Population ──────────────────────────────────────────

    def add_scalar(self, key: str, value: Any) -> None:
        """Store a JSON-serializable scalar (int, float, str, list, dict)."""
        self.scalars[key] = value

    def add_array(self, key: str, value: np.ndarray) -> None:
        """Store a numpy array (saved to data.npz)."""
        self.arrays[key] = value

    def add_scalars(self, mapping: dict[str, Any]) -> None:
        """Bulk-add JSON-safe scalars."""
        self.scalars.update(mapping)

    def add_arrays(self, mapping: dict[str, np.ndarray]) -> None:
        """Bulk-add numpy arrays."""
        self.arrays.update(mapping)

    # ── Persistence ─────────────────────────────────────────

    def save(self) -> Path:
        """Write meta.json, result.json, and data.npz to ``output_dir``."""
        # meta.json
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2, default=_json_default)

        # result.json (scalars only)
        if self.scalars:
            with open(self.output_dir / "result.json", "w") as f:
                json.dump(self.scalars, f, indent=2, default=_json_default)

        # data.npz (arrays)
        if self.arrays:
            np.savez(self.output_dir / "data.npz", **self.arrays)

        return self.output_dir

    @classmethod
    def load(cls, path: Path) -> "ExperimentResult":
        """Reconstruct an ExperimentResult from a saved directory."""
        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No meta.json in {path}")

        with open(meta_path) as f:
            meta = json.load(f)

        obj = cls(name=meta.get("experiment", "unknown"), output_dir=path)
        obj.meta = meta

        result_path = path / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                obj.scalars = json.load(f)

        npz_path = path / "data.npz"
        if npz_path.exists():
            data = dict(np.load(npz_path, allow_pickle=True))
            # Convert 0-d arrays to scalars for convenience
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    data[k] = v.item()
            obj.arrays = data

        return obj

    def __repr__(self) -> str:
        n_s = len(self.scalars)
        n_a = len(self.arrays)
        return f"ExperimentResult('{self.name}', scalars={n_s}, arrays={n_a})"


# =====================================================================
# Abstract Base Experiment
# =====================================================================


class BaseExperiment(abc.ABC):
    """Abstract base class for all experiments.

    Subclasses must define:
    - ``name``: unique string identifier (used in CLI and on-disk paths)
    - ``description``: one-line human-readable description
    - ``run()``: execute the experiment and return results

    The base class provides:
    - Automatic output directory management
    - ``save_results()`` / ``load_results()`` for disk persistence
    - Timing and metadata tracking
    - Automatic registration into ``ExperimentRegistry``

    Parameters
    ----------
    config : PipelineConfig
        The full pipeline configuration.
    output_dir : Path or str, optional
        Override the output directory (default: ``{config.output_dir}/{config.name}/experiments/{cls.name}``).
    """

    # Subclasses MUST override these
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses into the registry."""
        super().__init_subclass__(**kwargs)
        if cls.name and not getattr(cls, "__abstractmethods__", None):
            ExperimentRegistry._registry[cls.name] = cls

    def __init__(
        self,
        config: Any,  # PipelineConfig (Any to avoid circular import)
        output_dir: Path | str | None = None,
    ) -> None:
        from ..config.schema import PipelineConfig

        if not isinstance(config, PipelineConfig):
            raise TypeError(
                f"Expected PipelineConfig, got {type(config).__name__}"
            )

        self.config = config

        if output_dir is not None:
            self._output_dir = Path(output_dir)
        else:
            self._output_dir = (
                Path(config.output_dir)
                / config.name
                / "experiments"
                / self.name
            )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._result: ExperimentResult | None = None
        self._elapsed: float = 0.0

    @property
    def output_dir(self) -> Path:
        """Directory where this experiment's artifacts are stored."""
        return self._output_dir

    # ── Abstract interface ──────────────────────────────────

    @abc.abstractmethod
    def run(self, verbose: bool = True) -> dict[str, Any]:
        """Execute the experiment.

        Must return a dict of results.  Implementations should call
        ``self.save_results(data)`` before returning if they want
        automatic disk persistence.

        Parameters
        ----------
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        dict[str, Any]
            Experiment-specific results.
        """
        ...

    # ── Persistence helpers ─────────────────────────────────

    def save_results(
        self,
        scalars: dict[str, Any] | None = None,
        arrays: dict[str, np.ndarray] | None = None,
    ) -> ExperimentResult:
        """Persist results to disk.

        Parameters
        ----------
        scalars : dict
            JSON-serializable key-value pairs.
        arrays : dict
            Numpy arrays to save in data.npz.

        Returns
        -------
        ExperimentResult
        """
        er = ExperimentResult(self.name, self._output_dir)
        er.meta.update({
            "config_name": self.config.name,
            "elapsed_s": self._elapsed,
        })
        if scalars:
            er.add_scalars(scalars)
        if arrays:
            er.add_arrays(arrays)
        er.save()
        self._result = er
        return er

    def load_results(self) -> ExperimentResult | None:
        """Load previously saved results from disk, or None."""
        meta_path = self._output_dir / "meta.json"
        if not meta_path.exists():
            return None
        return ExperimentResult.load(self._output_dir)

    # ── Execution wrapper ───────────────────────────────────

    def execute(self, verbose: bool = True) -> dict[str, Any]:
        """Run with timing and banner (preferred over calling run directly).

        Parameters
        ----------
        verbose : bool
            Print progress.

        Returns
        -------
        dict[str, Any]
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  EXPERIMENT: {self.name}")
            print(f"  {self.description}")
            print(f"  Output: {self._output_dir}")
            print(f"{'=' * 60}")

        t0 = time.time()
        result = self.run(verbose=verbose)
        self._elapsed = time.time() - t0

        if verbose:
            print(f"\n  [{self.name}] done in {self._elapsed:.1f}s")
            print(f"  Results saved to: {self._output_dir}\n")

        return result


# =====================================================================
# Experiment Registry
# =====================================================================


class ExperimentRegistry:
    """Global registry of available experiments.

    Experiments register themselves automatically via ``__init_subclass__``
    when they define a non-empty ``name`` class variable.

    Can also be used as a decorator::

        @ExperimentRegistry.register
        class MyExperiment(BaseExperiment):
            name = "my_exp"
            ...

    Usage
    -----
    >>> ExperimentRegistry.list_experiments()
    ['convergence', 'rate_sweep', ...]
    >>> exp_cls = ExperimentRegistry.get("convergence")
    >>> exp = exp_cls(config)
    >>> exp.execute()
    """

    _registry: ClassVar[dict[str, type[BaseExperiment]]] = {}

    @classmethod
    def register(cls, experiment_cls: type[BaseExperiment]) -> type[BaseExperiment]:
        """Decorator to explicitly register an experiment class."""
        if not experiment_cls.name:
            raise ValueError(
                f"Experiment class {experiment_cls.__name__} must define 'name'"
            )
        cls._registry[experiment_cls.name] = experiment_cls
        return experiment_cls

    @classmethod
    def get(cls, name: str) -> type[BaseExperiment]:
        """Look up an experiment class by name.

        Parameters
        ----------
        name : str
            Registered experiment name.

        Returns
        -------
        type[BaseExperiment]

        Raises
        ------
        KeyError
            If no experiment with that name is registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Unknown experiment '{name}'. Available: [{available}]"
            )
        return cls._registry[name]

    @classmethod
    def list_experiments(cls) -> list[str]:
        """Return sorted list of registered experiment names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_detailed(cls) -> list[dict[str, str]]:
        """Return list of dicts with name + description for each experiment."""
        return [
            {"name": name, "description": exp_cls.description}
            for name, exp_cls in sorted(cls._registry.items())
        ]

    @classmethod
    def create(
        cls,
        name: str,
        config: Any,
        output_dir: Path | str | None = None,
    ) -> BaseExperiment:
        """Convenience: look up + instantiate in one call.

        Parameters
        ----------
        name : str
            Registered experiment name.
        config : PipelineConfig
            Pipeline configuration.
        output_dir : Path, optional
            Override output directory.

        Returns
        -------
        BaseExperiment
        """
        exp_cls = cls.get(name)
        return exp_cls(config=config, output_dir=output_dir)


# =====================================================================
# JSON helper (shared)
# =====================================================================


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
