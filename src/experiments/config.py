"""
Experiment Configuration
========================
Centralized, JSON-serializable configuration for all pipeline stages.

Supports:
- Named presets (``quick_test``, ``paper``)
- JSON save / load
- CLI overrides via ``with_overrides()``
- Parameter grid generation via ``ParameterGrid``

Examples
--------
>>> cfg = ExperimentConfig.quick_test()
>>> cfg = ExperimentConfig.from_json("my_config.json")
>>> cfg = ExperimentConfig(N=8, K=2, name="small_system")
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator


# Ordered list of all available pipeline stages
ALL_STAGES: list[str] = [
    "convergence",
    "rate_sweep",
    "dataset",
    "gan_train",
    "waveform_eval",
    "plots",
    "report",
]


# =====================================================================
# Experiment Config
# =====================================================================

@dataclass
class ExperimentConfig:
    """Single source-of-truth for one experiment run.

    Every parameter the system needs lives here.  Unused params are
    silently ignored by stages that do not need them.
    """

    # ── Identity ────────────────────────────────────────────
    name: str = "default"
    output_dir: str = "results"
    seed: int = 42

    # ── System physics ──────────────────────────────────────
    N: int = 16           # transmit antennas
    K: int = 4            # users
    L: int = 20           # frame length
    PT: float = 1.0       # total transmit power
    SNR_dB: float = 10.0  # signal-to-noise ratio (dB)

    # ── BnB solver ──────────────────────────────────────────
    bnb_rule: str = "ARS"
    bnb_lb: str = "gp"
    bnb_ub: str = "gp"
    bnb_tol: float = 1e-3
    bnb_max_iter: int = 200
    bnb_gp_iters: int = 100

    # ── Convergence stage ───────────────────────────────────
    conv_epsilon: float = 1.0

    # ── Rate-sweep stage ────────────────────────────────────
    rate_epsilons: list[float] = field(
        default_factory=lambda: [0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    )
    rate_n_trials: int = 5
    rate_tol: float = 5e-3
    rate_max_iter: int = 40

    # ── Dataset stage ───────────────────────────────────────
    ds_n_samples: int = 1000
    ds_epsilons: list[float] = field(default_factory=lambda: [0.3, 0.7, 1.0])
    ds_chunk_size: int = 50

    # ── GAN training stage ──────────────────────────────────
    gan_latent_dim: int = 128
    gan_hidden_g: list[int] = field(default_factory=lambda: [512, 512, 512])
    gan_hidden_c: list[int] = field(default_factory=lambda: [512, 256, 128])
    gan_n_epochs: int = 200
    gan_batch_size: int = 64
    gan_lr: float = 1e-4
    gan_n_critic: int = 5
    gan_lambda_gp: float = 10.0
    gan_eval_every: int = 5
    gan_save_every: int = 50

    # ── Waveform-eval stage ─────────────────────────────────
    eval_n_samples: int = 20
    eval_epsilons: list[float] = field(default_factory=lambda: [0.3, 0.7, 1.0])

    # ── Plotting ────────────────────────────────────────────
    plot_formats: list[str] = field(default_factory=lambda: ["png"])
    plot_dpi: int = 300

    # ── Stages to run ───────────────────────────────────────
    stages: list[str] = field(default_factory=lambda: ALL_STAGES.copy())

    # -----------------------------------------------------------------
    # Derived config objects
    # -----------------------------------------------------------------

    @property
    def sys_config(self):
        """Build a frozen ``SystemConfig``."""
        from ..utils.config import SystemConfig
        return SystemConfig(
            N=self.N, K=self.K, L=self.L, PT=self.PT, SNR_dB=self.SNR_dB,
        )

    @property
    def bnb_config(self):
        """Build a frozen ``BnBConfig``."""
        from ..utils.config import BnBConfig
        return BnBConfig(
            rule=self.bnb_rule,
            lb_solver=self.bnb_lb,
            ub_solver=self.bnb_ub,
            tol=self.bnb_tol,
            max_iter=self.bnb_max_iter,
            gp_max_iter=self.bnb_gp_iters,
        )

    # -----------------------------------------------------------------
    # Presets
    # -----------------------------------------------------------------

    @classmethod
    def quick_test(cls, **kw) -> ExperimentConfig:
        """Tiny system for fast validation (< 30 s)."""
        defaults = dict(
            name="quick_test",
            N=8, K=2, L=4,
            bnb_max_iter=20, bnb_gp_iters=30, bnb_tol=1e-2,
            rate_n_trials=1, rate_max_iter=10, rate_tol=1e-1,
            rate_epsilons=[0.3, 0.7, 1.0],
            ds_n_samples=16, ds_chunk_size=8,
            gan_latent_dim=64,
            gan_hidden_g=[128, 128],
            gan_hidden_c=[128, 64],
            gan_n_epochs=10, gan_batch_size=8,
            gan_eval_every=1, gan_save_every=0,
            eval_n_samples=4,
        )
        defaults.update(kw)
        return cls(**defaults)

    @classmethod
    def paper(cls, **kw) -> ExperimentConfig:
        """Parameters matching arXiv:1711.05220."""
        defaults = dict(
            name="paper",
            N=16, K=4, L=20, PT=1.0, SNR_dB=10.0,
            bnb_max_iter=200, bnb_gp_iters=100, bnb_tol=1e-4,
            rate_n_trials=5, rate_max_iter=40,
            ds_n_samples=5000, ds_chunk_size=100,
            gan_n_epochs=500, gan_batch_size=64,
            plot_formats=["png", "eps"],
        )
        defaults.update(kw)
        return cls(**defaults)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def with_overrides(self, **kw) -> ExperimentConfig:
        """Return a new config with selected fields overridden.

        ``None`` values are silently ignored so you can pass CLI args
        directly.
        """
        return replace(self, **{k: v for k, v in kw.items() if v is not None})

    def __str__(self) -> str:
        return (
            f"ExperimentConfig(name='{self.name}', "
            f"N={self.N}, K={self.K}, L={self.L})"
        )


# =====================================================================
# Parameter Grid
# =====================================================================

class ParameterGrid:
    """Generate experiment configs for all combinations of parameter values.

    Examples
    --------
    >>> base = ExperimentConfig.quick_test()
    >>> grid = ParameterGrid(base, {"N": [8, 16], "K": [2, 4]})
    >>> len(grid)
    4
    >>> for cfg in grid:
    ...     print(cfg.name, cfg.N, cfg.K)
    quick_test_N8_K2 8 2
    quick_test_N8_K4 8 4
    quick_test_N16_K2 16 2
    quick_test_N16_K4 16 4
    """

    def __init__(self, base: ExperimentConfig, axes: dict[str, list]):
        self.base = base
        self.axes = axes
        for k in axes:
            if k not in ExperimentConfig.__dataclass_fields__:
                raise ValueError(f"Unknown config parameter: '{k}'")

    def __iter__(self) -> Iterator[ExperimentConfig]:
        keys = list(self.axes.keys())
        for combo in itertools.product(*self.axes.values()):
            overrides = dict(zip(keys, combo))
            tag = "_".join(f"{k}{v}" for k, v in overrides.items())
            name = f"{self.base.name}_{tag}"
            yield replace(self.base, name=name, **overrides)

    def __len__(self) -> int:
        r = 1
        for v in self.axes.values():
            r *= len(v)
        return r

    def configs(self) -> list[ExperimentConfig]:
        """Materialise all configs as a list."""
        return list(self)

    @classmethod
    def from_json(cls, base: ExperimentConfig, path: str | Path) -> ParameterGrid:
        """Load sweep axes from a JSON file.

        File format: ``{"N": [8, 16], "K": [2, 4]}``
        """
        with open(path) as f:
            return cls(base, json.load(f))
