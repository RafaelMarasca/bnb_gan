"""
Pydantic Configuration Schema
==============================
All pipeline parameters live here as validated pydantic models.
Each section maps 1:1 to a top-level key in the YAML config file.

The hierarchy:

    PipelineConfig
    ├── system    → SystemConfig       (physics: N, K, L, PT, SNR)
    ├── bnb       → BnBConfig          (branch-and-bound solver)
    ├── dataset   → DatasetConfig      (HDF5 dataset generation)
    ├── gan       → GANConfig          (WGAN-GP architecture + training)
    ├── eval      → EvalConfig         (waveform evaluation)
    ├── plot      → PlotConfig         (figure output)
    ├── convergence → ConvergenceStageConfig
    ├── rate_sweep  → RateSweepStageConfig
    └── (top-level) name, output_dir, seed, stages
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# =====================================================================
# Sub-configs
# =====================================================================


class SystemConfig(BaseModel):
    """Physical system parameters (antenna array, users, power)."""

    N: int = Field(16, ge=1, description="Number of transmit antennas")
    K: int = Field(4, ge=1, description="Number of single-antenna users")
    L: int = Field(20, ge=1, description="Frame length / radar pulses")
    PT: float = Field(1.0, gt=0, description="Total transmit power")
    SNR_dB: float = Field(10.0, description="Signal-to-noise ratio (dB)")

    @property
    def N0(self) -> float:
        """Noise power: N0 = PT / 10^(SNR_dB/10)."""
        return self.PT / (10.0 ** (self.SNR_dB / 10.0))

    @property
    def scale(self) -> float:
        """Per-element amplitude: sqrt(PT / N)."""
        import numpy as np
        return float(np.sqrt(self.PT / self.N))


class BnBConfig(BaseModel):
    """Branch-and-Bound algorithm settings."""

    rule: Literal["ARS", "BRS"] = Field("ARS", description="Subdivision rule")
    lb_solver: Literal["cvxpy", "gp"] = Field("gp", description="Lower-bound solver")
    ub_solver: Literal["slsqp", "gp"] = Field("gp", description="Upper-bound solver")
    tol: float = Field(1e-3, gt=0, description="Convergence tolerance (UB - LB)")
    max_iter: int = Field(200, ge=1, description="Maximum BnB iterations")
    gp_max_iter: int = Field(100, ge=1, description="Gradient-projection iterations per bound")
    gp_tol: float = Field(1e-6, gt=0, description="GP convergence tolerance")
    verbose: bool = Field(False, description="Print BnB progress")
    verbose_interval: int = Field(20, ge=1, description="Print every N iterations")


class ConvergenceStageConfig(BaseModel):
    """Settings for the BnB convergence analysis stage."""

    epsilon: float = Field(1.0, ge=0, description="Similarity tolerance for convergence test")


class RateSweepStageConfig(BaseModel):
    """Settings for the rate-vs-epsilon sweep stage."""

    epsilons: list[float] = Field(
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0],
        description="Epsilon values to sweep",
    )
    n_trials: int = Field(5, ge=1, description="Independent channel realizations per epsilon")
    tol: float = Field(5e-3, gt=0, description="BnB tolerance for rate sweep")
    max_iter: int = Field(40, ge=1, description="BnB max iterations for rate sweep")


class DatasetConfig(BaseModel):
    """HDF5 dataset generation settings."""

    n_samples: int = Field(1000, ge=1, description="Number of optimization samples to generate")
    epsilons: list[float] = Field(
        default=[0.3, 0.7, 1.0],
        description="Epsilon values to cycle through during generation",
    )
    chunk_size: int = Field(50, ge=1, description="HDF5 write chunk size")
    n_workers: int = Field(
        1, ge=0,
        description="Parallel workers for generation. 0 = all CPU cores, 1 = sequential.",
    )


class GANConfig(BaseModel):
    """WGAN-GP architecture and training hyperparameters."""

    # Architecture
    latent_dim: int = Field(128, ge=1, description="Noise vector dimension")
    hidden_g: list[int] = Field(
        default=[512, 512, 512], description="Generator hidden layer sizes"
    )
    hidden_c: list[int] = Field(
        default=[512, 256, 128], description="Critic hidden layer sizes"
    )

    # Training
    n_epochs: int = Field(200, ge=1, description="Training epochs")
    batch_size: int = Field(64, ge=1, description="Mini-batch size")
    learning_rate: float = Field(1e-4, gt=0, description="Adam learning rate (both G and C)")
    n_critic: int = Field(5, ge=1, description="Critic updates per generator update")
    lambda_gp: float = Field(10.0, ge=0, description="Gradient penalty coefficient")

    # Checkpointing / eval
    eval_every: int = Field(5, ge=1, description="Compute physics metrics every N epochs")
    save_every: int = Field(50, ge=0, description="Save checkpoint every N epochs (0 = off)")


class EvalConfig(BaseModel):
    """Waveform evaluation (BnB vs GAN) settings."""

    n_samples: int = Field(20, ge=1, description="Number of evaluation samples")
    epsilons: list[float] = Field(
        default=[0.3, 0.7, 1.0],
        description="Epsilon values to evaluate",
    )


class PlotConfig(BaseModel):
    """Figure output settings."""

    formats: list[str] = Field(default=["png"], description="Output formats (png, eps, pdf)")
    dpi: int = Field(300, ge=72, description="Figure DPI")


# =====================================================================
# Top-level pipeline config
# =====================================================================


# All available pipeline stages (order matters)
ALL_STAGES: list[str] = [
    "convergence",
    "rate_sweep",
    "dataset",
    "gan_train",
    "waveform_eval",
    "plots",
    "report",
]


class PipelineConfig(BaseModel):
    """Root configuration — single source of truth for the entire pipeline.

    Every parameter the system needs is reachable from here.
    Maps 1:1 to the top-level YAML config file.
    """

    # ── Identity ────────────────────────────────────────────
    name: str = Field("default", description="Experiment name (used as output subdirectory)")
    output_dir: str = Field("outputs", description="Root output directory")
    seed: int = Field(42, ge=0, description="Global random seed")

    # ── Stages to run ───────────────────────────────────────
    stages: list[str] = Field(
        default_factory=lambda: ALL_STAGES.copy(),
        description="Pipeline stages to execute (in order)",
    )

    # ── Sub-configs ─────────────────────────────────────────
    system: SystemConfig = Field(default_factory=SystemConfig)
    bnb: BnBConfig = Field(default_factory=BnBConfig)
    convergence: ConvergenceStageConfig = Field(default_factory=ConvergenceStageConfig)
    rate_sweep: RateSweepStageConfig = Field(default_factory=RateSweepStageConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    gan: GANConfig = Field(default_factory=GANConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)

    model_config = {"extra": "forbid"}

    # ── Validators ──────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_stages(self) -> "PipelineConfig":
        for s in self.stages:
            if s not in ALL_STAGES:
                raise ValueError(
                    f"Unknown stage '{s}'. Valid stages: {ALL_STAGES}"
                )
        return self

    # ── Convenience builders ────────────────────────────────

    def with_overrides(self, **kw: Any) -> "PipelineConfig":
        """Return a copy with top-level or nested overrides.

        Supports flat keys like ``seed=123`` and dotted keys like
        ``system.N=8`` (as keyword args with underscores:
        ``system__N=8`` or via a dict).
        """
        data = self.model_dump()
        for key, val in kw.items():
            if val is None:
                continue
            # Support dotted or double-underscore nesting: "system.N" / "system__N"
            parts = key.replace("__", ".").split(".")
            target = data
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = val
        return PipelineConfig.model_validate(data)

    # ── Legacy bridge (maps to old SystemConfig / BnBConfig) ─

    @property
    def sys_config(self):
        """Build a frozen legacy ``SystemConfig`` (from src.utils.config)."""
        from ..utils.config import SystemConfig as LegacySystemConfig
        return LegacySystemConfig(
            N=self.system.N,
            K=self.system.K,
            L=self.system.L,
            PT=self.system.PT,
            SNR_dB=self.system.SNR_dB,
        )

    @property
    def bnb_legacy(self):
        """Build a frozen legacy ``BnBConfig`` (from src.utils.config)."""
        from ..utils.config import BnBConfig as LegacyBnBConfig
        return LegacyBnBConfig(
            rule=self.bnb.rule,
            lb_solver=self.bnb.lb_solver,
            ub_solver=self.bnb.ub_solver,
            tol=self.bnb.tol,
            max_iter=self.bnb.max_iter,
            gp_max_iter=self.bnb.gp_max_iter,
        )

    # ── Serialization ───────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Dump to plain dict (JSON-compatible)."""
        return self.model_dump()

    def to_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        import yaml
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def to_json(self, path: str | Path) -> None:
        """Write config to a JSON file (backwards-compatible)."""
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load config from a YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Load config from a JSON file."""
        import json
        with open(path) as f:
            return cls.model_validate(json.load(f))
