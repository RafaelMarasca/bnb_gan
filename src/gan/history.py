"""
Training History & Plotting
============================
Records per-epoch metrics during WGAN-GP training and provides
publication-quality plots for analysing training dynamics.

Tracked Metrics
---------------
- **critic_loss** / **generator_loss** — raw WGAN losses
- **wasserstein_dist** — estimated W-distance (critic_real − critic_fake)
- **gradient_penalty** — GP term magnitude
- **rate_real** / **rate_fake** — mean sum-rate of real vs generated waveforms
- **power_violation** — mean column-power deviation from P_T
- **similarity_violation** — mean ‖X_gen − X0‖_F / (√(N·L)·ε) (ratio > 1 = violation)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Use non-interactive backend when running headless / in scripts
if os.environ.get("MPLBACKEND") is None:
    import matplotlib
    matplotlib.use("Agg")


@dataclass
class TrainingHistory:
    """Accumulates per-epoch training metrics and generates plots.

    Usage
    -----
    >>> hist = TrainingHistory()
    >>> for epoch in range(n_epochs):
    ...     hist.record(epoch=epoch, critic_loss=..., generator_loss=..., ...)
    >>> hist.plot_all("figures/gan_training.png")
    """

    records: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, **kwargs: Any) -> None:
        """Append one epoch's metrics.  Must include ``epoch`` key."""
        self.records.append(kwargs)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _get(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records if key in r])

    @property
    def epochs(self) -> np.ndarray:
        return self._get("epoch")

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_losses(self, ax=None, **kwargs):
        """Critic and Generator loss vs epoch."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ep = self.epochs
        ax.plot(ep, self._get("critic_loss"), label="Critic", linewidth=1.2)
        ax.plot(ep, self._get("generator_loss"), label="Generator", linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("WGAN-GP Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_wasserstein(self, ax=None, **kwargs):
        """Estimated Wasserstein distance vs epoch."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ep = self.epochs
        ax.plot(ep, self._get("wasserstein_dist"), color="tab:purple", linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$W$-distance estimate")
        ax.set_title("Wasserstein Distance")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_gradient_penalty(self, ax=None, **kwargs):
        """Gradient penalty term vs epoch."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ep = self.epochs
        ax.plot(ep, self._get("gradient_penalty"), color="tab:orange", linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Penalty")
        ax.set_title("Gradient Penalty")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_rates(self, ax=None, **kwargs):
        """Real vs generated sum-rate comparison."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ep = self.epochs
        ax.plot(ep, self._get("rate_real"), label="BnB (real)", linewidth=1.2)
        ax.plot(ep, self._get("rate_fake"), label="GAN (generated)", linewidth=1.2,
                linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Sum-Rate (bits/s/Hz)")
        ax.set_title("Sum-Rate Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_power_violation(self, ax=None, **kwargs):
        """Power constraint violation vs epoch."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ep = self.epochs
        ax.plot(ep, self._get("power_violation"), color="tab:red", linewidth=1.2)
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean $|\\|x_t\\|^2 - P_T|$")
        ax.set_title("Power Constraint Violation")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_similarity(self, ax=None, **kwargs):
        """Waveform similarity constraint ratio vs epoch."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ep = self.epochs
        ax.plot(ep, self._get("similarity_violation"), color="tab:green", linewidth=1.2)
        ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", label="Feasibility boundary")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$\\|X_{\\mathrm{gen}} - X_0\\|_F / (\\sqrt{NL}\\,\\varepsilon)$")
        ax.set_title("Similarity Constraint")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # Combined dashboard
    # ------------------------------------------------------------------

    def plot_all(
        self,
        save_path: str | Path | None = None,
        dpi: int = 150,
        figsize: tuple[float, float] = (16, 10),
    ):
        """Six-panel training dynamics dashboard.

        Parameters
        ----------
        save_path : str or Path, optional
            If provided, save figure to this path (PNG/PDF/EPS).
        dpi : int
            Resolution for raster formats.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("WGAN-GP Training Dynamics", fontsize=14, fontweight="bold")

        self.plot_losses(axes[0, 0])
        self.plot_wasserstein(axes[0, 1])
        self.plot_gradient_penalty(axes[0, 2])
        self.plot_rates(axes[1, 0])
        self.plot_power_violation(axes[1, 1])
        self.plot_similarity(axes[1, 2])

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Training dashboard saved to {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save training history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingHistory":
        """Load training history from JSON."""
        with open(path) as f:
            records = json.load(f)
        return cls(records=records)

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"TrainingHistory(epochs={len(self)})"


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
