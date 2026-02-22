"""
Results Loading and Aggregation
===============================
Load experiment results from disk, compare across experiments, and
generate summary tables.

Usage
-----
>>> from src.experiments.results import ExperimentResult, ResultsAggregator
>>> result = ExperimentResult("results/quick_test")
>>> print(result.summary())
>>> wf = result.load_waveform(0)
>>> print(wf["X_bnb"].shape, wf["rate_bnb"])

>>> agg = ResultsAggregator()
>>> agg.add_dir("results/")
>>> print(agg.comparison_table())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ExperimentResult:
    """Load and inspect results from a completed experiment on disk.

    Parameters
    ----------
    path : str or Path
        Root directory of the experiment (must contain ``config.json``).
    """

    def __init__(self, path: str | Path):
        self.root = Path(path)
        if not (self.root / "config.json").exists():
            raise FileNotFoundError(
                f"No config.json found in {self.root}. "
                "Is this an experiment directory?"
            )
        from .config import ExperimentConfig
        self.config = ExperimentConfig.from_json(self.root / "config.json")

    # ── Waveform samples ────────────────────────────────────

    @property
    def waveforms_dir(self) -> Path:
        return self.root / "waveforms"

    @property
    def waveform_files(self) -> list[Path]:
        d = self.waveforms_dir
        return sorted(d.glob("sample_*.npz")) if d.exists() else []

    @property
    def n_waveforms(self) -> int:
        return len(self.waveform_files)

    def load_waveform(self, idx: int) -> dict[str, Any]:
        """Load a single waveform sample by index.

        Returns a dict with keys like ``H``, ``S``, ``X0``, ``X_bnb``,
        ``X_gan``, ``epsilon``, ``rate_bnb``, ``rate_gan``, etc.
        """
        f = self.waveform_files[idx]
        data = dict(np.load(f, allow_pickle=True))
        # Convert 0-d arrays to scalars
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim == 0:
                data[k] = v.item()
        return data

    def load_all_waveforms(self) -> list[dict[str, Any]]:
        """Load all waveform samples."""
        return [self.load_waveform(i) for i in range(self.n_waveforms)]

    # ── Stage results (JSON / NPZ) ──────────────────────────

    def _json(self, *parts) -> Any:
        p = self.root.joinpath(*parts)
        return json.loads(p.read_text()) if p.exists() else None

    def _npz(self, *parts) -> dict | None:
        p = self.root.joinpath(*parts)
        if not p.exists():
            return None
        data = dict(np.load(p, allow_pickle=True))
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim == 0:
                data[k] = v.item()
        return data

    @property
    def convergence_summary(self):
        return self._json("stages", "convergence", "summary.json")

    @property
    def convergence_data(self):
        return self._npz("stages", "convergence", "data.npz")

    @property
    def rate_sweep_summary(self):
        return self._json("stages", "rate_sweep", "summary.json")

    @property
    def rate_sweep_data(self):
        return self._npz("stages", "rate_sweep", "data.npz")

    @property
    def gan_summary(self):
        return self._json("stages", "gan_train", "summary.json")

    @property
    def gan_history(self):
        p = self.root / "stages" / "gan_train" / "history.json"
        if not p.exists():
            return None
        from ..gan.history import TrainingHistory
        return TrainingHistory.load(p)

    @property
    def waveform_summary(self):
        return self._json("waveforms", "summary.json")

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def figure_files(self) -> list[Path]:
        d = self.figures_dir
        return sorted(d.glob("*.*")) if d.exists() else []

    # ── High-level summary ──────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a high-level dict summarising all available results."""
        c = self.config
        s: dict[str, Any] = {
            "name": c.name,
            "system": f"N={c.N}, K={c.K}, L={c.L}",
            "N": c.N, "K": c.K, "L": c.L,
            "PT": c.PT, "SNR_dB": c.SNR_dB,
        }

        # Convergence
        cs = self.convergence_summary
        if cs:
            s["convergence"] = {
                "n_combos": len(cs),
                "best_objective": min(
                    c.get("objective", float("inf")) for c in cs
                ),
            }

        # Rate sweep
        rs = self.rate_sweep_data
        if rs and "rate_bnb" in rs:
            rb = rs["rate_bnb"]
            if hasattr(rb, "__len__"):
                s["rate_sweep"] = {
                    "max_rate_bnb": float(np.max(rb)),
                }

        # Waveform evaluation
        ws = self.waveform_summary
        if ws is None and self.n_waveforms > 0:
            # Compute from individual files
            rb, rg = [], []
            for f in self.waveform_files:
                d = np.load(f, allow_pickle=True)
                rb.append(float(d["rate_bnb"]))
                if "rate_gan" in d:
                    rg.append(float(d["rate_gan"]))
            ws = {
                "n_samples": self.n_waveforms,
                "mean_rate_bnb": float(np.mean(rb)),
            }
            if rg:
                ws["mean_rate_gan"] = float(np.mean(rg))
                ws["rate_ratio"] = float(
                    np.mean(rg) / max(np.mean(rb), 1e-12)
                )
        if ws:
            s["waveform_eval"] = ws

        s["n_figures"] = len(self.figure_files)
        return s

    def __repr__(self) -> str:
        return (
            f"ExperimentResult('{self.root}', "
            f"waveforms={self.n_waveforms})"
        )


# =====================================================================
# Aggregator
# =====================================================================

class ResultsAggregator:
    """Compare results across multiple experiments.

    Usage
    -----
    >>> agg = ResultsAggregator()
    >>> agg.add_dir("results/")  # auto-discover experiments
    >>> print(agg.comparison_table())
    """

    def __init__(self):
        self.experiments: list[ExperimentResult] = []

    def add(self, path: str | Path) -> ExperimentResult:
        """Add a single experiment."""
        exp = ExperimentResult(path)
        self.experiments.append(exp)
        return exp

    def add_dir(self, parent: str | Path) -> int:
        """Auto-discover and add all experiments in *parent*."""
        parent = Path(parent)
        n = 0
        for d in sorted(parent.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                self.add(d)
                n += 1
        return n

    def summaries(self) -> list[dict[str, Any]]:
        return [e.summary() for e in self.experiments]

    def comparison_table(self) -> str:
        """Return a Markdown table comparing all loaded experiments."""
        if not self.experiments:
            return "_No experiments found._"

        rows = self.summaries()
        headers = [
            "Name", "System", "BnB Rate", "GAN Rate", "Ratio", "Figures",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for s in rows:
            we = s.get("waveform_eval", {})
            rb = we.get("mean_rate_bnb")
            rg = we.get("mean_rate_gan")
            ratio = we.get("rate_ratio")
            lines.append("| " + " | ".join([
                s.get("name", "?"),
                s.get("system", "?"),
                f"{rb:.3f}" if isinstance(rb, (int, float)) else "\u2014",
                f"{rg:.3f}" if isinstance(rg, (int, float)) else "\u2014",
                f"{ratio:.3f}" if isinstance(ratio, (int, float)) else "\u2014",
                str(s.get("n_figures", 0)),
            ]) + " |")

        return "\n".join(lines)

    def generate_comparison_report(self) -> str:
        """Full Markdown comparison report."""
        now = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [
            "# Parameter Sweep — Comparison Report",
            f"\n**Generated**: {now}",
            f"**Experiments**: {len(self.experiments)}",
            "",
            "## Summary",
            "",
            self.comparison_table(),
        ]
        return "\n".join(parts)
