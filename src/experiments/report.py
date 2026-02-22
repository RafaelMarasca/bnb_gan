"""
Report Generator
================
Produces a comprehensive Markdown report from a completed experiment,
including configuration summary, stage-by-stage results, metric tables,
and references to generated figures.

Usage
-----
>>> from src.experiments.results import ExperimentResult
>>> from src.experiments.report import ReportGenerator
>>> result = ExperimentResult("results/quick_test")
>>> gen = ReportGenerator(result)
>>> gen.save()  # writes results/quick_test/report.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .results import ExperimentResult


class ReportGenerator:
    """Generate a Markdown report from experiment results on disk."""

    def __init__(self, result: ExperimentResult):
        self.result = result
        self.cfg = result.config

    # ── Public API ──────────────────────────────────────────

    def generate(self) -> str:
        """Build the full Markdown document."""
        sections = [
            self._header(),
            self._config_section(),
            self._convergence_section(),
            self._rate_sweep_section(),
            self._gan_section(),
            self._waveform_eval_section(),
            self._figures_section(),
            self._footer(),
        ]
        return "\n\n".join(s for s in sections if s)

    def save(self, path: str | Path | None = None) -> Path:
        """Write the report to disk."""
        path = Path(path) if path else self.result.root / "report.md"
        path.write_text(self.generate(), encoding="utf-8")
        return path

    # ── Header ──────────────────────────────────────────────

    def _header(self) -> str:
        c = self.cfg
        return (
            f"# Experiment Report: {c.name}\n\n"
            f"| Parameter | Value |\n"
            f"|---|---|\n"
            f"| Date | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n"
            f"| Antennas (N) | {c.N} |\n"
            f"| Users (K) | {c.K} |\n"
            f"| Frame length (L) | {c.L} |\n"
            f"| Power (PT) | {c.PT} |\n"
            f"| SNR | {c.SNR_dB} dB |\n"
            f"| Seed | {c.seed} |"
        )

    # ── Config ──────────────────────────────────────────────

    def _config_section(self) -> str:
        c = self.cfg
        return (
            "## Configuration\n\n"
            f"- **BnB**: rule={c.bnb_rule}, lb={c.bnb_lb}, ub={c.bnb_ub}, "
            f"tol={c.bnb_tol}, max_iter={c.bnb_max_iter}\n"
            f"- **Dataset**: {c.ds_n_samples} samples, "
            f"\u03b5 \u2208 {c.ds_epsilons}\n"
            f"- **GAN**: {c.gan_n_epochs} epochs, batch={c.gan_batch_size}, "
            f"lr={c.gan_lr}\n"
            f"- **Evaluation**: {c.eval_n_samples} samples, "
            f"\u03b5 \u2208 {c.eval_epsilons}"
        )

    # ── Convergence ─────────────────────────────────────────

    def _convergence_section(self) -> str | None:
        cs = self.result.convergence_summary
        if not cs:
            return None

        lines = [
            "## BnB Convergence\n",
            "| Solver Combo | Objective | Iters | Gap | Time |",
            "|---|---|---|---|---|",
        ]
        for c in cs:
            gap = c.get("gap")
            gap_s = f"{gap:.6f}" if gap is not None else "\u2014"
            lines.append(
                f"| {c['label']} | {c['objective']:.6f} "
                f"| {c['n_iterations']} | {gap_s} | {c['elapsed_s']:.1f}s |"
            )

        fig = self._find_figure("convergence")
        if fig:
            lines.append(f"\n![Convergence](figures/{fig.name})")

        return "\n".join(lines)

    # ── Rate sweep ──────────────────────────────────────────

    def _rate_sweep_section(self) -> str | None:
        data = self.result.rate_sweep_data
        if data is None:
            return None

        lines = ["## Rate vs Epsilon Sweep\n"]

        if "epsilons" in data and "rate_bnb" in data:
            eps = np.atleast_1d(data["epsilons"])
            bnb = np.atleast_1d(data["rate_bnb"])
            relx = np.atleast_1d(data.get("rate_relaxed", []))

            lines += [
                "| \u03b5 | BnB Rate | Relaxed Rate |",
                "|---|---|---|",
            ]
            for i, e in enumerate(eps):
                b = f"{bnb[i]:.3f}" if i < len(bnb) else "\u2014"
                r = f"{relx[i]:.3f}" if i < len(relx) else "\u2014"
                lines.append(f"| {e:.2f} | {b} | {r} |")

        fig = self._find_figure("rate_vs_epsilon")
        if fig:
            lines.append(f"\n![Rate vs Epsilon](figures/{fig.name})")

        return "\n".join(lines)

    # ── GAN ─────────────────────────────────────────────────

    def _gan_section(self) -> str | None:
        gs = self.result.gan_summary
        if gs is None:
            return None

        lines = [
            "## GAN Training\n",
            f"- **Epochs**: {gs.get('n_epochs', '?')}",
            f"- **Device**: {gs.get('device', '?')}",
            f"- **Generator params**: {gs.get('g_params', '?'):,}",
            f"- **Critic params**: {gs.get('c_params', '?'):,}",
        ]

        fig = self._find_figure("gan_training")
        if fig:
            lines.append(f"\n![GAN Training](figures/{fig.name})")

        return "\n".join(lines)

    # ── Waveform evaluation ─────────────────────────────────

    def _waveform_eval_section(self) -> str | None:
        n = self.result.n_waveforms
        if n == 0:
            return None

        lines = [
            "## Waveform Evaluation\n",
            f"**{n} samples evaluated**\n",
        ]

        # Aggregate from individual files
        rb, rg, eps_list = [], [], []
        fb, fg = [], []
        for f in self.result.waveform_files:
            d = np.load(f, allow_pickle=True)
            rb.append(float(d["rate_bnb"]))
            eps_list.append(float(d["epsilon"]))
            if "feasible_bnb" in d:
                fb.append(bool(d["feasible_bnb"]))
            if "rate_gan" in d:
                rg.append(float(d["rate_gan"]))
            if "feasible_gan" in d:
                fg.append(bool(d["feasible_gan"]))

        has_gan = len(rg) > 0

        # Summary table
        lines += [
            "### Summary Statistics\n",
            "| Metric | BnB | GAN |",
            "|---|---|---|",
        ]
        gan_rate_s = f"{np.mean(rg):.3f}" if has_gan else "\u2014"
        gan_std_s  = f"{np.std(rg):.3f}" if has_gan else "\u2014"
        lines.append(
            f"| Mean Rate (bps/Hz) | {np.mean(rb):.3f} | {gan_rate_s} |"
        )
        lines.append(
            f"| Std Rate | {np.std(rb):.3f} | {gan_std_s} |"
        )
        if has_gan:
            ratio = np.mean(rg) / max(np.mean(rb), 1e-12)
            lines.append(f"| Rate Ratio (GAN/BnB) | \u2014 | {ratio:.3f} |")
        if fb:
            fb_pct = f"{100 * np.mean(fb):.0f}%"
            fg_pct = f"{100 * np.mean(fg):.0f}%" if fg else "\u2014"
            lines.append(f"| Feasible | {fb_pct} | {fg_pct} |")

        # Per-epsilon breakdown
        eps_unique = sorted(set(eps_list))
        if len(eps_unique) > 1:
            lines += [
                "\n### Per-Epsilon Breakdown\n",
                "| \u03b5 | N | BnB Rate | GAN Rate | Ratio |",
                "|---|---|---|---|---|",
            ]
            for eps in eps_unique:
                mask = [e == eps for e in eps_list]
                r_b = [r for r, m in zip(rb, mask) if m]
                r_g = [r for r, m in zip(rg, mask) if m] if has_gan else []
                rb_m = np.mean(r_b) if r_b else 0
                rg_m = np.mean(r_g) if r_g else 0
                ratio_s = (f"{rg_m / max(rb_m, 1e-12):.3f}"
                           if r_g else "\u2014")
                rg_ms = f"{rg_m:.3f}" if r_g else "\u2014"
                lines.append(
                    f"| {eps:.2f} | {len(r_b)} "
                    f"| {rb_m:.3f} | {rg_ms} | {ratio_s} |"
                )

        # Individual sample table
        lines += [
            "\n### Individual Samples\n",
            "| # | \u03b5 | BnB Rate | GAN Rate | Ratio |",
            "|---|---|---|---|---|",
        ]
        for i, f in enumerate(self.result.waveform_files):
            d = np.load(f, allow_pickle=True)
            r_b = float(d["rate_bnb"])
            eps = float(d["epsilon"])
            if "rate_gan" in d:
                r_g = float(d["rate_gan"])
                ratio = r_g / max(r_b, 1e-12)
                lines.append(
                    f"| {i} | {eps:.2f} | {r_b:.3f} "
                    f"| {r_g:.3f} | {ratio:.3f} |"
                )
            else:
                lines.append(
                    f"| {i} | {eps:.2f} | {r_b:.3f} | \u2014 | \u2014 |"
                )

        # Embed evaluation figures
        for name in ("eval_rate_comparison", "eval_rate_scatter",
                     "eval_rate_boxplot"):
            fig = self._find_figure(name)
            if fig:
                lines.append(f"\n![{name}](figures/{fig.name})")

        return "\n".join(lines)

    # ── Figures index ───────────────────────────────────────

    def _figures_section(self) -> str | None:
        figs = self.result.figure_files
        if not figs:
            return None
        lines = ["## All Figures\n"]
        for f in figs:
            lines.append(f"- [{f.name}](figures/{f.name})")
        return "\n".join(lines)

    # ── Footer ──────────────────────────────────────────────

    def _footer(self) -> str:
        return "---\n*Generated by RadCom Waveform Design Pipeline*"

    # ── Utility ─────────────────────────────────────────────

    def _find_figure(self, stem: str) -> Path | None:
        """Find a figure file by stem prefix."""
        for f in self.result.figure_files:
            if f.stem == stem or f.stem.startswith(stem):
                return f
        return None
