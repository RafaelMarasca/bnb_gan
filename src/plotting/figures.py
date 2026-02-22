"""
Plot Functions
==============
Unified plotting for every result type the system produces:

- BnB convergence (Fig. 7)
- Rate vs epsilon (Fig. 8)
- Pulse compression (Fig. 9)
- GAN training dynamics (6-panel dashboard)
- Dataset statistics
- Waveform comparison (BnB vs GAN)
- Metric summaries

All functions accept an optional ``ax`` parameter for embedding
individual panels into larger figures, and return the ``Axes``
used.  Combined dashboard functions return the ``Figure``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .style import (
    apply_style, new_fig, save_fig, annotate_metric,
    PALETTE, PALETTE_CYCLE, color,
)

apply_style()

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# =====================================================================
# 1.  BnB Convergence  (Fig. 7 style)
# =====================================================================

def plot_convergence(
    results: list,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot BnB convergence curves (UB solid, LB dashed, gap fill).

    Parameters
    ----------
    results : list[ConvergenceResult]
        From ``run_convergence_experiment()``.
    """
    if ax is None:
        _, ax = plt.subplots()

    for i, r in enumerate(results):
        c = PALETTE_CYCLE[i % len(PALETTE_CYCLE)]
        iters = np.arange(1, len(r.ub_history) + 1)
        ax.plot(iters, r.ub_history, color=c, linestyle="-",
                label=f"{r.label} (UB)")
        ax.plot(iters, r.lb_history, color=c, linestyle="--",
                label=f"{r.label} (LB)", alpha=0.7)
        ax.fill_between(iters, r.lb_history, r.ub_history,
                        color=c, alpha=0.07)

    ax.set_xlabel("BnB Iteration")
    ax.set_ylabel("Objective")
    ax.set_title(title or "BnB Convergence")
    ax.legend(fontsize=8, ncol=2, loc="best")
    return ax


def plot_convergence_grid(
    results: list,
    save_path: str | Path | None = None,
) -> Figure:
    """2×2 convergence grid (one panel per solver combo)."""
    fig, axes = new_fig(2, 2, suptitle="BnB Convergence Behaviour")
    axes_flat = np.asarray(axes).ravel()

    for i, r in enumerate(results[:4]):
        ax = axes_flat[i]
        c = PALETTE_CYCLE[i]
        iters = np.arange(1, len(r.ub_history) + 1)
        ax.plot(iters, r.ub_history, color=c, linestyle="-", label="UB")
        ax.plot(iters, r.lb_history, color=c, linestyle="--", label="LB")
        ax.fill_between(iters, r.lb_history, r.ub_history, color=c, alpha=0.08)
        gap = r.ub_history[-1] - r.lb_history[-1] if r.ub_history else 0
        ax.set_title(r.label, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.legend(fontsize=9)
        annotate_metric(ax, f"gap={gap:.5f}\n{r.n_iterations} iters")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        save_fig(fig, save_path, close=False)
    return fig


# =====================================================================
# 2.  Rate vs Epsilon  (Fig. 8 style)
# =====================================================================

def plot_rate_vs_epsilon(
    result,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Sum-rate vs similarity tolerance.

    Parameters
    ----------
    result : RateVsEpsilonResult
    """
    if ax is None:
        _, ax = plt.subplots()

    eps = result.epsilons
    ax.plot(eps, result.rate_bnb, color=color("red"), marker="o",
            markersize=5, label="BnB (optimal)")
    ax.plot(eps, result.rate_relaxed, color=color("blue"), marker="s",
            markersize=5, linestyle="--", label="Convex relaxation")
    ax.axhline(result.awgn_capacity, color=color("grey"), linestyle=":",
               linewidth=1.2, label="AWGN capacity")

    ax.set_xlabel(r"Similarity tolerance $\varepsilon$")
    ax.set_ylabel("Sum-Rate (bits/s/Hz)")
    ax.set_title(title or "Sum-Rate vs $\\varepsilon$")
    ax.legend()
    annotate_metric(ax, f"{result.n_trials} trials, {result.elapsed_s:.0f}s")
    return ax


# =====================================================================
# 3.  Pulse Compression  (Fig. 9 style)
# =====================================================================

def plot_pulse_compression(
    waveforms: dict[str, NDArray],
    n_fft: int = 160,
    taylor_nbar: int = 4,
    taylor_sll: float = 35.0,
    ax: Axes | None = None,
    title: str | None = None,
    ylim: tuple[float, float] = (-90, 5),
    xlim: tuple[float, float] | None = (-80, 80),
) -> Axes:
    """Overlay pulse-compression profiles for multiple waveforms.

    Parameters
    ----------
    waveforms : dict
        ``{label: waveform_vector}`` — complex (N,) arrays.
    """
    from ..signal_proc.pulse_compression import pulse_compress

    if ax is None:
        _, ax = plt.subplots()

    for i, (label, wf) in enumerate(waveforms.items()):
        bins, mag_dB = pulse_compress(wf, n_fft, taylor_nbar, taylor_sll)
        c = PALETTE_CYCLE[i % len(PALETTE_CYCLE)]
        ax.plot(bins, mag_dB, color=c, label=label, alpha=0.85)

    ax.set_xlabel("Range bin")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title or "Pulse Compression")
    ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(fontsize=9)
    return ax


def plot_pulse_compression_grid(
    ref_waveform: NDArray,
    opt_waveforms: dict[str, NDArray],
    save_path: str | Path | None = None,
    **pc_kwargs,
) -> Figure:
    """1×N grid: reference vs optimised at different epsilons."""
    n = len(opt_waveforms)
    fig, axes = new_fig(1, n, figsize=(6 * n, 5),
                        suptitle="Pulse Compression Comparison")
    if n == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).ravel()

    for i, (label, wf) in enumerate(opt_waveforms.items()):
        plot_pulse_compression(
            {"Reference chirp": ref_waveform, label: wf},
            ax=axes[i], title=label, **pc_kwargs,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        save_fig(fig, save_path, close=False)
    return fig


# =====================================================================
# 4.  GAN Training Dynamics  (6-panel dashboard)
# =====================================================================

def plot_gan_losses(history, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    ep = history.epochs
    ax.plot(ep, history._get("critic_loss"), color=color("blue"),
            label="Critic")
    ax.plot(ep, history._get("generator_loss"), color=color("red"),
            label="Generator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("WGAN-GP Losses")
    ax.legend()
    return ax


def plot_gan_wasserstein(history, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history.epochs, history._get("wasserstein_dist"),
            color=color("purple"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$W$-distance estimate")
    ax.set_title("Wasserstein Distance")
    return ax


def plot_gan_gp(history, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history.epochs, history._get("gradient_penalty"),
            color=color("orange"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Penalty")
    ax.set_title("Gradient Penalty")
    return ax


def plot_gan_rates(history, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    ep = history.epochs
    ax.plot(ep, history._get("rate_real"), color=color("blue"),
            label="BnB (real)")
    ax.plot(ep, history._get("rate_fake"), color=color("red"),
            linestyle="--", label="GAN (generated)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Sum-Rate (bits/s/Hz)")
    ax.set_title("Sum-Rate Comparison")
    ax.legend()
    return ax


def plot_gan_power(history, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history.epochs, history._get("power_violation"),
            color=color("red"))
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean $|\\|x_t\\|^2 - P_T|$")
    ax.set_title("Power Constraint Violation")
    return ax


def plot_gan_similarity(history, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history.epochs, history._get("similarity_violation"),
            color=color("green"))
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--",
               label="Feasibility boundary")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$\\|X_{\\mathrm{gen}} - X_0\\|_F / (\\sqrt{NL}\\,\\varepsilon)$")
    ax.set_title("Similarity Constraint")
    ax.legend()
    return ax


def plot_gan_dashboard(
    history,
    save_path: str | Path | None = None,
) -> Figure:
    """Six-panel GAN training dynamics dashboard."""
    fig, axes = new_fig(2, 3, figsize=(17, 10),
                        suptitle="WGAN-GP Training Dynamics")
    ax = np.asarray(axes).ravel()

    plot_gan_losses(history, ax[0])
    plot_gan_wasserstein(history, ax[1])
    plot_gan_gp(history, ax[2])
    plot_gan_rates(history, ax[3])
    plot_gan_power(history, ax[4])
    plot_gan_similarity(history, ax[5])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        save_fig(fig, save_path, close=False)
    return fig


# =====================================================================
# 5.  Dataset Statistics
# =====================================================================

def plot_dataset_stats(
    dataset,
    save_path: str | Path | None = None,
) -> Figure:
    """Four-panel overview of an HDF5 dataset.

    Panels: epsilon distribution, sum-rate distribution,
    sum-rate vs epsilon scatter, column-power histogram.
    """
    n = len(dataset)
    epsilons = np.array([dataset[i][4] for i in range(n)])
    rates = np.array([dataset[i][5] for i in range(n)])

    # Column powers from X_opt
    powers = []
    for i in range(n):
        X_opt = dataset[i][3]  # (N, L)
        col_pwr = np.sum(np.abs(X_opt) ** 2, axis=0)
        powers.extend(col_pwr.tolist())
    powers = np.array(powers)

    fig, axes = new_fig(2, 2, suptitle="Dataset Statistics")
    ax = np.asarray(axes).ravel()

    # Epsilon histogram
    ax[0].hist(epsilons, bins=max(10, len(np.unique(epsilons))),
               color=color("blue"), alpha=0.75, edgecolor="white")
    ax[0].set_xlabel(r"$\varepsilon$")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Epsilon Distribution")

    # Sum-rate histogram
    ax[1].hist(rates, bins=30, color=color("green"), alpha=0.75,
               edgecolor="white")
    ax[1].set_xlabel("Sum-Rate (bits/s/Hz)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Sum-Rate Distribution")
    annotate_metric(ax[1], f"\u03bc={rates.mean():.2f}, \u03c3={rates.std():.2f}")

    # Rate vs epsilon
    ax[2].scatter(epsilons, rates, s=12, alpha=0.55, color=color("purple"),
                  edgecolors="none")
    ax[2].set_xlabel(r"$\varepsilon$")
    ax[2].set_ylabel("Sum-Rate")
    ax[2].set_title("Rate vs $\\varepsilon$")

    # Column power histogram — handle degenerate case (all equal)
    pwr_range = powers.max() - powers.min()
    if pwr_range < 1e-12:
        ax[3].bar([f"{powers.mean():.4f}"], [len(powers)],
                  color=color("orange"), alpha=0.75, edgecolor="white")
        ax[3].set_xlabel("Column Power $\\|x_t\\|^2$")
        ax[3].set_ylabel("Count")
        ax[3].set_title("Power Distribution (constant)")
    else:
        n_power_bins = min(50, max(5, len(np.unique(powers)) // 2))
        ax[3].hist(powers, bins=n_power_bins, color=color("orange"), alpha=0.75,
                   edgecolor="white")
        ax[3].axvline(np.mean(powers), color="k", linestyle="--", linewidth=1,
                      label=f"mean={np.mean(powers):.4f}")
        ax[3].set_xlabel("Column Power $\\|x_t\\|^2$")
        ax[3].set_ylabel("Count")
        ax[3].set_title("Power Distribution")
        ax[3].legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        save_fig(fig, save_path, close=False)
    return fig


# =====================================================================
# 6.  Waveform Comparison  (BnB vs GAN)
# =====================================================================

def plot_waveform_comparison(
    X_ref: NDArray,
    X_bnb: NDArray,
    X_gan: NDArray,
    col_idx: int = 0,
    save_path: str | Path | None = None,
) -> Figure:
    """Side-by-side phase/magnitude comparison for one time-frame column.

    Three panels: (a) amplitude, (b) phase, (c) constellation diagram.
    """
    fig, axes = new_fig(1, 3, figsize=(16, 4.5),
                        suptitle=f"Waveform Column {col_idx}")
    ax = np.asarray(axes).ravel()

    x_ref = X_ref[:, col_idx] if X_ref.ndim == 2 else X_ref
    x_bnb = X_bnb[:, col_idx] if X_bnb.ndim == 2 else X_bnb
    x_gan = X_gan[:, col_idx] if X_gan.ndim == 2 else X_gan
    n = np.arange(len(x_ref))

    # (a) Amplitude
    ax[0].stem(n, np.abs(x_ref), linefmt="-", markerfmt="o",
               basefmt=" ", label="Reference")
    ax[0].stem(n, np.abs(x_bnb), linefmt="--", markerfmt="s",
               basefmt=" ", label="BnB")
    ax[0].stem(n, np.abs(x_gan), linefmt=":", markerfmt="^",
               basefmt=" ", label="GAN")
    ax[0].set_xlabel("Antenna index")
    ax[0].set_ylabel("$|x_n|$")
    ax[0].set_title("Amplitude")
    ax[0].legend(fontsize=9)

    # (b) Phase
    ax[1].plot(n, np.angle(x_ref, deg=True), "o-", color=color("blue"),
               label="Reference", markersize=4)
    ax[1].plot(n, np.angle(x_bnb, deg=True), "s--", color=color("red"),
               label="BnB", markersize=4)
    ax[1].plot(n, np.angle(x_gan, deg=True), "^:", color=color("green"),
               label="GAN", markersize=4)
    ax[1].set_xlabel("Antenna index")
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_title("Phase")
    ax[1].legend(fontsize=9)

    # (c) Constellation
    for arr, lbl, c, m in [
        (x_ref, "Reference", color("blue"), "o"),
        (x_bnb, "BnB", color("red"), "s"),
        (x_gan, "GAN", color("green"), "^"),
    ]:
        ax[2].scatter(arr.real, arr.imag, c=c, marker=m, s=30,
                      label=lbl, alpha=0.7, edgecolors="none")
    ax[2].set_xlabel("Real")
    ax[2].set_ylabel("Imag")
    ax[2].set_title("Constellation")
    ax[2].set_aspect("equal")
    ax[2].legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        save_fig(fig, save_path, close=False)
    return fig


# =====================================================================
# 7.  Metric Summary Bar Chart
# =====================================================================

def plot_metric_bars(
    labels: Sequence[str],
    values: Sequence[float],
    ax: Axes | None = None,
    title: str = "Metric Summary",
    ylabel: str = "Value",
    horizontal: bool = False,
) -> Axes:
    """Simple grouped bar chart for metric comparison."""
    if ax is None:
        _, ax = plt.subplots()

    colours = [PALETTE_CYCLE[i % len(PALETTE_CYCLE)] for i in range(len(labels))]
    x = np.arange(len(labels))

    if horizontal:
        ax.barh(x, values, color=colours, edgecolor="white", height=0.6)
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel)
    else:
        ax.bar(x, values, color=colours, edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    return ax


# =====================================================================
# 8.  Rate Scatter (BnB vs GAN)
# =====================================================================

def plot_rate_scatter(
    rates_bnb: Sequence[float],
    rates_gan: Sequence[float],
    epsilons: Sequence[float] | None = None,
    ax: Axes | None = None,
    title: str = "BnB vs GAN Sum-Rate",
) -> Axes:
    """Scatter plot of BnB vs GAN per-sample rates."""
    if ax is None:
        _, ax = plt.subplots()

    rb = np.asarray(rates_bnb)
    rg = np.asarray(rates_gan)

    if epsilons is not None:
        eps = np.asarray(epsilons)
        scatter = ax.scatter(rb, rg, c=eps, cmap="viridis", s=25,
                             edgecolors="none", alpha=0.75)
        plt.colorbar(scatter, ax=ax, label=r"$\varepsilon$")
    else:
        ax.scatter(rb, rg, color=color("blue"), s=25, edgecolors="none",
                   alpha=0.75)

    lo = min(rb.min(), rg.min()) * 0.9
    hi = max(rb.max(), rg.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5,
            label="Parity (GAN = BnB)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("BnB Sum-Rate (bits/s/Hz)")
    ax.set_ylabel("GAN Sum-Rate (bits/s/Hz)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    return ax
