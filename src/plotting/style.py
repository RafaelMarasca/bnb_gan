"""
Unified Plot Style
==================
A single source-of-truth for every figure the system produces.
Import ``apply_style()`` at the top of any plotting code to get
consistent, publication-quality visuals.

The palette and layout are modelled on the BnB convergence plots
(Fig. 7) and extended to all downstream figures.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

# Ensure non-interactive backend for headless / script usage
if os.environ.get("MPLBACKEND") is None:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.ticker as ticker

# =====================================================================
# Colour palette  (8 colours that look great together)
# =====================================================================

PALETTE = {
    "blue":    "#2176FF",
    "red":     "#E63946",
    "green":   "#06D6A0",
    "orange":  "#FF9F1C",
    "purple":  "#7B2D8E",
    "teal":    "#118AB2",
    "grey":    "#6C757D",
    "yellow":  "#FFD166",
}

# Ordered list for automatic cycling
PALETTE_CYCLE = [
    PALETTE["blue"],
    PALETTE["red"],
    PALETTE["green"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["teal"],
    PALETTE["grey"],
    PALETTE["yellow"],
]

# =====================================================================
# Global rcParams — call apply_style() once at import time
# =====================================================================

_STYLE_APPLIED = False


def apply_style() -> None:
    """Apply the unified RadCom plot style globally."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    _STYLE_APPLIED = True

    plt.rcParams.update({
        # --- Figure ---
        "figure.figsize":       (8, 5),
        "figure.dpi":           120,
        "figure.facecolor":     "white",
        "figure.edgecolor":     "none",
        "figure.autolayout":    False,

        # --- Font ---
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Inter", "Segoe UI", "Helvetica", "Arial"],
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      10,

        # --- Axes ---
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "axes.grid.which":      "major",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.prop_cycle":      plt.cycler(color=PALETTE_CYCLE),

        # --- Grid ---
        "grid.alpha":           0.25,
        "grid.linewidth":       0.6,
        "grid.linestyle":       "--",

        # --- Lines ---
        "lines.linewidth":      1.8,
        "lines.markersize":     6,

        # --- Legend ---
        "legend.framealpha":    0.85,
        "legend.edgecolor":     "#CCCCCC",
        "legend.fancybox":      True,
        "legend.borderpad":     0.5,

        # --- Save ---
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.08,

        # --- LaTeX ---
        "text.usetex":          False,
        "mathtext.fontset":     "stixsans",
    })


# Auto-apply on import
apply_style()


# =====================================================================
# High-level figure helpers
# =====================================================================

def new_fig(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray | Axes]:
    """Create a styled figure with subplots.

    Returns
    -------
    fig, axes
        ``axes`` is a single ``Axes`` for 1×1, otherwise a numpy array.
    """
    if figsize is None:
        w = 6 * ncols + 1.5
        h = 4.2 * nrows + 0.8
        figsize = (min(w, 20), min(h, 14))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="600", y=1.01)

    return fig, axes


def save_fig(
    fig: Figure,
    path: str | Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> list[Path]:
    """Save a figure in one or more formats and optionally close it.

    Parameters
    ----------
    fig : Figure
    path : str or Path
        Base path **without** extension (extension comes from *formats*).
    formats : sequence of str
        e.g. ``("png", "eps", "pdf")``.
    dpi : int
    close : bool
        Close the figure after saving.

    Returns
    -------
    list[Path]
        Paths of saved files.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for fmt in formats:
        p = base.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(p)

    if close:
        plt.close(fig)

    return saved


# =====================================================================
# Accessory helpers
# =====================================================================

def annotate_metric(
    ax: Axes,
    text: str,
    xy: tuple[float, float] = (0.97, 0.05),
    fontsize: int = 9,
    **kwargs,
) -> None:
    """Place a small annotation box (e.g. final value) in an axes corner."""
    defaults = dict(
        ha="right", va="bottom",
        transform=ax.transAxes,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
    )
    defaults.update(kwargs)
    ax.annotate(text, xy=xy, **defaults)


def color(name: str) -> str:
    """Look up a named colour from the palette."""
    return PALETTE[name]
