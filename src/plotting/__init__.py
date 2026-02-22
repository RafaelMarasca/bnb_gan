"""
RadCom Plotting Package
=======================
Unified styles and figure functions for every result the system produces.
"""

from .style import (
    apply_style,
    new_fig,
    save_fig,
    annotate_metric,
    PALETTE,
    PALETTE_CYCLE,
    color,
)
from . import figures

__all__ = [
    "apply_style",
    "new_fig",
    "save_fig",
    "annotate_metric",
    "PALETTE",
    "PALETTE_CYCLE",
    "color",
    "figures",
]
