"""
Signal Processing Package
=========================
Waveform generation, pulse compression, and radar metric calculations.
"""

from .waveform import generate_chirp, generate_channel, generate_symbols
from .pulse_compression import pulse_compress, autocorrelation

__all__ = [
    "generate_chirp",
    "generate_channel",
    "generate_symbols",
    "pulse_compress",
    "autocorrelation",
]
