"""
GAN Package
===========
Conditional WGAN-GP for RadCom waveform generation.

Modules
-------
- networks : Generator & Critic architectures
- trainer  : WGAN-GP training loop
- history  : Training metrics logging & plotting
- utils    : Complex ↔ real tensor conversion helpers
"""

from .networks import Generator, Critic
from .trainer import WGANGPTrainer, TrainerConfig
from .history import TrainingHistory
from .utils import complex_to_real, real_to_complex

__all__ = [
    "Generator",
    "Critic",
    "WGANGPTrainer",
    "TrainerConfig",
    "TrainingHistory",
    "complex_to_real",
    "real_to_complex",
]
