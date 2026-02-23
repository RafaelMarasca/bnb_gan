"""
Generator & Critic Networks
============================
Conditional WGAN-GP architectures for RadCom waveform generation.

The condition vector contains (H, S, X0) but **not** epsilon — one GAN
is trained per epsilon value.

Generator
---------
Maps condition vector + latent noise → waveform matrix X_opt.
The final layer applies a **power projection** that normalises each
time-frame column so that ‖x_t‖² = P_T exactly.  This enforces the
transmit power constraint by construction.

Critic (Discriminator)
----------------------
Maps waveform + condition → scalar Wasserstein critic value.
Uses **LayerNorm** (not BatchNorm) as recommended for WGAN-GP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .utils import condition_dim


# =====================================================================
# Building blocks
# =====================================================================

def _gen_block(in_dim: int, out_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    """Generator block: Linear → BatchNorm → ReLU [→ Dropout]."""
    layers: list[nn.Module] = [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _critic_block(in_dim: int, out_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    """Critic block: Linear → LayerNorm → LeakyReLU [→ Dropout]."""
    layers: list[nn.Module] = [
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# =====================================================================
# Power Projection Layer
# =====================================================================

class PowerProjection(nn.Module):
    """Project each column of the output waveform to satisfy ‖x_t‖² = P_T.

    The raw generator output (batch, N, L, 2) is normalised column-wise
    so that each time-frame column has total power exactly P_T.  This is
    a differentiable projection (like L2-normalise + scale).

    Parameters
    ----------
    PT : float
        Target per-column power.
    N : int
        Number of antennas (rows).
    """

    def __init__(self, PT: float, N: int) -> None:
        super().__init__()
        self.register_buffer("target_norm", torch.tensor(math.sqrt(PT)))
        self.N = N

    def forward(self, x: Tensor) -> Tensor:
        """x : (batch, N, L, 2) → normalised (batch, N, L, 2)."""
        # Column norm: ‖x_t‖ = sqrt(sum over n of (re² + im²))
        col_norm = torch.sqrt(
            (x ** 2).sum(dim=(1, 3)).clamp(min=1e-12)
        )  # (batch, L)
        scale = self.target_norm / col_norm  # (batch, L)
        # Broadcast: (batch, 1, L, 1) * (batch, N, L, 2)
        return x * scale[:, None, :, None]


# =====================================================================
# Generator
# =====================================================================

class Generator(nn.Module):
    """Conditional generator: (condition, noise) → X_opt.

    Parameters
    ----------
    N, K, L : int
        System dimensions (antennas, users, frame length).
    PT : float
        Transmit power.
    latent_dim : int
        Dimension of the noise vector z.
    hidden_dims : tuple of int
        Hidden layer sizes.
    dropout : float
        Dropout rate in hidden layers (0 = disabled).
    """

    def __init__(
        self,
        N: int,
        K: int,
        L: int,
        PT: float = 1.0,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 512, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.K = K
        self.L = L
        self.latent_dim = latent_dim

        cond_dim = condition_dim(N, K, L)
        in_dim = cond_dim + latent_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(_gen_block(prev, h, dropout=dropout))
            prev = h
        layers.append(nn.Linear(prev, 2 * N * L))  # output: real+imag
        layers.append(nn.Tanh())  # bound activations before projection

        self.net = nn.Sequential(*layers)
        self.projection = PowerProjection(PT, N)

    def forward(self, condition: Tensor, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        condition : (batch, cond_dim)
        z : (batch, latent_dim)

        Returns
        -------
        X_opt : (batch, N, L, 2) — power-normalised waveform (real/imag).
        """
        x = self.net(torch.cat([condition, z], dim=1))
        x = x.view(-1, self.N, self.L, 2)
        return self.projection(x)


# =====================================================================
# Critic (Discriminator)
# =====================================================================

class Critic(nn.Module):
    """Conditional Wasserstein critic: (X, condition) → scalar.

    Parameters
    ----------
    N, K, L : int
        System dimensions.
    hidden_dims : tuple of int
        Hidden layer sizes.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        N: int,
        K: int,
        L: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cond_dim = condition_dim(N, K, L)
        in_dim = 2 * N * L + cond_dim  # flattened waveform + condition

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(_critic_block(prev, h, dropout=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, X: Tensor, condition: Tensor) -> Tensor:
        """
        Parameters
        ----------
        X : (batch, N, L, 2) — waveform.
        condition : (batch, cond_dim)

        Returns
        -------
        score : (batch, 1) — critic value.
        """
        x_flat = X.reshape(X.size(0), -1)
        return self.net(torch.cat([x_flat, condition], dim=1))
