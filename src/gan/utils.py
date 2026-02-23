"""
Tensor Conversion Utilities
============================
Helpers for converting between NumPy complex arrays and PyTorch real
tensors with shape (..., 2) for real/imaginary channels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from numpy.typing import NDArray


def complex_to_real(z: NDArray) -> Tensor:
    """Convert a complex NumPy array to a real float32 tensor.

    The last dimension is expanded into two channels: [real, imag].

    Parameters
    ----------
    z : ndarray, complex
        Arbitrary-shape complex array.

    Returns
    -------
    Tensor, float32, shape ``(*z.shape, 2)``
    """
    z = np.asarray(z)
    stacked = np.stack([z.real, z.imag], axis=-1)
    return torch.from_numpy(stacked.astype(np.float32))


def real_to_complex(t: Tensor) -> NDArray:
    """Convert a real tensor with trailing dim=2 back to complex NumPy array.

    Parameters
    ----------
    t : Tensor, shape ``(..., 2)``

    Returns
    -------
    ndarray, complex128, shape ``t.shape[:-1]``
    """
    arr = t.detach().cpu().numpy().astype(np.float64)
    return arr[..., 0] + 1j * arr[..., 1]


def flatten_condition(
    H: NDArray,
    S: NDArray,
    X0: NDArray,
) -> Tensor:
    """Flatten and concatenate the conditioning inputs into a single vector.

    Epsilon is **not** included — train one GAN per epsilon instead.

    Parameters
    ----------
    H : ndarray (K, N) complex
    S : ndarray (K, L) complex
    X0 : ndarray (N, L) complex

    Returns
    -------
    Tensor, float32, shape ``(2*K*N + 2*K*L + 2*N*L,)``
    """
    h_flat = complex_to_real(H).reshape(-1)
    s_flat = complex_to_real(S).reshape(-1)
    x0_flat = complex_to_real(X0).reshape(-1)
    return torch.cat([h_flat, s_flat, x0_flat])


def condition_dim(N: int, K: int, L: int) -> int:
    """Return the total condition vector dimension for given system sizes.

    Epsilon is excluded — one GAN is trained per epsilon value.
    """
    return 2 * K * N + 2 * K * L + 2 * N * L
