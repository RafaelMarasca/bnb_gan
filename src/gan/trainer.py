"""
WGAN-GP Trainer
===============
Wasserstein GAN with Gradient Penalty training loop for conditional
RadCom waveform generation.

The trainer operates on an HDF5 dataset produced by ``NNDatasetGenerator``
and logs rich per-epoch metrics to a ``TrainingHistory`` object for
downstream plot generation.

Algorithm
---------
1.  For each *critic step* (n_critic per generator step):
    a.  Sample real batch ``(H, S, X0, X_opt, ε, rate)`` from dataset.
    b.  Generate fake waveform: ``X_fake = G(cond, z)``.
    c.  Compute critic scores on real & fake, gradient penalty on
        interpolated samples, and update critic.
2.  Update generator to maximise critic score on ``G(cond, z)``.
3.  Log Wasserstein distance, GP, losses, and physics metrics (rate,
    power violation, similarity violation).

References
----------
- Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
- Fan Liu et al., "Towards Dual-functional Radar-Communication Systems",
  IEEE TSP 2018 (arXiv:1711.05220).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .networks import Generator, Critic
from .history import TrainingHistory
from .utils import complex_to_real, real_to_complex, flatten_condition


# =====================================================================
# Dataset Wrapper (converts HDF5 samples → tensors with condition)
# =====================================================================

class _ConditionedDataset(Dataset):
    """Wraps ``RadComHDF5Dataset`` to return (condition, X_real, epsilon, rate)."""

    def __init__(self, hdf5_dataset) -> None:
        self._ds = hdf5_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        H, S, X0, X_opt, eps, rate = self._ds[idx]
        cond = flatten_condition(H, S, X0, eps)
        X_real = complex_to_real(X_opt)  # (N, L, 2)
        X0_real = complex_to_real(X0)    # (N, L, 2)
        return cond, X_real, X0_real, torch.tensor(eps, dtype=torch.float32), torch.tensor(rate, dtype=torch.float32)


# =====================================================================
# Trainer Config
# =====================================================================

@dataclass
class TrainerConfig:
    """WGAN-GP training hyper-parameters.

    Parameters
    ----------
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr_gen : float
        Generator learning rate.
    lr_critic : float
        Critic learning rate.
    n_critic : int
        Number of critic updates per generator update.
    lambda_gp : float
        Gradient penalty coefficient (λ in the WGAN-GP paper).
    latent_dim : int
        Dimension of the generator's noise input z.
    beta1, beta2 : float
        Adam optimizer betas.
    eval_every : int
        Compute physics metrics every N epochs (rate, power, similarity).
        Set > 1 to speed up training at the cost of coarser history.
    save_every : int
        Save checkpoint every N epochs (0 = disabled).
    checkpoint_dir : str
        Directory for checkpoints.
    """

    n_epochs: int = 200
    batch_size: int = 64
    lr_gen: float = 1e-4
    lr_critic: float = 1e-4
    n_critic: int = 5
    lambda_gp: float = 10.0
    latent_dim: int = 128
    beta1: float = 0.0
    beta2: float = 0.9
    eval_every: int = 1
    save_every: int = 0
    checkpoint_dir: str = "checkpoints"


# =====================================================================
# Trainer
# =====================================================================

class WGANGPTrainer:
    """WGAN-GP trainer for conditional RadCom waveform generation.

    Parameters
    ----------
    generator : Generator
    critic : Critic
    config : TrainerConfig
    PT : float
        Transmit power (for physics metrics).
    N0 : float
        Noise power (for rate computation).
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        config: TrainerConfig | None = None,
        PT: float = 1.0,
        N0: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.cfg = config or TrainerConfig()
        self.device = torch.device(device)
        self.PT = PT
        self.N0 = N0

        self.G = generator.to(self.device)
        self.C = critic.to(self.device)

        self.opt_G = torch.optim.Adam(
            self.G.parameters(), lr=self.cfg.lr_gen,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        self.opt_C = torch.optim.Adam(
            self.C.parameters(), lr=self.cfg.lr_critic,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

        self.history = TrainingHistory()

    # ------------------------------------------------------------------
    # Gradient Penalty
    # ------------------------------------------------------------------

    def _gradient_penalty(
        self, real: Tensor, fake: Tensor, cond: Tensor,
    ) -> Tensor:
        """Compute gradient penalty on interpolated samples."""
        batch = real.size(0)
        alpha = torch.rand(batch, 1, 1, 1, device=self.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.C(interp, cond)

        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads = grads.reshape(batch, -1)
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    # ------------------------------------------------------------------
    # Physics metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _physics_metrics(
        self,
        X_fake: Tensor,
        X0_real: Tensor,
        cond: Tensor,
        eps_batch: Tensor,
        H_batch: np.ndarray | None = None,
        S_batch: np.ndarray | None = None,
        X_opt_batch: np.ndarray | None = None,
    ) -> dict:
        """Compute power violation, similarity ratio, and optional rates."""
        N = self.G.N
        L = self.G.L

        # Power violation: mean |‖x_t‖² − PT| across batch & columns
        col_power = (X_fake ** 2).sum(dim=(1, 3))  # (batch, L)
        power_viol = (col_power - self.PT).abs().mean().item()

        # Similarity: ‖X_fake − X0‖_F / (√(N·L) · ε)
        diff = X_fake - X0_real
        fro_norm = diff.reshape(diff.size(0), -1).norm(2, dim=1)  # (batch,)
        eps_safe = eps_batch.clamp(min=1e-8)
        ratio = (fro_norm / (math.sqrt(N * L) * eps_safe)).mean().item()

        return {"power_violation": power_viol, "similarity_violation": ratio}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataset,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Run the full training loop.

        Parameters
        ----------
        dataset : RadComHDF5Dataset
            HDF5 dataset from ``NNDatasetGenerator``.
        verbose : bool
            Print per-epoch progress.

        Returns
        -------
        TrainingHistory
            Recorded per-epoch metrics (also available as ``self.history``).
        """
        wrapped = _ConditionedDataset(dataset)
        loader = DataLoader(
            wrapped,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # HDF5 not fork-safe on Windows
        )

        t0 = time.time()
        total_critic_steps = 0

        epoch_pbar = tqdm(
            range(self.cfg.n_epochs),
            desc="Training",
            unit="epoch",
            disable=not verbose,
        )
        for epoch in epoch_pbar:
            ep_critic_loss = 0.0
            ep_gen_loss = 0.0
            ep_w_dist = 0.0
            ep_gp = 0.0
            n_critic_updates = 0
            n_gen_updates = 0

            # --- Accumulators for physics metrics ---
            ep_power_viol = 0.0
            ep_sim_viol = 0.0
            ep_rate_real = 0.0
            ep_rate_fake = 0.0
            n_physics = 0

            batch_pbar = tqdm(
                loader,
                desc=f"  Epoch {epoch+1:>{len(str(self.cfg.n_epochs))}}/{self.cfg.n_epochs}",
                unit="batch",
                leave=False,
                disable=not verbose,
            )
            for batch_idx, (cond, X_real, X0_real, eps_batch, rate_batch) in enumerate(batch_pbar):
                cond = cond.to(self.device)
                X_real = X_real.to(self.device)
                X0_real = X0_real.to(self.device)
                eps_batch = eps_batch.to(self.device)
                bs = cond.size(0)

                # ======================================
                # Update Critic
                # ======================================
                self.C.train()
                self.G.train()

                z = torch.randn(bs, self.cfg.latent_dim, device=self.device)
                X_fake = self.G(cond, z).detach()

                d_real = self.C(X_real, cond)
                d_fake = self.C(X_fake, cond)
                gp = self._gradient_penalty(X_real, X_fake, cond)

                loss_C = d_fake.mean() - d_real.mean() + self.cfg.lambda_gp * gp

                self.opt_C.zero_grad()
                loss_C.backward()
                self.opt_C.step()

                w_dist = d_real.mean().item() - d_fake.mean().item()
                ep_critic_loss += loss_C.item()
                ep_w_dist += w_dist
                ep_gp += gp.item()
                n_critic_updates += 1
                total_critic_steps += 1

                # ======================================
                # Update Generator (every n_critic steps)
                # ======================================
                if total_critic_steps % self.cfg.n_critic == 0:
                    z = torch.randn(bs, self.cfg.latent_dim, device=self.device)
                    X_fake = self.G(cond, z)
                    loss_G = -self.C(X_fake, cond).mean()

                    self.opt_G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                    ep_gen_loss += loss_G.item()
                    n_gen_updates += 1

                # ======================================
                # Physics metrics (every eval_every epochs, every batch)
                # ======================================
                if epoch % self.cfg.eval_every == 0:
                    with torch.no_grad():
                        z_eval = torch.randn(bs, self.cfg.latent_dim, device=self.device)
                        X_fake_eval = self.G(cond, z_eval)
                    metrics = self._physics_metrics(
                        X_fake_eval, X0_real, cond, eps_batch,
                    )
                    ep_power_viol += metrics["power_violation"]
                    ep_sim_viol += metrics["similarity_violation"]
                    ep_rate_real += rate_batch.mean().item()
                    ep_rate_fake += self._batch_rate_fake(
                        X_fake_eval, cond, dataset,
                    )
                    n_physics += 1

            # --- Average metrics ---
            has_physics = n_physics > 0
            record = {
                "epoch": epoch,
                "critic_loss": ep_critic_loss / max(n_critic_updates, 1),
                "generator_loss": ep_gen_loss / max(n_gen_updates, 1),
                "wasserstein_dist": ep_w_dist / max(n_critic_updates, 1),
                "gradient_penalty": ep_gp / max(n_critic_updates, 1),
                "rate_real": ep_rate_real / n_physics if has_physics else float("nan"),
                "rate_fake": ep_rate_fake / n_physics if has_physics else float("nan"),
                "power_violation": ep_power_viol / n_physics if has_physics else float("nan"),
                "similarity_violation": ep_sim_viol / n_physics if has_physics else float("nan"),
            }
            self.history.record(**record)

            # Update epoch progress bar postfix with key metrics
            epoch_pbar.set_postfix({
                "C": f"{record['critic_loss']:+.3f}",
                "G": f"{record['generator_loss']:+.3f}",
                "W": f"{record['wasserstein_dist']:.3f}",
                "r_real": f"{record['rate_real']:.2f}",
                "r_fake": f"{record['rate_fake']:.2f}",
            })

            # --- Checkpoint ---
            if self.cfg.save_every > 0 and (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch)

        if verbose:
            print(f"\nTraining complete in {time.time() - t0:.1f}s")

        return self.history

    # ------------------------------------------------------------------
    # Rate computation for fake samples
    # ------------------------------------------------------------------

    def _batch_rate_fake(self, X_fake: Tensor, cond: Tensor, dataset) -> float:
        """Estimate mean sum-rate for a batch of generated waveforms.

        We reconstruct H, S from the condition vector and compute the
        rate metric in NumPy (reusing the existing rate function).
        """
        from ..metrics.rate import sum_rate as _sum_rate

        N, K, L = self.G.N, self.G.K, self.G.L
        X_np = real_to_complex(X_fake.cpu())  # (batch, N, L)
        cond_np = cond.cpu().numpy()

        total = 0.0
        bs = X_np.shape[0]
        h_size = 2 * K * N
        s_size = 2 * K * L

        for i in range(bs):
            c = cond_np[i]
            # Reconstruct H, S from condition vector
            h_flat = c[:h_size].reshape(K, N, 2)
            H = h_flat[..., 0] + 1j * h_flat[..., 1]

            s_flat = c[h_size:h_size + s_size].reshape(K, L, 2)
            S = s_flat[..., 0] + 1j * s_flat[..., 1]

            total += _sum_rate(H, X_np[i], S, self.N0)

        return total / bs

    # ------------------------------------------------------------------
    # Checkpoint & Inference
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int) -> Path:
        """Save generator + critic + optimizers + history."""
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"wgan_epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "generator_state": self.G.state_dict(),
            "critic_state": self.C.state_dict(),
            "opt_G_state": self.opt_G.state_dict(),
            "opt_C_state": self.opt_C.state_dict(),
            "config": self.cfg,
            "history": self.history.records,
        }, path)
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint.  Returns the epoch number."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.G.load_state_dict(ckpt["generator_state"])
        self.C.load_state_dict(ckpt["critic_state"])
        self.opt_G.load_state_dict(ckpt["opt_G_state"])
        self.opt_C.load_state_dict(ckpt["opt_C_state"])
        self.history = TrainingHistory(records=ckpt.get("history", []))
        return ckpt["epoch"]

    @torch.no_grad()
    def generate(
        self,
        H: np.ndarray,
        S: np.ndarray,
        X0: np.ndarray,
        epsilon: float,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Generate waveform(s) using the trained generator.

        Parameters
        ----------
        H : (K, N) complex
        S : (K, L) complex
        X0 : (N, L) complex
        epsilon : float
        n_samples : int
            Number of independent waveform realisations to draw.

        Returns
        -------
        ndarray (n_samples, N, L) complex   (or (N, L) if n_samples=1)
        """
        self.G.eval()
        cond = flatten_condition(H, S, X0, epsilon).to(self.device)
        cond = cond.unsqueeze(0).expand(n_samples, -1)  # (n, cond_dim)
        z = torch.randn(n_samples, self.cfg.latent_dim, device=self.device)
        X_fake = self.G(cond, z)  # (n, N, L, 2)
        result = real_to_complex(X_fake.cpu())
        return result.squeeze(0) if n_samples == 1 else result
