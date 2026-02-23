"""
Built-in Experiments — GAN Training
====================================
Train a WGAN-GP on the generated HDF5 dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseExperiment


class GANTrainingExperiment(BaseExperiment):
    """Train a conditional WGAN-GP for waveform generation.

    Config keys used
    ----------------
    - ``system.*`` (N, K, L, PT, SNR_dB)
    - ``gan.*`` (architecture + training hyperparams)
    - ``seed``

    Requires a dataset to exist (run ``dataset`` experiment first).
    """

    name = "gan_train"
    description = "Train WGAN-GP on generated dataset"

    def run(self, verbose: bool = True) -> dict[str, Any]:
        import torch
        from ..data.nn_dataset import RadComHDF5Dataset
        from ..gan import Generator, Critic, WGANGPTrainer, TrainerConfig

        cfg = self.config

        # ── Locate dataset ──────────────────────────────
        ds_path = self._find_dataset()
        if ds_path is None:
            raise RuntimeError(
                "No dataset found. Run 'dataset' experiment first or place "
                "an .h5 file under the experiment output directory."
            )
        if verbose:
            print(f"  Dataset: {ds_path}")

        dataset = RadComHDF5Dataset(ds_path)

        # ── Build networks ──────────────────────────────
        G = Generator(
            cfg.system.N, cfg.system.K, cfg.system.L, cfg.system.PT,
            latent_dim=cfg.gan.latent_dim,
            hidden_dims=tuple(cfg.gan.hidden_g),
        )
        C = Critic(
            cfg.system.N, cfg.system.K, cfg.system.L,
            hidden_dims=tuple(cfg.gan.hidden_c),
        )

        ckpt_dir = str(self.output_dir / "checkpoints")
        tcfg = TrainerConfig(
            n_epochs=cfg.gan.n_epochs,
            batch_size=cfg.gan.batch_size,
            lr_gen=cfg.gan.learning_rate,
            lr_critic=cfg.gan.learning_rate,
            n_critic=cfg.gan.n_critic,
            lambda_gp=cfg.gan.lambda_gp,
            latent_dim=cfg.gan.latent_dim,
            eval_every=cfg.gan.eval_every,
            save_every=cfg.gan.save_every,
            checkpoint_dir=ckpt_dir,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = WGANGPTrainer(
            G, C, config=tcfg,
            PT=cfg.system.PT,
            N0=cfg.system.N0,
            device=device,
        )

        if verbose:
            print(f"  Device:    {device}")
            print(f"  Generator: {sum(p.numel() for p in G.parameters()):,} params")
            print(f"  Critic:    {sum(p.numel() for p in C.parameters()):,} params")

        history = trainer.train(dataset, verbose=verbose)

        # ── Persist ─────────────────────────────────────
        history.save(self.output_dir / "history.json")
        trainer.save_checkpoint(cfg.gan.n_epochs - 1)

        self.save_results(
            scalars={
                "n_epochs": cfg.gan.n_epochs,
                "device": device,
                "dataset": str(ds_path),
                "g_params": sum(p.numel() for p in G.parameters()),
                "c_params": sum(p.numel() for p in C.parameters()),
            },
        )

        dataset.close()
        return {"history": history, "trainer": trainer}

    # ── Helpers ─────────────────────────────────────────

    def _find_dataset(self) -> Path | None:
        """Search for .h5 dataset files in known locations."""
        search_dirs = [
            self.output_dir / "nn_datasets",
            self.output_dir.parent / "dataset" / "nn_datasets",
            self.output_dir.parent / "dataset",
            Path(self.config.output_dir) / self.config.name / "stages" / "dataset" / "nn_datasets",
            Path(self.config.output_dir) / self.config.name / "stages" / "dataset",
            Path(self.config.output_dir) / self.config.name / "experiments" / "dataset" / "nn_datasets",
            Path(self.config.output_dir) / self.config.name / "experiments" / "dataset",
        ]
        for d in search_dirs:
            if d.exists():
                h5s = sorted(d.glob("*.h5"))
                if h5s:
                    return h5s[-1]
        return None
