"""
Built-in Experiments — GAN Training
====================================
Train one WGAN-GP per epsilon value on the generated HDF5 dataset.

Each epsilon gets its own generator / critic pair and checkpoint
directory, so the network never sees epsilon as an input feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseExperiment


class GANTrainingExperiment(BaseExperiment):
    """Train one conditional WGAN-GP per epsilon for waveform generation.

    The HDF5 dataset (which may contain multiple epsilon values) is
    filtered so that each GAN only sees samples matching its epsilon.

    Config keys used
    ----------------
    - ``system.*`` (N, K, L, PT, SNR_dB)
    - ``gan.*`` (architecture + training hyperparams)
    - ``dataset.epsilons`` (the epsilon values to train GANs for)
    - ``seed``

    Requires a dataset to exist (run ``dataset`` experiment first).
    """

    name = "gan_train"
    description = "Train one WGAN-GP per epsilon on generated dataset"

    def run(self, verbose: bool = True) -> dict[str, Any]:
        import torch
        from ..data.nn_dataset import RadComHDF5Dataset, EpsilonFilteredDataset
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

        base_dataset = RadComHDF5Dataset(ds_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Epsilon values to train on — one GAN per epsilon
        epsilons = cfg.dataset.epsilons
        all_results: dict[float, Any] = {}

        for eps in epsilons:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Training GAN for epsilon = {eps}")
                print(f"{'='*60}")

            # Filter dataset for this epsilon
            filtered = EpsilonFilteredDataset(base_dataset, epsilon=eps)
            if len(filtered) == 0:
                if verbose:
                    print(f"  WARNING: No samples for epsilon={eps}, skipping.")
                continue
            if verbose:
                print(f"  Filtered dataset: {len(filtered)} samples")

            # ── Build fresh networks ────────────────────
            G = Generator(
                cfg.system.N, cfg.system.K, cfg.system.L, cfg.system.PT,
                latent_dim=cfg.gan.latent_dim,
                hidden_dims=tuple(cfg.gan.hidden_g),
            )
            C = Critic(
                cfg.system.N, cfg.system.K, cfg.system.L,
                hidden_dims=tuple(cfg.gan.hidden_c),
            )

            eps_tag = f"eps_{eps:.4f}".replace(".", "p")
            ckpt_dir = str(self.output_dir / "checkpoints" / eps_tag)
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

            trainer = WGANGPTrainer(
                G, C, config=tcfg,
                PT=cfg.system.PT,
                N0=cfg.system.N0,
                epsilon=eps,
                device=device,
            )

            if verbose:
                print(f"  Device:    {device}")
                print(f"  Generator: {sum(p.numel() for p in G.parameters()):,} params")
                print(f"  Critic:    {sum(p.numel() for p in C.parameters()):,} params")

            history = trainer.train(filtered, verbose=verbose)

            # ── Persist ─────────────────────────────────
            history.save(self.output_dir / f"history_{eps_tag}.json")
            trainer.save_checkpoint(cfg.gan.n_epochs - 1)

            all_results[eps] = {"history": history, "trainer": trainer}

        # ── Save summary ────────────────────────────────
        self.save_results(
            scalars={
                "n_epochs": cfg.gan.n_epochs,
                "device": device,
                "dataset": str(ds_path),
                "epsilons_trained": epsilons,
                "g_params": sum(p.numel() for p in G.parameters()),
                "c_params": sum(p.numel() for p in C.parameters()),
            },
        )

        base_dataset.close()
        return all_results

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
