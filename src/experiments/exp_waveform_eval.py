"""
Built-in Experiments — Waveform Evaluation
==========================================
Compare BnB (optimal) vs GAN (generated) waveforms on fresh samples.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseExperiment


class WaveformEvalExperiment(BaseExperiment):
    """Evaluate BnB + GAN waveforms side-by-side.

    For each sample saves an ``.npz`` file containing inputs (H, S, X0, ε)
    and outputs (X_bnb, X_gan, rates, power, similarity metrics, and
    execution times for both methods).

    Config keys used
    ----------------
    - ``system.*`` (N, K, L, PT, SNR_dB)
    - ``bnb.*`` (solver settings)
    - ``eval.n_samples``, ``eval.epsilons``
    - ``gan.*`` (for loading trained model)
    - ``seed``
    """

    name = "waveform_eval"
    description = "Evaluate BnB vs GAN waveform quality"

    def run(self, verbose: bool = True) -> dict[str, Any]:
        from ..signal_proc.waveform import (
            generate_channel, generate_symbols, generate_chirp,
        )
        from ..metrics.rate import sum_rate
        from ..metrics.similarity import WaveformSimilarityMetric
        from ..optimizer.waveform_optimizer import WaveformMatrixOptimizer

        cfg = self.config
        N0 = cfg.system.N0
        optimizer = WaveformMatrixOptimizer(cfg.sys_config, cfg.bnb_legacy)
        sim_metric = WaveformSimilarityMetric()
        X0 = generate_chirp(cfg.system.N, cfg.system.L, cfg.system.PT)

        # Try to load a trained GAN
        trainer = self._load_gan_trainer()
        gan_ok = trainer is not None
        if verbose:
            print(f"  GAN available: {gan_ok}")

        waveforms_dir = self.output_dir / "waveforms"
        waveforms_dir.mkdir(exist_ok=True)
        records: list[dict] = []

        for i in range(cfg.eval.n_samples):
            np.random.seed(cfg.seed + 10_000 + i)
            H = generate_channel(cfg.system.K, cfg.system.N)
            S = generate_symbols(cfg.system.K, cfg.system.L)
            eps = cfg.eval.epsilons[i % len(cfg.eval.epsilons)]

            # ── BnB ─────────────────────────────────────
            t_bnb_start = time.perf_counter()
            X_bnb, _ = optimizer.optimize(H, S, X0, eps)
            t_bnb = time.perf_counter() - t_bnb_start

            rate_bnb = sum_rate(H, X_bnb, S, N0)
            sim_bnb = sim_metric.compute(x=X_bnb, x0=X0, epsilon=eps)
            power_bnb = np.sum(np.abs(X_bnb) ** 2, axis=0)

            save_data: dict[str, Any] = {
                "H": H, "S": S, "X0": X0, "epsilon": eps,
                "X_bnb": X_bnb,
                "rate_bnb": rate_bnb,
                "power_bnb": power_bnb,
                "l2_bnb": sim_bnb.values["l2_dist"],
                "feasible_bnb": sim_bnb.values["feasible"],
                "time_bnb": t_bnb,
            }
            rec: dict[str, Any] = {
                "sample": i,
                "epsilon": eps,
                "rate_bnb": rate_bnb,
                "l2_bnb": float(sim_bnb.values["l2_dist"]),
                "feasible_bnb": bool(sim_bnb.values["feasible"]),
                "time_bnb_s": t_bnb,
            }

            # ── GAN ──────────────────────────────────────
            if gan_ok:
                t_gan_start = time.perf_counter()
                X_gan = trainer.generate(H, S, X0, eps)
                t_gan = time.perf_counter() - t_gan_start

                rate_gan = sum_rate(H, X_gan, S, N0)
                sim_gan = sim_metric.compute(x=X_gan, x0=X0, epsilon=eps)
                power_gan = np.sum(np.abs(X_gan) ** 2, axis=0)

                save_data.update({
                    "X_gan": X_gan,
                    "rate_gan": rate_gan,
                    "power_gan": power_gan,
                    "l2_gan": sim_gan.values["l2_dist"],
                    "feasible_gan": sim_gan.values["feasible"],
                    "time_gan": t_gan,
                })
                rec.update({
                    "rate_gan": rate_gan,
                    "rate_ratio": rate_gan / max(rate_bnb, 1e-12),
                    "l2_gan": float(sim_gan.values["l2_dist"]),
                    "feasible_gan": bool(sim_gan.values["feasible"]),
                    "time_gan_s": t_gan,
                    "speedup": t_bnb / max(t_gan, 1e-12),
                })

            np.savez(waveforms_dir / f"sample_{i:04d}.npz", **save_data)
            records.append(rec)

            if verbose:
                parts = [
                    f"  [{i+1}/{cfg.eval.n_samples}]  "
                    f"\u03b5={eps:.2f}  BnB={rate_bnb:.3f} ({t_bnb:.3f}s)"
                ]
                if gan_ok:
                    parts.append(
                        f"  GAN={rec['rate_gan']:.3f} ({t_gan:.4f}s)  "
                        f"ratio={rec['rate_ratio']:.3f}  "
                        f"speedup={rec['speedup']:.1f}x"
                    )
                print("".join(parts))

        # ── Save summary ────────────────────────────────
        summary_scalars: dict[str, Any] = {"records": records}

        # Aggregate timing statistics
        if records:
            t_bnb_all = [r["time_bnb_s"] for r in records]
            summary_scalars["time_bnb_mean_s"] = float(np.mean(t_bnb_all))
            summary_scalars["time_bnb_std_s"] = float(np.std(t_bnb_all))
            summary_scalars["time_bnb_total_s"] = float(np.sum(t_bnb_all))

            if gan_ok:
                t_gan_all = [r["time_gan_s"] for r in records if "time_gan_s" in r]
                speedups = [r["speedup"] for r in records if "speedup" in r]
                summary_scalars["time_gan_mean_s"] = float(np.mean(t_gan_all))
                summary_scalars["time_gan_std_s"] = float(np.std(t_gan_all))
                summary_scalars["time_gan_total_s"] = float(np.sum(t_gan_all))
                summary_scalars["speedup_mean"] = float(np.mean(speedups))
                summary_scalars["speedup_min"] = float(np.min(speedups))
                summary_scalars["speedup_max"] = float(np.max(speedups))

        self.save_results(scalars=summary_scalars)

        if verbose and records:
            rb = [r["rate_bnb"] for r in records]
            t_bnb_all = [r["time_bnb_s"] for r in records]
            print(f"\n  BnB rate: mean={np.mean(rb):.3f}")
            print(
                f"  BnB time: mean={np.mean(t_bnb_all):.4f}s, "
                f"total={np.sum(t_bnb_all):.2f}s"
            )
            if gan_ok:
                ratios = [r["rate_ratio"] for r in records if "rate_ratio" in r]
                t_gan_all = [r["time_gan_s"] for r in records if "time_gan_s" in r]
                speedups = [r["speedup"] for r in records if "speedup" in r]
                print(
                    f"  GAN/BnB ratio: mean={np.mean(ratios):.3f}, "
                    f"min={np.min(ratios):.3f}, max={np.max(ratios):.3f}"
                )
                print(
                    f"  GAN time: mean={np.mean(t_gan_all):.4f}s, "
                    f"total={np.sum(t_gan_all):.2f}s"
                )
                print(
                    f"  Speedup (BnB/GAN): mean={np.mean(speedups):.1f}x, "
                    f"min={np.min(speedups):.1f}x, max={np.max(speedups):.1f}x"
                )

        return {"records": records, "waveforms_dir": waveforms_dir}

    # ── Helpers ─────────────────────────────────────────

    def _load_gan_trainer(self) -> Any:
        """Try to rebuild a GAN trainer from a saved checkpoint."""
        search_dirs = [
            self.output_dir.parent / "gan_train" / "checkpoints",
            Path(self.config.output_dir) / self.config.name / "stages" / "gan_train" / "checkpoints",
            Path(self.config.output_dir) / self.config.name / "experiments" / "gan_train" / "checkpoints",
        ]
        ckpt_dir = None
        for d in search_dirs:
            if d.exists() and sorted(d.glob("*.pt")):
                ckpt_dir = d
                break

        if ckpt_dir is None:
            return None

        try:
            import torch
            from ..gan import Generator, Critic, WGANGPTrainer, TrainerConfig

            cfg = self.config
            G = Generator(
                cfg.system.N, cfg.system.K, cfg.system.L, cfg.system.PT,
                latent_dim=cfg.gan.latent_dim,
                hidden_dims=tuple(cfg.gan.hidden_g),
            )
            C = Critic(
                cfg.system.N, cfg.system.K, cfg.system.L,
                hidden_dims=tuple(cfg.gan.hidden_c),
            )
            tcfg = TrainerConfig(
                n_epochs=cfg.gan.n_epochs,
                batch_size=cfg.gan.batch_size,
                lr_gen=cfg.gan.learning_rate,
                lr_critic=cfg.gan.learning_rate,
                n_critic=cfg.gan.n_critic,
                lambda_gp=cfg.gan.lambda_gp,
                latent_dim=cfg.gan.latent_dim,
                checkpoint_dir=str(ckpt_dir),
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = WGANGPTrainer(
                G, C, config=tcfg,
                PT=cfg.system.PT,
                N0=cfg.system.N0,
                device=device,
            )
            ckpts = sorted(ckpt_dir.glob("*.pt"))
            trainer.load_checkpoint(ckpts[-1])
            return trainer
        except Exception:
            return None
