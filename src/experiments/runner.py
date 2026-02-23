"""
Experiment Runner
=================
Step-wise executor for a single experiment.

Each stage is an independent method that:
1. Runs the computation.
2. Saves **all** artifacts (waveforms, summaries, configs) to disk.
3. Returns a structured result.

Stages can be run individually or in sequence.

Directory Layout
----------------
::

    results/{name}/
        config.json
        stages/
            convergence/  summary.json  data.npz
            rate_sweep/   summary.json  data.npz
            dataset/      summary.json  nn_datasets/*.h5
            gan_train/    summary.json  history.json  checkpoints/
        waveforms/
            summary.json
            sample_0000.npz   (H, S, X0, X_bnb, X_gan, ε, rates, ...)
            sample_0001.npz
        figures/
            *.png / *.eps
        report.md

Usage
-----
>>> from src.experiments import ExperimentConfig, ExperimentRunner
>>> cfg = ExperimentConfig.quick_test()
>>> runner = ExperimentRunner(cfg)
>>> runner.run()                                  # all stages
>>> runner.run(stages=["convergence", "plots"])    # selective
>>> runner.run_convergence()                       # single stage
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ExperimentConfig, ALL_STAGES


class ExperimentRunner:
    """Orchestrates a single experiment with step-by-step execution.

    Parameters
    ----------
    config : ExperimentConfig
        Full configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config

        # Directory structure
        self.root = Path(config.output_dir) / config.name
        self.stages_dir = self.root / "stages"
        self.waveforms_dir = self.root / "waveforms"
        self.figures_dir = self.root / "figures"
        for d in (self.stages_dir, self.waveforms_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Persist config
        config.to_json(self.root / "config.json")

        # In-memory artifacts (populated as stages execute)
        self._convergence: dict | None = None
        self._rate_sweep: Any = None
        self._dataset_path: Path | None = None
        self._gan_trainer: Any = None
        self._gan_history: Any = None
        self._waveform_records: list[dict] | None = None

    # =================================================================
    # Public API
    # =================================================================

    def run(
        self,
        stages: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run selected (or all) stages in order.

        Parameters
        ----------
        stages : list of str, optional
            Subset of ``ALL_STAGES``.  Default: the config's stage list.
        verbose : bool
            Print stage-level banners and progress.

        Returns
        -------
        dict  — keys are stage names, values are each stage's result.
        """
        stages = stages or self.cfg.stages
        results: dict[str, Any] = {}
        t0 = time.time()

        for stage in stages:
            if stage not in ALL_STAGES:
                raise ValueError(
                    f"Unknown stage '{stage}'. Choose from: {ALL_STAGES}"
                )
            if verbose:
                self._banner(stage)

            st = time.time()
            method = getattr(self, f"run_{stage}")
            results[stage] = method(verbose=verbose)

            if verbose:
                print(f"  [{stage}] done in {time.time() - st:.1f}s\n")

        if verbose:
            total = time.time() - t0
            print(f"{'=' * 60}")
            print(f"  ALL DONE — {total:.1f}s total")
            print(f"  Results: {self.root.resolve()}")
            print(f"{'=' * 60}\n")

        return results

    # =================================================================
    # Stage 1 — Convergence
    # =================================================================

    def run_convergence(self, verbose: bool = True) -> dict:
        """BnB convergence analysis (4 solver combos)."""
        from ..data.experiments import run_convergence_experiment

        out = self.stages_dir / "convergence"
        out.mkdir(exist_ok=True)

        result = run_convergence_experiment(
            N=self.cfg.N, K=self.cfg.K,
            epsilon=self.cfg.conv_epsilon,
            PT=self.cfg.PT,
            max_iter=self.cfg.bnb_max_iter,
            tol=self.cfg.bnb_tol,
            gp_iters=self.cfg.bnb_gp_iters,
            seed=self.cfg.seed,
            verbose=verbose,
        )
        self._convergence = result

        # ── Persist ──────────────────────────────────────────
        save_data: dict[str, Any] = {
            "H": result["H"], "s": result["s"], "x0": result["x0"],
        }
        summary = []
        for i, r in enumerate(result["results"]):
            save_data[f"ub_history_{i}"] = np.array(r.ub_history)
            save_data[f"lb_history_{i}"] = np.array(r.lb_history)
            summary.append({
                "label": r.label,
                "rule":  r.rule,
                "lb_solver": r.lb_solver,
                "ub_solver": r.ub_solver,
                "objective": r.objective,
                "n_iterations": r.n_iterations,
                "elapsed_s": r.elapsed_s,
                "gap": (r.ub_history[-1] - r.lb_history[-1]
                        if r.ub_history else None),
            })

        np.savez(out / "data.npz", **save_data)
        _save_json(summary, out / "summary.json")
        return result

    # =================================================================
    # Stage 2 — Rate Sweep
    # =================================================================

    def run_rate_sweep(self, verbose: bool = True):
        """Sum-rate vs similarity tolerance sweep."""
        from ..data.experiments import run_rate_vs_epsilon_experiment

        out = self.stages_dir / "rate_sweep"
        out.mkdir(exist_ok=True)

        result = run_rate_vs_epsilon_experiment(
            N=self.cfg.N, K=self.cfg.K, L=self.cfg.L,
            PT=self.cfg.PT, SNR_dB=self.cfg.SNR_dB,
            epsilons=np.array(self.cfg.rate_epsilons),
            n_trials=self.cfg.rate_n_trials,
            bnb_tol=self.cfg.rate_tol,
            bnb_max_iter=self.cfg.rate_max_iter,
            gp_iters=self.cfg.bnb_gp_iters,
            seed=self.cfg.seed,
            verbose=verbose,
        )
        self._rate_sweep = result

        # ── Persist ──────────────────────────────────────────
        np.savez(
            out / "data.npz",
            epsilons=result.epsilons,
            rate_bnb=result.rate_bnb,
            rate_relaxed=result.rate_relaxed,
            awgn_capacity=np.array([result.awgn_capacity]),
        )
        _save_json({
            "epsilons": result.epsilons.tolist(),
            "rate_bnb": result.rate_bnb.tolist(),
            "rate_relaxed": result.rate_relaxed.tolist(),
            "awgn_capacity": result.awgn_capacity,
            "n_trials": result.n_trials,
            "elapsed_s": result.elapsed_s,
        }, out / "summary.json")
        return result

    # =================================================================
    # Stage 3 — Dataset Generation
    # =================================================================

    def run_dataset(self, verbose: bool = True) -> Path:
        """Generate HDF5 training dataset."""
        from ..data.nn_dataset import NNDatasetGenerator

        out = self.stages_dir / "dataset"
        out.mkdir(exist_ok=True)

        gen = NNDatasetGenerator(
            sys_config=self.cfg.sys_config,
            bnb_config=self.cfg.bnb_config,
            output_dir=str(out),
            chunk_size=self.cfg.ds_chunk_size,
        )
        path = gen.generate(
            n_samples=self.cfg.ds_n_samples,
            epsilons=self.cfg.ds_epsilons,
            seed=self.cfg.seed,
            verbose=verbose,
        )
        self._dataset_path = path

        _save_json({
            "dataset_path": str(path),
            "n_samples": self.cfg.ds_n_samples,
            "epsilons": self.cfg.ds_epsilons,
        }, out / "summary.json")
        return path

    # =================================================================
    # Stage 4 — GAN Training
    # =================================================================

    def run_gan_train(self, verbose: bool = True):
        """Train one WGAN-GP per epsilon on the generated dataset."""
        import torch
        from ..data.nn_dataset import RadComHDF5Dataset, EpsilonFilteredDataset
        from ..gan import Generator, Critic, WGANGPTrainer, TrainerConfig

        # Locate dataset
        ds_path = self._find_dataset()
        if ds_path is None:
            raise RuntimeError(
                "No dataset found. Run 'dataset' stage first or place "
                "an .h5 file under stages/dataset/nn_datasets/."
            )
        if verbose:
            print(f"  Dataset: {ds_path}")

        base_dataset = RadComHDF5Dataset(ds_path)
        cfg = self.cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"

        epsilons = cfg.ds_epsilons
        all_histories = {}

        for eps in epsilons:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Training GAN for epsilon = {eps}")
                print(f"{'='*60}")

            filtered = EpsilonFilteredDataset(base_dataset, epsilon=eps)
            if len(filtered) == 0:
                if verbose:
                    print(f"  WARNING: No samples for epsilon={eps}, skipping.")
                continue
            if verbose:
                print(f"  Filtered dataset: {len(filtered)} samples")

            G = Generator(
                cfg.N, cfg.K, cfg.L, cfg.PT,
                latent_dim=cfg.gan_latent_dim,
                hidden_dims=tuple(cfg.gan_hidden_g),
            )
            C = Critic(
                cfg.N, cfg.K, cfg.L,
                hidden_dims=tuple(cfg.gan_hidden_c),
            )

            eps_tag = f"eps_{eps:.4f}".replace(".", "p")
            ckpt_dir = str(self.stages_dir / "gan_train" / "checkpoints" / eps_tag)
            tcfg = TrainerConfig(
                n_epochs=cfg.gan_n_epochs,
                batch_size=cfg.gan_batch_size,
                lr_gen=cfg.gan_lr, lr_critic=cfg.gan_lr,
                n_critic=cfg.gan_n_critic,
                lambda_gp=cfg.gan_lambda_gp,
                latent_dim=cfg.gan_latent_dim,
                eval_every=cfg.gan_eval_every,
                save_every=cfg.gan_save_every,
                checkpoint_dir=ckpt_dir,
            )

            trainer = WGANGPTrainer(
                G, C, config=tcfg,
                PT=cfg.PT, N0=cfg.sys_config.N0,
                epsilon=eps,
                device=device,
            )

            if verbose:
                print(f"  Device:    {device}")
                print(f"  Generator: {sum(p.numel() for p in G.parameters()):,} params")
                print(f"  Critic:    {sum(p.numel() for p in C.parameters()):,} params")

            history = trainer.train(filtered, verbose=verbose)
            all_histories[eps] = history

            # Persist per-epsilon
            gan_dir = self.stages_dir / "gan_train"
            gan_dir.mkdir(exist_ok=True)
            history.save(gan_dir / f"history_{eps_tag}.json")
            trainer.save_checkpoint(cfg.gan_n_epochs - 1)

        self._gan_trainer = None  # no single trainer anymore
        self._gan_history = all_histories

        # ── Persist summary ──────────────────────────────────
        gan_dir = self.stages_dir / "gan_train"
        gan_dir.mkdir(exist_ok=True)
        _save_json({
            "n_epochs": cfg.gan_n_epochs,
            "device": device,
            "dataset": str(ds_path),
            "epsilons_trained": epsilons,
        }, gan_dir / "summary.json")

        base_dataset.close()
        return all_histories

    # =================================================================
    # Stage 5 — Waveform Evaluation
    # =================================================================

    def run_waveform_eval(self, verbose: bool = True) -> list[dict]:
        """Evaluate BnB + GAN, save complete waveform data per sample.

        For each sample this stage saves an ``.npz`` file containing:
        - ``H, S, X0, epsilon`` (inputs)
        - ``X_bnb, rate_bnb, power_bnb, l2_bnb, feasible_bnb`` (BnB)
        - ``X_gan, rate_gan, power_gan, l2_gan, feasible_gan`` (GAN, if available)
        """
        from ..signal_proc.waveform import (
            generate_channel, generate_symbols, generate_chirp,
        )
        from ..metrics.rate import sum_rate
        from ..metrics.similarity import WaveformSimilarityMetric
        from ..optimizer.waveform_optimizer import WaveformMatrixOptimizer

        cfg = self.cfg
        N0 = cfg.sys_config.N0
        optimizer = WaveformMatrixOptimizer(cfg.sys_config, cfg.bnb_config)
        sim_metric = WaveformSimilarityMetric()
        X0 = generate_chirp(cfg.N, cfg.L, cfg.PT)

        # Try to get a GAN generator for each epsilon
        gan_trainers: dict[float, Any] = {}
        for eps_val in cfg.eval_epsilons:
            t = self._load_gan_trainer(eps_val)
            if t is not None:
                gan_trainers[eps_val] = t
        gan_ok = len(gan_trainers) > 0
        if verbose:
            print(f"  GANs available: {len(gan_trainers)}/{len(cfg.eval_epsilons)}")

        records: list[dict] = []

        for i in range(cfg.eval_n_samples):
            np.random.seed(cfg.seed + 10_000 + i)
            H = generate_channel(cfg.K, cfg.N)
            S = generate_symbols(cfg.K, cfg.L)
            eps = cfg.eval_epsilons[i % len(cfg.eval_epsilons)]

            # ── BnB ─────────────────────────────────────────
            X_bnb, _ = optimizer.optimize(H, S, X0, eps)
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
            }

            rec: dict[str, Any] = {
                "sample": i, "epsilon": eps,
                "rate_bnb": rate_bnb,
                "l2_bnb": float(sim_bnb.values["l2_dist"]),
                "feasible_bnb": bool(sim_bnb.values["feasible"]),
            }

            # ── GAN ──────────────────────────────────────────
            trainer = gan_trainers.get(eps)
            if trainer is not None:
                X_gan = trainer.generate(H, S, X0)
                rate_gan = sum_rate(H, X_gan, S, N0)
                sim_gan = sim_metric.compute(x=X_gan, x0=X0, epsilon=eps)
                power_gan = np.sum(np.abs(X_gan) ** 2, axis=0)

                save_data.update({
                    "X_gan": X_gan,
                    "rate_gan": rate_gan,
                    "power_gan": power_gan,
                    "l2_gan": sim_gan.values["l2_dist"],
                    "feasible_gan": sim_gan.values["feasible"],
                })
                rec.update({
                    "rate_gan": rate_gan,
                    "rate_ratio": rate_gan / max(rate_bnb, 1e-12),
                    "l2_gan": float(sim_gan.values["l2_dist"]),
                    "feasible_gan": bool(sim_gan.values["feasible"]),
                })

            # ── Save .npz ────────────────────────────────────
            np.savez(self.waveforms_dir / f"sample_{i:04d}.npz", **save_data)
            records.append(rec)

            if verbose:
                parts = [
                    f"  [{i+1}/{cfg.eval_n_samples}]  "
                    f"\u03b5={eps:.2f}  BnB={rate_bnb:.3f}"
                ]
                if trainer is not None:
                    parts.append(
                        f"  GAN={rate_gan:.3f}  "
                        f"ratio={rec['rate_ratio']:.3f}"
                    )
                print("".join(parts))

        self._waveform_records = records
        _save_json(records, self.waveforms_dir / "summary.json")

        # Print summary
        if verbose and records:
            rb = [r["rate_bnb"] for r in records]
            print(f"\n  BnB rate: mean={np.mean(rb):.3f}")
            if any("rate_ratio" in r for r in records):
                ratios = [r["rate_ratio"] for r in records if "rate_ratio" in r]
                print(
                    f"  GAN/BnB ratio: mean={np.mean(ratios):.3f}, "
                    f"min={np.min(ratios):.3f}, max={np.max(ratios):.3f}"
                )

        return records

    # =================================================================
    # Stage 6 — Plots
    # =================================================================

    def run_plots(self, verbose: bool = True) -> list[Path]:
        """Generate all figures from available results."""
        from ..plotting import figures as F
        from ..plotting.style import save_fig, new_fig, apply_style, PALETTE_CYCLE

        apply_style()
        saved: list[Path] = []
        fmts = self.cfg.plot_formats

        # 1. Convergence
        conv = self._convergence or self._load_convergence()
        if conv is not None:
            if verbose:
                print("  Plotting convergence...")
            fig = F.plot_convergence_grid(conv["results"])
            saved += save_fig(fig, self.figures_dir / "convergence", fmts)

        # 2. Rate sweep
        rate = self._rate_sweep or self._load_rate_sweep()
        if rate is not None:
            if verbose:
                print("  Plotting rate vs epsilon...")
            fig, ax = new_fig()
            F.plot_rate_vs_epsilon(rate, ax=ax)
            saved += save_fig(fig, self.figures_dir / "rate_vs_epsilon", fmts)

        # 3. GAN dashboard
        hist = self._gan_history or self._load_gan_history()
        if hist is not None and len(hist) > 0:
            if verbose:
                print("  Plotting GAN dashboard...")
            fig = F.plot_gan_dashboard(hist)
            saved += save_fig(fig, self.figures_dir / "gan_training", fmts)

        # 4. Dataset stats
        ds_path = self._find_dataset()
        if ds_path is not None and ds_path.exists():
            if verbose:
                print("  Plotting dataset statistics...")
            from ..data.nn_dataset import RadComHDF5Dataset
            ds = RadComHDF5Dataset(ds_path)
            fig = F.plot_dataset_stats(ds)
            saved += save_fig(fig, self.figures_dir / "dataset_stats", fmts)
            ds.close()

        # 5. Waveform evaluation plots
        wf_recs = self._waveform_records or self._load_waveform_summary()
        if wf_recs:
            if verbose:
                print("  Plotting evaluation results...")
            saved += self._plot_eval(wf_recs, fmts, F, new_fig, save_fig,
                                     PALETTE_CYCLE)

        if verbose:
            print(f"  {len(saved)} figure(s) saved to {self.figures_dir}")
        return saved

    # =================================================================
    # Stage 7 — Report
    # =================================================================

    def run_report(self, verbose: bool = True) -> Path:
        """Generate a Markdown report."""
        from .results import ExperimentResult
        from .report import ReportGenerator

        result = ExperimentResult(self.root)
        gen = ReportGenerator(result)
        path = gen.save()
        if verbose:
            print(f"  Report saved: {path}")
        return path

    # =================================================================
    # Private helpers
    # =================================================================

    def _banner(self, stage: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  STAGE: {stage}")
        print(f"{'=' * 60}")

    # ── Artifact discovery ──────────────────────────────────

    def _find_dataset(self) -> Path | None:
        """Locate the HDF5 dataset (memory → disk)."""
        if self._dataset_path and self._dataset_path.exists():
            return self._dataset_path
        for d in [
            self.stages_dir / "dataset" / "nn_datasets",
            self.root / "nn_datasets",
        ]:
            if d.exists():
                h5s = sorted(d.glob("*.h5"))
                if h5s:
                    return h5s[-1]
        return None

    def _load_gan_trainer(self, epsilon: float = 1.0):
        """Try to rebuild a GAN trainer from a saved checkpoint for a specific epsilon."""
        eps_tag = f"eps_{epsilon:.4f}".replace(".", "p")
        candidates = [
            self.stages_dir / "gan_train" / "checkpoints" / eps_tag,
            # Fallback: legacy flat checkpoint dir
            self.stages_dir / "gan_train" / "checkpoints",
        ]
        ckpt_dir = None
        for d in candidates:
            if d.exists():
                ckpts = sorted(d.glob("*.pt"))
                if ckpts:
                    ckpt_dir = d
                    break

        if ckpt_dir is None:
            return None
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        if not ckpts:
            return None

        try:
            import torch
            from ..gan import Generator, Critic, WGANGPTrainer, TrainerConfig

            cfg = self.cfg
            G = Generator(
                cfg.N, cfg.K, cfg.L, cfg.PT,
                latent_dim=cfg.gan_latent_dim,
                hidden_dims=tuple(cfg.gan_hidden_g),
            )
            C = Critic(
                cfg.N, cfg.K, cfg.L,
                hidden_dims=tuple(cfg.gan_hidden_c),
            )
            tcfg = TrainerConfig(
                n_epochs=cfg.gan_n_epochs,
                batch_size=cfg.gan_batch_size,
                lr_gen=cfg.gan_lr, lr_critic=cfg.gan_lr,
                n_critic=cfg.gan_n_critic,
                lambda_gp=cfg.gan_lambda_gp,
                latent_dim=cfg.gan_latent_dim,
                checkpoint_dir=str(ckpt_dir),
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = WGANGPTrainer(
                G, C, config=tcfg,
                PT=cfg.PT, N0=cfg.sys_config.N0,
                epsilon=epsilon,
                device=device,
            )
            trainer.load_checkpoint(ckpts[-1])
            return trainer
        except Exception:
            return None

    def _load_convergence(self) -> dict | None:
        """Reconstruct convergence data from disk."""
        s_path = self.stages_dir / "convergence" / "summary.json"
        d_path = self.stages_dir / "convergence" / "data.npz"
        if not s_path.exists() or not d_path.exists():
            return None

        from ..data.experiments import ConvergenceResult
        summary = json.loads(s_path.read_text())
        data = dict(np.load(d_path, allow_pickle=True))

        results = []
        for i, s in enumerate(summary):
            results.append(ConvergenceResult(
                label=s["label"],
                rule=s.get("rule", ""),
                lb_solver=s.get("lb_solver", ""),
                ub_solver=s.get("ub_solver", ""),
                lb_history=list(data.get(f"lb_history_{i}", [])),
                ub_history=list(data.get(f"ub_history_{i}", [])),
                objective=s["objective"],
                elapsed_s=s["elapsed_s"],
                n_iterations=s["n_iterations"],
            ))
        return {
            "results": results,
            "H": data.get("H"),
            "s": data.get("s"),
            "x0": data.get("x0"),
        }

    def _load_rate_sweep(self):
        """Reconstruct RateVsEpsilonResult from disk."""
        d_path = self.stages_dir / "rate_sweep" / "data.npz"
        s_path = self.stages_dir / "rate_sweep" / "summary.json"
        if not d_path.exists():
            return None

        from ..data.experiments import RateVsEpsilonResult
        data = dict(np.load(d_path, allow_pickle=True))
        summary = json.loads(s_path.read_text()) if s_path.exists() else {}

        return RateVsEpsilonResult(
            epsilons=data["epsilons"],
            rate_bnb=data["rate_bnb"],
            rate_relaxed=data["rate_relaxed"],
            awgn_capacity=float(data.get("awgn_capacity", [0])[0]
                                if hasattr(data.get("awgn_capacity", 0), '__len__')
                                else data.get("awgn_capacity", 0)),
            n_trials=summary.get("n_trials", 0),
            elapsed_s=summary.get("elapsed_s", 0),
        )

    def _load_gan_history(self):
        """Load training history from disk.

        Returns a single ``TrainingHistory`` (first found) or a dict
        mapping epsilon → history if multiple per-epsilon files exist.
        Falls back to legacy ``history.json`` for backward compatibility.
        """
        from ..gan.history import TrainingHistory

        gan_dir = self.stages_dir / "gan_train"
        # Try per-epsilon histories
        per_eps = sorted(gan_dir.glob("history_eps_*.json"))
        if per_eps:
            # For the GAN dashboard, merge or return the first one
            # (dashboard currently expects a single history)
            return TrainingHistory.load(per_eps[0])

        # Legacy single-history fallback
        p = gan_dir / "history.json"
        if not p.exists():
            return None
        return TrainingHistory.load(p)

    def _load_waveform_summary(self) -> list[dict] | None:
        """Load waveform eval summary from disk."""
        p = self.waveforms_dir / "summary.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    # ── Plot helpers ────────────────────────────────────────

    def _plot_eval(self, records, fmts, F, new_fig, save_fig,
                   PALETTE_CYCLE) -> list[Path]:
        """Generate all waveform-evaluation figures."""
        saved: list[Path] = []
        has_gan = any("rate_gan" in r for r in records)
        mean_bnb = np.mean([r["rate_bnb"] for r in records])

        # Rate comparison bar chart
        if has_gan:
            mean_gan = np.mean([
                r["rate_gan"] for r in records if "rate_gan" in r
            ])
            fig, ax = new_fig()
            F.plot_metric_bars(
                ["BnB (optimal)", "GAN (generated)"],
                [mean_bnb, mean_gan],
                ax=ax, title="Mean Sum-Rate Comparison",
                ylabel="Sum-Rate (bps/Hz)",
            )
            saved += save_fig(fig, self.figures_dir / "eval_rate_comparison",
                              fmts)

        # Rate scatter — BnB vs GAN
        if has_gan:
            rb = [r["rate_bnb"] for r in records if "rate_gan" in r]
            rg = [r["rate_gan"] for r in records if "rate_gan" in r]
            ep = [r["epsilon"] for r in records if "rate_gan" in r]
            fig, ax = new_fig()
            F.plot_rate_scatter(rb, rg, ep, ax=ax)
            saved += save_fig(fig, self.figures_dir / "eval_rate_scatter", fmts)

        # Boxplot by epsilon
        if has_gan:
            import matplotlib.pyplot as plt
            fig, ax = new_fig()
            eps_vals = sorted(set(r["epsilon"] for r in records))
            bp_data, bp_labels = [], []
            for eps in eps_vals:
                ratios = [
                    r["rate_ratio"]
                    for r in records
                    if r["epsilon"] == eps and "rate_ratio" in r
                ]
                if ratios:
                    bp_data.append(ratios)
                    bp_labels.append(f"\u03b5={eps}")
            if bp_data:
                bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True)
                for patch, c in zip(bp["boxes"], PALETTE_CYCLE):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
                ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8,
                           label="Parity")
                ax.set_ylabel("Rate Ratio (GAN / BnB)")
                ax.set_title("GAN Quality by $\\varepsilon$")
                ax.legend(fontsize=9)
            saved += save_fig(fig, self.figures_dir / "eval_rate_boxplot", fmts)

        # Waveform comparison for first few samples
        n_show = min(3, len(list(self.waveforms_dir.glob("sample_*.npz"))))
        for idx in range(n_show):
            f = self.waveforms_dir / f"sample_{idx:04d}.npz"
            if not f.exists():
                continue
            data = dict(np.load(f, allow_pickle=True))
            if "X_gan" in data:
                fig = F.plot_waveform_comparison(
                    data["X0"], data["X_bnb"], data["X_gan"], col_idx=0,
                )
                saved += save_fig(
                    fig, self.figures_dir / f"waveform_sample_{idx:04d}", fmts,
                )

        # Pulse compression for first sample
        s0 = self.waveforms_dir / "sample_0000.npz"
        if s0.exists():
            data = dict(np.load(s0, allow_pickle=True))
            wfs = {
                "Reference (chirp)": data["X0"][:, 0],
                "BnB": data["X_bnb"][:, 0],
            }
            if "X_gan" in data:
                wfs["GAN"] = data["X_gan"][:, 0]
            fig, ax = new_fig()
            F.plot_pulse_compression(wfs, ax=ax)
            saved += save_fig(fig, self.figures_dir / "pulse_compression", fmts)

        return saved


# =====================================================================
# JSON helpers
# =====================================================================

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
