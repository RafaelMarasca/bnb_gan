"""
Built-in Experiments — Dataset Generation
==========================================
Generate HDF5 training datasets for supervised / GAN training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseExperiment


class DatasetGenerationExperiment(BaseExperiment):
    """Generate an HDF5 dataset of (H, S, X0, ε) → X_opt pairs.

    Config keys used
    ----------------
    - ``system.*`` (N, K, L, PT, SNR_dB)
    - ``bnb.*`` (solver settings)
    - ``dataset.n_samples``, ``dataset.epsilons``, ``dataset.chunk_size``
    - ``seed``
    """

    name = "dataset"
    description = "Generate HDF5 training dataset via BnB optimizer"

    def run(self, verbose: bool = True) -> dict[str, Any]:
        from ..data.nn_dataset import NNDatasetGenerator

        cfg = self.config
        gen = NNDatasetGenerator(
            sys_config=cfg.sys_config,
            bnb_config=cfg.bnb_legacy,
            output_dir=str(self.output_dir),
            chunk_size=cfg.dataset.chunk_size,
        )
        path = gen.generate(
            n_samples=cfg.dataset.n_samples,
            epsilons=cfg.dataset.epsilons,
            seed=cfg.seed,
            verbose=verbose,
            n_workers=cfg.dataset.n_workers,
        )

        self.save_results(
            scalars={
                "dataset_path": str(path),
                "n_samples": cfg.dataset.n_samples,
                "epsilons": cfg.dataset.epsilons,
            },
        )
        return {"dataset_path": path}
