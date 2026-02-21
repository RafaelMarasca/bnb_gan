"""
Dataset Generator
=================
Simulate and serialize (H, S) -> X_opt datasets for ML integration.
Saves as HDF5 with clear metadata for reproducibility.

The generator runs the full BnB waveform optimizer for each sample,
producing (H, S, X0, epsilon) -> X_opt input–output pairs suitable
for supervised learning or offline analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from ..signal_proc.waveform import generate_channel, generate_chirp, generate_symbols
from ..utils.config import BnBConfig, SystemConfig
from ..optimizer.waveform_optimizer import WaveformMatrixOptimizer


@dataclass
class DatasetSample:
    """A single optimization input–output pair.

    Attributes
    ----------
    H : ndarray, shape (K, N)
        Channel realization.
    S : ndarray, shape (K, L)
        Communication symbols.
    X0 : ndarray, shape (N, L)
        Reference waveform.
    X_opt : ndarray, shape (N, L)
        BnB-optimized waveform.
    epsilon : float
        Similarity tolerance used.
    sum_rate : float
        Achieved sum-rate with X_opt.
    """

    H: NDArray
    S: NDArray
    X0: NDArray
    X_opt: NDArray
    epsilon: float
    sum_rate: float


class DatasetGenerator:
    """Generate and serialize waveform optimization datasets.

    Creates training samples of the form:
        Input:  (H, S, X0, epsilon)
        Output: X_opt (optimized waveform)

    Serialized as HDF5 with metadata for reproducibility.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save dataset files.
    sys_config : SystemConfig, optional
        Physical system parameters (defaults to paper setup).
    bnb_config : BnBConfig, optional
        BnB algorithm configuration (defaults to standard settings).
    """

    def __init__(
        self,
        output_dir: str | Path = "datasets",
        sys_config: SystemConfig | None = None,
        bnb_config: BnBConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sys_config = sys_config or SystemConfig()
        self.bnb_config = bnb_config or BnBConfig()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples: int,
        epsilon: float = 1.0,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[DatasetSample]:
        """Generate a dataset of optimization input–output pairs.

        For each sample a fresh (H, S) realization is drawn, the reference
        waveform X0 is built from the orthogonal chirp, and the BnB optimizer
        produces X_opt.  The achieved sum-rate is also recorded.

        Parameters
        ----------
        n_samples : int
            Number of (H, S, X_opt) samples to generate.
        epsilon : float
            Similarity tolerance.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Print progress per sample.

        Returns
        -------
        list[DatasetSample]
            Generated samples.
        """
        from ..metrics.rate import sum_rate as _sum_rate  # local to avoid circular

        cfg = self.sys_config
        optimizer = WaveformMatrixOptimizer(cfg, self.bnb_config)
        X0 = generate_chirp(cfg.N, cfg.L, cfg.PT)

        samples: list[DatasetSample] = []
        t0 = time.time()

        for i in range(n_samples):
            np.random.seed(seed + i)
            H = generate_channel(cfg.K, cfg.N)
            S = generate_symbols(cfg.K, cfg.L)

            X_opt, _ = optimizer.optimize(H, S, X0, epsilon)
            sr = _sum_rate(H, X_opt, S, cfg.N0)

            samples.append(
                DatasetSample(
                    H=H, S=S, X0=X0, X_opt=X_opt,
                    epsilon=epsilon, sum_rate=sr,
                )
            )

            if verbose:
                elapsed = time.time() - t0
                print(
                    f"  Sample {i + 1}/{n_samples}  "
                    f"rate={sr:.4f} bps/Hz  "
                    f"({elapsed:.1f}s elapsed)"
                )

        return samples

    # ------------------------------------------------------------------
    # HDF5 serialization
    # ------------------------------------------------------------------

    def save_hdf5(
        self,
        samples: list[DatasetSample],
        filename: str = "dataset.h5",
    ) -> Path:
        """Serialize samples to an HDF5 file with full metadata.

        File layout::

            /metadata           (JSON attrs: sys_config, bnb_config, n_samples)
            /samples/0/H        (K, N) complex128
            /samples/0/S        (K, L) complex128
            /samples/0/X0       (N, L) complex128
            /samples/0/X_opt    (N, L) complex128
            /samples/0/epsilon  scalar
            /samples/0/sum_rate scalar
            ...

        Parameters
        ----------
        samples : list[DatasetSample]
            Samples to serialize.
        filename : str
            Output filename (relative to output_dir).

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        path = self.output_dir / filename

        with h5py.File(path, "w") as f:
            # -- metadata --
            meta = f.create_group("metadata")
            meta.attrs["n_samples"] = len(samples)
            meta.attrs["sys_config"] = json.dumps({
                "N": self.sys_config.N,
                "K": self.sys_config.K,
                "L": self.sys_config.L,
                "PT": self.sys_config.PT,
                "SNR_dB": self.sys_config.SNR_dB,
            })
            meta.attrs["bnb_config"] = json.dumps({
                "rule": self.bnb_config.rule,
                "lb_solver": self.bnb_config.lb_solver,
                "ub_solver": self.bnb_config.ub_solver,
                "tol": self.bnb_config.tol,
                "max_iter": self.bnb_config.max_iter,
                "gp_max_iter": self.bnb_config.gp_max_iter,
            })

            # -- samples --
            grp = f.create_group("samples")
            for i, s in enumerate(samples):
                sg = grp.create_group(str(i))
                sg.create_dataset("H", data=s.H)
                sg.create_dataset("S", data=s.S)
                sg.create_dataset("X0", data=s.X0)
                sg.create_dataset("X_opt", data=s.X_opt)
                sg.attrs["epsilon"] = s.epsilon
                sg.attrs["sum_rate"] = s.sum_rate

        return path.resolve()

    @staticmethod
    def load_hdf5(path: str | Path) -> tuple[list[DatasetSample], dict[str, Any]]:
        """Load a dataset from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Path to HDF5 file.

        Returns
        -------
        samples : list[DatasetSample]
            Loaded samples.
        metadata : dict
            Config metadata (sys_config, bnb_config dicts).
        """
        path = Path(path)
        samples: list[DatasetSample] = []
        metadata: dict[str, Any] = {}

        with h5py.File(path, "r") as f:
            # -- metadata --
            meta = f["metadata"]
            metadata["n_samples"] = int(meta.attrs["n_samples"])
            metadata["sys_config"] = json.loads(meta.attrs["sys_config"])
            metadata["bnb_config"] = json.loads(meta.attrs["bnb_config"])

            # -- samples --
            grp = f["samples"]
            for i in range(metadata["n_samples"]):
                sg = grp[str(i)]
                samples.append(
                    DatasetSample(
                        H=np.array(sg["H"]),
                        S=np.array(sg["S"]),
                        X0=np.array(sg["X0"]),
                        X_opt=np.array(sg["X_opt"]),
                        epsilon=float(sg.attrs["epsilon"]),
                        sum_rate=float(sg.attrs["sum_rate"]),
                    )
                )

        return samples, metadata

    # ------------------------------------------------------------------
    # Convenience: generate + save in one call
    # ------------------------------------------------------------------

    def generate_and_save(
        self,
        n_samples: int,
        epsilon: float = 1.0,
        seed: int = 42,
        filename: str = "dataset.h5",
        verbose: bool = True,
    ) -> Path:
        """Generate samples and save to HDF5 in one call.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        epsilon : float
            Similarity tolerance.
        seed : int
            Random seed.
        filename : str
            Output HDF5 filename.
        verbose : bool
            Print progress.

        Returns
        -------
        Path
            Path to the saved HDF5 file.
        """
        samples = self.generate(n_samples, epsilon, seed, verbose)
        path = self.save_hdf5(samples, filename)
        if verbose:
            print(f"\n  Dataset saved: {path}  ({len(samples)} samples)")
        return path
