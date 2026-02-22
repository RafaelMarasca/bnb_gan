"""
Streaming NN Dataset Generator
==============================
Generates large-scale waveform optimization datasets for neural network
training **without loading the entire dataset into RAM**.

Samples are produced in configurable chunks and flushed to an HDF5 file
with resizable datasets after each chunk.  This means:

*  RAM usage stays bounded (≈ ``chunk_size`` samples at a time).
*  On interrupt / crash, all previously-flushed chunks are safely on disk.
*  The resulting HDF5 file uses chunked storage, ideal for random-access
   data loaders (PyTorch ``DataLoader`` with ``num_workers > 0``).

HDF5 Layout
-----------
::

    /H          (n_samples, K, N)       complex128   — channel matrices
    /S          (n_samples, K, L)       complex128   — symbol matrices
    /X0         (n_samples, N, L)       complex128   — reference waveforms
    /X_opt      (n_samples, N, L)       complex128   — optimized waveforms
    /epsilon    (n_samples,)            float64      — similarity tolerance
    /sum_rate   (n_samples,)            float64      — achieved rate
    /metadata   attrs: sys_config, bnb_config, creation_time, …

Usage
-----
>>> from src.data.nn_dataset import NNDatasetGenerator, RadComHDF5Dataset
>>> gen = NNDatasetGenerator(sys_config=cfg, bnb_config=bnb_cfg)
>>> path = gen.generate(n_samples=10_000, epsilons=[0.2, 0.6, 1.0])
>>> ds = RadComHDF5Dataset(path)       # lazy loader
>>> H, S, X0, X_opt, eps, rate = ds[0] # numpy arrays, zero-copy
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from numpy.typing import NDArray

from ..signal_proc.waveform import generate_channel, generate_chirp, generate_symbols
from ..utils.config import BnBConfig, SystemConfig
from ..optimizer.waveform_optimizer import WaveformMatrixOptimizer


# =====================================================================
# Generator
# =====================================================================

class NNDatasetGenerator:
    """Streaming dataset generator with chunked HDF5 output.

    Parameters
    ----------
    sys_config : SystemConfig, optional
        Physical system parameters.
    bnb_config : BnBConfig, optional
        BnB solver configuration.  For speed on large datasets, consider
        using GP solvers: ``BnBConfig(lb_solver='gp', ub_solver='gp')``.
    output_dir : str or Path
        Root directory for saved datasets.  A ``nn_datasets/`` subfolder
        is created automatically.
    chunk_size : int
        Number of samples held in RAM before flushing to disk.
        Lower values save memory; higher values reduce I/O overhead.
    """

    DEFAULT_DIR = "nn_datasets"

    def __init__(
        self,
        sys_config: SystemConfig | None = None,
        bnb_config: BnBConfig | None = None,
        output_dir: str | Path = ".",
        chunk_size: int = 50,
    ) -> None:
        self.sys_config = sys_config or SystemConfig()
        self.bnb_config = bnb_config or BnBConfig()
        self.output_dir = Path(output_dir) / self.DEFAULT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, chunk_size)

    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples: int,
        epsilons: float | Sequence[float] = 1.0,
        seed: int = 0,
        filename: str | None = None,
        verbose: bool = True,
    ) -> Path:
        """Generate *n_samples* and stream them to an HDF5 file.

        For each sample a random (H, S) realization is drawn.  The epsilon
        value is cycled from *epsilons* so the dataset contains a balanced
        mix of similarity tolerances.

        Parameters
        ----------
        n_samples : int
            Total number of samples to generate.
        epsilons : float or sequence of float
            Similarity tolerance(s).  If a list is provided, samples cycle
            through the values (round-robin).
        seed : int
            Base random seed (sample *i* uses ``seed + i``).
        filename : str, optional
            HDF5 filename.  Defaults to an auto-generated name with a
            timestamp and sample count.
        verbose : bool
            Print chunk-level progress.

        Returns
        -------
        Path
            Absolute path to the saved HDF5 file.
        """
        from ..metrics.rate import sum_rate as _sum_rate

        cfg = self.sys_config
        eps_list = [epsilons] if isinstance(epsilons, (int, float)) else list(epsilons)

        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"radcom_N{cfg.N}_K{cfg.K}_L{cfg.L}_{n_samples}samples_{ts}.h5"

        path = self.output_dir / filename
        optimizer = WaveformMatrixOptimizer(cfg, self.bnb_config)
        X0 = generate_chirp(cfg.N, cfg.L, cfg.PT)

        t0 = time.time()
        written = 0

        with h5py.File(path, "w") as f:
            # --- Create resizable datasets (initially empty) ---
            ds_H = f.create_dataset(
                "H", shape=(0, cfg.K, cfg.N), maxshape=(None, cfg.K, cfg.N),
                dtype=np.complex128, chunks=(min(self.chunk_size, n_samples), cfg.K, cfg.N),
            )
            ds_S = f.create_dataset(
                "S", shape=(0, cfg.K, cfg.L), maxshape=(None, cfg.K, cfg.L),
                dtype=np.complex128, chunks=(min(self.chunk_size, n_samples), cfg.K, cfg.L),
            )
            ds_X0 = f.create_dataset(
                "X0", shape=(0, cfg.N, cfg.L), maxshape=(None, cfg.N, cfg.L),
                dtype=np.complex128, chunks=(min(self.chunk_size, n_samples), cfg.N, cfg.L),
            )
            ds_Xopt = f.create_dataset(
                "X_opt", shape=(0, cfg.N, cfg.L), maxshape=(None, cfg.N, cfg.L),
                dtype=np.complex128, chunks=(min(self.chunk_size, n_samples), cfg.N, cfg.L),
            )
            ds_eps = f.create_dataset(
                "epsilon", shape=(0,), maxshape=(None,),
                dtype=np.float64, chunks=(min(self.chunk_size, n_samples),),
            )
            ds_rate = f.create_dataset(
                "sum_rate", shape=(0,), maxshape=(None,),
                dtype=np.float64, chunks=(min(self.chunk_size, n_samples),),
            )

            # --- Write metadata ---
            meta = f.create_group("metadata")
            meta.attrs["n_samples_target"] = n_samples
            meta.attrs["epsilons"] = json.dumps(eps_list)
            meta.attrs["seed"] = seed
            meta.attrs["chunk_size"] = self.chunk_size
            meta.attrs["creation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta.attrs["sys_config"] = json.dumps({
                "N": cfg.N, "K": cfg.K, "L": cfg.L,
                "PT": cfg.PT, "SNR_dB": cfg.SNR_dB,
            })
            meta.attrs["bnb_config"] = json.dumps({
                "rule": self.bnb_config.rule,
                "lb_solver": self.bnb_config.lb_solver,
                "ub_solver": self.bnb_config.ub_solver,
                "tol": self.bnb_config.tol,
                "max_iter": self.bnb_config.max_iter,
                "gp_max_iter": self.bnb_config.gp_max_iter,
            })

            # --- Generate in chunks ---
            while written < n_samples:
                cs = min(self.chunk_size, n_samples - written)

                buf_H = np.empty((cs, cfg.K, cfg.N), dtype=np.complex128)
                buf_S = np.empty((cs, cfg.K, cfg.L), dtype=np.complex128)
                buf_X0 = np.empty((cs, cfg.N, cfg.L), dtype=np.complex128)
                buf_Xopt = np.empty((cs, cfg.N, cfg.L), dtype=np.complex128)
                buf_eps = np.empty(cs, dtype=np.float64)
                buf_rate = np.empty(cs, dtype=np.float64)

                for j in range(cs):
                    idx = written + j
                    np.random.seed(seed + idx)

                    H = generate_channel(cfg.K, cfg.N)
                    S = generate_symbols(cfg.K, cfg.L)
                    eps_val = eps_list[idx % len(eps_list)]

                    X_opt, _ = optimizer.optimize(H, S, X0, eps_val)
                    sr = _sum_rate(H, X_opt, S, cfg.N0)

                    buf_H[j] = H
                    buf_S[j] = S
                    buf_X0[j] = X0
                    buf_Xopt[j] = X_opt
                    buf_eps[j] = eps_val
                    buf_rate[j] = sr

                # --- Flush chunk to disk ---
                new_len = written + cs
                ds_H.resize(new_len, axis=0);    ds_H[written:new_len] = buf_H
                ds_S.resize(new_len, axis=0);    ds_S[written:new_len] = buf_S
                ds_X0.resize(new_len, axis=0);   ds_X0[written:new_len] = buf_X0
                ds_Xopt.resize(new_len, axis=0); ds_Xopt[written:new_len] = buf_Xopt
                ds_eps.resize(new_len, axis=0);  ds_eps[written:new_len] = buf_eps
                ds_rate.resize(new_len, axis=0); ds_rate[written:new_len] = buf_rate

                f.flush()  # force write to OS
                written = new_len

                if verbose:
                    elapsed = time.time() - t0
                    rate_per_s = written / elapsed if elapsed > 0 else 0
                    eta = (n_samples - written) / rate_per_s if rate_per_s > 0 else 0
                    print(
                        f"  [{written:>{len(str(n_samples))}}/{n_samples}] "
                        f"flushed {cs} samples  "
                        f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)"
                    )

            # --- Finalize metadata ---
            meta.attrs["n_samples"] = written
            meta.attrs["total_time_s"] = time.time() - t0

        if verbose:
            sz_mb = path.stat().st_size / (1024 * 1024)
            print(f"\n  Dataset saved: {path.resolve()}")
            print(f"  {written} samples, {sz_mb:.1f} MB, {time.time() - t0:.1f}s total")

        return path.resolve()


# =====================================================================
# Lazy HDF5 Dataset (PyTorch-compatible)
# =====================================================================

class RadComHDF5Dataset:
    """Lazy-loading HDF5 dataset for training neural networks.

    Reads samples on demand — the file is memory-mapped by HDF5, so only
    the requested sample is loaded into RAM.

    Compatible with ``torch.utils.data.Dataset`` (implements
    ``__len__`` and ``__getitem__``).

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file produced by ``NNDatasetGenerator``.
    transform : callable, optional
        Optional transform applied to each sample tuple.

    Examples
    --------
    >>> ds = RadComHDF5Dataset("nn_datasets/data.h5")
    >>> len(ds)
    10000
    >>> H, S, X0, X_opt, eps, rate = ds[42]

    With PyTorch:

    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    """

    def __init__(
        self,
        path: str | Path,
        transform=None,
    ) -> None:
        self.path = Path(path)
        self.transform = transform
        # Open in read-only mode; stays open for the dataset's lifetime
        self._file = h5py.File(self.path, "r")
        self._n = self._file["epsilon"].shape[0]

        # Cache references (not data)
        self._H = self._file["H"]
        self._S = self._file["S"]
        self._X0 = self._file["X0"]
        self._X_opt = self._file["X_opt"]
        self._epsilon = self._file["epsilon"]
        self._sum_rate = self._file["sum_rate"]

    # --- Dataset interface ---

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._n}")

        sample = (
            np.array(self._H[idx]),
            np.array(self._S[idx]),
            np.array(self._X0[idx]),
            np.array(self._X_opt[idx]),
            float(self._epsilon[idx]),
            float(self._sum_rate[idx]),
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    # --- Metadata helpers ---

    @property
    def metadata(self) -> dict:
        """Return stored metadata as a dict."""
        meta = self._file["metadata"]
        return {k: meta.attrs[k] for k in meta.attrs}

    @property
    def shape_info(self) -> dict:
        """Return shapes of all stored arrays."""
        return {
            "H": self._H.shape,
            "S": self._S.shape,
            "X0": self._X0.shape,
            "X_opt": self._X_opt.shape,
            "n_samples": self._n,
        }

    # --- Cleanup ---

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        self._file.close()

    def __del__(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"RadComHDF5Dataset(path='{self.path}', "
            f"n_samples={self._n}, "
            f"shapes={self.shape_info})"
        )
