"""
Config Loader
=============
Convenience functions for loading and merging configurations from
YAML/JSON files, named presets, and CLI overrides.

Usage
-----
>>> cfg = load_config("configs/default.yaml")
>>> cfg = load_config("configs/quick.yaml", overrides={"seed": 99})
>>> cfg = load_preset("quick")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import PipelineConfig


# ── Built-in presets (no YAML file needed) ──────────────────


_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "name": "quick_test",
        "system": {"N": 8, "K": 2, "L": 4},
        "bnb": {"max_iter": 20, "gp_max_iter": 30, "tol": 1e-2},
        "rate_sweep": {
            "n_trials": 1,
            "max_iter": 10,
            "tol": 1e-1,
            "epsilons": [0.3, 0.7, 1.0],
        },
        "dataset": {"n_samples": 16, "chunk_size": 8},
        "gan": {
            "latent_dim": 64,
            "hidden_g": [128, 128],
            "hidden_c": [128, 64],
            "n_epochs": 10,
            "batch_size": 8,
            "eval_every": 1,
            "save_every": 0,
        },
        "eval": {"n_samples": 4},
    },
    "paper": {
        "name": "paper",
        "system": {"N": 16, "K": 4, "L": 20, "PT": 1.0, "SNR_dB": 10.0},
        "bnb": {"max_iter": 200, "gp_max_iter": 100, "tol": 1e-4},
        "rate_sweep": {"n_trials": 5, "max_iter": 40},
        "dataset": {"n_samples": 5000, "chunk_size": 100},
        "gan": {"n_epochs": 500, "batch_size": 64},
        "plot": {"formats": ["png", "eps"]},
    },
}


def load_preset(name: str, **overrides: Any) -> PipelineConfig:
    """Load a built-in preset by name.

    Parameters
    ----------
    name : str
        One of ``'quick'``, ``'paper'``.
    **overrides
        Additional overrides applied on top of the preset.

    Returns
    -------
    PipelineConfig
    """
    if name not in _PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(_PRESETS.keys())}"
        )
    data = _PRESETS[name].copy()
    # Deep-merge overrides
    _deep_merge(data, overrides)
    return PipelineConfig.model_validate(data)


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load config from a YAML or JSON file, with optional overrides.

    Parameters
    ----------
    path : str or Path
        Path to a ``.yaml`` / ``.yml`` or ``.json`` config file.
    overrides : dict, optional
        Key-value pairs to overlay on top of the loaded file.

    Returns
    -------
    PipelineConfig
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    elif suffix == ".json":
        import json
        with open(p) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .yaml or .json")

    if overrides:
        _deep_merge(data, overrides)

    return PipelineConfig.model_validate(data)


# ── Helpers ─────────────────────────────────────────────────


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (in-place)."""
    for key, val in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(val, dict)
        ):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base
