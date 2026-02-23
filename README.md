# RadCom Waveform Design

Production-grade, modular Python system for **Dual-Functional Radar-Communication (RadCom) Waveform Design** via Branch-and-Bound optimization and GAN-based waveform learning.

Implements the complete algorithm from:

> Fan Liu, Christos Masouros, Ang Li, Huafei Sun, and Lajos Hanzo,
> *"Towards Dual-functional Radar-Communication Systems: Optimal Waveform Design,"*
> IEEE Trans. Signal Processing, vol. 66, no. 16, pp. 4264–4279, 2018.
> ([arXiv:1711.05220](https://arxiv.org/abs/1711.05220))

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [Mode Shortcuts](#1-mode-shortcuts---mode)
  - [Single Experiment](#2-single-experiment-experiment)
  - [Full Pipeline](#3-full-pipeline-pipeline)
  - [Legacy Commands](#4-legacy-commands-run--sweep--report)
- [Configuration](#configuration)
  - [YAML Config Files](#yaml-config-files)
  - [Built-in Presets](#built-in-presets)
  - [CLI Overrides](#cli-parameter-overrides)
  - [Configuration Precedence](#configuration-precedence)
  - [All Parameters](#all-configuration-parameters)
- [Experiment Framework](#experiment-framework)
  - [Available Experiments](#available-experiments)
  - [Adding a Custom Experiment](#adding-a-custom-experiment)
- [Python API](#python-api)
- [Output Structure](#output-structure)
- [Running Tests](#running-tests)
- [Reproducing Paper Figures](#reproducing-paper-figures)
- [Equation–Module Mapping](#equationmodule-mapping)
- [Solver Registry](#solver-registry)
- [Design Principles](#design-principles)

---

## Features

- **Branch-and-Bound (Algorithm 2)** with ARS (eq. 36) and BRS (eq. 35) subdivision rules
- **Pluggable solvers** via registry pattern:
  - **LB**: CVXPY/SCS interior-point (eq. 40), Accelerated GP with PR₂ (eqs. 43–44)
  - **UB**: scipy SLSQP (eq. 42), GP with PR₁
- **Exact projections**: PR₁ onto unit-circle arc (eq. 41), PR₂ onto convex hull (eq. 62, corrected)
- **Multi-column optimizer** exploiting eq. 27 separability
- **WGAN-GP** for learned waveform generation with physics-aware evaluation
- **Signal processing**: orthogonal chirp generation (eq. 33), FFT pulse compression with Taylor window
- **Metrics**: sum-rate (eqs. 4–5), convergence analysis, ISL/PSL radar metrics
- **Centralized YAML configuration** with pydantic validation and built-in presets
- **Extensible experiment framework** — add new experiments in a single file
- **Unified CLI** with mode shortcuts, experiment runner, and full pipeline orchestration
- **Dataset generator**: HDF5 serialization for ML integration
- **Cross-validated**: 0.00e+00 numerical difference against reference implementation

---

## Project Structure

```
bnb_gan/
├── main.py                      # ★ Unified CLI entry point
├── run.py                       # Legacy CLI (still works)
├── pyproject.toml               # Package config, dependencies
├── README.md                    # This file
│
├── configs/                     # YAML configuration files
│   ├── default.yaml             # Full parameter set with comments
│   ├── quick.yaml               # Small system for fast iteration (< 30 s)
│   └── paper.yaml               # Full-scale paper reproduction params
│
├── src/                         # Main package
│   ├── __init__.py
│   ├── __main__.py              # `python -m src` entry point
│   │
│   ├── config/                  # ★ Centralized config (pydantic)
│   │   ├── schema.py            # PipelineConfig + all sub-configs
│   │   └── loader.py            # load_config(), load_preset()
│   │
│   ├── experiments/             # ★ Experiment framework
│   │   ├── base.py              # BaseExperiment ABC + ExperimentRegistry
│   │   ├── exp_convergence.py   # BnB convergence analysis
│   │   ├── exp_rate_sweep.py    # Sum-rate vs epsilon sweep
│   │   ├── exp_dataset.py       # HDF5 dataset generation
│   │   ├── exp_gan_train.py     # WGAN-GP training
│   │   ├── exp_waveform_eval.py # BnB vs GAN evaluation
│   │   ├── config.py            # Legacy ExperimentConfig
│   │   ├── runner.py            # Legacy ExperimentRunner
│   │   ├── results.py           # Legacy results aggregation
│   │   ├── report.py            # Markdown report generation
│   │   └── sweep.py             # Parameter sweep engine
│   │
│   ├── utils/                   # Configuration & helpers
│   │   ├── config.py            # SystemConfig, BnBConfig (frozen dataclasses)
│   │   └── math_helpers.py      # angle_diff, Lipschitz step, initial bounds
│   │
│   ├── optimizer/               # Core BnB engine
│   │   ├── bnb.py               # BranchAndBoundSolver, bnb_solve()
│   │   ├── node.py              # BnBNode with priority queue support
│   │   ├── projections.py       # PR1 (arc) + PR2 (convex hull)
│   │   ├── waveform_optimizer.py  # Multi-column WaveformMatrixOptimizer
│   │   └── solvers/
│   │       ├── base.py          # ABC + SolverRegistry
│   │       ├── lb_cvxpy.py      # CVXPY/SCS lower-bound (eq. 40)
│   │       ├── lb_gp.py         # FISTA gradient projection LB (eqs. 43–44)
│   │       ├── ub_slsqp.py      # scipy SLSQP upper-bound (eq. 42)
│   │       └── ub_gp.py         # GP + PR1 upper-bound
│   │
│   ├── gan/                     # WGAN-GP implementation
│   │   ├── networks.py          # Generator & Critic architectures
│   │   ├── trainer.py           # WGANGPTrainer with gradient penalty
│   │   ├── history.py           # Training history tracking
│   │   └── utils.py             # GAN utility functions
│   │
│   ├── signal_proc/             # Waveform & radar signal processing
│   │   ├── waveform.py          # generate_chirp, generate_channel, generate_symbols
│   │   └── pulse_compression.py # pulse_compress (FFT + Taylor), autocorrelation
│   │
│   ├── metrics/                 # Performance metrics
│   │   ├── base.py              # MetricBase ABC, MetricResult dataclass
│   │   ├── convergence.py       # ConvergenceMetric (gap history)
│   │   ├── rate.py              # RateMetric, sum_rate() (eqs. 4–5)
│   │   ├── radar.py             # ISLMetric, PSLMetric
│   │   ├── similarity.py        # Waveform similarity metrics
│   │   └── pulse_comp_metrics.py # Pulse compression metrics
│   │
│   ├── data/                    # Dataset generation
│   │   ├── generator.py         # DatasetGenerator (HDF5 I/O)
│   │   ├── experiments.py       # run_convergence_experiment, run_rate_vs_epsilon_experiment
│   │   └── nn_dataset.py        # NNDatasetGenerator + RadComHDF5Dataset (PyTorch)
│   │
│   └── plotting/                # Visualization
│       ├── figures.py           # Figure generation
│       └── style.py             # Plot styling
│
├── tests/                       # Test suite (pytest)
│   ├── test_projections.py
│   ├── test_solvers.py
│   ├── test_bnb.py
│   ├── test_signal_proc.py
│   └── test_refactored.py       # Config + registry + CLI tests
│
├── notebooks/
│   └── benchmark_analysis.ipynb # Reproduces Figs. 7, 8, 9
│
├── examples/
│   └── basic_usage.py           # Complete pipeline demo
│
├── figures/                     # Generated plots (PNG + EPS)
└── docs/
    └── README.md                # Architecture notes & equation mapping
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url> bnb_gan
cd bnb_gan

# Install in development mode (recommended)
pip install -e .
```

This installs all required dependencies automatically. The full dependency list:

| Package | Min Version | Purpose |
|---------|-------------|---------|
| numpy | ≥ 1.22 | Array operations |
| scipy | ≥ 1.9 | Optimization (SLSQP) |
| cvxpy | ≥ 1.3 | Convex solver (SCS) |
| h5py | ≥ 3.7 | HDF5 dataset I/O |
| matplotlib | ≥ 3.6 | Plotting |
| torch | ≥ 2.0 | GAN training |
| pydantic | ≥ 2.0 | Config validation |
| pyyaml | ≥ 6.0 | YAML config files |

**Python ≥ 3.9** is required.

To install dependencies manually (without `pip install -e .`):

```bash
pip install numpy scipy cvxpy h5py matplotlib torch pydantic pyyaml
```

---

## Quick Start

### 30-Second Demo (CLI)

```bash
# Run the full pipeline with a fast preset (~30 seconds)
python main.py pipeline --preset quick

# Or run individual stages
python main.py --mode generate  --preset quick    # Generate HDF5 dataset
python main.py --mode train     --preset quick    # Train WGAN-GP
python main.py --mode evaluate  --preset quick    # Compare BnB vs GAN
```

### 30-Second Demo (Python)

```python
from src.config import PipelineConfig, load_config
from src.experiments import ExperimentRegistry

# Load a preset and run one experiment
cfg = load_config("configs/quick.yaml")
exp = ExperimentRegistry.create("convergence", cfg)
result = exp.execute(verbose=True)
```

---

## CLI Reference

The unified CLI entry point is **`main.py`**. There are four ways to invoke it:

```
python main.py --mode <MODE>              # 1. Mode shortcut
python main.py experiment --name <NAME>   # 2. Single experiment
python main.py pipeline                   # 3. Full pipeline
python main.py run|sweep|report           # 4. Legacy commands
```

You can also invoke via `python -m src` (which delegates to `main.py`).

### 1. Mode Shortcuts (`--mode`)

The fastest way to run a single stage. Maps directly to registered experiments:

| Mode | Experiment Run | Description |
|------|---------------|-------------|
| `generate` | `dataset` | Generate HDF5 training dataset |
| `train` | `gan_train` | Train the WGAN-GP |
| `evaluate` | `waveform_eval` | Compare BnB vs GAN waveforms |

**Syntax:**

```bash
python main.py --mode <generate|train|evaluate> [OPTIONS]
```

**Examples:**

```bash
# Generate dataset with default config
python main.py --mode generate

# Generate dataset using a YAML config
python main.py --mode generate --config configs/default.yaml

# Generate dataset using the quick preset
python main.py --mode generate --preset quick

# Train GAN with custom parameters
python main.py --mode train --preset quick --gan-epochs 100 --gan-lr 0.0002

# Evaluate with paper parameters and custom sample count
python main.py --mode evaluate --preset paper --eval-n-samples 50

# All modes support --quiet to suppress progress output
python main.py --mode train --preset quick --quiet
```

### 2. Single Experiment (`experiment`)

Run any registered experiment by name. This is more flexible than `--mode` since it can access all 5 registered experiments, not just the 3 mode aliases.

**List all available experiments:**

```bash
python main.py experiment --list
```

Output:

```
Available experiments:
------------------------------------------------------------
  convergence           BnB convergence analysis (4 solver combos)
  rate_sweep            Sum-rate vs epsilon sweep
  dataset               Generate HDF5 dataset for GAN training
  gan_train             Train WGAN-GP on BnB dataset
  waveform_eval         Evaluate BnB vs GAN waveforms
```

**Run a specific experiment:**

```bash
# Run convergence analysis
python main.py experiment --name convergence --config configs/default.yaml

# Run rate sweep with paper params
python main.py experiment --name rate_sweep --preset paper

# Run dataset generation with overrides
python main.py experiment --name dataset --preset quick --ds-n-samples 500

# Run GAN training
python main.py experiment --name gan_train --config configs/quick.yaml
```

### 3. Full Pipeline (`pipeline`)

Run multiple stages in sequence. By default, runs all 7 stages defined in the config.

**Syntax:**

```bash
python main.py pipeline [--stages STAGE1 STAGE2 ...] [OPTIONS]
```

**Examples:**

```bash
# Run the full pipeline (all 7 stages)
python main.py pipeline --preset quick

# Run the full pipeline with paper parameters
python main.py pipeline --preset paper

# Run only specific stages (in order)
python main.py pipeline --preset quick --stages convergence rate_sweep

# Run analysis + plots
python main.py pipeline --preset paper --stages convergence rate_sweep plots report

# Run with custom output directory and name
python main.py pipeline --preset quick --run-name my_experiment --output-dir results

# Run with system parameter overrides
python main.py pipeline --preset quick --N 32 --K 8 --seed 123
```

**Available stages** (executed in this order):

| # | Stage | Description |
|---|-------|-------------|
| 1 | `convergence` | BnB convergence analysis across 4 solver combinations |
| 2 | `rate_sweep` | Sum-rate vs epsilon sweep over multiple channel realizations |
| 3 | `dataset` | Generate HDF5 dataset of BnB-optimized waveforms |
| 4 | `gan_train` | Train WGAN-GP on the generated dataset |
| 5 | `waveform_eval` | Side-by-side evaluation of BnB vs GAN waveforms |
| 6 | `plots` | Generate publication-quality figures (legacy stage) |
| 7 | `report` | Auto-generate Markdown report with all results (legacy stage) |

### 4. Legacy Commands (`run` / `sweep` / `report`)

Backward-compatible with the original `run.py` interface. These use the legacy `ExperimentRunner`.

```bash
# Run full experiment (all stages)
python main.py run --preset quick
python main.py run --preset paper
python main.py run --preset quick --stages convergence rate_sweep plots

# Run with parameter overrides
python main.py run --preset quick --N 16 --K 4 --gan-epochs 50

# Parameter sweep
python main.py sweep --preset quick --axis N=8,16,32
python main.py sweep --preset quick --axis N=8,16,32 --axis K=2,4
python main.py sweep --preset quick --grid sweep_grid.json

# Generate reports
python main.py report results/quick_test
python main.py report results/ --compare
```

The original `run.py` still works identically:

```bash
python run.py run --preset quick
python run.py sweep --preset quick --axis N=8,16,32
python run.py report results/quick_test --compare
```

### Global CLI Options

These options are available across all commands:

| Option | Type | Description |
|--------|------|-------------|
| `--config PATH` | str | Path to a YAML or JSON config file |
| `--preset NAME` | choice | Built-in preset: `quick` or `paper` |
| `--run-name NAME` | str | Experiment name (used as output subdirectory) |
| `--output-dir DIR` | str | Root output directory (default: `outputs`) |
| `--seed INT` | int | Random seed for reproducibility |
| `--quiet` | flag | Suppress progress output |
| `--N INT` | int | Number of transmit antennas |
| `--K INT` | int | Number of users |
| `--L INT` | int | Frame length |
| `--PT FLOAT` | float | Total transmit power |
| `--snr-db FLOAT` | float | Signal-to-noise ratio (dB) |
| `--bnb-tol FLOAT` | float | BnB convergence tolerance |
| `--bnb-max-iter INT` | int | BnB max iterations |
| `--ds-n-samples INT` | int | Dataset generation sample count |
| `--gan-epochs INT` | int | GAN training epochs |
| `--gan-batch-size INT` | int | GAN mini-batch size |
| `--gan-lr FLOAT` | float | GAN learning rate |
| `--eval-n-samples INT` | int | Evaluation sample count |

---

## Configuration

The system uses a centralized configuration approach: one `PipelineConfig` object holds every parameter the pipeline needs.

### YAML Config Files

Config files live in `configs/` and control all parameters. Three are provided out of the box:

| File | Purpose | Typical Runtime |
|------|---------|----------------|
| `configs/default.yaml` | Full parameter set with comments | ~10 min |
| `configs/quick.yaml` | Small system for fast iteration | < 30 seconds |
| `configs/paper.yaml` | Paper reproduction parameters | ~1 hour+ |

**Load a config file:**

```bash
python main.py --mode train --config configs/quick.yaml
python main.py pipeline --config configs/paper.yaml
```

You can copy and edit any YAML file to create your own:

```bash
cp configs/default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml, then:
python main.py pipeline --config configs/my_experiment.yaml
```

### Built-in Presets

For convenience, `--preset quick` and `--preset paper` are available without specifying a file path:

```bash
python main.py --mode generate --preset quick    # Small/fast
python main.py --mode generate --preset paper    # Full-scale
```

### CLI Parameter Overrides

Any config parameter can be overridden from the command line. CLI overrides take highest priority:

```bash
# Override system parameters
python main.py pipeline --preset quick --N 32 --K 8 --L 10

# Override BnB solver settings
python main.py pipeline --config configs/default.yaml --bnb-tol 0.0001 --bnb-max-iter 500

# Override GAN training
python main.py --mode train --preset paper --gan-epochs 1000 --gan-lr 0.0002 --gan-batch-size 128

# Override dataset size
python main.py --mode generate --preset quick --ds-n-samples 5000

# Override evaluation
python main.py --mode evaluate --preset quick --eval-n-samples 100

# Override seed and output
python main.py pipeline --preset quick --seed 0 --run-name ablation_01 --output-dir results
```

### Configuration Precedence

Parameters are resolved in this order (later overrides earlier):

```
1. Built-in defaults (PipelineConfig defaults)
2. --preset (quick / paper)
3. --config (YAML/JSON file)
4. CLI flags (--N, --gan-epochs, etc.)
```

### Programmatic Configuration

```python
from src.config import PipelineConfig, load_config
from src.config.loader import load_preset

# From defaults
cfg = PipelineConfig()

# From YAML file
cfg = load_config("configs/quick.yaml")

# From preset
cfg = load_preset("quick")

# With overrides (dot notation for nested keys)
cfg = cfg.with_overrides(**{"system.N": 32, "gan.n_epochs": 500, "seed": 0})

# Serialize / deserialize
cfg.to_yaml("my_config.yaml")
cfg2 = PipelineConfig.from_yaml("my_config.yaml")

cfg.to_json("my_config.json")
cfg3 = PipelineConfig.from_json("my_config.json")
```

### All Configuration Parameters

<details>
<summary>Click to expand the full parameter reference</summary>

```yaml
# ── Top-level ─────────────────────────────────────────────
name: default             # Experiment name (output subdirectory)
output_dir: outputs       # Root output directory
seed: 42                  # Global random seed
stages:                   # Pipeline stages to run (in order)
  - convergence
  - rate_sweep
  - dataset
  - gan_train
  - waveform_eval
  - plots
  - report

# ── System Physics ────────────────────────────────────────
system:
  N: 16           # Transmit antennas (≥ 1)
  K: 4            # Single-antenna users (≥ 1)
  L: 20           # Frame length / radar pulses (≥ 1)
  PT: 1.0         # Total transmit power (> 0)
  SNR_dB: 10.0    # Signal-to-noise ratio in dB

# ── Branch-and-Bound Solver ──────────────────────────────
bnb:
  rule: ARS           # Subdivision rule: "ARS" or "BRS"
  lb_solver: gp       # Lower-bound solver: "cvxpy" or "gp"
  ub_solver: gp       # Upper-bound solver: "slsqp" or "gp"
  tol: 0.001          # Convergence tolerance (> 0)
  max_iter: 200       # Max BnB iterations (≥ 1)
  gp_max_iter: 100    # Gradient-projection iters per bound (≥ 1)
  gp_tol: 0.000001   # GP convergence tolerance (> 0)
  verbose: false      # Print BnB iteration progress
  verbose_interval: 20

# ── Convergence Analysis Stage ───────────────────────────
convergence:
  epsilon: 1.0        # Similarity tolerance (≥ 0)

# ── Rate Sweep Stage ─────────────────────────────────────
rate_sweep:
  epsilons: [0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
  n_trials: 5         # Channel realizations per epsilon (≥ 1)
  tol: 0.005          # BnB tolerance for sweep (> 0)
  max_iter: 40        # BnB max iterations for sweep (≥ 1)

# ── Dataset Generation ───────────────────────────────────
dataset:
  n_samples: 1000     # Number of optimization samples (≥ 1)
  epsilons: [0.3, 0.7, 1.0]
  chunk_size: 50      # HDF5 write chunk size (≥ 1)

# ── GAN (WGAN-GP) ────────────────────────────────────────
gan:
  latent_dim: 128     # Noise vector dimension (≥ 1)
  hidden_g: [512, 512, 512]      # Generator hidden layers
  hidden_c: [512, 256, 128]      # Critic hidden layers
  n_epochs: 200       # Training epochs (≥ 1)
  batch_size: 64      # Mini-batch size (≥ 1)
  learning_rate: 0.0001  # Adam LR (> 0)
  n_critic: 5         # Critic updates per G update (≥ 1)
  lambda_gp: 10.0     # Gradient penalty coefficient (≥ 0)
  eval_every: 5       # Evaluate every N epochs (≥ 1)
  save_every: 50      # Checkpoint every N epochs (0 = off)

# ── Evaluation (BnB vs GAN) ─────────────────────────────
eval:
  n_samples: 20       # Evaluation samples (≥ 1)
  epsilons: [0.3, 0.7, 1.0]

# ── Plotting ─────────────────────────────────────────────
plot:
  formats: [png]      # Output formats: png, eps, pdf
  dpi: 300            # Figure DPI (≥ 72)
```

</details>

---

## Experiment Framework

The experiment framework uses an auto-registration pattern. Every subclass of `BaseExperiment` is automatically discoverable via the CLI and Python API.

### Available Experiments

| Name | Description | Requires |
|------|-------------|----------|
| `convergence` | BnB convergence analysis across 4 solver combinations (ARS/BRS × CVXPY/GP) | Nothing |
| `rate_sweep` | Sum-rate vs epsilon sweep over multiple channel realizations | Nothing |
| `dataset` | Generate HDF5 dataset of BnB-optimized waveforms | Nothing |
| `gan_train` | Train WGAN-GP on generated dataset | Dataset (`.h5` file) |
| `waveform_eval` | Side-by-side BnB vs GAN evaluation with per-sample metrics | Trained GAN checkpoint |

### Running Experiments

**Via CLI:**

```bash
python main.py experiment --name convergence --preset quick
python main.py experiment --name rate_sweep --preset paper
python main.py experiment --name dataset --preset quick
python main.py experiment --name gan_train --config configs/quick.yaml
python main.py experiment --name waveform_eval --config configs/quick.yaml
```

**Via Python:**

```python
from src.config import load_config
from src.experiments import ExperimentRegistry

cfg = load_config("configs/quick.yaml")

# Run one experiment
exp = ExperimentRegistry.create("convergence", cfg)
result = exp.execute(verbose=True)

# Run several in sequence
for name in ["convergence", "rate_sweep", "dataset"]:
    exp = ExperimentRegistry.create(name, cfg)
    exp.execute(verbose=True)
```

### Adding a Custom Experiment

Create a single file in `src/experiments/` — it is auto-registered:

```python
# src/experiments/exp_my_analysis.py
from src.experiments.base import BaseExperiment

class MyAnalysis(BaseExperiment):
    name = "my_analysis"
    description = "Custom analysis of something cool"

    def run(self, verbose: bool = True) -> dict:
        cfg = self.config  # PipelineConfig object
        if verbose:
            print(f"Running with N={cfg.system.N}, K={cfg.system.K}")

        result = {"answer": 42, "rates": [1.0, 2.0, 3.0]}
        self.save_results(result)  # Persists to output_dir
        return result
```

Then add the import in `src/experiments/__init__.py`:

```python
from . import exp_my_analysis
```

The experiment is now available everywhere:

```bash
python main.py experiment --list          # Shows my_analysis
python main.py experiment --name my_analysis --preset quick
```

---

## Python API

### Core Optimization

```python
import numpy as np
from src.utils.config import SystemConfig, BnBConfig
from src.signal_proc import generate_chirp, generate_channel, generate_symbols
from src.optimizer import WaveformMatrixOptimizer
from src.metrics import sum_rate

# 1. Configure system (paper defaults)
sys_cfg = SystemConfig(N=16, K=4, L=20, PT=1.0, SNR_dB=10.0)
bnb_cfg = BnBConfig(rule="ARS", lb_solver="cvxpy", ub_solver="slsqp", tol=1e-3)

# 2. Generate data
np.random.seed(42)
H  = generate_channel(sys_cfg.K, sys_cfg.N)
S  = generate_symbols(sys_cfg.K, sys_cfg.L)
X0 = generate_chirp(sys_cfg.N, sys_cfg.L, sys_cfg.PT)

# 3. Optimize
optimizer = WaveformMatrixOptimizer(sys_cfg, bnb_cfg)
X_opt, results = optimizer.optimize(H, S, X0, epsilon=1.0)

# 4. Evaluate
rate = sum_rate(H, X_opt, S, sys_cfg.N0)
print(f"Optimized sum-rate: {rate:.4f} bps/Hz")
```

### Functional API

```python
from src.optimizer import bnb_solve, optimize_waveform

# Single-column BnB
x_opt, obj, lb_hist, ub_hist = bnb_solve(Ht, s, x0, epsilon=1.0)

# Full matrix (convenience wrapper)
X_opt = optimize_waveform(H, S, X0, epsilon=1.0, PT=1.0)
```

### Dataset Generation (HDF5)

```python
from src.data import DatasetGenerator

gen = DatasetGenerator(output_dir="datasets", sys_config=sys_cfg, bnb_config=bnb_cfg)
path = gen.generate_and_save(n_samples=100, epsilon=1.0, seed=42)

# Load later
samples, metadata = DatasetGenerator.load_hdf5(path)
```

### Configuration API

```python
from src.config import PipelineConfig, load_config
from src.config.loader import load_preset

# Load from file
cfg = load_config("configs/quick.yaml")

# Modify programmatically
cfg = cfg.with_overrides(**{
    "system.N": 32,
    "system.K": 8,
    "gan.n_epochs": 500,
    "seed": 0,
})

# Access sub-configs
print(cfg.system.N)          # 32
print(cfg.system.N0)         # Noise power (computed property)
print(cfg.bnb.rule)          # "ARS"
print(cfg.gan.learning_rate) # 0.0001

# Bridge to legacy frozen dataclasses
sys_cfg = cfg.sys_config     # src.utils.config.SystemConfig
bnb_cfg = cfg.bnb_legacy     # src.utils.config.BnBConfig
```

---

## Output Structure

All outputs are written under `<output_dir>/<name>/`. The default is `outputs/default/`.

```
outputs/quick_test/
├── config.json                  # Full experiment config snapshot
├── report.md                    # Auto-generated Markdown report
├── figures/                     # All plots (PNG/EPS/PDF)
│   ├── convergence.png
│   ├── rate_vs_epsilon.png
│   ├── gan_training.png
│   ├── eval_rate_comparison.png
│   └── ...
├── waveforms/                   # Per-sample waveform data
│   ├── sample_0000.npz
│   ├── sample_0001.npz
│   └── summary.json
├── stages/                      # Per-stage artifacts
│   ├── convergence/
│   │   ├── meta.json
│   │   ├── result.json
│   │   └── data.npz
│   ├── rate_sweep/
│   ├── dataset/
│   └── gan_train/
└── checkpoints/                 # GAN model checkpoints
    ├── generator_epoch_50.pt
    ├── critic_epoch_50.pt
    └── ...
```

Each `sample_XXXX.npz` contains: `H`, `S`, `X0`, `X_bnb`, `X_gan`, `epsilon`, `rate_bnb`, `rate_gan`, `power_bnb`, `power_gan`, `l2_bnb`, `l2_gan`, `feasible_bnb`, `feasible_gan`.

---

## Running Tests

The test suite uses **pytest**:

```bash
# Run all tests
python -m pytest tests/ -v

# Run only the config/registry/CLI tests
python -m pytest tests/test_refactored.py -v

# Run only the core algorithm tests
python -m pytest tests/test_bnb.py tests/test_projections.py tests/test_solvers.py -v

# Run with short traceback
python -m pytest tests/ -v --tb=short
```

**Test files:**

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_refactored.py` | 36 | Config validation, registry, CLI parsing |
| `tests/test_bnb.py` | — | Branch-and-Bound solver correctness |
| `tests/test_projections.py` | — | PR₁ / PR₂ projection accuracy |
| `tests/test_solvers.py` | — | LB/UB solver correctness |
| `tests/test_signal_proc.py` | — | Chirp generation, pulse compression |

---

## Reproducing Paper Figures

### Option 1: Full Pipeline

```bash
# Quick verification (< 30 seconds)
python main.py pipeline --preset quick --stages convergence rate_sweep plots

# Full-scale reproduction (paper parameters)
python main.py pipeline --preset paper --stages convergence rate_sweep plots report
```

### Option 2: Jupyter Notebook

```bash
cd notebooks
jupyter notebook benchmark_analysis.ipynb
```

The notebook generates:
- **Fig. 7** — BnB convergence (2×2 grid: ARS/BRS × CVXPY/GP)
- **Fig. 8** — Sum-rate vs ε (BnB, convex relaxation, AWGN capacity)
- **Fig. 9** — Pulse compression range profiles (ε = 0.05, 0.4, 1.0)

### Option 3: Python Script

```bash
python examples/basic_usage.py
```

### Option 4: Experiment API

```python
from src.data.experiments import run_convergence_experiment, run_rate_vs_epsilon_experiment

# Fig. 7 — convergence
data = run_convergence_experiment(N=16, K=4, epsilon=1.0, seed=42)
for r in data["results"]:
    print(f"{r.label}: {r.n_iterations} iters, gap={r.ub_history[-1]-r.lb_history[-1]:.6f}")

# Fig. 8 — rate sweep
result = run_rate_vs_epsilon_experiment(n_trials=5, seed=0)
```

---

## Equation–Module Mapping

| Paper Equation | Description | Module |
|:---:|---|---|
| Eq. 4–5 | SINR & sum-rate | `metrics/rate.py` |
| Eq. 27 | Per-column separability | `optimizer/waveform_optimizer.py` |
| Eq. 30 | Arc similarity constraint | `utils/math_helpers.py` |
| Eq. 33 | Orthogonal chirp reference | `signal_proc/waveform.py` |
| Eq. 35 | BRS subdivision rule | `optimizer/bnb.py` |
| Eq. 36 | ARS subdivision rule | `optimizer/bnb.py` |
| Eq. 40 | QP-LB (convex relaxation) | `optimizer/solvers/lb_cvxpy.py` |
| Eq. 41 | PR₁ (arc projection) | `optimizer/projections.py` |
| Eq. 42 | QP-UB (non-convex local) | `optimizer/solvers/ub_slsqp.py` |
| Eqs. 43–44 | FISTA gradient projection | `optimizer/solvers/lb_gp.py` |
| Eq. 62 | PR₂ (convex hull, corrected) | `optimizer/projections.py` |
| Alg. 2 | BnB framework | `optimizer/bnb.py` |

> **Note:** Paper eq. 62 has a typo in the denominator (|T| instead of |T|²).
> This implementation uses the corrected |T|² form, matching the reference code.

---

## Solver Registry

Solvers are pluggable via the registry pattern:

```python
from src.optimizer.solvers.base import SolverRegistry, default_registry

# List available solvers
print(default_registry.list_lb_solvers())  # ['cvxpy', 'gp']
print(default_registry.list_ub_solvers())  # ['slsqp', 'gp']

# Select via config
cfg = BnBConfig(lb_solver="gp", ub_solver="gp")  # All-GP (fastest)
cfg = BnBConfig(lb_solver="cvxpy", ub_solver="slsqp")  # Most accurate
```

---

## Design Principles

1. **Centralized config** — single `PipelineConfig` (pydantic) is the source of truth; no scattered magic numbers
2. **Strict separation** — optimizer has zero dependency on matplotlib or signal_proc
3. **Extensible experiments** — add a new experiment by subclassing `BaseExperiment` in one file
4. **Frozen core configs** — legacy `SystemConfig` and `BnBConfig` remain immutable dataclasses
5. **Functional + OOP APIs** — every module provides both class-based and functional interfaces
6. **Cross-validated** — all core functions match the reference implementation to machine precision (0.00e+00 diff)
7. **Reproducible** — HDF5 datasets include full config metadata; seeded RNG throughout

---

## License

MIT
