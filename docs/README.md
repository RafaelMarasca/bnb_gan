# Architecture & Design Notes

## Paper Reference

Fan Liu et al., "Towards Dual-functional Radar-Communication Systems:
Optimal Waveform Design", IEEE TSP 2018 (arXiv:1711.05220).

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  (notebooks/benchmark_analysis.ipynb, examples/, CLI)    │
├──────────┬──────────────┬───────────────┬───────────────┤
│  data/   │  metrics/    │  signal_proc/ │               │
│ Generator│ Convergence  │  Waveform     │   No cross-   │
│ Exper.   │ Rate, Radar  │  PulseCompr.  │  dependencies │
├──────────┴──────────────┴───────────────┤   between     │
│              optimizer/                  │   columns     │
│  BnB Engine ← Solvers (pluggable)       │               │
│  PR1/PR2 Projections                    │               │
│  Node, Waveform Optimizer               │               │
├─────────────────────────────────────────┤               │
│              utils/                      │               │
│  SystemConfig, BnBConfig (frozen)       │               │
│  math_helpers                            │               │
└─────────────────────────────────────────────────────────┘
```

### Dependency Flow

```
utils/ ← (no deps)
optimizer/ ← utils/
signal_proc/ ← (numpy/scipy only)
metrics/ ← (numpy only)
data/ ← optimizer/, signal_proc/, metrics/, utils/
notebooks/ ← all packages
```

**Key constraint:** `optimizer/` has zero dependency on matplotlib, signal_proc, or metrics.

## Key Equations

| Module | Paper Reference | Description |
|--------|----------------|-------------|
| `optimizer/bnb.py` | Algorithm 2 | BnB framework with priority queue |
| `optimizer/solvers/lb_cvxpy.py` | Eq. 40 | QP-LB via CVXPY/SCS interior-point |
| `optimizer/solvers/lb_gp.py` | Eqs. 43-44 | FISTA accelerated gradient projection + PR₂ |
| `optimizer/solvers/ub_slsqp.py` | Eq. 42 | QP-UB via scipy SLSQP + PR₁ |
| `optimizer/solvers/ub_gp.py` | — | GP with PR₁, best-tracking |
| `optimizer/projections.py` | Eq. 41 | PR₁: projection onto unit-circle arc |
| `optimizer/projections.py` | Eq. 62 (corrected) | PR₂: projection onto convex hull |
| `optimizer/bnb.py` | Eq. 36 | ARS: Adaptive Rectangular Subdivision |
| `optimizer/bnb.py` | Eq. 35 | BRS: Basic Rectangular Subdivision |
| `optimizer/waveform_optimizer.py` | Eq. 27 | Column-wise separability |
| `signal_proc/waveform.py` | Eq. 33 | Orthogonal chirp reference |
| `signal_proc/waveform.py` | Eq. 30 | Chordal similarity ↔ arc half-width |
| `metrics/rate.py` | Eqs. 4-5 | SINR and sum-rate |

## PR₂ Correction

Paper eq. 62 contains a typo in Region M4 (chord projection).
The denominator should be |T|² (squared), not |T|.
This matches the reference implementation and produces correct results.

## Solver Registry Pattern

```python
class SolverRegistry:
    """Central registry for pluggable LB/UB solvers."""
    _lb: dict[str, type[LBSolverBase]]
    _ub: dict[str, type[UBSolverBase]]

    def register_lb(name, cls) → None
    def register_ub(name, cls) → None
    def get_lb(name) → LBSolverBase
    def get_ub(name) → UBSolverBase
```

New solvers can be added by subclassing `LBSolverBase`/`UBSolverBase` and
registering with `default_registry.register_lb("name", MyLBSolver)`.

## Validation Results

| Test | Metric | Result |
|------|--------|--------|
| PR₁ projection | max &#124;ref - ours&#124; | 0.00e+00 |
| PR₂ projection | max &#124;ref - ours&#124; | 0.00e+00 |
| BnB objective | &#124;obj_ref - obj_ours&#124; | 0.00e+00 |
| BnB solution | max &#124;x_ref - x_ours&#124; | 0.00e+00 |
| sum_rate | &#124;rate_ref - rate_ours&#124; | 0.00e+00 |
| Multi-column X_opt | max &#124;X_ref - X_ours&#124; | 0.00e+00 |
| HDF5 round-trip | All arrays | Exact |
| Constant-modulus | max &#124;mod - √(PT/N)&#124; | 1.11e-16 |
