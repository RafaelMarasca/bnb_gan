"""
Basic Usage Example
===================
Demonstrates the full pipeline: generate data, optimize waveform,
compute metrics, and (optionally) serialize to HDF5.

Run from the radcom_waveform/ directory:
    python examples/basic_usage.py
"""

import sys
import os
import time

# Ensure the package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.utils.config import SystemConfig, BnBConfig
from src.signal_proc import generate_chirp, generate_channel, generate_symbols
from src.optimizer import WaveformMatrixOptimizer, bnb_solve
from src.metrics import RateMetric, ConvergenceMetric, sum_rate


def main():
    """Run a basic waveform optimization example."""
    print("=" * 60)
    print("  RadCom Waveform Design — Basic Usage Example")
    print("=" * 60)

    # ---- 1. System setup ----
    sys_cfg = SystemConfig(N=16, K=4, L=20, PT=1.0, SNR_dB=10.0)
    bnb_cfg = BnBConfig(
        rule="ARS",
        lb_solver="cvxpy",
        ub_solver="slsqp",
        tol=1e-3,
        max_iter=200,
        verbose=False,
    )
    print(f"\nSystem: N={sys_cfg.N}, K={sys_cfg.K}, L={sys_cfg.L}, "
          f"PT={sys_cfg.PT}, SNR={sys_cfg.SNR_dB} dB")
    print(f"BnB:    rule={bnb_cfg.rule}, LB={bnb_cfg.lb_solver}, "
          f"UB={bnb_cfg.ub_solver}, tol={bnb_cfg.tol}")

    # ---- 2. Generate data ----
    np.random.seed(42)
    H = generate_channel(sys_cfg.K, sys_cfg.N)
    S = generate_symbols(sys_cfg.K, sys_cfg.L)
    X0 = generate_chirp(sys_cfg.N, sys_cfg.L, sys_cfg.PT)
    print(f"\nGenerated: H {H.shape}, S {S.shape}, X0 {X0.shape}")

    # ---- 3. Baseline rate ----
    rate_ref = sum_rate(H, X0, S, sys_cfg.N0)
    print(f"Reference waveform rate: {rate_ref:.4f} bps/Hz")

    # ---- 4. Single-column BnB demo ----
    print("\n--- Single-column BnB demo ---")
    sc = sys_cfg.scale
    Ht = sc * H
    x0_col = X0[:, 0] / sc
    s_col = S[:, 0]
    epsilon = 1.0

    t0 = time.time()
    x_opt, obj, lb_hist, ub_hist = bnb_solve(
        Ht, s_col, x0_col, epsilon,
        rule="ARS", lb_solver="cvxpy", ub_solver="slsqp",
        tol=1e-3, max_iter=200,
    )
    dt = time.time() - t0
    print(f"  Objective: {obj:.6f}  ({len(ub_hist)} iters, {dt:.2f}s)")
    print(f"  Gap: {ub_hist[-1] - lb_hist[-1]:.2e}")

    # Convergence metric
    cm = ConvergenceMetric()
    conv = cm.compute(lb_history=lb_hist, ub_history=ub_hist, tol=1e-3)
    print(f"  Iters to tol: {conv.values['iters_to_tol']}")

    # ---- 5. Full matrix optimization ----
    print(f"\n--- Full matrix optimization (epsilon={epsilon}) ---")
    optimizer = WaveformMatrixOptimizer(sys_cfg, bnb_cfg)
    t0 = time.time()
    X_opt, col_results = optimizer.optimize(H, S, X0, epsilon)
    dt = time.time() - t0
    print(f"  Completed {len(col_results)} columns in {dt:.1f}s")

    # ---- 6. Rate comparison ----
    rate_opt = sum_rate(H, X_opt, S, sys_cfg.N0)
    improvement = rate_opt - rate_ref
    print(f"\n  Reference rate: {rate_ref:.4f} bps/Hz")
    print(f"  Optimized rate: {rate_opt:.4f} bps/Hz")
    print(f"  Improvement:    {improvement:+.4f} bps/Hz ({improvement / rate_ref * 100:+.1f}%)")

    # AWGN capacity bound
    awgn_cap = sys_cfg.K * np.log2(1.0 + 1.0 / sys_cfg.N0)
    print(f"  AWGN capacity:  {awgn_cap:.4f} bps/Hz")

    # ---- 7. Rate metric ----
    rm = RateMetric()
    rate_result = rm.compute(H=H, X=X_opt, S=S, N0=sys_cfg.N0)
    print(f"\n  Per-user rates: {[f'{r:.2f}' for r in rate_result.values['per_user_rate']]}")

    # ---- 8. Constant-modulus check ----
    mods = np.abs(X_opt)
    target_mod = np.sqrt(sys_cfg.PT / sys_cfg.N)
    max_dev = np.max(np.abs(mods - target_mod))
    print(f"\n  CM deviation: max |mod - sqrt(PT/N)| = {max_dev:.2e}")
    assert max_dev < 1e-6, f"Constant-modulus violated! max_dev={max_dev}"
    print("  Constant-modulus constraint satisfied ✓")

    print("\n" + "=" * 60)
    print("  Example complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
