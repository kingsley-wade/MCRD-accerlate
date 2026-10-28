"""Quick demo: run a short MCRD simulation with the Fig.2b parameter profile."""

from __future__ import annotations

import time

import numpy as np

from mcrd_accel.config import SolverConfig
from mcrd_accel.mcrd import run


def main():
    print("MCRD Simulation Demo (Fig.2b profile)")
    print("=" * 50)

    # Use a smaller grid and fewer steps for a quick demo
    config = SolverConfig.fig2b(
        Nx=128,
        Ny=128,
        n_steps=1000,
        snapshot_stride=200,  # Save snapshots every 200 steps
    )

    print(f"Grid: {config.Nx} x {config.Ny}")
    print(f"Spacing: dx={config.dx}, dy={config.dy}")
    print(f"Time step: dt={config.dt}")
    print(f"Total steps: {config.n_steps}")
    print(f"Final time: {config.final_time:.2f}")
    print(f"Snapshot stride: {config.snapshot_stride}")
    print()

    print("Running simulation...")
    start = time.time()
    result = run(config, checkpoint_stride=200)
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.3f} seconds")
    print()

    print("Metadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")
    print()

    print(f"Checkpoints: {len(result.checkpoints)}")
    if result.checkpoints:
        print("  step | time   | mass      | drift       | finite")
        print("  " + "-" * 55)
        for cp in result.checkpoints:
            print(
                f"  {cp.step:4d} | {cp.time:6.2f} | {cp.mass:9.4f} | "
                f"{cp.mass_drift:+.2e} | {cp.all_finite}"
            )
    print()

    print(f"Snapshots: {len(result.snapshots)}")
    if result.snapshots:
        print(f"  Saved at times: {[f'{t:.2f}' for t in result.snapshot_times]}")
        # Show some stats from the last snapshot
        last = result.snapshots[-1]
        print(f"  Last snapshot u: min={last.u.min():.4f}, max={last.u.max():.4f}, mean={last.u.mean():.4f}")
        print(f"  Last snapshot v: min={last.v.min():.4f}, max={last.v.max():.4f}, mean={last.v.mean():.4f}")
    print()

    print("Final state:")
    u_final = result.final_state.u
    v_final = result.final_state.v
    print(f"  u: min={u_final.min():.4f}, max={u_final.max():.4f}, mean={u_final.mean():.4f}")
    print(f"  v: min={v_final.min():.4f}, max={v_final.max():.4f}, mean={v_final.mean():.4f}")
    print(f"  All finite: {np.all(np.isfinite(u_final)) and np.all(np.isfinite(v_final))}")
    print()

    print("Done!")


if __name__ == "__main__":
    main()
