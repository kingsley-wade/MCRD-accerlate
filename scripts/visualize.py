"""Visualize MCRD simulation snapshots."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from mcrd_accel.config import SolverConfig
from mcrd_accel.mcrd import run


def main():
    parser = argparse.ArgumentParser(description="Visualize MCRD simulation")
    parser.add_argument("--Nx", type=int, default=128, help="Grid size X")
    parser.add_argument("--Ny", type=int, default=128, help="Grid size Y")
    parser.add_argument("--n-steps", type=int, default=1000, help="Total steps")
    parser.add_argument("--snapshot-idx", type=int, default=-1, help="Snapshot index to plot (-1 for last)")
    parser.add_argument("--save", type=str, help="Save plot to file")
    args = parser.parse_args()

    config = SolverConfig.fig2b(
        Nx=args.Nx,
        Ny=args.Ny,
        n_steps=args.n_steps,
        snapshot_stride=max(1, args.n_steps // 5),  # ~5 snapshots
    )

    print(f"Running simulation: {config.Nx}x{config.Ny}, {config.n_steps} steps...")
    result = run(config, checkpoint_stride=0)

    if not result.snapshots:
        print("No snapshots saved. Re-run with snapshot_stride > 0.")
        return

    idx = args.snapshot_idx
    snapshot = result.snapshots[idx]
    time = result.snapshot_times[idx]

    print(f"Plotting snapshot {idx} at time {time:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot u field
    im0 = axes[0].imshow(
        snapshot.u,
        cmap="viridis",
        extent=[0, config.Lx, 0, config.Ly],
        origin="lower",
    )
    axes[0].set_title(f"u (DNA-bound protein) at t={time:.2f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], label="Concentration")

    # Plot v field
    im1 = axes[1].imshow(
        snapshot.v,
        cmap="viridis",
        extent=[0, config.Lx, 0, config.Ly],
        origin="lower",
    )
    axes[1].set_title(f"v (free protein) at t={time:.2f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], label="Concentration")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
