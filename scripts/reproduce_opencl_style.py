"""Reproduce the legacy p53 MCRD speckled morphology with a deterministic seed."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/opencl-style-1024"))
    parser.add_argument("--backend", choices=("jax-cuda",), default="jax-cuda")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--domain-length", type=float, default=25.6)
    parser.add_argument("--final-time", type=float, default=40.0)
    parser.add_argument("--reference-dx", type=float, default=0.2)
    parser.add_argument("--reference-dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--u-limits", nargs=2, type=float, default=(15.0, 23.0))
    parser.add_argument("--v-limits", nargs=2, type=float, default=(6.8, 7.4))
    args = parser.parse_args()

    if args.size < 16 or args.domain_length <= 0 or args.final_time <= 0:
        parser.error("size must be at least 16; domain length and final time must be positive")
    if args.reference_dx <= 0 or args.reference_dt <= 0:
        parser.error("reference dx and dt must be positive")
    if args.u_limits[0] >= args.u_limits[1] or args.v_limits[0] >= args.v_limits[1]:
        parser.error("color limits must be increasing pairs")

    dx = args.domain_length / args.size
    # Preserve the legacy run's 4*Dv*dt/dx^2 diffusion number while refining the grid.
    dt = args.reference_dt * (dx / args.reference_dx) ** 2
    n_steps = round(args.final_time / dt)
    actual_final_time = n_steps * dt
    if not np.isclose(actual_final_time, args.final_time, rtol=0.0, atol=dt * 1e-9):
        parser.error("final time must be an integer multiple of the derived dt")

    from mcrd_accel.backends import get_jax_device
    from mcrd_accel.config import Backend, DType, SolverConfig
    from mcrd_accel.mcrd.solver import State, run

    backend = Backend(args.backend)
    config = SolverConfig.fig2b(
        Nx=args.size,
        Ny=args.size,
        dx=dx,
        dy=dx,
        dt=dt,
        n_steps=n_steps,
        dtype=DType(args.dtype),
        backend=backend,
        seed=args.seed,
        snapshot_stride=0,
    )
    # Build random fluctuations at the legacy physical cell scale, then map them
    # to the finer grid. Independent 1024² noise is almost entirely diffused away
    # by t=40 and does not recreate the legacy speckled morphology.
    reference_size = round(args.domain_length / args.reference_dx)
    rng = np.random.default_rng(args.seed)
    ref_u = rng.uniform(0.25, 0.75, (reference_size, reference_size))
    ref_v = rng.uniform(24.5, 25.5, (reference_size, reference_size))
    source_index = np.minimum(
        (np.arange(args.size) * reference_size // args.size).astype(np.int64),
        reference_size - 1,
    )
    numpy_dtype = np.float32 if args.dtype == "float32" else np.float64
    initial = State(
        u=ref_u[np.ix_(source_index, source_index)].astype(numpy_dtype),
        v=ref_v[np.ix_(source_index, source_index)].astype(numpy_dtype),
    )
    initial_stats = {
        "u_mean": float(np.mean(initial.u)), "u_std": float(np.std(initial.u)),
        "v_mean": float(np.mean(initial.v)), "v_std": float(np.std(initial.v)),
    }
    device = get_jax_device(backend)
    if device.platform != "gpu":
        raise RuntimeError(f"CUDA backend resolved to unexpected device {device}")
    device_kind = getattr(device, "device_kind", str(device))
    if "H100" not in device_kind.upper():
        raise RuntimeError(f"Expected an H100 CUDA device, got {device_kind}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"START backend={backend.value} grid={args.size}² dx={dx:.8g} "
        f"dt={dt:.8g} steps={n_steps} final_time={actual_final_time:.8g} seed={args.seed}",
        flush=True,
    )
    start = time.perf_counter()
    result = run(config, initial_state=initial, checkpoint_stride=0)
    elapsed = time.perf_counter() - start
    u = np.asarray(result.final_state.u)
    v = np.asarray(result.final_state.v)

    np.savez_compressed(
        args.output_dir / f"p53-random-speckle-{args.size}-t{actual_final_time:g}.npz",
        u=u, v=v, Lx=args.domain_length, Ly=args.domain_length, dx=dx, dy=dx,
        dt=dt, n_steps=n_steps, time=actual_final_time,
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, field, title, limits, label in (
        (axes[0], u, "u (DNA-bound protein)", args.u_limits, "u concentration"),
        (axes[1], v, "v (free protein)", args.v_limits, "v concentration"),
    ):
        image = ax.imshow(
            field, origin="lower", extent=(0, args.domain_length, 0, args.domain_length),
            cmap="viridis", vmin=limits[0], vmax=limits[1], interpolation="nearest",
        )
        ax.set_title(f"{title} at t={actual_final_time:.2f} ({args.size}²)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label=label)
    image_path = args.output_dir / f"p53-random-speckle-{args.size}-t{actual_final_time:g}.png"
    fig.savefig(image_path, dpi=160)
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "experiment": "legacy_opencl_style_random_initial_condition_reproduction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "backend": backend.value,
        "device": str(device),
        "device_kind": device_kind,
        "grid": {"Nx": args.size, "Ny": args.size, "Lx": args.domain_length, "Ly": args.domain_length, "dx": dx, "dy": dx},
        "time": {
            "target_final_time": args.final_time, "actual_final_time": actual_final_time,
            "dt": dt, "steps": n_steps,
            "dt_scaling": "reference_dt * (dx / reference_dx)^2",
            "reference_dx": args.reference_dx, "reference_dt": args.reference_dt,
        },
        "model": config.to_dict(),
        "dtype": args.dtype,
        "seed": args.seed,
        "initial_condition": (
            f"seeded uniform random fields generated at reference size {reference_size}² "
            f"(dx={args.reference_dx:g}), then nearest-neighbor mapped to {args.size}²; "
            "u~U(0.25,0.75), v~U(24.5,25.5)"
        ),
        "initial_field_statistics": initial_stats,
        "integrator": "explicit_euler",
        "boundary": "periodic roll / synchronous out-of-place update",
        "runtime_seconds": elapsed,
        "final_field_statistics": {
            "u": {"mean": float(u.mean()), "std": float(u.std()), "min": float(u.min()), "max": float(u.max())},
            "v": {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())},
        },
        "plot": {"file": image_path.name, "cmap": "viridis", "u_limits": args.u_limits, "v_limits": args.v_limits},
        "interpretation": "Reproduces the random-field speckled morphology; not a bitwise replay of the race-prone legacy OpenCL kernel.",
    }
    (args.output_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"DONE runtime={elapsed:.2f}s image={image_path} data={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
