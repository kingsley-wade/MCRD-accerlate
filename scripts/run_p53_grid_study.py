"""Run the fixed-domain p53 MCRD spatial study on JAX CPU or CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _initial_fields(
    size: int, phases_u: np.ndarray, phases_v: np.ndarray, dtype: str
):
    modes = ((1, 0), (0, 1), (1, 1), (2, -1), (-1, 2), (2, 1))
    x = np.arange(size, dtype=np.float64) / size
    xx, yy = np.meshgrid(x, x)

    def field(phases):
        values = np.zeros((size, size), dtype=np.float64)
        for (kx, ky), phase in zip(modes, phases, strict=True):
            values += np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phase)
        return values / len(modes)

    numpy_dtype = np.float64 if dtype == "float64" else np.float32
    u = (0.5 + 0.25 * field(phases_u)).astype(numpy_dtype)
    v = (25.0 + 0.5 * field(phases_v)).astype(numpy_dtype)
    return u, v


def _parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid sizes must be comma-separated integers") from exc
    if sizes != sorted(set(sizes)) or any(size < 16 for size in sizes):
        raise argparse.ArgumentTypeError("grid sizes must be unique, ascending, and at least 16")
    if any(right != 2 * left for left, right in zip(sizes, sizes[1:])):
        raise argparse.ArgumentTypeError("each grid size must be twice the previous size")
    if sizes[-1] != 4096:
        raise argparse.ArgumentTypeError("the final grid must be 4096 for the requested figure")
    return sizes


def _limit_cpu_affinity(requested: int) -> list[int]:
    if requested <= 0:
        raise ValueError("CPU core limit must be positive")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU core pinning requires Linux sched_getaffinity support")
    available = sorted(os.sched_getaffinity(0))
    if len(available) < requested:
        raise RuntimeError(
            f"ACP grants {len(available)} CPU cores, but {requested} were requested"
        )
    selected = available[:requested]
    os.sched_setaffinity(0, set(selected))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("jax-cpu", "jax-cuda"), default="jax-cuda")
    parser.add_argument("--cpu-cores", type=int, default=8)
    parser.add_argument("--grid-sizes", type=_parse_sizes, default=_parse_sizes("64,128,256,512,1024,2048,4096"))
    parser.add_argument("--domain-length", type=float, default=25.6)
    parser.add_argument("--final-time", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--chunk-steps", type=int, default=10240)
    parser.add_argument("--relative-change-tolerance", type=float, default=1e-5)
    parser.add_argument("--consecutive-windows", type=int, default=3)
    args = parser.parse_args()

    if args.domain_length <= 0 or args.final_time <= 0 or args.chunk_steps <= 0:
        parser.error("domain length, final time, and chunk steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    affinity = _limit_cpu_affinity(args.cpu_cores) if args.backend == "jax-cpu" else None

    import jax
    import jax.numpy as jnp
    import jaxlib

    from mcrd_accel.backends import get_jax_device
    from mcrd_accel.config import Backend, DType, SolverConfig
    from mcrd_accel.mcrd.jax_solver import JAXState, run_jax
    from mcrd_accel.mcrd.models import ReactionModel

    backend = Backend(args.backend)
    device = get_jax_device(backend)
    expected_platform = "cpu" if backend is Backend.JAX_CPU else "gpu"
    if device.platform != expected_platform:
        raise RuntimeError(
            f"backend {args.backend} selected platform {device.platform}, "
            f"expected {expected_platform}"
        )
    device_kind = getattr(device, "device_kind", str(device))
    if backend is Backend.JAX_CUDA and "H100" not in device_kind.upper():
        raise RuntimeError(f"Expected an H100 MIG device, detected: {device_kind}")
    if args.dtype == "float64" and not jax.config.read("jax_enable_x64"):
        raise RuntimeError("float64 was requested but JAX_ENABLE_X64 is not enabled")

    solver_dtype = DType(args.dtype)

    sizes = args.grid_sizes
    finest_dx = args.domain_length / sizes[-1]
    diffusion_max = 0.17
    dt = 0.5 * finest_dx**2 / (4.0 * diffusion_max)
    n_steps = round(args.final_time / dt)
    actual_final_time = n_steps * dt
    if n_steps % 4:
        raise RuntimeError("step count must divide evenly into four plot snapshots")
    if n_steps % args.chunk_steps:
        raise RuntimeError("chunk steps must divide the full run evenly; use 10240 for this setup")
    snapshot_steps = {n_steps * k // 4 for k in (1, 2, 3, 4)}
    if any(step % args.chunk_steps for step in snapshot_steps):
        raise RuntimeError("chunk steps must align with the four snapshot times")

    rng = np.random.default_rng(args.seed)
    phases_u = rng.uniform(0.0, 2.0 * np.pi, 6)
    phases_v = rng.uniform(0.0, 2.0 * np.pi, 6)
    model_config = SolverConfig.fig2b(
        Nx=sizes[0], Ny=sizes[0], dx=args.domain_length / sizes[0],
        dy=args.domain_length / sizes[0], dt=dt, n_steps=1,
        dtype=solver_dtype, backend=backend, seed=args.seed,
    )
    model = ReactionModel.p53_mcrd(model_config)

    manifest = {
        "schema_version": 1,
        "experiment": "p53_fixed_domain_spatial_convergence_t40",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform_machine": platform.machine(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "device": str(device),
        "device_kind": device_kind,
        "backend": args.backend,
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
        "cpu_threads_requested": args.cpu_cores if affinity is not None else None,
        "dtype": args.dtype,
        "integrator": "explicit_euler",
        "boundary": "periodic",
        "physical_domain": {"Lx": args.domain_length, "Ly": args.domain_length},
        "grid_sizes": sizes,
        "dt": dt,
        "dt_rule": "0.5 * dx_4096^2 / (4 * Dv), with Dv=0.17",
        "steps_per_grid": n_steps,
        "target_final_time": args.final_time,
        "actual_final_time": actual_final_time,
        "seed": args.seed,
        "initial_condition": "same seeded smooth periodic six-mode field sampled on every grid",
        "adaptive_stop": {
            "relative_rms_change_threshold": args.relative_change_tolerance,
            "consecutive_windows": args.consecutive_windows,
            "window_steps": args.chunk_steps,
            "window_time": args.chunk_steps * dt,
            "behavior": "diagnostic only; every grid continues to the fixed final time",
        },
    }
    (args.output_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    records = []
    previous_state = None
    evolution_snapshots = None
    for size in sizes:
        dx = args.domain_length / size
        config = SolverConfig.fig2b(
            Nx=size, Ny=size, dx=dx, dy=dx, dt=dt, n_steps=n_steps,
            dtype=solver_dtype, backend=backend, seed=args.seed,
        )
        initial_u, initial_v = _initial_fields(size, phases_u, phases_v, args.dtype)
        state = JAXState(
            u=jax.device_put(jnp.asarray(initial_u), device),
            v=jax.device_put(jnp.asarray(initial_v), device),
        )
        initial_mass = float(jnp.mean(state.u + state.v))
        previous_monitor = state
        consecutive = 0
        first_threshold_crossing_time = None
        threshold_recrossed = False
        checkpoints = []
        snapshots = {
            0.0: (
                initial_u[:: max(1, size // 512), :: max(1, size // 512)],
                initial_v[:: max(1, size // 512), :: max(1, size // 512)],
            )
        }
        elapsed_solver = 0.0
        start = time.perf_counter()
        print(f"START grid={size}x{size} steps={n_steps} dt={dt:.12g}", flush=True)

        for completed in range(args.chunk_steps, n_steps + 1, args.chunk_steps):
            chunk_config = replace(config, n_steps=args.chunk_steps, snapshot_stride=0)
            result = run_jax(chunk_config, state, checkpoint_stride=0, model=model)
            state = result.final_state
            elapsed_solver += result.runtime_seconds

            rel_u = float(
                jnp.sqrt(jnp.mean(jnp.square(state.u - previous_monitor.u)))
                / jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(state.u))), 1e-12)
            )
            rel_v = float(
                jnp.sqrt(jnp.mean(jnp.square(state.v - previous_monitor.v)))
                / jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(state.v))), 1e-12)
            )
            finite = bool(jnp.all(jnp.isfinite(state.u)) & jnp.all(jnp.isfinite(state.v)))
            current_mass = float(jnp.mean(state.u + state.v))
            mass_drift = abs(current_mass - initial_mass) / max(abs(initial_mass), 1e-30)
            current_time = completed * dt
            max_change = max(rel_u, rel_v)
            if max_change < args.relative_change_tolerance:
                consecutive += 1
                if (
                    consecutive >= args.consecutive_windows
                    and first_threshold_crossing_time is None
                ):
                    first_threshold_crossing_time = current_time
            else:
                consecutive = 0
                if first_threshold_crossing_time is not None:
                    threshold_recrossed = True

            checkpoint = {
                "step": completed,
                "time": current_time,
                "relative_change_u": rel_u,
                "relative_change_v": rel_v,
                "relative_mass_drift": mass_drift,
                "finite": finite,
            }
            checkpoints.append(checkpoint)
            if not finite:
                raise FloatingPointError(f"nonfinite state at grid {size}, step {completed}")

            previous_monitor = state
            if completed in snapshot_steps:
                stride = max(1, size // 512)
                snapshots[float(current_time)] = (
                    np.asarray(state.u)[::stride, ::stride],
                    np.asarray(state.v)[::stride, ::stride],
                )
            if len(checkpoints) % 10 == 0 or completed == n_steps:
                min_u, max_u = float(jnp.min(state.u)), float(jnp.max(state.u))
                min_v, max_v = float(jnp.min(state.v)), float(jnp.max(state.v))
                print(
                    f"PROGRESS grid={size} step={completed}/{n_steps} t={current_time:.6g} "
                    f"elapsed={time.perf_counter()-start:.1f}s relchange={max_change:.3g} "
                    f"mass_drift={mass_drift:.3g} finite={finite} "
                    f"u_range=({min_u:.4g},{max_u:.4g}) v_range=({min_v:.4g},{max_v:.4g})",
                    flush=True,
                )

        final_u, final_v = np.asarray(state.u), np.asarray(state.v)
        record = {
            "grid_size": size,
            "dx": dx,
            "dy": dx,
            "physical_domain": {"Lx": args.domain_length, "Ly": args.domain_length},
            "dt": dt,
            "steps": n_steps,
            "final_time": actual_final_time,
            "runtime_seconds_including_compile_and_diagnostics": time.perf_counter() - start,
            "solver_runtime_seconds": elapsed_solver,
            "relative_mass_drift": checkpoints[-1]["relative_mass_drift"],
            "all_finite": all(item["finite"] for item in checkpoints),
            "state_range": {
                "u": [float(final_u.min()), float(final_u.max())],
                "v": [float(final_v.min()), float(final_v.max())],
            },
            "adaptive_convergence_diagnostic": {
                **manifest["adaptive_stop"],
                "first_threshold_crossing_time": first_threshold_crossing_time,
                "threshold_recrossed_after_first_crossing": threshold_recrossed,
                "below_threshold_at_final_time": max(rel_u, rel_v)
                < args.relative_change_tolerance,
                "remained_below_threshold_after_first_crossing": (
                    first_threshold_crossing_time is not None and not threshold_recrossed
                ),
                "continued_to_fixed_final_time": True,
            },
            "device_memory_stats": device.memory_stats(),
            "checkpoints": checkpoints,
        }

        if previous_state is not None:
            coarse_u, coarse_v = previous_state
            du = final_u[::2, ::2].astype(np.float64) - coarse_u.astype(np.float64)
            dv = final_v[::2, ::2].astype(np.float64) - coarse_v.astype(np.float64)
            error_u = float(np.sqrt(np.mean(du * du)))
            error_v = float(np.sqrt(np.mean(dv * dv)))
            record["difference_from_previous_grid"] = {
                "coarse_grid_size": size // 2,
                "fine_grid_size": size,
                "u_rms_difference": error_u,
                "v_rms_difference": error_v,
                "u_relative_rms_difference": error_u
                / max(float(np.sqrt(np.mean(coarse_u.astype(np.float64) ** 2))), 1e-30),
                "v_relative_rms_difference": error_v
                / max(float(np.sqrt(np.mean(coarse_v.astype(np.float64) ** 2))), 1e-30),
            }
        previous_state = (final_u, final_v)
        records.append(record)
        evolution_snapshots = snapshots

        if size == sizes[-1]:
            np.savez_compressed(
                args.output_dir / "p53-final-state-4096-t40.npz",
                u=final_u,
                v=final_v,
                Lx=args.domain_length,
                Ly=args.domain_length,
                dx=dx,
                dy=dx,
                dt=dt,
                n_steps=n_steps,
                time=actual_final_time,
            )
        (args.output_dir / f"p53-checkpoints-{size}.json").write_text(
            json.dumps(checkpoints, indent=2) + "\n"
        )
        print(
            f"DONE grid={size} runtime={record['runtime_seconds_including_compile_and_diagnostics']:.1f}s "
            f"mass_drift={record['relative_mass_drift']:.3g} "
            f"threshold_crossing_at={first_threshold_crossing_time} "
            f"threshold_recrossed={threshold_recrossed}",
            flush=True,
        )

    for index in range(len(records) - 2):
        current_error = records[index + 1].get("difference_from_previous_grid")
        finer_error = records[index + 2].get("difference_from_previous_grid")
        if current_error and finer_error:
            for species in ("u", "v"):
                coarse = current_error[f"{species}_rms_difference"]
                fine = finer_error[f"{species}_rms_difference"]
                current_error[f"{species}_observed_order"] = (
                    math.log(coarse / fine, 2) if coarse > 0 and fine > 0 else None
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = sorted(evolution_snapshots)
    u_min = min(float(np.min(evolution_snapshots[t][0])) for t in times)
    u_max = max(float(np.max(evolution_snapshots[t][0])) for t in times)
    v_min = min(float(np.min(evolution_snapshots[t][1])) for t in times)
    v_max = max(float(np.max(evolution_snapshots[t][1])) for t in times)
    fig, axes = plt.subplots(2, len(times), figsize=(4 * len(times), 8), constrained_layout=True)
    for column, snapshot_time in enumerate(times):
        u, v = evolution_snapshots[snapshot_time]
        image_u = axes[0, column].imshow(
            u, origin="lower", extent=(0, args.domain_length, 0, args.domain_length),
            cmap="viridis", vmin=u_min, vmax=u_max, interpolation="nearest",
        )
        image_v = axes[1, column].imshow(
            v, origin="lower", extent=(0, args.domain_length, 0, args.domain_length),
            cmap="viridis", vmin=v_min, vmax=v_max, interpolation="nearest",
        )
        axes[0, column].set_title(f"u at t={snapshot_time:.2f}")
        axes[1, column].set_title(f"v at t={snapshot_time:.2f}")
        axes[0, column].set_xlabel("x")
        axes[1, column].set_xlabel("x")
        axes[0, column].set_ylabel("y")
        axes[1, column].set_ylabel("y")
    fig.colorbar(image_u, ax=axes[0, :].tolist(), label="u concentration", shrink=0.85)
    fig.colorbar(image_v, ax=axes[1, :].tolist(), label="v concentration", shrink=0.85)
    fig.savefig(args.output_dir / "p53-evolution-4096-t0-to-t40.png", dpi=160)
    plt.close(fig)

    final_u, final_v = previous_state
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    image_u = axes[0].imshow(
        final_u, origin="lower", extent=(0, args.domain_length, 0, args.domain_length),
        cmap="viridis", interpolation="nearest",
    )
    image_v = axes[1].imshow(
        final_v, origin="lower", extent=(0, args.domain_length, 0, args.domain_length),
        cmap="viridis", interpolation="nearest",
    )
    axes[0].set_title(f"p53 MCRD u, t={actual_final_time:.4f}, 4096²")
    axes[1].set_title(f"p53 MCRD v, t={actual_final_time:.4f}, 4096²")
    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(image_u, ax=axes[0], label="u concentration")
    fig.colorbar(image_v, ax=axes[1], label="v concentration")
    fig.savefig(args.output_dir / "p53-t40-4096-heatmap.png", dpi=200)
    plt.close(fig)

    report = {"manifest": manifest, "cases": [{k: v for k, v in row.items() if k != "checkpoints"} for row in records]}
    (args.output_dir / "p53-grid-study.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "grid_size", "dx", "dt", "steps", "final_time",
        "runtime_seconds_including_compile_and_diagnostics", "relative_mass_drift",
        "all_finite", "u_min", "u_max", "v_min", "v_max",
        "adaptive_first_threshold_crossing_time",
    ]
    with (args.output_dir / "p53-grid-study.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "grid_size": row["grid_size"],
                "dx": row["dx"],
                "dt": row["dt"],
                "steps": row["steps"],
                "final_time": row["final_time"],
                "runtime_seconds_including_compile_and_diagnostics": row["runtime_seconds_including_compile_and_diagnostics"],
                "relative_mass_drift": row["relative_mass_drift"],
                "all_finite": row["all_finite"],
                "u_min": row["state_range"]["u"][0],
                "u_max": row["state_range"]["u"][1],
                "v_min": row["state_range"]["v"][0],
                "v_max": row["state_range"]["v"][1],
                "adaptive_first_threshold_crossing_time": row["adaptive_convergence_diagnostic"]["first_threshold_crossing_time"],
            })

    summary = [
        f"# p53 MCRD fixed-domain grid study ({args.backend})",
        "",
        f"- Physical region: {args.domain_length:g} x {args.domain_length:g}",
        f"- Target time: {actual_final_time:.8g}",
        f"- Grid sizes: {', '.join(map(str, sizes))}",
        f"- Backend: {args.backend} ({device_kind}), {args.dtype}",
        f"- Shared time step: {dt:.10g}; {n_steps:,} steps at every grid size",
        "- Adaptive stop check: relative RMS change < tolerance for consecutive windows; diagnostic only, all cases continue to the target time.",
        "- The spatial comparison uses the same seeded smooth periodic six-mode initial field sampled on each grid.",
        "",
        "Machine-readable timings and diagnostics are in p53-grid-study.csv and p53-grid-study.json.",
        "Figures: p53-t40-4096-heatmap.png and p53-evolution-4096-t0-to-t40.png.",
        "The compressed final fields are in p53-final-state-4096-t40.npz.",
        "",
        "This is a single-backend numerical grid study. Use compare_full_workflows.py for a matched CPU/H100 comparison.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(summary))
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
