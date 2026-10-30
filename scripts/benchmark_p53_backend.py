"""Matched short-run p53 MCRD benchmark for JAX CPU and CUDA backends."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_p53_grid_study import _initial_fields, _parse_sizes


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


def _max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ACP is Linux, where ru_maxrss is reported in KiB.
    return int(value * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax-cpu", "jax-cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grid-sizes",
        type=_parse_sizes,
        default=_parse_sizes("64,128,256,512,1024,2048,4096"),
    )
    parser.add_argument("--domain-length", type=float, default=25.6)
    parser.add_argument("--steps", type=int, default=10240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu-cores", type=int, default=8)
    args = parser.parse_args()

    if args.steps <= 0 or args.repeats <= 0 or args.warmups < 0:
        parser.error("steps and repeats must be positive; warmups must be non-negative")
    if args.domain_length <= 0:
        parser.error("domain length must be positive")
    if args.grid_sizes[-1] != 4096:
        parser.error("include 4096 as the largest grid so CPU/GPU parity is measured there")

    affinity = None
    if args.backend == "jax-cpu":
        affinity = _limit_cpu_affinity(args.cpu_cores)

    import jax
    import jax.numpy as jnp
    import jaxlib

    from mcrd_accel.backends import get_jax_device
    from mcrd_accel.config import Backend, DType, SolverConfig
    from mcrd_accel.mcrd.jax_solver import JAXState, run_jax
    from mcrd_accel.mcrd.models import ReactionModel

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("JAX_ENABLE_X64=true is required for the matched p53 benchmark")
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
        raise RuntimeError(f"Expected an H100 device for the CUDA reference, got {device_kind}")

    sizes = args.grid_sizes
    finest_dx = args.domain_length / sizes[-1]
    dt = 0.5 * finest_dx**2 / (4.0 * 0.17)
    actual_time = args.steps * dt
    rng = np.random.default_rng(args.seed)
    phases_u = rng.uniform(0.0, 2.0 * np.pi, 6)
    phases_v = rng.uniform(0.0, 2.0 * np.pi, 6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "matched_p53_jax_backend_short_run",
        "backend": args.backend,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_enable_x64": jax.config.read("jax_enable_x64"),
        "device": str(device),
        "device_kind": device_kind,
        "platform": platform.platform(),
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
        "cpu_threads_requested": args.cpu_cores if affinity is not None else None,
        "physical_domain": {"Lx": args.domain_length, "Ly": args.domain_length},
        "grid_sizes": sizes,
        "dtype": "float64",
        "integrator": "explicit_euler",
        "boundary": "periodic",
        "dt": dt,
        "dt_rule": "0.5 * dx_4096^2 / (4 * Dv), with Dv=0.17",
        "steps_per_grid": args.steps,
        "simulated_time_per_grid": actual_time,
        "seed": args.seed,
        "initial_condition": "same seeded smooth periodic six-mode field as the fixed-time p53 study",
        "warmups": args.warmups,
        "timed_repeats": args.repeats,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    records = []
    for size in sizes:
        dx = args.domain_length / size
        config = SolverConfig.fig2b(
            Nx=size,
            Ny=size,
            dx=dx,
            dy=dx,
            dt=dt,
            n_steps=args.steps,
            dtype=DType.FLOAT64,
            backend=backend,
            seed=args.seed,
        )
        model = ReactionModel.p53_mcrd(config)
        u0, v0 = _initial_fields(size, phases_u, phases_v, "float64")
        initial_state = JAXState(
            u=jax.device_put(jnp.asarray(u0), device),
            v=jax.device_put(jnp.asarray(v0), device),
        )
        print(f"START backend={args.backend} grid={size}² steps={args.steps}", flush=True)

        cold_start = time.perf_counter()
        cold_result = run_jax(config, initial_state, checkpoint_stride=0, model=model)
        cold_wall_seconds = time.perf_counter() - cold_start
        initial_mass = cold_result.initial_mass

        warmup_seconds = []
        for warmup_index in range(args.warmups):
            start = time.perf_counter()
            run_jax(config, initial_state, checkpoint_stride=0, model=model)
            warmup_seconds.append(time.perf_counter() - start)
            print(
                f"WARMUP backend={args.backend} grid={size}² "
                f"index={warmup_index + 1}/{args.warmups} "
                f"seconds={warmup_seconds[-1]:.3f}",
                flush=True,
            )

        timed_seconds = []
        final_result = cold_result
        for repeat_index in range(args.repeats):
            start = time.perf_counter()
            final_result = run_jax(config, initial_state, checkpoint_stride=0, model=model)
            timed_seconds.append(time.perf_counter() - start)
            print(
                f"TIMED backend={args.backend} grid={size}² "
                f"repeat={repeat_index + 1}/{args.repeats} "
                f"seconds={timed_seconds[-1]:.3f}",
                flush=True,
            )

        final_mass = final_result.final_mass
        finite = bool(
            jnp.all(jnp.isfinite(final_result.final_state.u))
            & jnp.all(jnp.isfinite(final_result.final_state.v))
        )
        record = {
            "grid_size": size,
            "dx": dx,
            "dt": dt,
            "steps": args.steps,
            "simulated_time": actual_time,
            "cold_first_run_seconds_including_compile": cold_result.runtime_seconds,
            "cold_first_run_wall_seconds": cold_wall_seconds,
            "warmup_wall_seconds": warmup_seconds,
            "timed_run_seconds": timed_seconds,
            "steady_median_seconds": float(np.median(timed_seconds)),
            "steady_min_seconds": float(np.min(timed_seconds)),
            "steady_std_seconds": float(np.std(timed_seconds)),
            "steps_per_second": args.steps / float(np.median(timed_seconds)),
            "grid_steps_per_second": size * size * args.steps / float(np.median(timed_seconds)),
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "relative_mass_drift": abs(final_mass - initial_mass) / max(abs(initial_mass), 1e-300),
            "all_finite": finite,
            "device_memory_stats": device.memory_stats(),
            "process_max_rss_bytes": _max_rss_bytes(),
        }
        records.append(record)
        print(
            f"DONE backend={args.backend} grid={size}² "
            f"cold={record['cold_first_run_seconds_including_compile']:.3f}s "
            f"median={record['steady_median_seconds']:.3f}s "
            f"mass_drift={record['relative_mass_drift']:.3g} finite={finite}",
            flush=True,
        )

        if size == 4096:
            np.savez_compressed(
                args.output_dir / "p53-4096-final-state.npz",
                u=np.asarray(final_result.final_state.u),
                v=np.asarray(final_result.final_state.v),
                Lx=args.domain_length,
                Ly=args.domain_length,
                dx=dx,
                dy=dx,
                dt=dt,
                n_steps=args.steps,
                simulated_time=actual_time,
                dtype="float64",
                seed=args.seed,
            )

    report = {"manifest": manifest, "cases": records}
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "benchmark.csv").open("w", newline="") as stream:
        import csv

        fields = [
            "grid_size",
            "steps",
            "simulated_time",
            "cold_first_run_seconds_including_compile",
            "steady_median_seconds",
            "steady_min_seconds",
            "steady_std_seconds",
            "steps_per_second",
            "grid_steps_per_second",
            "relative_mass_drift",
            "all_finite",
            "process_max_rss_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields})

    summary = [
        "# Matched p53 MCRD backend benchmark",
        "",
        f"- Backend: {args.backend} ({device_kind})",
        f"- Dtype: float64; JAX/JAXlib: {jax.__version__}/{jaxlib.__version__}",
        f"- CPU affinity: {affinity if affinity is not None else 'not pinned by this job'}",
        f"- Region: {args.domain_length:g} x {args.domain_length:g}",
        f"- Grid sizes: {', '.join(map(str, sizes))}",
        f"- Steps per grid: {args.steps}; dt: {dt:.12g}; simulated time: {actual_time:.8g}",
        f"- Timed repeats: {args.repeats}; warmups: {args.warmups}",
        "",
        "Use benchmark.csv for steady medians. The first run includes compilation and is reported separately.",
        "The 4096² final state is saved for CPU/H100 numerical comparison.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(summary))
    print(f"COMPLETE backend={args.backend} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
