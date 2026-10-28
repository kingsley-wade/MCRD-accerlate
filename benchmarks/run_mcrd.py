"""Benchmark representative SciSim-Core PDE cases on NumPy and JAX CPU."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tracemalloc

import numpy as np

from mcrd_accel.config import Backend, SolverConfig
from mcrd_accel.mcrd import ReactionModel, run, run_jax


def benchmark_numpy(
    config: SolverConfig, n_warmup: int = 2, model: ReactionModel | None = None
) -> dict:
    """Benchmark NumPy solver."""
    # Warmup
    for _ in range(n_warmup):
        run(config, checkpoint_stride=0, model=model)

    # Timed runs
    times = []
    peak_memory = 0
    for _ in range(3):
        tracemalloc.start()
        tracemalloc.reset_peak()
        start = time.time()
        result = run(config, checkpoint_stride=0, model=model)
        times.append(time.time() - start)
        peak_memory = max(peak_memory, tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    return {
        "backend": "numpy",
        "reaction_model": "p53_mcrd" if model is None else model.name,
        "setup_seconds": 0.0,
        "warmup_count": n_warmup,
        "n_steps": config.n_steps,
        "grid": f"{config.Nx}x{config.Ny}",
        "shape": [config.Ny, config.Nx],
        "dtype": config.dtype.value,
        "config": config.to_dict(),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "steps_per_second": config.n_steps / np.mean(times),
        "grid_steps_per_second": config.Nx * config.Ny * config.n_steps / np.mean(times),
        "peak_memory_bytes": int(peak_memory),
        "peak_memory_method": "tracemalloc_host_allocations",
        "mass_drift": abs(result.metadata["final_mass"] - result.metadata["initial_mass"]) / result.metadata["initial_mass"],
    }


def benchmark_jax(
    config: SolverConfig, n_warmup: int = 2, model: ReactionModel | None = None
) -> dict:
    """Benchmark JAX solver."""
    # Warmup includes compilation and device setup.
    setup_start = time.time()
    for _ in range(n_warmup):
        run_jax(config, model=model)
    setup_seconds = time.time() - setup_start

    # Timed runs
    times = []
    peak_memory = 0
    for _ in range(3):
        tracemalloc.start()
        tracemalloc.reset_peak()
        result = run_jax(config, model=model)
        times.append(result.runtime_seconds)
        peak_memory = max(peak_memory, tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    return {
        "backend": config.backend.value,
        "reaction_model": "p53_mcrd" if model is None else model.name,
        "setup_seconds": setup_seconds,
        "warmup_count": n_warmup,
        "n_steps": config.n_steps,
        "grid": f"{config.Nx}x{config.Ny}",
        "shape": [config.Ny, config.Nx],
        "dtype": config.dtype.value,
        "config": config.to_dict(),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "steps_per_second": config.n_steps / np.mean(times),
        "grid_steps_per_second": config.Nx * config.Ny * config.n_steps / np.mean(times),
        "peak_memory_bytes": int(peak_memory),
        "peak_memory_method": "tracemalloc_host_allocations",
        "mass_drift": abs(result.final_mass - result.initial_mass) / result.initial_mass,
    }


def model_for_case(name: str, config: SolverConfig) -> ReactionModel:
    if name == "linear_exchange":
        return ReactionModel.linear_exchange(k_forward=0.8, k_reverse=0.3)
    if name == "nonlinear_autocatalysis":
        return ReactionModel.nonlinear_autocatalysis(rate=0.4)
    if name == "p53_mcrd":
        return ReactionModel.p53_mcrd(config)
    raise ValueError(f"Unknown PDE case {name!r}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark selected PDE cases on NumPy and JAX CPU.")
    parser.add_argument(
        "--case",
        choices=("linear_exchange", "nonlinear_autocatalysis", "p53_mcrd", "all"),
        default="p53_mcrd",
    )
    parser.add_argument("--grid-sizes", default="64,128,256")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_mcrd.json"))
    args = parser.parse_args()

    print("SciSim-Core PDE Benchmark: NumPy vs JAX CPU")
    print("=" * 60)

    grids = [int(size) for size in args.grid_sizes.split(",")]
    configs = [SolverConfig.fig2b(Nx=n, Ny=n, n_steps=args.steps) for n in grids]
    names = (
        ("linear_exchange", "nonlinear_autocatalysis", "p53_mcrd")
        if args.case == "all"
        else (args.case,)
    )

    results = []

    for name in names:
        for config in configs:
            model = model_for_case(name, config)
            print(f"\nCase: {name}, Grid: {config.Nx}x{config.Ny}, Steps: {config.n_steps}")
            print("-" * 60)

            # NumPy
            np_result = benchmark_numpy(config, model=model)
            print(f"NumPy:     {np_result['mean_time']*1000:.1f} ms +/- {np_result['std_time']*1000:.1f} ms | {np_result['steps_per_second']:.0f} steps/s | drift={np_result['mass_drift']:.2e}")
            results.append(np_result)

            # JAX CPU
            config_jax = replace(config, backend=Backend.JAX_CPU)
            jax_result = benchmark_jax(config_jax, model=model)
            print(f"JAX CPU:   {jax_result['mean_time']*1000:.1f} ms +/- {jax_result['std_time']*1000:.1f} ms | {jax_result['steps_per_second']:.0f} steps/s | drift={jax_result['mass_drift']:.2e}")
            results.append(jax_result)

            # Speedup
            speedup = np_result['mean_time'] / jax_result['mean_time']
            print(f"Speedup:   {speedup:.1f}x")

    # Save results
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        import jax
        jax_version = jax.__version__
    except ImportError:
        jax_version = None
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": revision,
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "jax": jax_version,
            },
            "records": results,
        }, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
