"""Run bounded smoke checks for parabolic, elliptic, and hyperbolic PDEs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mcrd_accel.backends import JAX_BACKENDS, get_jax_device
from mcrd_accel.config import Backend, DType, SolverConfig, ValidationConfig
from mcrd_accel.mcrd import ReactionModel
from mcrd_accel.mcrd.validation import (
    run_conservation_check,
    run_spatial_convergence,
    run_stability_sweep,
    run_temporal_convergence,
)
from mcrd_accel.pde import GridConfig, run_heat, run_wave, solve_periodic_poisson


def _grid(size: int, backend: Backend, dtype: DType) -> GridConfig:
    return GridConfig(Nx=size, Ny=size, dx=1.0 / size, dy=1.0 / size, backend=backend, dtype=dtype)


def _run_heat(grid: GridConfig, steps: int, requested_dt: float) -> dict:
    alpha = 0.1
    dt = min(requested_dt, 0.4 / (alpha * (grid.dx**-2 + grid.dy**-2)))
    x = np.arange(grid.Nx) * grid.dx
    initial = 0.75 + np.broadcast_to(np.cos(2 * np.pi * x), grid.shape).copy()
    result = run_heat(grid, alpha=alpha, dt=dt, n_steps=steps, initial_state=initial)
    eigenvalue = -4.0 * np.sin(np.pi / grid.Nx) ** 2 / grid.dx**2
    expected_amplitude = (1.0 + dt * alpha * eigenvalue) ** steps
    expected = 0.75 + expected_amplitude * (initial - 0.75)
    mode_error = float(np.max(np.abs(np.asarray(result.final_state) - expected)))
    return {
        "grid": grid.to_dict(),
        "steps": steps,
        "dt": dt,
        "mode_max_error": mode_error,
        "mass_drift": result.final_mass - result.initial_mass,
        "stability_number": result.diagnostics["stability_number"],
        "runtime_seconds": result.runtime_seconds,
    }


def _run_poisson(grid: GridConfig) -> dict:
    rng = np.random.default_rng(grid.seed)
    rhs = rng.standard_normal(grid.shape).astype(grid.numpy_dtype)
    rhs -= rhs.mean(dtype=np.float64)
    result = solve_periodic_poisson(grid, rhs, tolerance=1e-6, max_iterations=2000)
    return {
        "grid": grid.to_dict(),
        "rhs_mean": result.rhs_mean,
        "relative_residual": result.relative_residual,
        "iterations": result.iterations,
        "converged": result.converged,
        "runtime_seconds": result.runtime_seconds,
    }


def _run_wave(grid: GridConfig, steps: int, requested_dt: float) -> dict:
    speed = 0.4
    dt = min(requested_dt, 0.8 / (speed * np.sqrt(grid.dx**-2 + grid.dy**-2)))
    x = np.arange(grid.Nx) * grid.dx
    initial = 0.75 + np.broadcast_to(np.cos(2 * np.pi * x), grid.shape).copy()
    result = run_wave(
        grid,
        wave_speed=speed,
        dt=dt,
        n_steps=steps,
        initial_displacement=initial,
        snapshot_stride=max(1, steps // 4),
    )
    eigenvalue = -4.0 * np.sin(np.pi / grid.Nx) ** 2 / grid.dx**2
    omega_dt = np.arccos(1.0 + 0.5 * (dt * speed) ** 2 * eigenvalue)
    expected = 0.75 + np.cos(steps * omega_dt) * (initial - 0.75)
    mode_error = float(np.max(np.abs(np.asarray(result.final_state) - expected)))
    return {
        "grid": grid.to_dict(),
        "steps": steps,
        "dt": dt,
        "cfl_number": result.diagnostics["cfl_number"],
        "mode_max_error": mode_error,
        "energy_relative_drift": result.diagnostics["relative_energy_drift"],
        "snapshot_steps": result.snapshot_steps,
        "mass_drift": result.final_mass - result.initial_mass,
        "runtime_seconds": result.runtime_seconds,
    }


def _run_p53(sizes: list[int], steps: int, dt: float, backend: Backend, dtype: DType) -> dict:
    size = max(sizes)
    config = SolverConfig.fig2b(
        Nx=size,
        Ny=size,
        dx=1.0 / size,
        dy=1.0 / size,
        dt=dt,
        n_steps=steps,
        backend=backend,
        dtype=dtype,
    )
    validation = ValidationConfig(
        solver=config,
        grid_sizes=sizes,
        time_steps=[dt, dt / 2.0],
    )
    model = ReactionModel.p53_mcrd(config)
    return {
        "conservation": asdict(run_conservation_check(config, model=model)),
        "spatial": asdict(run_spatial_convergence(validation, model=model)),
        "temporal": asdict(run_temporal_convergence(validation, model=model)),
        "stability": asdict(run_stability_sweep(validation, model=model)),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("heat", "poisson", "wave", "p53_mcrd", "all"),
        default="all",
    )
    parser.add_argument("--grid-sizes", default="16,32", help="Comma-separated smoke grid sizes")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument(
        "--backend", choices=("numpy", "jax-cpu", "jax-metal", "jax-cuda"), default="numpy"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=repo_root / "reports/s1_pde_smoke.json")
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be non-negative")
    sizes = [int(size) for size in args.grid_sizes.split(",")]
    if not sizes or any(size < 2 for size in sizes):
        parser.error("--grid-sizes must contain integers >= 2")

    backend, dtype = Backend(args.backend), DType(args.dtype)
    selected = ("heat", "poisson", "wave", "p53_mcrd") if args.case == "all" else (args.case,)
    cases: dict[str, object] = {}
    for name in selected:
        if name == "p53_mcrd":
            cases[name] = _run_p53(sizes, args.steps, args.dt, backend, dtype)
            continue
        case_results: dict[str, object] = {}
        for size in sizes:
            grid = _grid(size, backend, dtype)
            if name == "heat":
                case_results[str(size)] = _run_heat(grid, args.steps, args.dt)
            elif name == "poisson":
                case_results[str(size)] = _run_poisson(grid)
            elif name == "wave":
                case_results[str(size)] = _run_wave(grid, args.steps, args.dt)
        cases[name] = case_results

    runtime = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    if backend in JAX_BACKENDS:
        import jax

        device = get_jax_device(backend)
        runtime.update({
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "device": str(device),
            "device_kind": getattr(device, "device_kind", None),
            "device_platform": getattr(device, "platform", None),
        })

    report = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "explicit_smoke",
        "runtime": runtime,
        "case_families": {
            "heat": "parabolic",
            "poisson": "elliptic",
            "wave": "hyperbolic",
            "p53_mcrd": "scientific reaction-diffusion workload",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
