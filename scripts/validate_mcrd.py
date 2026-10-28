"""Run the bounded full-series MCRD convergence and stability studies."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from mcrd_accel.config import DType, ResourceProfile, SolverConfig, ValidationConfig
from mcrd_accel.mcrd import ReactionModel
from mcrd_accel.mcrd.validation import (
    run_spatial_convergence,
    run_stability_sweep,
    run_temporal_convergence,
)


def build_report() -> dict:
    model = ReactionModel.p53_mcrd(
        SolverConfig.fig2b(Nx=64, Ny=64, dx=1 / 64, dy=1 / 64, dt=0.01, n_steps=8)
    )
    grids = [64, 128, 256, 512, 1024, 2048, 4096]
    spatial_dt = 0.5 * (1 / 4096) ** 2 / (4 * 0.17)
    spatial_solver = SolverConfig.fig2b(
        Nx=4096, Ny=4096, dx=1 / 4096, dy=1 / 4096,
        dt=spatial_dt, n_steps=8, dtype=DType.FLOAT32,
    )
    spatial_config = ValidationConfig(
        solver=spatial_solver,
        grid_sizes=grids,
        time_steps=[spatial_dt, spatial_dt / 2, spatial_dt / 4, spatial_dt / 8],
        resource=ResourceProfile(max_resident_bytes=1024**3, max_steps=64),
    )
    spatial = run_spatial_convergence(spatial_config, model=model)

    temporal_solver = SolverConfig.fig2b(
        Nx=64, Ny=64, dx=1 / 64, dy=1 / 64, dt=0.0001, n_steps=8,
        dtype=DType.FLOAT32,
    )
    temporal_config = ValidationConfig(
        solver=temporal_solver,
        grid_sizes=[64],
        time_steps=[0.0001, 0.00005, 0.000025, 0.0000125],
        resource=ResourceProfile(max_resident_bytes=512 * 1024**2, max_steps=128),
    )
    temporal_model = ReactionModel.p53_mcrd(temporal_solver)
    temporal = run_temporal_convergence(temporal_config, model=temporal_model)

    stability_solver = SolverConfig.fig2b(
        Nx=64, Ny=64, dx=1 / 64, dy=1 / 64, dt=0.00005, n_steps=512,
        dtype=DType.FLOAT32,
    )
    stability_config = ValidationConfig(
        solver=stability_solver,
        grid_sizes=[64],
        time_steps=[0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016],
        resource=ResourceProfile(max_resident_bytes=512 * 1024**2, max_steps=1024),
    )
    stability_model = ReactionModel.p53_mcrd(stability_solver)
    stability = run_stability_sweep(stability_config, model=stability_model)

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "profile": "bounded_local_m4",
        "spatial": asdict(spatial),
        "temporal": asdict(temporal),
        "stability": asdict(stability),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "reports/mcrd_validation.json")
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
