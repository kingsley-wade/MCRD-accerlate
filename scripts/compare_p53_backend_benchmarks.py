"""Compare matched CPU and H100 p53 benchmark output directories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cpu_report = json.loads((args.cpu_dir / "benchmark.json").read_text())
    gpu_report = json.loads((args.gpu_dir / "benchmark.json").read_text())
    cpu_manifest, gpu_manifest = cpu_report["manifest"], gpu_report["manifest"]
    if cpu_manifest.get("backend") != "jax-cpu":
        raise ValueError("--cpu-dir must contain a jax-cpu benchmark")
    if gpu_manifest.get("backend") != "jax-cuda":
        raise ValueError("--gpu-dir must contain a jax-cuda benchmark")

    matching_fields = (
        "jax",
        "jaxlib",
        "jax_enable_x64",
        "physical_domain",
        "grid_sizes",
        "dtype",
        "dt",
        "steps_per_grid",
        "simulated_time_per_grid",
        "seed",
    )
    mismatches = {
        field: {"cpu": cpu_manifest.get(field), "gpu": gpu_manifest.get(field)}
        for field in matching_fields
        if cpu_manifest.get(field) != gpu_manifest.get(field)
    }
    if mismatches:
        raise ValueError(f"benchmark configurations do not match: {mismatches}")

    cpu_cases = {item["grid_size"]: item for item in cpu_report["cases"]}
    gpu_cases = {item["grid_size"]: item for item in gpu_report["cases"]}
    if cpu_cases.keys() != gpu_cases.keys():
        raise ValueError("CPU and GPU reports do not have the same grid sizes")

    rows = []
    for size in sorted(cpu_cases):
        cpu = cpu_cases[size]
        gpu = gpu_cases[size]
        if cpu["steps"] != gpu["steps"] or cpu["dt"] != gpu["dt"]:
            raise ValueError(f"step count or dt differs at grid {size}")
        rows.append(
            {
                "grid_size": size,
                "steps": cpu["steps"],
                "cpu_cores": cpu_manifest.get("cpu_affinity_count"),
                "cpu_steady_median_seconds": cpu["steady_median_seconds"],
                "h100_steady_median_seconds": gpu["steady_median_seconds"],
                "steady_speedup_cpu_over_h100": cpu["steady_median_seconds"]
                / gpu["steady_median_seconds"],
                "cpu_first_compile_run_seconds": cpu["cold_first_run_seconds_including_compile"],
                "h100_first_compile_run_seconds": gpu["cold_first_run_seconds_including_compile"],
                "first_run_speedup_cpu_over_h100": cpu["cold_first_run_seconds_including_compile"]
                / gpu["cold_first_run_seconds_including_compile"],
                "cpu_relative_mass_drift": cpu["relative_mass_drift"],
                "h100_relative_mass_drift": gpu["relative_mass_drift"],
                "cpu_all_finite": cpu["all_finite"],
                "h100_all_finite": gpu["all_finite"],
            }
        )

    cpu_npz = np.load(args.cpu_dir / "p53-4096-final-state.npz")
    gpu_npz = np.load(args.gpu_dir / "p53-4096-final-state.npz")
    parity = {}
    for field in ("u", "v"):
        cpu_field = cpu_npz[field].astype(np.float64)
        gpu_field = gpu_npz[field].astype(np.float64)
        if cpu_field.shape != gpu_field.shape:
            raise ValueError(f"4096² {field} array shapes differ")
        difference = cpu_field - gpu_field
        rms = float(np.sqrt(np.mean(difference**2)))
        parity[field] = {
            "max_absolute_error": float(np.max(np.abs(difference))),
            "rms_error": rms,
            "relative_rms_error": rms
            / max(float(np.sqrt(np.mean(gpu_field**2))), np.finfo(np.float64).tiny),
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matched_configuration": {field: cpu_manifest[field] for field in matching_fields},
        "cpu_device": cpu_manifest.get("device_kind"),
        "cpu_affinity": cpu_manifest.get("cpu_affinity"),
        "h100_device": gpu_manifest.get("device_kind"),
        "speedups": rows,
        "4096_field_parity": parity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cpu-vs-h100.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# p53 CPU versus H100 benchmark",
        "",
        f"- CPU: {report['cpu_device']} with {report['cpu_affinity']} pinned cores",
        f"- H100: {report['h100_device']}",
        f"- Matched JAX/JAXlib: {cpu_manifest['jax']}/{cpu_manifest['jaxlib']}; float64",
        f"- Steps per grid: {cpu_manifest['steps_per_grid']}; dt: {cpu_manifest['dt']:.12g}",
        "- Speedup is CPU median time divided by H100 median time; values above 1 mean H100 was faster.",
        "",
        "| Grid | CPU median (s) | H100 median (s) | H100 speedup |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['grid_size']}² | {row['cpu_steady_median_seconds']:.4f} | "
            f"{row['h100_steady_median_seconds']:.4f} | "
            f"{row['steady_speedup_cpu_over_h100']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## 4096² output parity",
            "",
            f"- u max absolute error: {parity['u']['max_absolute_error']:.6g}; relative RMS: {parity['u']['relative_rms_error']:.6g}",
            f"- v max absolute error: {parity['v']['max_absolute_error']:.6g}; relative RMS: {parity['v']['relative_rms_error']:.6g}",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(f"WROTE {args.output_dir / 'cpu-vs-h100.json'}")


if __name__ == "__main__":
    main()
