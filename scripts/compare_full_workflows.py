"""Compare complete matched CPU and H100 PDE + p53 workflow result directories."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _workflow_wall_seconds(root: Path) -> float | None:
    manifest = root / "workflow-manifest.json"
    if manifest.exists():
        data = _load(manifest)
        if data.get("status") != "success":
            raise ValueError(f"workflow did not succeed: {manifest}")
        return float(data["wall_seconds"])
    log = root / "job.log"
    if not log.exists():
        return None
    contents = log.read_text(errors="replace")
    start = re.search(r"started_utc=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z", contents)
    end = re.search(r"finished_utc=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z", contents)
    if start and end:
        from datetime import datetime
        start_dt = datetime.strptime(start.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return (end_dt - start_dt).total_seconds()
    return None


def _pde_config(report: dict) -> dict:
    def without_backend(value):
        if isinstance(value, dict):
            return {k: without_backend(v) for k, v in value.items() if k != "backend"}
        if isinstance(value, list):
            return [without_backend(item) for item in value]
        return value

    cases = report["cases"]
    configs: dict[str, object] = {}
    for family, results in cases.items():
        if family == "p53_mcrd":
            # The bounded validation suite emits nested configs rather than a grid table.
            configs[family] = without_backend(results.get("conservation", {}).get("config", {}))
        else:
            configs[family] = {
                size: {k: v for k, v in item["grid"].items() if k != "backend"}
                for size, item in results.items()
            }
    return configs


def _pde_rows(cpu: dict, gpu: dict) -> list[dict]:
    rows = []
    for family in ("heat", "poisson", "wave"):
        cset, gset = cpu["cases"][family], gpu["cases"][family]
        if cset.keys() != gset.keys():
            raise ValueError(f"PDE smoke grids differ for {family}")
        for size in sorted(cset, key=int):
            c, g = cset[size], gset[size]
            for key in ("steps", "dt"):
                if key in c and c.get(key) != g.get(key):
                    raise ValueError(f"PDE {family} {size} {key} differs")
            ct, gt = c["runtime_seconds"], g["runtime_seconds"]
            metrics = {}
            for key in ("mode_max_error", "mass_drift", "relative_residual", "energy_relative_drift", "converged"):
                if key in c or key in g:
                    metrics[key] = {"cpu": c.get(key), "h100": g.get(key)}
            rows.append({
                "family": family, "grid_size": int(size),
                "cpu_runtime_seconds": ct, "h100_runtime_seconds": gt,
                "cpu_over_h100_speedup": ct / gt if gt else None,
                "metrics": metrics,
            })
    return rows


def _save_figures(out: Path, cpu_cases: dict, gpu_cases: dict, parity: dict, pde_rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generated = []
    sizes = sorted(cpu_cases)
    cpu_t = [cpu_cases[s]["runtime_seconds_including_compile_and_diagnostics"] for s in sizes]
    gpu_t = [gpu_cases[s]["runtime_seconds_including_compile_and_diagnostics"] for s in sizes]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.loglog(sizes, cpu_t, "o-", label="CPU")
    ax.loglog(sizes, gpu_t, "o-", label="H100 MIG")
    ax.set(xlabel="Grid size (N × N)", ylabel="Full t=40 runtime (s)", title="p53 MCRD full-run runtime")
    ax.grid(True, which="both", alpha=.25); ax.legend()
    path = out / "p53-full-runtime-cpu-vs-h100.png"
    fig.savefig(path, dpi=180); plt.close(fig); generated.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, species in zip(axes, ("u", "v"), strict=True):
        axis.loglog(sizes[1:], [cpu_cases[s]["difference_from_previous_grid"][f"{species}_rms_difference"] for s in sizes[1:]], "o-", label="CPU")
        axis.loglog(sizes[1:], [gpu_cases[s]["difference_from_previous_grid"][f"{species}_rms_difference"] for s in sizes[1:]], "o-", label="H100")
        axis.set(title=f"{species}: successive-grid RMS difference", xlabel="Fine grid size", ylabel="RMS difference")
        axis.grid(True, which="both", alpha=.25); axis.legend()
    path = out / "p53-spatial-grid-differences.png"
    fig.savefig(path, dpi=180); plt.close(fig); generated.append(path.name)

    path_npz = [out.parent / "_cpu_tmp.npz", out.parent / "_gpu_tmp.npz"]
    # Arrays are loaded by caller and attached temporarily as NumPy sidecars.
    cpu_state, gpu_state = np.load(path_npz[0]), np.load(path_npz[1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row, species in enumerate(("u", "v")):
        c = cpu_state[species]; g = gpu_state[species]
        vmin, vmax = min(float(c.min()), float(g.min())), max(float(c.max()), float(g.max()))
        axes[row, 0].imshow(c, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(g, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        diff = np.abs(c.astype(np.float64) - g.astype(np.float64))
        image = axes[row, 2].imshow(diff, origin="lower", cmap="magma")
        axes[row, 0].set_title(f"CPU {species}, t=40")
        axes[row, 1].set_title(f"H100 {species}, t=40")
        axes[row, 2].set_title(f"|CPU − H100| {species}\nmax={parity[species]['max_absolute_error']:.2e}")
        fig.colorbar(image, ax=axes[row, 2], label="absolute error")
    path = out / "p53-final-field-parity-heatmaps.png"
    fig.savefig(path, dpi=180); plt.close(fig); generated.append(path.name)
    for p in path_npz:
        p.unlink(missing_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = [f"{r['family']} {r['grid_size']}²" for r in pde_rows]
    x = np.arange(len(labels)); width = .38
    ax.bar(x - width / 2, [r["cpu_runtime_seconds"] for r in pde_rows], width, label="CPU")
    ax.bar(x + width / 2, [r["h100_runtime_seconds"] for r in pde_rows], width, label="H100")
    ax.set_yscale("log"); ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set(ylabel="Smoke runtime (s, log scale)", title="PDE smoke timings")
    ax.grid(True, axis="y", which="both", alpha=.25); ax.legend()
    path = out / "pde-smoke-runtime-cpu-vs-h100.png"
    fig.savefig(path, dpi=180); plt.close(fig); generated.append(path.name)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True, help="Full workflow result directory")
    parser.add_argument("--h100-dir", type=Path, required=True, help="Full workflow or legacy H100 result directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cpu_pde, gpu_pde = _load(args.cpu_dir / "pde-smoke-to-128.json"), _load(args.h100_dir / "pde-smoke-to-128.json")
    if cpu_pde["runtime"].get("device_platform") != "cpu":
        raise ValueError("CPU result does not report a CPU device")
    if gpu_pde["runtime"].get("device_platform") != "gpu" or "H100" not in gpu_pde["runtime"].get("device_kind", "").upper():
        raise ValueError("H100 result does not report an H100 device")
    if _pde_config(cpu_pde) != _pde_config(gpu_pde):
        raise ValueError("CPU and H100 PDE smoke configurations differ")
    cpu_report = _load(args.cpu_dir / "p53-grid-study/p53-grid-study.json")
    gpu_report = _load(args.h100_dir / "p53-grid-study/p53-grid-study.json")
    cm, gm = cpu_report["manifest"], gpu_report["manifest"]
    fields = ("jax", "jaxlib", "dtype", "integrator", "boundary", "physical_domain", "grid_sizes", "dt", "steps_per_grid", "target_final_time", "actual_final_time", "seed", "initial_condition")
    differences = {f: {"cpu": cm.get(f), "h100": gm.get(f)} for f in fields if cm.get(f) != gm.get(f)}
    if differences:
        raise ValueError(f"p53 full workflow settings differ: {differences}")
    cc = {int(r["grid_size"]): r for r in cpu_report["cases"]}
    gc = {int(r["grid_size"]): r for r in gpu_report["cases"]}
    if cc.keys() != gc.keys():
        raise ValueError("CPU and H100 p53 grid sizes differ")
    rows = []
    for n in sorted(cc):
        c, g = cc[n], gc[n]
        for field in ("steps", "dt", "final_time"):
            if c[field] != g[field]:
                raise ValueError(f"p53 {n}² {field} differs")
        cpu_time = c["runtime_seconds_including_compile_and_diagnostics"]
        gpu_time = g["runtime_seconds_including_compile_and_diagnostics"]
        rows.append({
            "grid_size": n, "steps": c["steps"], "simulated_time": c["final_time"],
            "cpu_full_run_seconds": cpu_time, "h100_full_run_seconds": gpu_time,
            "cpu_over_h100_speedup": cpu_time / gpu_time if gpu_time else None,
            "cpu_mass_drift": c["relative_mass_drift"], "h100_mass_drift": g["relative_mass_drift"],
            "cpu_all_finite": c["all_finite"], "h100_all_finite": g["all_finite"],
            "cpu_u_observed_order": c.get("difference_from_previous_grid", {}).get("u_observed_order"),
            "h100_u_observed_order": g.get("difference_from_previous_grid", {}).get("u_observed_order"),
            "cpu_v_observed_order": c.get("difference_from_previous_grid", {}).get("v_observed_order"),
            "h100_v_observed_order": g.get("difference_from_previous_grid", {}).get("v_observed_order"),
        })
    cpu_npz_path = args.cpu_dir / "p53-grid-study/p53-final-state-4096-t40.npz"
    gpu_npz_path = args.h100_dir / "p53-grid-study/p53-final-state-4096-t40.npz"
    cpu_npz, gpu_npz = np.load(cpu_npz_path), np.load(gpu_npz_path)
    parity = {}
    for key in ("u", "v"):
        cf, gf = cpu_npz[key].astype(np.float64), gpu_npz[key].astype(np.float64)
        if cf.shape != gf.shape:
            raise ValueError(f"4096² {key} output shapes differ")
        delta = cf - gf; rms = float(np.sqrt(np.mean(delta * delta)))
        parity[key] = {"max_absolute_error": float(np.max(np.abs(delta))), "rms_error": rms,
                       "relative_rms_error": rms / max(float(np.sqrt(np.mean(gf * gf))), np.finfo(float).tiny)}
    pde_rows = _pde_rows(cpu_pde, gpu_pde)
    cpu_wall, gpu_wall = _workflow_wall_seconds(args.cpu_dir), _workflow_wall_seconds(args.h100_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Use temporary sidecars in output parent so the figure helper keeps a small interface.
    np.savez(args.output_dir.parent / "_cpu_tmp.npz", u=cpu_npz["u"], v=cpu_npz["v"])
    np.savez(args.output_dir.parent / "_gpu_tmp.npz", u=gpu_npz["u"], v=gpu_npz["v"])
    figures = _save_figures(args.output_dir, cc, gc, parity, pde_rows)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_device": cpu_pde["runtime"].get("device_kind"), "cpu_cores": cm.get("cpu_affinity_count"),
        "h100_device": gpu_pde["runtime"].get("device_kind"),
        "matched_p53_configuration": {field: cm.get(field) for field in fields},
        "whole_workflow": {"cpu_wall_seconds": cpu_wall, "h100_wall_seconds": gpu_wall,
                           "cpu_over_h100_speedup": cpu_wall / gpu_wall if cpu_wall and gpu_wall else None},
        "p53_grid_results": rows, "pde_smoke_results": pde_rows,
        "p53_4096_t40_field_parity": parity, "figures": figures,
    }
    (args.output_dir / "full-workflow-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Full CPU versus H100 workflow comparison", "",
             f"- CPU: {report['cpu_device']} ({report['cpu_cores']} pinned cores)",
             f"- H100: {report['h100_device']}",
             f"- Matched p53 run: float64, t={cm['actual_final_time']:.8g}, {cm['steps_per_grid']:,} steps/grid, grids {', '.join(map(str, cm['grid_sizes']))}",
             "- Both workflows include heat, Poisson, wave, p53 validation smoke checks and the fixed-time p53 spatial study.", "",
             "## Whole workflow", "", "| CPU wall time | H100 wall time | CPU/H100 speedup |", "|---:|---:|---:|"]
    if cpu_wall is not None and gpu_wall is not None:
        lines.append(f"| {cpu_wall/3600:.3f} h | {gpu_wall/60:.2f} min | {cpu_wall/gpu_wall:.2f}x |")
    else:
        lines.append("| unavailable | unavailable | unavailable |")
    lines += ["", "## p53 t=40 per-grid timings", "", "| Grid | CPU seconds | H100 seconds | speedup | CPU mass drift | H100 mass drift |", "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['grid_size']}² | {r['cpu_full_run_seconds']:.3f} | {r['h100_full_run_seconds']:.3f} | {r['cpu_over_h100_speedup']:.2f}x | {r['cpu_mass_drift']:.3g} | {r['h100_mass_drift']:.3g} |")
    lines += ["", "## PDE smoke timings", "", "| Case | Grid | CPU (s) | H100 (s) | CPU/H100 |", "|---|---:|---:|---:|---:|"]
    for r in pde_rows:
        lines.append(f"| {r['family']} | {r['grid_size']}² | {r['cpu_runtime_seconds']:.4g} | {r['h100_runtime_seconds']:.4g} | {r['cpu_over_h100_speedup']:.2f}x |")
    lines += ["", "## 4096² final-field parity", ""]
    for key in ("u", "v"):
        lines.append(f"- {key}: max absolute error {parity[key]['max_absolute_error']:.6g}; relative RMS {parity[key]['relative_rms_error']:.6g}.")
    lines += ["", "Figures include per-grid runtime, spatial grid differences, CPU/H100 final-field and absolute-error heatmaps, and PDE smoke timings.", "All timings are specific to these devices, software builds, and ACP allocations.", ""]
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(f"WROTE {args.output_dir / 'full-workflow-comparison.json'}")


if __name__ == "__main__":
    main()
