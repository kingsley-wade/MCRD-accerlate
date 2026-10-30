# Full CPU/H100 experiment workflow

This workflow compares the same numerical workload on an 8-core pinned CPU allocation and an H100 MIG allocation. It runs the bounded heat, periodic Poisson, wave, and p53 validation suite, then runs the full p53 MCRD fixed-domain grid study from 64² through 4096² to t=40 in float64. It records JSON/CSV, progress checkpoints, reproducibility metadata, SHA256 checksums, and heatmaps.

The full CPU sweep is expected to take about 35–40 hours on the previously measured 8-core ACP CPU allocation. The 4096² case alone is estimated around 23 hours. These are planning estimates; ACP scheduling and CPU allocation can change actual duration. Submit CPU as a long job with enough wall-time. The prepared H100 full sweep took about 15 minutes.

## Submit on ACP

Use the same project path and the matching environment under `/data/250010002/DNA-p53-MCRD-accelerated`. The shell entry point is non-interactive and does not install packages, use Git, or write outside that project directory.

CPU job command:

```bash
BACKEND=jax-cpu CPU_CORES=8 bash /data/250010002/DNA-p53-MCRD-accelerated/scripts/run_full_workflow_acp.sh
```

H100 job command (choose the CUDA 12.4 H100 MIG image matching `.venv312`):

```bash
BACKEND=jax-cuda bash /data/250010002/DNA-p53-MCRD-accelerated/scripts/run_full_workflow_acp.sh
```

Each submission creates a unique directory:

```text
/data/250010002/DNA-p53-MCRD-accelerated/reports/acp-full-workflow-<backend>-<UTC timestamp>/
```

The wrapper checks that the requested JAX backend is truly visible (and checks the GPU device name for H100), then runs the PDE smoke suite and p53 grid study. It exits nonzero on failure and writes `workflow-manifest.json` even on failure. For CPU, it pins execution to the requested number of currently allocated CPUs. Keep both jobs on the same Python/JAX/JAXlib build and use the same `CPU_CORES` allocation for reproducible comparisons.

## Data and figures produced per run

- `workflow-manifest.json`, `job.log`: total wall time, status, timestamps, backend and requested CPU core count; the actual JAX device identity is recorded in the PDE smoke JSON.
- `pde-smoke-to-128.json`: heat, Poisson, wave, and p53 verification metrics and timings at 16²–128².
- `p53-grid-study/run-manifest.json`: physical domain, dtype, backend, dt, steps, seed and software/device provenance.
- `p53-grid-study/p53-grid-study.json` and `.csv`: per-grid runtime, finite-state check, mass drift, field ranges, adaptive-change diagnostics and successive-grid spatial differences/observed order.
- `p53-grid-study/p53-checkpoints-<N>.json`: time-window diagnostics per grid.
- `p53-grid-study/p53-final-state-4096-t40.npz`: full-resolution final `u` and `v` arrays for parity checks. This generated array is over 200 MB and is intentionally kept outside the GitHub repository; the ACP run still writes it to its result directory.
- `p53-grid-study/p53-evolution-4096-t0-to-t40.png`: t=0, 10, 20, 30, 40 evolution heatmaps.
- `p53-grid-study/p53-t40-4096-heatmap.png`: final 4096² `u` and `v` heatmaps.
- `SHA256SUMS`: checksums for result artifacts (excluding the mutable job log and checksum file).

The adaptive-change threshold is diagnostic only. Both backends always run to fixed t=40; crossing a threshold is not presented as a steady-state proof.

## Compare after both jobs finish

Copy the completed result directories back to the local machine (or point the script at local copies). Existing H100 results can also be used at `reports/acp-h100-20261030T070325Z`; the comparison tool accepts its legacy result layout.

```bash
python3 scripts/compare_full_workflows.py \
  --cpu-dir reports/acp-full-workflow-jax-cpu-<UTC timestamp> \
  --h100-dir reports/acp-h100-20261030T070325Z \
  --output-dir reports/full-workflow-cpu-vs-h100
```

The comparison fails if the backend reports, shared PDE smoke inputs, JAX/JAXlib, p53 domain, grid sizes, dtype, timestep, step count, target time, seed, or initial-condition definition do not match. It emits `full-workflow-comparison.json` and `README.md`, plus:

- `p53-full-runtime-cpu-vs-h100.png`: full t=40 runtime by grid.
- `p53-spatial-grid-differences.png`: successive-grid RMS differences by backend.
- `p53-final-field-parity-heatmaps.png`: CPU field, H100 field, and absolute error for both species.
- `pde-smoke-runtime-cpu-vs-h100.png`: PDE smoke runtime comparison.

The final report separates solver/runtime measurements, numerical checks, and output parity. A speedup is CPU wall/runtime divided by H100 wall/runtime, so values above 1 indicate H100 was faster. Results apply to the recorded allocations and software versions; do not generalize them to all CPUs or H100 configurations.

## Existing results already available

The local H100 t=40 float64 spatial study is in `reports/acp-h100-20261030T070325Z/`; it includes both p53 heatmaps and all seven grid records. The matched short-run CPU/H100 benchmark is in `reports/p53-cpu-vs-h100-20261030T134645Z/`, but it covers only 10,240 steps per grid and is not a substitute for the full workflow comparison. The older float32 p53 run is numerically stalled and must not be used as convergence evidence.
