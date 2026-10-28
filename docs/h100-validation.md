# H100 Smoke Validation

This procedure is for a small device-execution smoke on the shared H100 host. It is not a performance benchmark or a full correctness study. Never stop another user's process. The last recorded probe found about 29 GiB free per H100, but device availability must be checked again immediately before each run.

## Current state

The project is uploaded to `/mnt/afs/gmt/DNA-p53-MCRD-accelerated`. The locked CUDA dependency sync resolved successfully but its package downloads made negligible progress for more than ten minutes. The install was stopped before any GPU work. The remote `.venv` is incomplete, and the old `reports/s1_pde_smoke_h100.json` is not attributable evidence because it lacks host and device provenance.

## Prepare the environment

```bash
cd /mnt/afs/gmt/DNA-p53-MCRD-accelerated
df -h /mnt/afs
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version --format=csv
UV_CACHE_DIR=/mnt/afs/gmt/.cache/uv-dna-p53 uv sync --locked --extra jax-cuda --python 3.12
```

If package downloads stall, stop and resolve the cluster's package-index/network path before retrying. Do not repeatedly create environments or download duplicate caches. The project requires Python 3.11 or newer; do not use the system Python 3.10.

After installation, verify JAX sees a GPU before running the project workload:

```bash
cd /mnt/afs/gmt/DNA-p53-MCRD-accelerated
uv run python -c 'import jax; print(jax.__version__); print(jax.devices("gpu"))'
```

## Run the bounded smoke

Recheck free VRAM. Select one device with at least 16 GiB free, then capture the device state before and after. This short run uses only 16 and 32 grids, float32, 20 steps, and one selected GPU.

```bash
mkdir -p reports/h100-smoke
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version --format=csv > reports/h100-smoke/gpu-before.csv
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/validate_local.py \
  --case all --grid-sizes 16,32 --steps 20 --dt 1e-5 \
  --backend jax-cuda --dtype float32 \
  --output reports/h100-smoke/report.json
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version --format=csv > reports/h100-smoke/gpu-after.csv
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version --format=csv
```

Keep the JSON report together with both CSV snapshots, the exact command, and the source revision or source archive SHA-256. Confirm `runtime.device` and `runtime.device_kind` identify the H100 before describing this as GPU execution. If the device is not available, the workload fails, or memory drops below the preflight threshold, stop without substituting a CPU run and labeling it CUDA.

## Interpretation and follow-up

- This smoke establishes only that the listed small workloads execute through the selected JAX CUDA backend on the recorded device.
- It does not establish speedup, production-scale stability, long-run conservation, or parity with NumPy, MATLAB, or the original OpenCL implementation.
- Compare NumPy and CUDA on identical inputs and explicit tolerances before making a numerical parity claim. Keep that comparison as a separate result.
- Do not run grid 4096, a throughput sweep, or concurrent benchmark jobs while this device is shared.
- Copy the new report and metadata back to the local `reports/` directory, review it, and update `docs/backend-matrix.md`. Preserve the old provenance-incomplete JSON as historical evidence; do not silently relabel it.
