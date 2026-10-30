# H100 Validation

This page records the bounded CUDA evidence for the MCRD migration. It does not establish production-scale behavior, compatibility with other GPUs, or parity with MATLAB/OpenCL. The benchmark used one shared NVIDIA H100 MIG 3g.40gb instance and did not interrupt other processes.

## Recorded run: 2026-10-30

The reports and exact commands are in [`reports/h100-validation-20261030/`](../reports/h100-validation-20261030/). They were generated from source revision `d0d11fe02df97e360fd8b8eae495683a1e9a32d6`.

| Item | Recorded value |
| --- | --- |
| Device | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| Driver | 580.95.05 |
| Python | 3.12.16 |
| JAX / jaxlib | 0.12.0.1 / 0.12.0.1 |
| CUDA runtime package | 12.9.79 |
| CUDA memory before / after benchmark | 39.21 / 39.11 GiB free of 39.50 GiB |

The project PDE smoke ran heat, periodic Poisson, wave, and p53 MCRD on 16² and 32² grids, float32, 20 steps. Both Poisson cases converged; both p53 stability cases were stable; all reported fields were finite.

The same-input p53 parity run used a 64² grid, float32, 100 steps, and seed 31415. NumPy and JAX CUDA were compared with `rtol=3e-5, atol=2e-6`. The maximum absolute errors were `9.54e-6` for `u` and `9.54e-7` for `v`; both fields passed the tolerance.

The repeated timing comparison used identical seeded initial fields and p53 parameters for both backends, float32, 200 steps, two warmups, and five timed repeats per backend:

| Grid | NumPy mean | JAX CUDA mean | NumPy/CUDA ratio |
| --- | ---: | ---: | ---: |
| 128 × 128 | 45.6 ms | 4.93 ms | 9.26× |
| 256 × 256 | 146.0 ms | 5.23 ms | 27.90× |

The CUDA timings cover the public solver call, including host-to-device input transfer and final-state transfer back to host memory. First-call wall time, which includes compilation and setup, is reported separately in the JSON. These are measurements for this MIG slice, this code revision, and these two workloads; they are not general GPU speed guarantees.

`nvidia-smi` denied memory-detail queries inside the container. The report therefore records free and total MIG memory through CUDA `cudaMemGetInfo` before and after runs. This is not a peak-memory measurement. The JAX allocator statistics are recorded separately and do not represent total process or device memory.

The remote environment was installed under `/data/250010002/DNA-p53-MCRD-accelerated/` using the Tsinghua PyPI mirror. `uv sync --locked` reported that the existing lockfile needs an update; the lockfile was not changed for this run, so the exact installed versions are recorded with the report.

## Reproduce on the recorded host

The prepared environment is `.venv312` in the project directory. Run one workload at a time and check MIG memory before starting. The exact commands used are in [`command.txt`](../reports/h100-validation-20261030/command.txt); the parity and benchmark scripts are stored beside the reports.

```bash
cd /data/250010002/DNA-p53-MCRD-accelerated
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 \
  .venv312/bin/python scripts/validate_local.py \
  --case all --grid-sizes 16,32 --steps 20 --dt 1e-5 \
  --backend jax-cuda --dtype float32 \
  --output reports/h100-validation-20261030/cuda-smoke.json
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 \
  .venv312/bin/python reports/h100-validation-20261030/parity-run.py
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 \
  .venv312/bin/python reports/h100-validation-20261030/benchmark-run.py
```

Do not run grid 4096 or concurrent throughput sweeps on this shared MIG host. Keep H100 and RTX 5090 evidence separate. The old `reports/s1_pde_smoke_h100.json` remains historical because it has no attributable host/device provenance.
