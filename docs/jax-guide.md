# JAX CUDA guide

This guide shows how to select the JAX CUDA backend explicitly, run the p53 solver, recreate the OpenCL-style 1024² figure, and reproduce the recorded benchmarks.

## Install and verify the CUDA backend

On a compatible Linux CUDA host, install the project's CUDA extra:

```bash
uv sync --extra jax-cuda
```

For the prepared ACP image, use the existing `.venv312` environment rather than installing packages inside each job. Before running, unset any platform override so JAX can discover the installed CUDA plugin. The package then requests the GPU explicitly and raises an error instead of silently falling back to CPU.

```bash
unset JAX_PLATFORMS
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

.venv312/bin/python - <<'PY'
from mcrd_accel.backends import get_jax_device
from mcrd_accel.config import Backend

device = get_jax_device(Backend.JAX_CUDA)
print(device, getattr(device, "device_kind", "unknown"))
PY
```

The printed device is the runtime check. A configured backend label alone does not prove that CUDA ran.

## Run a p53 MCRD case from Python

```python
from mcrd_accel.config import Backend, DType, SolverConfig
from mcrd_accel.mcrd import run

config = SolverConfig.fig2b(
    Nx=256,
    Ny=256,
    dx=0.2,
    dy=0.2,
    dt=0.02,
    n_steps=1_000,
    dtype=DType.FLOAT32,
    backend=Backend.JAX_CUDA,
    seed=0,
)
result = run(config, checkpoint_stride=250)
print(result.metadata["backend"], result.metadata["final_time"])
print(result.final_state.u.shape, result.final_state.v.shape)
```

Use `DType.FLOAT64` only after setting `JAX_ENABLE_X64=true` before importing JAX. The explicit Euler method requires a stable timestep for the chosen spacing and diffusion coefficients; reduce `dt` with grid refinement.

## Recreate the OpenCL-style 1024² figure

The H100 reproduction uses float32 and a random initial field generated at the old reference scale before mapping it to 1024². It scales `dt` with `dx²`, which gives `dt=0.0003125`, 128,000 steps, and t=40. The seed and all outputs are recorded in the run manifest.

On the prepared ACP H100 image, submit this shell job:

```bash
set -euo pipefail
cd /data/250010002/DNA-p53-MCRD-accelerated
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS

.venv312/bin/python scripts/reproduce_opencl_style.py \
  --backend jax-cuda \
  --size 1024 \
  --domain-length 25.6 \
  --final-time 40 \
  --dtype float32 \
  --seed 0 \
  --output-dir /data/250010002/DNA-p53-MCRD-accelerated/reports/opencl-style-1024-reference-init
```

Outputs are the PNG heatmap, a compressed NPZ with the final fields, and `run-manifest.json`. The reproduction aims for similar morphology; the legacy OpenCL kernel's in-place updates have race conditions, so pixel-for-pixel identity is not promised. See the [OpenCL migration guide](opencl-migration-guide.md).

## Reproduce the matched JAX CPU/H100 benchmark

The ACP wrapper runs the same p53 configuration on JAX CPU or JAX CUDA. Submit separate jobs so each run records its backend and result directory:

```bash
BACKEND=jax-cpu CPU_CORES=8 bash /data/250010002/DNA-p53-MCRD-accelerated/scripts/run_p53_backend_benchmark_acp.sh
BACKEND=jax-cuda bash /data/250010002/DNA-p53-MCRD-accelerated/scripts/run_p53_backend_benchmark_acp.sh
```

The default profile tests 64² through 4096², 10,240 steps per grid, float64, one warmup, and three timed repeats. It records the device, JAX/JAXlib versions, CPU affinity, per-run times, medians, and numerical checks. The current results are summarized in the [runtime comparison](runtime-comparison.md).

The separate [full workflow ACP script](../scripts/run_full_workflow_acp.sh) runs the PDE smoke suite and p53 fixed-time spatial study. Its full CPU sweep is substantially longer than the short matched benchmark.
