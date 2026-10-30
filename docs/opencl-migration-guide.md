# Migrating the p53 MCRD OpenCL run

This guide reproduces the random, speckled p53 MCRD morphology used by the earlier OpenCL run on H100/JAX CUDA, then maps its settings to the package solver. The reproduction is deterministic for a given seed and configuration. It aims for similar morphology, not pixel-for-pixel identity with the original OpenCL kernel.

## Reproduce a 1024² image

Submit this non-interactive shell command to ACP with the CUDA 12.4 H100 MIG image used by `.venv312`:

```bash
set -euo pipefail
cd /data/250010002/DNA-p53-MCRD-accelerated
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS
/data/250010002/DNA-p53-MCRD-accelerated/.venv312/bin/python \
  scripts/reproduce_opencl_style.py \
  --backend jax-cuda \
  --size 1024 \
  --domain-length 25.6 \
  --final-time 40 \
  --dtype float32 \
  --seed 0 \
  --output-dir /data/250010002/DNA-p53-MCRD-accelerated/reports/opencl-style-1024-reference-init
```

The program requires an H100 device and fails before simulation if CUDA/H100 is unavailable. Every generated file is written below `/data/250010002/`. This uses a new output directory so it preserves the first run for reference.

At 1024² with a 25.6 × 25.6 domain, `dx=0.025`. The script scales the old `dt=0.02` at `dx=0.2` by `(dx/0.2)²`, giving `dt=0.0003125` and 128,000 steps to t=40. This keeps the diffusion stability number comparable while refining the grid. Reusing `dt=0.02` at 1024² would be unstable for this explicit scheme.

The initial random field is generated on the legacy reference scale (`dx=0.2`, 128² cells over 25.6 units), then nearest-neighbor mapped to 1024². This preserves the old random field's physical feature scale; independent noise at all 1024² points nearly disappears by t=40. The command writes three reviewable artifacts:

- `p53-random-speckle-1024-t40.png`: the u/v heatmaps, with color limits set to match the earlier figure's visible ranges.
- `p53-random-speckle-1024-t40.npz`: full-resolution final arrays and grid/time metadata. The local example NPZ is retained on disk but omitted from the GitHub repository; the ACP command still generates it.
- `run-manifest.json`: exact command, seed, backend/device, parameters, time step, step count, statistics, and update/boundary semantics.

To reproduce the same new run, keep the command and seed unchanged. The old OpenCL notebook used unseeded NumPy randomness, so its original cell-by-cell initial field cannot be reconstructed from the saved figure alone. Seed 0 creates a deterministic field with the same independent uniform distributions.

## OpenCL-to-package mapping

| OpenCL notebook | Package setting | Migration note |
|---|---|---|
| `n`, `Grid_Width`, `Grid_Height` | `Nx`, `Ny` / `--size` | Use a square grid for the published example. |
| `dX`, `dY` | `dx`, `dy` | Set them from the intended physical domain and grid size. For this reproduction, `dx=dy=25.6/1024`. |
| `dT` | `dt` | Preserve stability when changing resolution. The reproduction scales the old value with `dx²`. |
| `endtime` and nested iteration count | `final_time`, `n_steps` | `n_steps = final_time / dt`; the 1024² example uses 128,000 steps. |
| `u0=0.5`, `u=u0+(rand-0.5)*0.5` | `u ~ U(0.25, 0.75)` at 128², then map to 1024² | The seed makes the replacement initialization repeatable while preserving the old cell scale. |
| `v0=25`, `v=v0+(rand-0.5)*1.0` | `v ~ U(24.5, 25.5)` at 128², then map to 1024² | Same distribution, at the same reference spatial scale. |
| `kd`, `k1`, `b`, `k2`, `rho`, `beta`, `Du`, `Dv`, `cv` | `SolverConfig.fig2b()` | Parameter values follow the executable Fig. 2b OpenCL kernel. |
| float OpenCL buffers | `--dtype float32` | Keeps the reproduction's precision close to the old device kernel. |
| `PeriodicBoundaries(...)` | periodic full-grid wrapping | Package solvers use periodic rolls instead of repeatedly writing ghost cells from boundary work items. |
| `u[current] += ...; v[current] += ...` | synchronous out-of-place update | The legacy kernel reads and writes neighboring locations in one launch. Its result can depend on work-item scheduling. NumPy/JAX read one complete old state and write a separate new state. |

## What transfers directly and what changes

The reaction expression and Fig. 2b coefficients transfer directly. The species reactions have opposite signs, preserving the `u+v` reaction invariant. The package also uses the explicit Euler method and periodic diffusion.

The old OpenCL kernel is not a safe numerical reference for exact field equality: each work item updates `u` and `v` in place while other work items may still read those locations for their Laplacian. That creates a scheduling-dependent read/write race. The package uses synchronous double buffering to make each step deterministic. The boundary implementation also differs because the OpenCL code uses ghost cells while the package wraps the full array.

For validation, first compare initial arrays, parameters, grid, `dt`, and final time. Then compare conserved mass, means, variances, spatial correlation, and morphology. Treat pixel-level differences from the legacy in-place kernel as expected until a race-free two-buffer OpenCL reference is available. The saved `reports/opencl_comparison.json` currently has no OpenCL reference arrays, so it is not a measured OpenCL parity report.

## Relation to the H100 4096² convergence study

The H100 grid study deliberately uses a smooth six-mode periodic initial condition shared across resolutions. That makes grid refinement meaningful, but it does not recreate the random speckled initial field. Use `reproduce_opencl_style.py` for an OpenCL-like visual starting point and use `run_p53_grid_study.py` for the smooth-field spatial convergence experiment. Keep their conclusions separate.
