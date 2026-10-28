# SciSim-Core PDE Case Matrix

The canonical verification matrix covers the three standard PDE classes with a
small periodic finite-difference problem for each. The p53-dsDNA MCRD system is
kept as a separate scientific workload because its coupled reaction-diffusion
physics and source-code parity questions are not represented by these scalar
equations.

| PDE class | Representative equation | Method | Main correctness checks |
| --- | --- | --- | --- |
| Parabolic | Heat equation `u_t = alpha Laplace(u)` | Explicit Euler, periodic 5-point Laplacian | Discrete Fourier-mode evolution, mass conservation, explicit stability limit |
| Elliptic | Periodic Poisson `-Laplace(u) = f` | Matrix-free conjugate gradients, zero-mean gauge | Compatibility (`mean(f)=0`), manufactured discrete solution, relative residual |
| Hyperbolic | Wave equation `u_tt = c^2 Laplace(u)` | Leapfrog, periodic 5-point Laplacian | Discrete Fourier-mode recurrence, mass and energy diagnostics, CFL limit, snapshot steps |
| Scientific workload | p53-dsDNA mass-conserving reaction-diffusion | Existing explicit Euler MCRD implementation | Stoichiometric conservation, convergence/stability reports, morphology and legacy comparison |

The heat equation rejects a step if

```text
dt * alpha * (dx^-2 + dy^-2) > 1/2
```

and the two-dimensional leapfrog wave solver rejects a step if

```text
c * dt * sqrt(dx^-2 + dy^-2) > 1.
```

These are stability constraints for the implemented 5-point periodic stencil;
passing them does not establish positivity or scientific accuracy. The Poisson
right-hand side must have zero mean for a periodic solution to exist. The
solver rejects incompatible inputs instead of silently projecting them.

## Bounded Local Verification

Run all three canonical cases plus the p53 workload on small grids:

```bash
uv run python scripts/validate_local.py --case all --grid-sizes 16,32 --steps 20 --dt 1e-5
```

Choose a single equation family or backend when narrowing a failure:

```bash
uv run python scripts/validate_local.py --case heat --backend jax-cpu --dtype float32 --grid-sizes 16,32 --steps 20
uv run python scripts/validate_local.py --case poisson --backend jax-cpu --dtype float32 --grid-sizes 16,32
uv run python scripts/validate_local.py --case wave --backend jax-cpu --dtype float32 --grid-sizes 16,32 --steps 20
```

On an H100 host, select `--backend jax-cuda`. Device selection is explicit:
`jax-cpu` is pinned to CPU, and `jax-cuda` raises an error if JAX cannot see a
GPU instead of silently falling back to CPU.

The command writes `reports/s1_pde_smoke.json`. The smoke profile uses explicit
small inputs and does not start the H100/RTX 5090 experiment matrix. To select a
larger p53 convergence run, use the MCRD benchmark and validation APIs directly.

## Validation Scope

The canonical PDE tests compare NumPy and JAX CPU results, use Fourier modes
whose discrete evolution is known, check Poisson against a manufactured
solution of the same discrete operator, and exercise invalid compatibility,
stability, and step-count inputs. JAX arrays use the configured dtype subject
to JAX's runtime precision configuration.

The p53 reaction follows the executable Fig. 2b OpenCL kernel. The notebook's
displayed `dv/dt` equation gives the degradation term the same sign in both
species equations, while the kernel negates the entire reaction rate. The port
follows the executable kernel and its mass-conserving implementation. The
legacy kernel also updates fields in place and uses ghost-cell boundaries, so
its trajectories are not expected to match the new double-buffered full-array
periodic implementation exactly.

The local Apple Silicon environment provides JAX CPU only. A remote H100 host
is reachable and an older CUDA smoke artifact is present, but that artifact
does not identify its host, GPU, driver, or JAX runtime. A fresh attributable
H100 validation and benchmark is required before citing H100 results. RTX 5090,
Ascend C, GPU memory telemetry, and Nsight evidence remain unmeasured.

JAX's conjugate-gradient API does not currently expose its iteration count in
the result contract used here. `PoissonResult.iterations` is therefore `None`
for JAX and a measured non-negative count for the SciPy NumPy solver; `-1` is
not used as an unknown-value sentinel.
