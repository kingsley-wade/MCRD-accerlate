# Numerical Methods

> This document records the discretization, integrator choice, and validation strategy for the MCRD solver. It is a living document and will be updated as the solver stabilizes.

## Model equations

The Fig. 2b executable OpenCL kernel uses one scalar reaction rate shared with opposite signs by the two species:

$$
R(u,v) = -k_d\,u + (k_1u+b)v - \rho\,\frac{k_2\beta u}{k_2+u+v}
$$

$$
\partial_t u = c_v R(u,v) + D_u\Delta u, \qquad
\partial_t v = -c_v R(u,v) + D_v\Delta v
$$

where

$$
f(u,v) = k_1u+b, \quad g(u,v) = \frac{k_2\beta}{k_2+u+v}
$$

The source notebook's markdown displays `-rho*g(u,v)*u` in both equations, which does not conserve `u+v`. The executable kernel instead negates the entire reaction expression for `v`, as shown above. This port follows the executable kernel and its stated mass-conserving intent. The legacy kernel also updates fields in place and uses ghost-cell boundary rows/columns; the NumPy/JAX solver uses out-of-place updates and periodic rolls over the full array. Therefore the reaction law and parameters match, while legacy trajectories are not expected to match exactly.

## Default parameter profile (Fig. 2b)

Values are taken from `DPICs-MCRD-Phase-separation/fig2/Fig2b_code/Fig2_b_sim.ipynb`:

| Parameter | Value | Notebook expression |
|-----------|-------|---------------------|
| $k_d$ | 0.36 | `kd=0.36` |
| $k_1$ | 1.35 | `k1=1.35` |
| $b$ | $60/6.7$ | `tildeb=60; b=tildeb/k3` with `k3=6.7` |
| $k_2$ | 2.0 | `k2=2.0` |
| $\rho$ | 9.0 | `rho=5.0*1.8` |
| $\beta$ | 19.58 | `beta=17.8*1.1` |
| $D_u$ | 0.0085 | `Du=0.17*0.05` |
| $D_v$ | 0.17 | `Dv=0.17` |
| $c_v$ | 0.1072 | `cv=0.016*6.7` |

**Note on the paper README:** the repository README states that "within the codes symbol of beta and k2 should be switched for their value." This package records the values as they appear in the notebook source, not the swapped interpretation. The OpenCL comparison script (task 4.4) will document any resulting divergence.

The Fig. 2b notebook initializes `u` uniformly on `[0.25, 0.75]` and `v` on `[24.5, 25.5]`. Its inline comment for the `v` range is inconsistent with the actual expression; this package follows the expression.

## Discretization

- Spatial: 5-point periodic Laplacian on a uniform grid with spacing $\Delta x, \Delta y$.
- Temporal: explicit Euler (compatibility baseline with the OpenCL notebooks).
- Update: out-of-place double buffering (no read/write race).

## Validation strategy

1. **Conservation:** relative drift of $\sum(u+v)$ at every checkpoint, dtype-aware tolerance.
2. **Spatial convergence:** grid series $N \in \{64, 128, 256, 512, 1024, 2048, 4096\}$ at fixed physical domain, compared against a finer reference or nested interpolation.
3. **Temporal convergence:** time-step series $\{dt_0, dt_0/2, dt_0/4, dt_0/8\}$ at fixed grid, compared against a finer-step reference.
4. **Stability sweep:** coarse-to-large $dt$ sequence, classifying each run as stable/unstable/nonfinite/bound-violation.
5. **OpenCL comparison:** invariant and field-norm comparison against the legacy implementation, with explicit labeling of legacy in-place-kernel differences.

## Local M4 Evidence

Reproduce the bounded full-series report with:

```bash
uv run python scripts/validate_mcrd.py
```

The generated report artifact is `reports/mcrd_validation.json`. It uses float32, a fixed unit-square domain, and an eight-step spatial horizon for `N={64,128,256,512,1024,2048,4096}`. All seven levels completed. The temporal study uses `dt={1e-4,5e-5,2.5e-5,1.25e-5}` and reports errors against the finest completed step. The stability study uses a fixed physical horizon with `dt={5e-5,1e-4,2e-4,4e-4,8e-4,1.6e-3}`; the first three cases were stable, while `4e-4` and above violated non-negative concentration bounds. The reported stable interval is the contiguous stable prefix of the tested step sizes, so it cannot span a failing case.

The spatial error decreases from `1.31e-2` at `N=64` through `2.08e-4` at `N=2048`; the `N=4096` reference error is zero by definition. Temporal error decreases from `6.90e-3` at `dt=1e-4` to `9.84e-4` at `dt=2.5e-5`; the finest reference error is zero by definition. These short-horizon measurements verify execution and refinement behavior, not long-time morphology or a general accuracy guarantee.

Benchmark artifacts are `reports/benchmark_mcrd.json` and `reports/benchmark_lpp.json`. Their peak-memory field is measured with Python `tracemalloc`, which captures host allocations and does not represent total resident or accelerator memory. The local records identify the ARM64 host, Python/NumPy/JAX versions, backend, configuration, setup/warmup, steady-state throughput, and schema version. No Metal, CUDA, or Ascend C performance was measured.
