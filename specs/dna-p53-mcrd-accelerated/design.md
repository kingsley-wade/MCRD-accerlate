# Design

## Spec Metadata

- Feature: `dna-p53-mcrd-accelerated`
- Status: `approved`
- Last updated: `2026-10-26`
- Requirements source: `requirements.md`

## Overview

Build a standalone Python package with two related but independent compute paths:

1. A race-free, JAX-first MCRD solver for the two fields `u` and `v`, with a small NumPy reference path and an explicit backend contract for future CUDA and Ascend C work.
2. A batched LPP image-analysis pipeline that reproduces the existing MATLAB overlapping-window 2D FFT behavior while replacing per-window loops and the large overlap-merge tensor.

The first implementation target is a reproducible Apple M4 run. JAX CUDA support for RTX 5090/H100 and Ascend C kernels are later backend work against the same public contracts.

## Repository Tree Impact

```text
DNA-p53-MCRD-accelerated/
  pyproject.toml                 # package metadata and optional backend extras
  uv.lock                        # reproducible dependency resolution
  README.md                      # local setup, examples, backend matrix
  src/mcrd_accel/
    __init__.py
    config.py                    # typed solver/LPP configurations
    backends.py                  # backend/device capability and error boundary
    mcrd/
      __init__.py
      equations.py               # reaction term and conserved quantities
      operators.py               # periodic finite-difference operators
      solver.py                  # functional stepping, scan/output policy
      validation.py              # conservation, convergence, stability metrics
    lpp/
      __init__.py
      windows.py                 # bounded window extraction and batching
      spectrum.py                # FFT, radial/angular bin aggregation
      peaks.py                   # MATLAB-compatible peak/weighted metrics
      merge.py                   # scatter accumulation and director averaging
      pipeline.py                # public LPP entry point
    io.py                        # compact state/metric serialization
  tests/
    test_mcrd_operators.py
    test_mcrd_solver.py
    test_validation.py
    test_lpp_synthetic.py
    test_lpp_contract.py
  benchmarks/
    run_mcrd.py
    run_lpp.py
    configs/
  scripts/
    compare_opencl.py
    validate_local.py
  docs/
    numerical-methods.md
    lpp-parity.md
    backend-matrix.md
  specs/dna-p53-mcrd-accelerated/
    requirements.md
    design.md
    tasks.md
```

Files in the existing `DPICs-MCRD-Phase-separation` repository are not modified or deleted. Generated benchmark results belong under a user-selected output directory and are not imported into source modules.

## Architecture

`config.py` defines immutable, serializable configurations. `mcrd.equations` contains the paper reaction term and parameter profiles. `mcrd.operators` contains periodic array operators. `mcrd.solver` composes these into a pure step function and a compiled time loop. `mcrd.validation` runs controlled experiments and returns structured records rather than plotting. Refinement is a matrix of runs, not a single pairwise comparison: the required spatial series is `N in {64, 128, 256, 512, 1024, 2048, 4096}` at fixed physical domain, with an optional finer reference only when resources allow. The local default smoke profile uses a smaller prefix; the full series is an explicit benchmark/validation profile. Time studies use at least four `dt` levels around the compatibility baseline, by default `{dt0, dt0/2, dt0/4, dt0/8}` plus a coarse stability sweep. The exact tested values, resource profile, and skipped cases are recorded in the report.

The solver's primary execution path is JAX: arrays are backend-owned, the step function is JIT-compilable, and repeated steps use `jax.lax.scan` or an equivalent compiled loop. The initial reference implementation uses out-of-place double buffering. Explicit Euler is retained as the compatibility baseline because it matches the OpenCL notebooks; a second-order explicit method may be added only after the baseline tests establish its stability and accuracy. The default method is selected by validation evidence and recorded in configuration.

`backends.py` exposes backend name, device, dtype, FFT availability, compilation state, and memory telemetry. The local backend may be NumPy for a guaranteed reference path and JAX CPU/Metal when available. NVIDIA later uses JAX CUDA. Ascend support is a separate adapter boundary: Ascend C kernels and CANN launch code may be added under an optional backend module without changing solver or LPP result schemas.

The LPP path is independent of the MCRD time loop. `windows.py` extracts a bounded batch of overlapping `w x w` windows. `spectrum.py` subtracts each window mean, applies the backend 2D FFT, forms power, and aggregates precomputed radial and angular bin indices. `peaks.py` implements the MATLAB rules for valid unique maxima, weighted peak locations, wavelength conversion, direction, and uniqueness. `merge.py` uses sum/count accumulators and circular director components; it never materializes the MATLAB `O * P * n^2` merge tensor. Connected-component filtering is an explicit postprocessing stage with a reference implementation and a backend capability flag.

## Components

| Component | Responsibility | Inputs | Outputs | Depends on |
| --- | --- | --- | --- | --- |
| `config` | Validate and serialize model/LPP/run settings | user config | typed config objects | standard library |
| `backends` | Select device, dtype, FFT and telemetry capabilities | backend request | backend handle/capability record | JAX/NumPy optional |
| `mcrd.equations` | Evaluate paper reaction and conserved total | `u`, `v`, parameters | reaction, `u+v` diagnostics | array API |
| `mcrd.operators` | Apply periodic discrete Laplacian | field, `dx`, `dy` | field derivative | array API |
| `mcrd.solver` | Advance state without read/write races | state, config | final/checkpoint states, metadata | equations, operators, backend |
| `mcrd.validation` | Compute error, convergence, stability, memory records | solver factory/configs | structured validation report | solver, NumPy metrics |
| `lpp.windows` | Extract bounded batches of overlapping windows | image, `w`, `dL` | batch tensor, coordinates | array API |
| `lpp.spectrum` | Transform and bin power | window batch, bin maps | radial/angular power | backend FFT |
| `lpp.peaks` | Match MATLAB peak and metric semantics | binned spectra | per-window LPP fields | array API |
| `lpp.merge` | Filter and overlap-merge fields | LPP fields, thresholds | merged wavelength/direction fields | array API, optional morphology |
| `lpp.pipeline` | Public image-analysis workflow | image, LPP config | LPP result object | windows, spectrum, peaks, merge |
| `benchmarks` | Warmup-aware performance/memory runs | configs | JSON/CSV records | public APIs |

## Data Flow

1. Load a configuration and deterministic initial state; record package revision, dtype, backend, and device.
2. Select the requested backend and validate required capabilities before allocating large arrays.
3. Build the periodic operator and reaction function. Compile one step/loop for the selected shape and dtype.
4. Advance `u` and `v` from a read-only old state into a new state. At checkpoints, compute `sum(u + v)`, finite/bounds checks, and optional copied snapshots.
5. For validation, run a fixed-domain spatial series and a time-step series. The spatial series samples the same deterministic continuous initial condition at `N={64,128,256,512,1024,2048,4096}`, keeps physical final time and parameters fixed, compares each level with a finer reference or nested interpolation, and reports per-level error, conservation drift, observed order, peak memory, and runtime. Large cases use short validation horizons and bounded checkpoint output; they are never implicit in ordinary unit tests. The time series keeps the spatial grid fixed, uses the same initial state, runs at multiple `dt` values including values above and below the expected explicit stability limit, and reports error, conservation drift, finite/bounds status, first failing step/time, and the estimated stable interval. Throughput and memory are recorded for every completed case.
6. For LPP, crop/extract windows in bounded batches, transform each batch, aggregate spectral bins, derive metrics, and write results into fixed-size output arrays.
7. Apply threshold and connected-component filters, accumulate valid overlap values by sum/count, use doubled-angle vector averaging for directions, and return merged fields plus validity masks.
8. Serialize metrics and optional states in a documented compact format. Plotting remains a separate consumer of those outputs.

## Interfaces and Contracts

### MCRD public contract

```python
run(config: SolverConfig, *, initial_state=None) -> SimulationResult
step(state: State, config: SolverConfig) -> State
validate(config: ValidationConfig) -> ValidationReport
```

`State` contains `u` and `v` arrays with shape `(height, width)` and the configured dtype. `SimulationResult` contains the final state, checkpoint times/states when requested, conservation diagnostics, and runtime metadata. `SolverConfig` includes paper parameters (`kd`, `k3`, `k1`, `b`, `k2`, `rho`, `cv`, `beta`, `Du`, `Dv`), grid/spacing, `dt`, final time or step count, integrator, boundary mode, dtype, backend, seed, and output stride. Invalid shapes, nonpositive spacing/time steps, unsupported boundaries, and nonfinite parameters fail before execution.

The conserved quantity is the domain sum of `u + v`; its reported relative drift is `(mass_t - mass_0) / max(abs(mass_0), tiny(dtype))`. The discrete periodic Laplacian must sum to zero up to reduction error. Physical-bound checks are configurable because the legacy equations may produce nonphysical values for unstable step sizes.

### LPP public contract

```python
analyze(image, config: LPPConfig, *, backend=None) -> LPPResult
```

`LPPConfig` includes `w`, `dL`, `minL`, `maxL`, radial tolerance, angular tolerance, peak ratio, batch size, minimum power, minimum patch size, and edge policy. `LPPResult` exposes cropped image shape, window-grid shape, per-window fields (`powmax`, `powmean`, `uniqueness`, `tile`, `within_range` for wavelength and direction), merged fields, masks, and metadata. NaN/invalid-window behavior follows the MATLAB contract unless a documented numerical edge case requires a separate mask.

The FFT convention is `fftshift(fft2(window - mean(window)))`, with power `abs(fft)**2`. Radial and angular bins are precomputed once per window shape. Direction is a director in `[0, pi)`, and merge averages use doubled-angle sine/cosine components.

### Refinement and stability contract

`ValidationConfig` accepts an explicit ordered `grid_sizes` list and an explicit ordered `time_steps` list; it must never silently replace a requested series with one value. The default full grid list is `[64, 128, 256, 512, 1024, 2048, 4096]`. A spatial case holds the physical domain `(Lx, Ly)` fixed and derives `dx=Lx/Nx`, `dy=Ly/Ny`, unless a compatibility profile explicitly requests paper-style fixed spacing. Each case has a resource profile with maximum resident bytes, maximum steps, checkpoint stride, and whether compilation cache reuse is allowed. If the estimated memory exceeds the configured budget, the case is reported as `skipped_resource_budget` before allocation. A temporal case holds the grid and final physical time fixed and records the number of steps actually taken. Each report contains the tested series, reference case, error norm, pairwise refinement ratio, observed order, conservation tolerance, resource profile, and status for every level. Stability sweeps must include a coarse-to-fine `dt` sequence and may also include a coarse-to-large-step sequence; each case is classified as `stable`, `unstable`, `nonfinite`, `bound_violation`, or `inconclusive`.

### Benchmark record contract

Each record is JSON-serializable and includes a schema version, git revision, UTC timestamp, backend/device/runtime versions, dtype, shape, configuration hash, warmup count, measured duration, setup/compile duration, throughput, peak memory if available, and status/error text.

## Error Handling

- Expected failure: invalid numerical parameters or incompatible array shapes. Handling: validate before compilation/allocation and raise a typed configuration error naming the field. User-visible result: actionable message and no partial output.
- Expected failure: requested backend, device, FFT, or Ascend C capability is unavailable. Handling: raise a typed backend error with detected capabilities and optionally fall back to the explicitly requested reference backend. User-visible result: backend status is recorded; no silent device substitution.
- Expected failure: nonfinite state, bound violation, or instability during a validation run. Handling: stop the run, record first failing step/time and diagnostic metric, and return a failed stability case rather than hiding it.
- Expected failure: LPP window has no unique radial/angular maximum or no points in a requested bin. Handling: return NaN fields plus validity masks consistent with MATLAB behavior; do not substitute zero.
- Expected failure: image too large for the configured batch/memory budget. Handling: reduce to bounded chunks only when permitted by configuration; otherwise fail before processing with the estimated requirement.

## Testing Strategy

- Unit tests: periodic Laplacian sum, reaction cancellation, state-shape/config validation, deterministic seeding, LPP bin maps, peak weighting, director averaging, and scatter merge.
- Integration tests: short MCRD runs across NumPy and available JAX backend; LPP synthetic sinusoid/checkerboard images with known wavelength/orientation; optional fixture tests against MATLAB outputs when supplied.
- Numerical tests: conservation tolerance by dtype, spatial refinement order, time refinement order, stability sweep, finite/bounds checks, and deterministic repeated runs.
- Performance checks: warmup-aware MCRD and LPP benchmarks on representative M4 shapes, with throughput and peak-memory records. Performance tests are reports, not brittle pass/fail thresholds until baseline measurements exist.
- Compatibility checks: OpenCL comparison script on a small matching configuration, explicitly labeling legacy in-place-kernel differences. Later CUDA and Ascend C checks reuse the same result schema and tolerances.
- Smoke/manual checks: `uv run` setup, one CLI/API example, serialized report inspection, and no import-time requirement for plotting libraries.

## Migration and Compatibility

This is an additive sibling repository. The original notebooks, OpenCL files, MATLAB scripts, images, and output layout remain untouched. The new package does not promise binary compatibility with `.mat` files initially; optional import/export adapters may be added after the numerical contracts stabilize. The public MCRD and LPP result schemas are backend-neutral so that CPU/M4, CUDA, and Ascend C implementations can be compared without changing analysis code.

## Risks and Tradeoffs

| Risk or Tradeoff | Impact | Mitigation |
| --- | --- | --- |
| JAX Metal support or FFT behavior differs across versions | Local acceleration may be unavailable or numerically vary | Keep NumPy reference path, record versions, and use tolerance-based parity tests |
| Explicit Euler is easy to compare with OpenCL but has a restrictive diffusion stability limit | Long simulations may require many steps | Measure stability first; add a validated higher-order method behind the same step contract |
| MATLAB LPP has ambiguous edge cases and toolbox-dependent morphology | Exact parity may fail on rare windows | Preserve masks/NaNs, add synthetic fixtures, document intentional differences, and accept user golden fixtures later |
| Full 4096-grid studies can consume substantial M4 memory/time | A long run could compete with normal desktop use or trigger swap | Estimate bytes before allocation, use short horizons and bounded checkpoints, make the full profile explicit, and report `skipped_resource_budget` rather than allocating unexpectedly |
| Fully vectorized LPP can exceed M4 memory | Compilation or batch execution can fail | Bound batch size, use precomputed bins, and stream sum/count merge accumulators |
| Ascend C is a separate toolchain from JAX | One implementation may not run unchanged on all devices | Keep backend-neutral contracts and isolate Ascend kernels/launch code under an optional adapter |
| Legacy OpenCL in-place updates may be nondeterministic | Direct output equality is not a valid acceptance criterion | Compare invariants and statistical/field norms, and label legacy divergence explicitly |

## Approval

- Design approved by user: `yes`
