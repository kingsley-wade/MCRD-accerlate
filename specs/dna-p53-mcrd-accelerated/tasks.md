# Tasks

## Spec Metadata

- Feature: `dna-p53-mcrd-accelerated`
- Status: `complete`
- Last updated: `2026-10-26`
- Requirements source: `requirements.md`
- Design source: `design.md`

## Status Legend

- `[ ]` pending
- `[~]` in progress
- `[x]` completed
- `[!]` blocked

## Execution Rules

- Work only on one `[~]` task at a time.
- Do not start a task until its dependencies are complete.
- Mark a task `[x]` only after recording acceptance evidence.
- If a task changes scope, pause and update the approved specs before continuing.
- Do not add plotting or paper-figure code to the compute modules.

## Task Checklist

### 1. Repository Structure and Environment

- [x] 1.1 Establish package scaffold
  - Module: `repository`
  - Goal: Create the `src/`, `tests/`, `benchmarks/`, `scripts/`, `docs/`, and package metadata layout from `design.md`.
  - Files: `pyproject.toml, README.md, src/mcrd_accel/__init__.py, src/mcrd_accel/{config.py,backends.py,io.py}, tests/, benchmarks/, scripts/, docs/`
  - Depends on: `requirements approved, design approved`
  - Acceptance evidence: `find` output matches the approved tree and the package imports in a clean environment.

- [x] 1.2 Configure `uv` and dependency extras
  - Module: `environment`
  - Goal: Define supported Python version, locked core dependencies, optional JAX/Metal/CUDA/analysis extras, and test/benchmark commands.
  - Files: `pyproject.toml, uv.lock, README.md`
  - Depends on: `1.1`
  - Acceptance evidence: `uv sync` completes and `uv run python -c "import mcrd_accel"` succeeds.

### 2. MCRD Core

- [x] 2.1 Implement typed configurations and paper parameter profiles
  - Module: `config`
  - Goal: Validate solver, validation, output, and backend settings and encode the Fig.2b-compatible parameter profile with provenance.
  - Files: `src/mcrd_accel/config.py, tests/test_mcrd_solver.py, docs/numerical-methods.md`
  - Depends on: `1.2`
  - Acceptance evidence: invalid shapes, spacing, `dt`, and nonfinite parameters fail with typed errors; a valid profile round-trips through serialization.

- [x] 2.2 Implement periodic operators and reaction equations
  - Module: `mcrd`
  - Goal: Implement the two-field reaction term and periodic discrete Laplacian with out-of-place array semantics.
  - Files: `src/mcrd_accel/mcrd/equations.py, src/mcrd_accel/mcrd/operators.py, tests/test_mcrd_operators.py`
  - Depends on: `2.1`
  - Acceptance evidence: periodic Laplacian sums to dtype-aware tolerance; reaction contributions cancel pointwise in `u+v`; operator tests pass on the reference backend.

- [x] 2.3 Implement solver stepping and output policy
  - Module: `mcrd.solver`
  - Goal: Implement deterministic double-buffer stepping, compiled loop integration where available, checkpoints, strided snapshots, and finite/bounds diagnostics.
  - Files: `src/mcrd_accel/mcrd/solver.py, src/mcrd_accel/mcrd/__init__.py, tests/test_mcrd_solver.py`
  - Depends on: `2.2`
  - Acceptance evidence: repeated seeded runs match; no in-place neighbor race is present; short runs return final state and requested checkpoints.

### 3. LPP Analysis

- [x] 3.1 Implement bounded overlapping-window extraction
  - Module: `lpp.windows`
  - Goal: Reproduce MATLAB window coordinates/cropping and process windows in configurable bounded batches.
  - Files: `src/mcrd_accel/lpp/windows.py, tests/test_lpp_contract.py`
  - Depends on: `1.2`
  - Acceptance evidence: synthetic image windows match MATLAB indexing for several `w`/`dL` combinations and batch sizes.

- [x] 3.2 Implement batched FFT spectral aggregation
  - Module: `lpp.spectrum`
  - Goal: Implement mean subtraction, shifted 2D FFT, power, radial bins, angular bins, and empty-bin behavior using backend FFT primitives.
  - Files: `src/mcrd_accel/lpp/spectrum.py, tests/test_lpp_synthetic.py, docs/lpp-parity.md`
  - Depends on: `3.1`
  - Acceptance evidence: synthetic sinusoid images identify expected radial frequency and orientation within documented tolerance; memory stays bounded by batch size.

- [x] 3.3 Implement MATLAB-compatible peak metrics and LPP result schema
  - Module: `lpp.peaks`
  - Goal: Implement unique-max validation, weighted peak selection, wavelength conversion, direction normalization, uniqueness, and NaN/mask semantics.
  - Files: `src/mcrd_accel/lpp/peaks.py, src/mcrd_accel/lpp/__init__.py, tests/test_lpp_synthetic.py, tests/test_lpp_contract.py`
  - Depends on: `3.2`
  - Acceptance evidence: hand-constructed spectra and synthetic images cover unique peaks, tied peaks, edge peaks, empty bins, and director angles.

- [x] 3.4 Implement memory-bounded LPP filtering and merge
  - Module: `lpp.merge, lpp.pipeline`
  - Goal: Apply range/power filters, optional connected-component filtering, scatter sum/count overlap merge, and doubled-angle director averaging without the MATLAB 3D merge tensor.
  - Files: `src/mcrd_accel/lpp/merge.py, src/mcrd_accel/lpp/pipeline.py, tests/test_lpp_contract.py, docs/lpp-parity.md`
  - Depends on: `3.3`
  - Acceptance evidence: a small known field matches a reference merge; peak memory scales with image area plus batch size rather than overlap count; public `analyze` API works end to end.

### 4. Validation and Compatibility

- [x] 4.1 Implement conservation and diagnostic reports
  - Module: `mcrd.validation`
  - Goal: Report conserved-mass drift, finite/bounds status, norm choices, and reproducibility metadata for every run.
  - Files: `src/mcrd_accel/mcrd/validation.py, tests/test_validation.py, docs/numerical-methods.md`
  - Depends on: `2.3`
  - Acceptance evidence: a periodic short run reports drift below the dtype-aware tolerance and includes backend/dtype/config metadata.

- [x] 4.2 Implement multi-level grid convergence study
  - Module: `mcrd.validation`
  - Goal: Run the configured full grid series `N={64,128,256,512,1024,2048,4096}` or a user-supplied series at fixed physical domain, with a finer reference/nested interpolation and per-level observed orders. Use short horizons, preflight memory estimates, bounded checkpoints, and explicit resource profiles for large cases.
  - Files: `src/mcrd_accel/mcrd/validation.py, tests/test_validation.py, scripts/validate_local.py, docs/numerical-methods.md`
  - Depends on: `4.1`
  - Acceptance evidence: report lists every requested grid, reference case, error norm, refinement ratio, observed order, peak memory, runtime, resource profile, and any skipped case with reason; ordinary tests do not allocate the full profile unless explicitly requested.

- [x] 4.3 Implement multi-level time convergence and stability sweep
  - Module: `mcrd.validation`
  - Goal: Run at least `{dt0, dt0/2, dt0/4, dt0/8}` for temporal convergence plus a coarse-to-large-step stability series, recording first failure step/time and classification.
  - Files: `src/mcrd_accel/mcrd/validation.py, tests/test_validation.py, scripts/validate_local.py, docs/numerical-methods.md`
  - Depends on: `4.1`
  - Acceptance evidence: report contains all requested `dt` values, stable/unstable/nonfinite/bound-violation status, temporal errors/orders, and the estimated stable interval.

- [x] 4.4 Add OpenCL comparison adapter
  - Module: `compatibility`
  - Goal: Load or generate a small matching case, run the legacy OpenCL path when available, and compare invariants and field norms while labeling legacy in-place-kernel differences.
  - Files: `scripts/compare_opencl.py, docs/numerical-methods.md, tests/test_validation.py`
  - Depends on: `2.3, 4.1`
  - Acceptance evidence: command reports whether OpenCL is available, records both configurations, and produces a structured comparison without modifying the original repository.

### 5. Benchmarks and Backend Boundaries

- [x] 5.1 Implement warmup-aware MCRD/LPP benchmarks
  - Module: `benchmarks`
  - Goal: Separate setup/compile from steady-state timing and record throughput, memory, device, dtype, shape, and revision.
  - Files: `benchmarks/run_mcrd.py, benchmarks/run_lpp.py, src/mcrd_accel/backends.py, tests/test_validation.py`
  - Depends on: `2.3, 3.4, 4.1`
  - Acceptance evidence: benchmark commands emit schema-versioned JSON records with warmup count, steady-state duration, throughput, and memory fields.
  - Recorded evidence: `reports/benchmark_mcrd.json` and `reports/benchmark_lpp.json` on the local ARM64 Apple Silicon host; memory is Python host allocation via `tracemalloc`.

- [x] 5.2 Add backend capability reporting and future adapter stubs
  - Module: `backends`
  - Goal: Report NumPy/JAX CPU/Metal capabilities now and define isolated extension points for JAX CUDA and Huawei Ascend C/CANN without claiming untested support.
  - Files: `src/mcrd_accel/backends.py, docs/backend-matrix.md, README.md`
  - Depends on: `1.2, 2.3, 3.4`
  - Acceptance evidence: local capability report is accurate; unavailable CUDA/Ascend C requests fail clearly; public result contracts remain backend-neutral.

### 6. Documentation and Final Verification

- [x] 6.1 Document local installation and API usage
  - Module: `docs`
  - Goal: Document `uv` setup, MCRD run, LPP analysis, benchmark commands, output schemas, and plotting as an optional downstream concern.
  - Files: `README.md, docs/numerical-methods.md, docs/lpp-parity.md, docs/backend-matrix.md`
  - Depends on: `4.4, 5.1, 5.2`
  - Acceptance evidence: a clean-machine walkthrough can execute the documented local smoke commands.

- [x] 6.2 Run final spec, test, and local benchmark validation
  - Module: `verification`
  - Goal: Validate all three specs, run the full test suite, run the multi-level grid/time reports, and preserve a concise local benchmark artifact.
  - Files: `specs/dna-p53-mcrd-accelerated/requirements.md, specs/dna-p53-mcrd-accelerated/design.md, specs/dna-p53-mcrd-accelerated/tasks.md, reports/`
  - Depends on: `6.1`
  - Acceptance evidence: spec validator passes; tests pass; grid and stability reports contain the full requested series; local M4 benchmark metadata is recorded; original repository remains unchanged.
  - Recorded evidence: `uv run python scripts/validate_specs.py`, `uv run pytest -q` (144 passed), `reports/mcrd_validation.json` (all seven grids, four time levels, stability outcomes), plus the two benchmark artifacts above. CUDA, Metal, H100, RTX 5090, and Ascend C execution remains unmeasured.

## Approval

- Task checklist approved by user: `no`
