# Requirements

## Spec Metadata

- Feature: `dna-p53-mcrd-accelerated`
- Owner: `User`
- Status: `approved`
- Last updated: `2026-10-18`

## Purpose

Create a standalone, high-performance Python package for the DNA-p53 mass-conserving reaction-diffusion (MCRD) model, with a locally validated implementation on the available Apple M4 system. The package must make numerical correctness measurable against analytical invariants, grid/time refinement, and the existing OpenCL implementation. It must also provide a custom, accelerated implementation of the existing Local Pattern Properties (LPP) image-analysis workflow, whose current MATLAB implementation is slow on overlapping image windows.

## Scope

### In Scope

- Implement the two-field MCRD model used by the paper simulation notebooks, with configurable model parameters, grid, time step, initial fields, and periodic boundaries.
- Implement a deterministic, race-free numerical update and validate a suitable time-integration scheme for the model.
- Provide a reproducible numerical reference path and compare results with available OpenCL outputs under matching configurations.
- Measure mass-conservation error, spatial and temporal convergence, time-step stability, simulation throughput, and memory use.
- Port the existing MATLAB LPP workflow into a standalone Python analysis module, including overlapping-window spectral analysis, radial/angular power summaries, wavelength/direction/uniqueness metrics, filters, and overlap merging.
- Optimize LPP by processing windows in batches or bounded-size chunks and avoiding the MATLAB implementation's large 3D overlap-merge intermediate.
- Use `uv` for Python version/environment management, dependency resolution, and lock-file reproducibility.
- Create an independent Git repository beside the existing paper-code repository. Preserve the original repository and its history.
- Validate and benchmark the package locally on the available Apple M4 system. Keep future device validation possible for NVIDIA RTX 5090, NVIDIA H100, and the user's NPU target.

### Out of Scope

- Editing, deleting, or reorganizing files in the original paper-code repository.
- Reimplementing the FFT algorithm itself (for example, writing a new FFT butterfly engine). The package owns the LPP spectral-analysis algorithm and may call the selected backend's optimized FFT primitive.
- Implementing a separate continuous or discrete wavelet transform in this milestone. The MATLAB directory is named `Fig3_code_wavelet_Exp`, but `LocalPattProps`/`LPPwindow` implement overlapping-window 2D FFT spectral analysis.
- Moving plotting, figure styling, experimental-image import workflows, or paper-specific figure scripts into the high-performance compute core.
- Claiming or measuring performance on RTX 5090, H100, or NPU hardware before those devices are available for testing.

## Users and Goals

- As a researcher, I want to run the DNA-p53 MCRD model from a reproducible Python API or command line so that simulations no longer depend on notebook-specific OpenCL setup.
- As a researcher, I want mass conservation, convergence, and time-step stability checked so that an accelerated result is supported by numerical evidence.
- As a researcher, I want the existing MATLAB LPP outputs reproduced by an implementation I can inspect and extend so that image analysis no longer depends on the slow per-window MATLAB loops.
- As a package maintainer, I want `uv`-managed dependencies and separate compute/analysis/benchmark modules so that local development is reproducible and future device backends can be tested independently.

## Functional Requirements

- `WHEN` the caller supplies valid model parameters, initial fields, grid spacing, and time controls, `THE SYSTEM SHALL` run the two-field MCRD equations with a race-free update and the configured boundary conditions.
- `WHEN` the caller selects periodic boundaries, `THE SYSTEM SHALL` use a periodic discrete diffusion operator whose domain-wide sum is conservative up to documented floating-point reduction error.
- `WHEN` a model run is configured for validation, `THE SYSTEM SHALL` report relative drift in the conserved total `sum(u + v)` at requested checkpoints.
- `WHEN` a grid-refinement validation is run at fixed physical domain and final time, `THE SYSTEM SHALL` report errors against a finer reference and the observed spatial convergence rate.
- `WHEN` a time-step refinement validation is run, `THE SYSTEM SHALL` report errors against a finer-step reference and the observed temporal convergence rate for the selected integrator.
- `WHEN` a time-step stability sweep is run, `THE SYSTEM SHALL` record whether each run remains finite and within configured physical bounds, and report the tested step-size range and failure point where instability occurs.
- `WHEN` compatible initial states and parameters are supplied, `THE SYSTEM SHALL` compare package checkpoints or summary metrics with available OpenCL results and identify precision, discretization, or legacy-kernel differences.
- `WHEN` the caller supplies an image and valid LPP parameters (`w`, `dL`, `minL`, and `maxL`), `THE SYSTEM SHALL` return per-window wavelength, direction, power, uniqueness, and range-validity fields with documented units and NaN behavior.
- `WHEN` LPP window analysis is requested, `THE SYSTEM SHALL` calculate all windows using bounded batching/chunking and backend FFT primitives without requiring a third-party wavelet-analysis package.
- `WHEN` the caller requests the LPP merge stage, `THE SYSTEM SHALL` apply the configured power/range/patch filters, merge overlapping wavelength and uniqueness values, merge orientation as a director angle, and return the edge-trimmed result.
- `WHEN` benchmark mode is enabled, `THE SYSTEM SHALL` report compilation/setup time separately from steady-state runtime, grid-steps per second for MCRD, windows per second for LPP, and peak memory with device/runtime metadata.
- `WHEN` output fields or time snapshots are requested, `THE SYSTEM SHALL` support final-state and configurable strided output so that full trajectories need not remain resident in memory.
- `WHEN` the package is installed or its development environment is synchronized, `THE SYSTEM SHALL` resolve dependencies through `uv` and record a reproducible lock file.
- `IF` a requested backend is unavailable or lacks a required operation, `THE SYSTEM SHALL` report the unsupported capability clearly and provide an available reference/fallback backend when configured.

## Non-Functional Requirements

- Performance: MCRD and LPP throughput must be measured after backend warmup, with compilation/setup cost reported separately. The spatial study must support and report the requested grid series from `N=64` through `N=4096`; local smoke tests may use a smaller subset, while full high-resolution runs must be explicitly selected. No unsupported accelerator speed claims are permitted.
- Reliability: The solver must not update a field in place while neighboring cells read it. Validation runs must be reproducible from recorded configuration and initial state.
- Numerical quality: Conservation, grid convergence, time convergence, and stability acceptance tolerances must be explicit, dtype-aware, and supported by test results. The default spatial order and time-integration order must be measured rather than assumed.
- Memory: Benchmarks must record peak memory and output policy. LPP processing must not require materializing every overlapping window's merged values in an `O(image area * window overlap count)` 3D array. The local environment must provide a bounded-memory profile so a `4096 x 4096` case does not run unintentionally during ordinary tests.
- Usability: Core APIs and CLI usage must be documented, with plotting kept optional and outside the compute core.
- Compatibility: The local development target is Apple Silicon with an isolated `uv`-managed Python environment. The package structure must allow NVIDIA and NPU backend validation without changing the model/LPP public contracts.
- Reproducibility: Benchmark records must include package revision, configuration, dtype, backend/device, grid/image shape, and relevant runtime versions.

## Constraints

- Repository constraints: The new project is a sibling directory/repository named `DNA-p53-MCRD-accelerated`; the existing `DPICs-MCRD-Phase-separation` repository remains untouched.
- Platform constraints: The current local machine is a MacBook Air with Apple M4 and 16 GB memory. RTX 5090, H100, and NPU tests are deferred until hardware access is provided.
- Dependency constraints: Use `uv` for environment and lock management. The virtual environment must be isolated to the new repository and must not modify system Python. Use an optimized backend FFT primitive while implementing LPP frequency aggregation, peak selection, and result merging in project code.
- Reference constraints: The paper repository contains OpenCL simulation notebooks and MATLAB analysis scripts. No standalone MATLAB MCRD solver has been identified there. The OpenCL kernel updates neighboring cells in-place, so it may be nondeterministic and must not be treated as an unquestioned numerical oracle.
- Model constraints: The initial solver scope is the two-field `u`/`v` model in Fig. 2/3/5 notebooks. Parameter profiles may vary by figure; the governing equations and parameter values must be traceable to an explicit source/configuration.
- Image-analysis constraints: Existing MATLAB LPP merge uses Image Processing Toolbox operations. The Python module must document any behavior that cannot be reproduced exactly and distinguish required numerical fields from optional plotting.

## Assumptions

- The first solver profile will use the two-field equations and parameter values from the Fig. 2b notebook, with additional figure profiles added only where their equations/configuration are confirmed.
- Periodic boundaries are the first solver boundary condition because they are used by the paper simulation notebooks.
- The numerical scheme will be selected by measured conservation, convergence, and stability results; a race-free reference implementation will be retained for validation.
- “Own LPP implementation” means the project will implement the windowing, spectral bins, peak metrics, filtering, and merge logic, while using a backend FFT primitive for the transform itself.
- The NPU target is Huawei Ascend hardware with Ascend C as the low-level kernel/optimization path. Exact device SKU and CANN/Ascend C version will be recorded when hardware testing starts.

## Open Questions

- Can the user later provide a few MATLAB LPP outputs or representative source images? This is optional; without them, parity tests will use synthetic images with known wavelengths/orientations and the existing MATLAB source as the behavioral reference.
- Are original MATLAB MCRD solver results available outside the checked paper-code repository? This is optional; without them, the numerical reference will be the race-free package solver plus conservation/convergence tests and the available OpenCL implementation as a legacy comparison.

## Approval

- Requirements approved by user: `yes`
