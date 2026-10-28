# GitHub Project Release Roadmap

The master task book for the full migration and portfolio release is [project-development-plan.md](project-development-plan.md). This document focuses on publication gates.

## Purpose

This roadmap turns the current repository into a clean, installable, testable public GitHub project. It separates the work needed for an honest public Alpha from the additional evidence needed for a research-grade release and hardware acceleration claims.

The existing approved implementation spec is marked complete, but that means its scoped implementation tasks are complete. It does not mean the repository is ready to publish: the worktree has no commits, package metadata contains placeholders, two parity/backend documents are placeholders, and the project has no CI or release hygiene files yet.

## Current Assessment

The repository already contains a useful foundation:

- A `src/` Python package with NumPy and JAX solver paths, LPP analysis, tests, scripts, benchmarks, a `uv.lock`, and numerical reports.
- Local test evidence: 146 tests passed on the current Apple Silicon environment after API hardening.
- A local MCRD validation artifact covering the requested seven spatial grids and four time-step levels.
- MCRD and LPP benchmark JSON artifacts with machine/runtime metadata.
- Requirements and design documents that distinguish implemented paths from future hardware targets.

The following release blockers remain:

- Git has no commits yet, so there is no reviewable project history or release baseline.
- A `.gitignore`, MIT `LICENSE`, initial GitHub Actions workflow, issue/PR templates, and contribution guide now exist locally, but CI has not yet run on GitHub.
- `pyproject.toml` uses `kingsley-wade` as author, declares `kingsley-wade/MCRD-accerlate` URLs, and distinguishes the `mcrd-accel` distribution from the `mcrd_accel` import package. The target repo exists as an empty private repository.
- `docs/backend-matrix.md` and `docs/lpp-parity.md` describe current behavior; MATLAB parity and fresh H100 provenance remain open evidence gaps.
- `reports/s1_pde_smoke_h100.json` must have its provenance verified before it is presented as hardware evidence. The local host for the current work is Apple Silicon.
- `reports/opencl_comparison.json` records that no OpenCL reference was supplied; it is not a parity result.
- The LPP benchmark reports roughly 948 MB of Python-traced peak allocations for a 256 x 256 image. This measurement needs investigation and an OS-level RSS measurement before making bounded-memory or performance claims.
- The current LPP tests use synthetic cases; MATLAB golden outputs and representative real images have not been supplied.
- The current benchmark records are local smoke evidence, not stable performance baselines. Their memory method does not include total process RSS or device allocations.
- Current device evidence is NumPy/JAX CPU on Apple Silicon. The CUDA dependency installation on H100 stalled before device execution. Metal, CUDA/H100, RTX 5090, and Ascend C are not currently validated by this checkout.

## Remote Access Check (2026-10-28)

- SSH batch access to the configured remote host succeeds. The host reports Ubuntu 22.04 x86_64 and eight NVIDIA H100 80 GB devices.
- At the latest check, every H100 had about 29 GiB free and about 52 GiB allocated by existing processes. Do not stop or interfere with those jobs; run only a small test after checking memory again.
- Base Python is 3.10 and is below this project's `requires-python >=3.11`. Python 3.12 and `uv` are available. The full project is uploaded at `/mnt/afs/gmt/DNA-p53-MCRD-accelerated`.
- `uv sync --locked --extra jax-cuda --python 3.12` resolved packages and started downloading, then stalled for over ten minutes with negligible cache growth. The install process was stopped; no GPU test ran. `/mnt/afs` had about 87 GiB available. There were no `sbatch`, `srun`, or `sinfo` commands in the inspected environment.
- Local `gh` identifies the username `kingsley-wade`, but its stored GitHub token is invalid. Local preparation can continue; repository creation, pushing, and release publication require GitHub re-authentication.

## Release Levels

### Public Alpha

The project can publish a useful Alpha once Phase 0 and Phase 1 are complete. It must install from a clean environment, pass CI on declared Python versions, include accurate licensing and usage documentation, and label numerical and backend limitations plainly. An Alpha does not require an untested accelerator backend or exact MATLAB parity.

### Research-Ready Release

Before claiming reliable scientific reproduction, complete Phase 2: obtain external reference fixtures where possible, establish reproducible long-enough validation, validate the LPP memory claim with an appropriate process-memory measure, and publish provenance for every result. Hardware-specific claims remain gated on actual hardware tests.

### Accelerator Support

Metal, CUDA, and Ascend C are separate support claims. Each requires a backend-specific CI or recorded hardware run, numerical parity tolerances, device/runtime metadata, and benchmark evidence. A capability stub or install extra alone is not support evidence.

## Phase 0: Repository Hygiene and Identity

These tasks should be completed before the first public push.

- [x] Add a root `.gitignore` for `.venv/`, Python bytecode, pytest/coverage caches, build outputs, local editor files, and temporary benchmark files. Keep explicitly chosen report artifacts trackable.
- [x] Decide and document the GitHub repository (`kingsley-wade/MCRD-accerlate`), import name (`mcrd_accel`), and distribution name (`mcrd-accel`).
- [x] Set the supplied GitHub username as author and add the intended repository URLs, Python classifiers, and package metadata to `pyproject.toml`.
- [x] Add the actual `LICENSE` file matching the declared MIT license. The final audit of source-code and image redistribution rights remains a publication gate.
- [ ] Decide which generated artifacts belong in Git. Keep small, provenance-rich examples; exclude caches and machine-specific output that cannot be reproduced. Label hardware-specific artifacts with verified device provenance.
- [ ] Complete the redistribution and provenance audit before pushing. The local scan found no credential patterns or private local absolute paths; copied source/data/image rights still need confirmation.
- [ ] Create an initial commit that contains only reviewed project files. Confirm `git status --short` is clean afterward.
- [ ] Create the GitHub repository and push the initial branch. Configure the default branch and protect it after CI is established.

**Acceptance gate:** a fresh clone contains no virtual environment or cache; `git status` is clean; licensing and project ownership are explicit; all checked-in results have traceable origins.

## Phase 1: Public Alpha Readiness

### Packaging and Installation

- [x] Declare Python >=3.11 and CI targets 3.11, 3.12, and 3.13. Hosted matrix execution remains pending.
- [x] Review dependency extras for core NumPy, development/JAX CPU, Metal, CUDA, and analysis dependencies.
- [x] Add project URLs, classifiers, keywords, and a `0.1.0` Alpha version; keep `uv.lock` for development only.
- [x] Build wheel and sdist, inspect package/license contents, and install the wheel outside the checkout.
- [x] Smoke the public import plus minimal MCRD and LPP APIs in an isolated environment.
- [x] Document the supported `mcrd_accel` import path; keep the public subpackage APIs described in the README examples.

**Acceptance gate:** `uv build` succeeds; wheel and sdist contain the intended package and license; clean-environment installation and the two README examples pass without relying on the source checkout.

### Continuous Integration and Repository Templates

- [x] Add a GitHub Actions CI workflow for Python 3.11, 3.12, and 3.13, including dependency sync, tests, spec validation, compile checks, and package build. Hosted CI has not run yet and remains a release gate.
- [ ] Add lint and formatting checks, then run them locally and fix all reported issues before making CI required.
- [ ] Add a lightweight dependency/security audit suitable for a scientific Python project. Document the policy for updating `uv.lock`.
- [x] Add issue templates for bug reports and feature requests, plus a pull request template covering tests, numerical impact, and backend/device evidence.
- [x] Add `CONTRIBUTING.md` with setup, test, benchmark, and pull request instructions.
- [x] Add a GitHub release checklist covering version bump, changelog, tests, package build, tag, and attached artifacts.

**Acceptance gate:** CI is green from a clean GitHub runner on every supported Python version; a failing test or build blocks merging to the default branch.

### README and User Documentation

- [x] Rewrite the README for a first-time GitHub visitor: problem and scope, Alpha status, supported backends, installation, minimal MCRD and LPP examples, validation, and links to detailed docs.
- [x] Replace `docs/backend-matrix.md` placeholder text with a table of implementation and validation status, including unverified CUDA artifact provenance.
- [x] Replace `docs/lpp-parity.md` placeholder text with the Python behavior map for coordinates/cropping, FFT convention, bins, peak/tie behavior, NaN and mask rules, filtering, edge handling, and unverified MATLAB differences.
- [ ] Remove stale `SciSim-Core` naming where it conflicts with the project identity, or document it as an internal umbrella name if it is intentional.
- [ ] Check every README command from a clean clone. Make optional dependencies explicit where a command needs them.
- [ ] Decide whether to include `CITATION.cff` now or at the first research release. If included, validate the citation metadata.

**Acceptance gate:** a researcher unfamiliar with the repository can install it, run each minimal workflow, identify tested backends, and understand the major scientific limitations without reading source code first.

## Phase 2: Numerical and Performance Evidence

This phase can proceed alongside the Alpha work, but its claims must not appear in release notes until the listed evidence exists.

### MCRD Validation

- [ ] Re-audit the Fig. 2b equation, parameter provenance, initialization, and boundary assumptions against the source notebook and executable OpenCL kernel. Keep disagreements explicit.
- [ ] Add a reproducible short-horizon fixture with fixed initial state and expected invariant/error tolerances.
- [ ] Confirm temporal and spatial refinement metrics use the intended norms, reference levels, grid ratios, and observed-order formula. Include tests where the expected order can be distinguished from zero-error or unstable cases.
- [ ] Run a longer, resource-bounded MCRD experiment that demonstrates the intended pattern-forming behavior, then record seed, full config, backend, runtime, and output provenance.
- [ ] Obtain and compare an OpenCL checkpoint if available. If none is available, keep the comparison labeled `not compared`; do not describe the placeholder report as parity evidence.
- [ ] Record peak resident memory and runtime using OS/process telemetry. Keep Python allocator estimates as a separate metric if useful.
- [ ] Explain what the current explicit Euler scheme does and does not establish. Do not generalize short-horizon convergence into a long-time accuracy claim.

### LPP Compatibility and Memory

- [ ] Request a small set of MATLAB golden outputs and representative source images. Store redistributable fixtures with provenance and expected outputs.
- [ ] Compare per-window coordinates and metrics separately from the merged output so a mismatch can be localized.
- [ ] Test edge policies, tied spectral peaks, empty bins, no-valid-window inputs, connected components, direction wraparound, and NaN behavior against MATLAB or a documented reference.
- [ ] Investigate the current 948 MB traced peak for a 256 x 256 benchmark image. Profile extraction, FFT batches, retained per-window metrics, and merge stages separately.
- [ ] Add a process RSS measurement and a scaling experiment over image size and batch size. Define a memory acceptance bound from measured data before calling the pipeline memory-bounded.
- [ ] Avoid retaining duplicate per-window result arrays where streaming or preallocation can preserve the public result contract at lower peak memory.

**Acceptance gate:** validation and parity artifacts can be regenerated from documented commands; artifacts include code revision/config/input provenance; resource claims use process/device measurements appropriate to the claim.

## Phase 3: Backend Support Claims

- [ ] NumPy: retain as the reference implementation and include it in required CI.
- [ ] JAX CPU: test the MCRD path in CI and document dtype/precision behavior. Add LPP JAX support only if it is implemented and tested, not because JAX exists in the environment.
- [ ] Apple Metal: install the optional extra on a supported Apple Silicon runner, verify device selection explicitly, compare results to the reference tolerances, and record device/runtime versions. Until then list it as unvalidated or unavailable.
- [ ] CUDA: use a CUDA-enabled runner or target host, verify the selected GPU, run numerical parity and memory checks, and record the exact GPU, driver, CUDA, and JAX versions. H100 and RTX 5090 are separate evidence entries.
- [ ] Ascend C/CANN: define the exact device and toolchain before implementation, add an isolated adapter, and require numerical and performance evidence before marking it supported.
- [ ] Ensure unavailable backend requests fail explicitly and never silently report a CPU fallback as accelerator execution.

**Acceptance gate:** each backend has a status, tested operation set, numerical tolerance, device/runtime provenance, and repeatable validation command. Unsupported backends remain clearly marked.

## Phase 4: First Public Release

- [ ] Confirm all Phase 0 and Phase 1 acceptance gates pass.
- [ ] Review the final diff, checked-in reports, licenses, and generated files. Verify that no original paper-code repository files were copied or changed without permission.
- [ ] Run the release checks from a clean clone: dependency sync, tests, lint, build, installation, and documented smoke examples.
- [ ] Update README release status and changelog with the actual tested scope and known limitations.
- [ ] Tag the first version as an Alpha (for example `0.1.0a1`) if API stability is not yet promised. Publish a GitHub release with source and validation summary.
- [ ] If publishing to PyPI, perform a TestPyPI upload and clean-environment install before the production upload. Keep PyPI publication separate from the GitHub repository release decision.
- [ ] After release, use issues for bugs and follow-up work; do not silently change numerical semantics without tests and release notes.

**Acceptance gate:** a new user can follow the release README on a clean supported machine, reproduce the smoke checks, and see exactly which scientific and backend claims are supported.

## Suggested Execution Order

1. Finish Phase 0 repository identity, ignore rules, license, artifact provenance, and first reviewed commit.
2. Establish CI, package build/install checks, linting, and contribution/release templates.
3. Replace placeholder docs and validate all README examples.
4. Investigate LPP peak memory and correct or qualify the bounded-memory claim.
5. Strengthen MCRD and LPP reference evidence without blocking an accurately labeled Alpha on unavailable hardware.
6. Publish an Alpha only after Phase 0 and Phase 1 acceptance gates pass.
7. Add Metal, CUDA, and Ascend support only through their separate evidence gates.

## Out of Scope for the First Public Alpha

- Claiming validated support for H100, RTX 5090, Metal, or Ascend C without a real device run.
- Claiming exact MATLAB LPP parity before golden outputs are available.
- Claiming OpenCL trajectory parity when no reference checkpoint exists and the legacy kernel updates in place.
- Promising a stable API or long-term support policy before maintainers and release cadence are established.
- Rewriting the original paper-code repository as part of publishing this new project.

## Project Decisions to Record Before Release

- Canonical GitHub URL and repository visibility.
- Maintainer or organization name and contact channel.
- Distribution name and whether/when to publish to PyPI.
- Declared supported Python versions and operating systems.
- Which benchmark/validation artifacts are committed, and the required provenance fields.
- Whether this project promises research reproducibility, an experimental toolkit, or both in the first release.
- API stability policy and versioning convention.
