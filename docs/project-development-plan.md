# Project Development Plan

## Project Goal

Migrate the original DNA-p53 reaction-diffusion and Local Pattern Properties workflows into a cleaner, faster, reproducible Python project that can be published as a GitHub portfolio project.

The portfolio story is the engineering migration itself: identify behavior in legacy notebook/OpenCL/MATLAB code, implement explicit numerical and image-analysis contracts, test them, measure performance fairly, and document the limits of the comparison. The project should be useful to another researcher while giving a reviewer enough evidence to assess the work.

The first public release is an experimental Alpha. It does not need to claim perfect MATLAB/OpenCL parity or support every accelerator. It does need to install cleanly, pass CI, show one reproducible end-to-end MCRD run and one LPP run, and make any unverified claims obvious.

## Portfolio Acceptance Criteria

The release is ready to feature on a resume when all of the following are true:

- A new user can clone the repository, install a documented extra, run tests and both examples, and build a wheel from a clean machine.
- The README presents the migration problem, design choices, supported backends, one correctness result, and one reproducible performance comparison.
- CI verifies the supported Python versions and prevents broken builds from merging.
- Numerical and performance results include the code revision, complete config, dtype, backend/device, runtime versions, and an honest memory measure.
- The implementation distinguishes measured results from planned NVIDIA/Metal/Ascend support.
- A reviewer can see which behavior came from legacy source inspection and which parity claims have external fixture evidence.
- Git history, license, ownership, and contribution instructions make the repository maintainable by someone other than its author.

## Current Snapshot

Observed on 2026-10-28:

- Implementation exists for the NumPy reference path, JAX MCRD/PDE paths, and NumPy LPP pipeline.
- The local Apple Silicon test suite passes: 146 tests after the current fixes.
- The wheel and sdist build successfully; a wheel installed in an isolated temporary environment and passed small MCRD and LPP API smoke checks.
- `uv lock --check` passes.
- A seven-level spatial report, four-level time report, local MCRD benchmark, LPP benchmark, and older CUDA smoke artifact exist under `reports/`.
- The older CUDA artifact is labeled `jax-cuda` but has no host/device/JAX provenance. Do not use it as attributable H100 evidence.
- The full source, tests, documentation, specs, and reports are uploaded under `/mnt/afs/gmt/DNA-p53-MCRD-accelerated`; the upload archive is `/mnt/afs/gmt/DNA-p53-MCRD-accelerated-full-project.tgz`. The isolated environment and uv cache are also targeted to `/mnt/afs/gmt`.
- The remote host is Ubuntu 22.04 x86_64 with eight NVIDIA H100 80 GB devices. At the latest check, each had about 29 GiB free and 52 GiB used by other jobs. Do not terminate or compete aggressively with those jobs.
- The remote base Python is 3.10, below the project minimum. Python 3.12 and `uv` are available. `uv sync --locked --extra jax-cuda --python 3.12` resolved and began downloading the locked CUDA stack, but made negligible progress for more than ten minutes; the process was stopped. The CUDA environment is incomplete and no new H100 result has been generated.
- `/mnt/afs` showed about 87 GiB free at the latest check. There are no visible Slurm commands in this account/session.
- Initial commit `4dbb014` is pushed to `https://github.com/kingsley-wade/MCRD-accerlate` on `main`. The repository is private, the local working tree is clean, and the first hosted CI run passed on Python 3.11, 3.12, and 3.13.
- Author metadata now uses `kingsley-wade`; an MIT license, `.gitignore`, CI workflow, templates, contribution guide, backend matrix, LPP behavior guide, and release roadmap have been added locally.

## Known Bugs and Risks

### Fixed in the current working tree

- **LPP backend argument was ignored.** `analyze(..., backend="jax")` silently ran NumPy. Unsupported names now fail explicitly, with a regression test.
- **LPP connected-component filtering silently disappeared without SciPy.** A missing optional dependency now raises an actionable error instead of returning unfiltered validity pixels. The default config requests filtering, so users should install the `analysis` extra for the default LPP behavior.
- **JAX Poisson iteration count used `-1` as an unknown sentinel.** The result now uses `None`; the JAX smoke report generator records this as JSON `null`.

### Still open

- **LPP peak memory needs a valid measurement.** The current `tracemalloc` report shows about 948 MB for a 256 x 256 float64 image with batch size 64. Profile with process RSS and vary batch size before making a bounded-memory or throughput claim. The implementation bounds FFT batches but keeps per-window metric arrays until merge.
- **H100 report provenance is incomplete.** The previous CUDA dependency installation stalled against the package index. Retry the isolated install in `/mnt/afs/gmt` when package downloads are viable, then generate a new report with device/runtime metadata. The GPU is shared and already heavily allocated; run a small smoke case only after checking available memory.
- **LPP MATLAB parity lacks golden fixtures.** The current tests establish synthetic behavior, not exact parity on experimental images.
- **OpenCL comparison has no reference output.** Keep it labeled un-compared until a checkpoint is supplied.
- **Package names intentionally differ.** The distribution is `mcrd-accel`, the import package is `mcrd_accel`, and the GitHub repository is `MCRD-accerlate`; the README and package metadata document these names.
- **The license file exists, but the final redistribution audit is still required.** Check the origin and licensing of copied code, data, and images before public push.
- **The target repository is still private.** Do not change its visibility without explicit instruction; finish the redistribution audit before proposing a public release.
- **Branch protection is not configured.** Confirm branch and repository settings before treating hosted CI as a merge gate.

## Development Phases

### Phase A: Finish the Migration Demonstration

Goal: make the actual code migration easy to understand and reproduce.

- [x] A1. Re-run the full local suite after current fixes and record the final count (146 passed on 2026-10-28).
- [ ] A2. Run the README MCRD and LPP examples in a clean environment with the documented extras.
- [ ] A3. Keep the fixed short-run conservation, convergence, and stability reports reproducible from commands in the README.
- [ ] A4. Create a small, deterministic MCRD example with a fixed seed and save a compact output/visualization only if its provenance and generation command are recorded.
- [ ] A5. Add a migration note mapping the old OpenCL/MATLAB stages to the Python modules and explaining deliberate changes: race-free double buffering, periodic full-grid boundary treatment, batched FFT, and sum/count director merge.
- [ ] A6. Avoid phrases like “same results as MATLAB/OpenCL” unless a fixture comparison actually demonstrates it.

**Acceptance:** a reviewer can trace each major legacy stage to new code and run both workflows from the documented commands.

### Phase B: Correctness and Resource Evidence

- [ ] B1. Regenerate the local reports after the Poisson `iterations=None` change; confirm JSON uses `null` and no prior `-1` result is described as current.
- [ ] B2. Complete the H100 environment setup in `/mnt/afs/gmt/DNA-p53-MCRD-accelerated/.venv`. Resolve the stalled CUDA package download, then confirm `jax.devices("gpu")`, exact device kind, JAX/jaxlib versions, driver/runtime, and one device selection. Follow [H100 validation](h100-validation.md).
- [ ] B3. Run only the 16/32 grid canonical PDE smoke and a short MCRD smoke on one H100 after checking free VRAM. Do not run the full 4096 grid or a throughput sweep while shared allocations remain high.
- [ ] B4. Save a new H100 JSON report with timestamp, host, GPU model, device index, free/used VRAM before and after, driver, Python, JAX/jaxlib, config, and all result metrics. Replace or clearly archive the provenance-incomplete report.
- [ ] B5. Compare JAX CUDA outputs with NumPy on the exact same inputs and tolerances. Report which equations and cases were exercised.
- [ ] B6. Add a process RSS measurement to MCRD/LPP benchmark records. Clearly separate RSS, Python allocator, and accelerator memory.
- [ ] B7. Profile LPP by stage and test at least two batch sizes. Fix avoidable copies or reduce the default batch if evidence shows it materially lowers peak memory without unacceptable throughput loss.
- [ ] B8. Keep the performance comparison reproducible and modest: same input/config, warmup separated, multiple repeats, device and runtime metadata included.

**Acceptance:** the H100 report is attributable and repeatable; LPP memory claims match RSS measurements; no shared processes were disturbed.

### Phase C: GitHub Repository Quality

- [x] C1. Add `.gitignore` for environments, caches, local files, and build outputs.
- [x] C2. Add an MIT license using the supplied GitHub username; finish the redistribution audit before push.
- [x] C3. Replace backend and LPP parity placeholders with current capability and behavior documents.
- [x] C4. Add a Python 3.11/3.12/3.13 GitHub Actions workflow for sync, tests, spec validation, compile checks, and build.
- [x] C5. Add issue forms, pull request template, and `CONTRIBUTING.md`.
- [x] C6. Document the repository slug (`kingsley-wade/MCRD-accerlate`), distribution name (`mcrd-accel`), supported Python versions (3.11–3.13 in CI), and public API import path (`mcrd_accel`).
- [x] C7. Run CI on GitHub after authentication is restored. The first push passed Python 3.11, 3.12, and 3.13 test/spec/compile/build jobs.
- [x] C8. Confirm the source distribution and wheel contain the package and license, and install the wheel outside the checkout. Local build/install smoke passed for the current `0.1.0` metadata; package contents were inspected on 2026-10-28.
- [ ] C9. Review every checked-in report and image. Keep only artifacts that are small, attributable, redistributable, and useful to a portfolio reviewer.
- [ ] C10. Add a concise changelog/release note and (optionally) `CITATION.cff` once maintainer and citation details are known.
- [x] C11. Create the first reviewed Git commit (`4dbb014`); local working tree is clean.
- [x] C12. Authenticate `gh`, push the initial commit to the existing private repository, and verify the `main` branch plus hosted CI. Keep the repository private until the redistribution audit and public release gates pass.

**Acceptance:** public repository has a clean clone, visible green CI, accurate ownership/license, useful contribution instructions, and a release page whose claims match evidence.

### Phase D: Portfolio Presentation and Release

- [ ] D1. Lead the README with the engineering problem and what the migration implements, not a generic “high performance” claim.
- [ ] D2. Include a compact architecture diagram or module map and links to numerical methods, LPP behavior, and backend support.
- [ ] D3. Include one correctness table (conservation, refinement, synthetic LPP behavior) and one benchmark table with full configuration and memory caveat.
- [ ] D4. Add a simple demo image only when generated from a checked-in script/config and labeled as an example result.
- [ ] D5. State Alpha/API stability, tested OS/Python/backend combinations, and missing MATLAB/OpenCL fixtures in the README.
- [ ] D6. Use an Alpha tag such as `0.1.0a1` only after CI and clean install pass. Keep PyPI publication optional and separate from the GitHub release.
- [ ] D7. Draft two truthful resume bullet variants after final results exist: one focused on solver migration/validation and one on batched image analysis/performance. Use only measured numbers with machine/config context.

**Acceptance:** a hiring reviewer can understand the problem, implementation, measured result, and evidence limitations in a few minutes.

### Phase E: Optional Research and Accelerator Follow-up

- [ ] E1. Request redistributable MATLAB outputs and representative microscopy images; add fixture tests only after rights/provenance are clear.
- [ ] E2. Request a matching legacy OpenCL checkpoint and compare invariants and field norms while explaining in-place race and boundary differences.
- [ ] E3. Validate Apple Metal on supported hardware, CUDA on H100 and RTX 5090 independently, and Ascend C only after the exact device/toolchain is known.
- [ ] E4. Keep LPP accelerator implementation separate until a measured bottleneck and backend contract justify it.
- [ ] E5. Consider a higher-order integrator only after preserving explicit Euler compatibility and adding convergence/stability evidence.

**Acceptance:** each optional capability has its own test report and remains unclaimed until that evidence exists.

## Dependency Order

```text
Phase A migration demo
  -> Phase B correctness/resource evidence
  -> Phase C GitHub quality and CI
  -> Phase D public portfolio release
  -> Phase E optional research/device support
```

Phase C repository housekeeping can proceed alongside Phase A/B. Do not block an honest Alpha on unavailable MATLAB fixtures or RTX/Ascend hardware; do block it on a failing clean install, absent CI, unclear licensing, missing provenance, or misleading claims.

## Release Commands

Run from a clean clone after CI is added:

```bash
uv sync --extra dev
uv run pytest -q
uv run python scripts/validate_specs.py
uv run python scripts/validate_mcrd.py
uv run python benchmarks/run_mcrd.py --case p53_mcrd --grid-sizes 64,128 --steps 1000
uv run python benchmarks/run_lpp.py --size 512 --window 64 --stride 8
uv build
```

For the documented LPP default connected-component filter, install `uv sync --extra analysis`. The LPP API now raises an actionable error if that optional SciPy dependency is missing while filtering is requested.

## External Decisions and Access

- GitHub username provided: `kingsley-wade`.
- Canonical repository slug is `kingsley-wade/MCRD-accerlate`; the private repository exists and is empty, awaiting its first push.
- GitHub CLI authentication is active after user device authorization; the initial push is complete.
- No MATLAB golden fixtures, original OpenCL checkpoint, RTX 5090, or Ascend C device are currently available in this workspace/session.
