# Backend Support Matrix

Backend status describes evidence in this repository. Installing an optional dependency or detecting a device does not by itself mean that every operation is supported.

| Backend | MCRD | Canonical PDE cases | LPP | Evidence and limits |
| --- | --- | --- | --- | --- |
| NumPy CPU | Implemented and tested | Implemented and tested | Implemented and tested | Reference path. Full unit suite runs locally. LPP is NumPy-only. |
| JAX CPU | Implemented and locally tested | Implemented and locally tested | Not implemented | Local Apple Silicon host used JAX CPU; results are in the test suite and `reports/benchmark_mcrd.json`. |
| JAX Metal | Adapter path exists; device unvalidated | Adapter path exists; device unvalidated | Not implemented | Metal runtime/device was unavailable in the local validation environment. Do not infer Metal support from JAX installation. |
| JAX CUDA | Implemented in MCRD and canonical PDE code; smoke evidence needs provenance | Smoke artifact present | Not implemented | `reports/s1_pde_smoke_h100.json` contains successful cases labeled `jax-cuda`, but omits host/device/runtime identity. An H100 cluster is reachable; a fresh, attributable project run is still required before claiming H100 validation. |
| Ascend C/CANN | Not implemented | Not implemented | Not implemented | Future adapter target only. No Ascend device or toolchain has been validated. |

## Explicit Device Routing

JAX device selection is explicit. `jax-cpu` requests a CPU device, `jax-cuda` requests a GPU device, and `jax-metal` requests a Metal device. An unavailable requested device raises an error instead of silently substituting CPU.

The project is uploaded to `/mnt/afs/gmt/DNA-p53-MCRD-accelerated` on the reachable Ubuntu 22.04 x86_64 host. It has eight H100 80 GB devices, but other jobs used about 52 GiB on each at the latest check. A locked JAX CUDA install under the project directory stalled while downloading from the package index and was stopped before any GPU run. The environment is incomplete; no new H100 result exists yet. The `/mnt/afs` volume had about 87 GiB free. No Slurm commands were available in the login environment. Retry installation and check available device memory before running the small smoke; do not terminate other users' processes.

## What Each Status Means

- `Implemented`: code path exists in this repository.
- `Tested`: automated tests exercise the path.
- `Locally tested`: a recorded command completed on the named local device/runtime.
- `Smoke evidence`: a small run exists but does not establish broad numerical correctness or performance.
- `Unvalidated`: no attributable end-to-end run exists for that device/backend combination.
- `Not implemented`: the package has no execution path for the operation/backend combination.

## Requirements for a Backend Claim

Before marking a backend `locally tested`, attach a report with the exact device model, device count, driver/runtime, Python/JAX/jaxlib versions, dtype, configuration, and command. The report must include a numerical comparison against the NumPy reference within declared tolerances and the correct device name returned by JAX. Benchmark claims additionally require warmup-separated timings and a memory metric appropriate to the device.

H100 and RTX 5090 are separate targets. CUDA test evidence on one GPU does not establish performance or compatibility on the other. Ascend C requires its own adapter, CANN/Ascend C versions, device identity, correctness checks, and measurements.
