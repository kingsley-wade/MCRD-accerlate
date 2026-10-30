# Backend Support Matrix

Backend status describes evidence in this repository. Installing an optional dependency or detecting a device does not by itself mean that every operation is supported.

| Backend | MCRD | Canonical PDE cases | LPP | Evidence and limits |
| --- | --- | --- | --- | --- |
| NumPy CPU | Implemented and tested | Implemented and tested | Implemented and tested | Reference path. Full unit suite runs locally. LPP is NumPy-only. |
| JAX CPU | Implemented and locally tested | Implemented and locally tested | Not implemented | Local Apple Silicon host used JAX CPU; results are in the test suite and `reports/benchmark_mcrd.json`. |
| JAX Metal | Adapter path exists; device unvalidated | Adapter path exists; device unvalidated | Not implemented | Metal runtime/device was unavailable in the local validation environment. Do not infer Metal support from JAX installation. |
| JAX CUDA | Implemented in MCRD and canonical PDE code; H100 MIG validation and 64²–4096² p53 run recorded | Heat, periodic Poisson, wave, and p53 smoke cases recorded through 128² | Not implemented | `reports/h100-validation-20261030/` records parity and an earlier NumPy comparison. `reports/acp-h100-20261030T070325Z/` records the float64 t=40 p53 grid study; `reports/p53-cpu-vs-h100-20261030T134645Z/` records matched short-run timings against an 8-core pinned JAX CPU. Performance claims apply only to each recorded configuration. |
| Ascend C/CANN | Not implemented | Not implemented | Not implemented | Future adapter target only. No Ascend device or toolchain has been validated. |

## Explicit Device Routing

JAX device selection is explicit. `jax-cpu` requests a CPU device, `jax-cuda` requests a GPU device, and `jax-metal` requests a Metal device. An unavailable requested device raises an error instead of silently substituting CPU.

The 2026-10-30 reports were produced on an NVIDIA H100 3g.40gb MIG instance with JAX 0.12.0.1. The matched CPU/H100 short-run benchmark uses float64, 10,240 steps per grid, one warmup, and three timed repeats. The separate fixed-time grid study runs p53 through 4096² to t=40. The historical OpenCL runtime was recorded on an Apple M2 Ultra and is not a same-device or same-configuration speedup comparison. See [runtime comparison](runtime-comparison.md) for the protocols and limitations.

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
