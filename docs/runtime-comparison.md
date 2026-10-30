# Runtime comparison: OpenCL and JAX

This page puts the available OpenCL, JAX CPU, and JAX CUDA timing evidence in one place. The OpenCL record and JAX records come from different devices and numerical settings. Treat the cross-system comparisons below as runtime references, not controlled backend speedup measurements.

## Reference workload and available timings

The original Fig. 2b OpenCL notebook records one 256² run on an Apple M2 Ultra. Its `endtime=30000` is a requested simulated-time horizon, not 30,000 iterations. With `dt=0.02` and `nplot=100`, it computes 15,000 inner steps per output interval and executes 99 × 14,999 = 1,484,901 kernel launches. The recorded runtime is 168.1 seconds. The nominal integrated time is 29,698.02 because the loops stop before the final nominal output interval.

| Backend and device | Grid | Steps | Precision | Runtime | Record type |
|---|---:|---:|---|---:|---|
| Legacy OpenCL, Apple M2 Ultra | 256² | 1,484,901 | float32 | 168.1 s | One recorded run |
| JAX CPU, ACP host with 8 pinned cores | 256² | 10,240 | float64 | 5.243 s | Median of 3 warm runs; 1 warmup |
| JAX CPU, same benchmark scaled to OpenCL step count | 256² | 1,484,901 | float64 | ~760 s | Linear estimate, not run |
| JAX CUDA, H100 MIG, full grid study | 256² | 1,392,640 | float64 | 17.95 s total; 16.84 s solver | Recorded run, close in step count |
| JAX CUDA, same grid scaled to OpenCL step count | 256² | 1,484,901 | float64 | ~19 s total | Linear estimate, not run |

The JAX CPU estimate scales the measured 5.243 seconds for 10,240 steps by `1,484,901 / 10,240`. For CUDA, the estimate scales the recorded 256² H100 full-study runtime by `1,484,901 / 1,392,640`. The matched short benchmark gives a similar steady-state CUDA estimate of about 17 seconds before full-run diagnostics.

The observed OpenCL-to-H100 runtime ratio is therefore roughly 9× for these near-equal grid/update counts. The JAX CPU estimate is about 4.5× longer than the OpenCL measurement. Both ratios cross devices: OpenCL ran on an M2 Ultra, JAX CPU on an ACP x86 host, and JAX CUDA on an H100 MIG slice. They are useful context, not portable performance claims.

## Why these are not an apples-to-apples solver benchmark

The OpenCL run used `dx=dy=0.2`, `dt=0.02`, float32, a random unseeded initial field, and a nominal simulated time near 30,000. The JAX H100 grid study used a 25.6 × 25.6 domain, `dx=dy=0.1` at 256², `dt≈2.87224e-5`, float64, a seeded smooth six-mode initial field, and t=40. The JAX CPU timing used the same short-run input and settings as its H100 timing, but only 10,240 steps per grid.

The legacy OpenCL kernel updates neighboring values in place, so its result can depend on work-item scheduling. JAX updates from a complete old state into a separate new state. This can affect numerical results and makes exact state parity an invalid assumption.

The original notebook does not contain a 4096² run. It sets `n=256`; its `endtime=30000` value is simulated time. A 4096², 30,000-step OpenCL time of roughly 14.5 minutes can be estimated by scaling the recorded run by grid-cell count and step count:

```text
168.1 s × (4096 / 256)² × 30000 / 1,484,901 ≈ 869 s ≈ 14.5 min
```

That estimate assumes constant site-update throughput; it is not a measured 4096² OpenCL result. The original notebook also allocates 100 snapshots for each species with NumPy's default float64. At 4096², `Us` and `Vs` together would need about 26.8 GB of host memory for those snapshots alone.

## Matched JAX CPU versus H100 benchmark

This is the cleanest current speedup result because CPU and H100 used the same JAX/JAXlib versions, float64 inputs, seed, smooth initial fields, grid series, timestep, and 10,240 steps per grid. The CPU process was pinned to 8 cores; each case used 1 warmup and 3 timed repeats. Values are steady-state medians.

| Grid | JAX CPU (s) | JAX CUDA H100 (s) | CPU/H100 speedup |
|---:|---:|---:|---:|
| 64² | 0.1892 | 0.1164 | 1.63× |
| 128² | 0.7290 | 0.1166 | 6.25× |
| 256² | 5.2434 | 0.1174 | 44.65× |
| 512² | 27.3653 | 0.1160 | 235.94× |
| 1024² | 113.1891 | 0.3032 | 373.29× |
| 2048² | 130.7076 | 1.1001 | 118.81× |
| 4096² | 603.6509 | 4.1021 | 147.16× |

These timings cover 10,240 steps, or about 0.294 simulated time units. They do not represent the complete t=40 CPU-versus-H100 workflow. The exact reports are [`CPU benchmark JSON`](../reports/acp-p53-backend-jax-cpu-20261030T083432Z/benchmark.json), [`H100 benchmark JSON`](../reports/acp-p53-backend-jax-cuda-20261030T134645Z/benchmark.json), and the [comparison summary](../reports/p53-cpu-vs-h100-20261030T134645Z/README.md).

A separate earlier NumPy-versus-JAX CUDA benchmark on the H100 MIG slice measured 128² and 256², float32, 200 steps, two warmups, and five repeats. Its 9.26× and 27.90× ratios answer a different question from the JAX CPU-versus-H100 table; see the [H100 validation report](h100-validation.md).

## Reproduction image and data

The [1024² JAX CUDA speckled-field figure](../reports/opencl-style-1024-reference-init/p53-random-speckle-1024-t40.png) uses a seeded random field generated at the 128², `dx=0.2` reference scale and mapped to 1024². Its measured solver runtime is 2.77 seconds on the recorded H100 MIG device. This is a visual reproduction workload, not a matched OpenCL timing case. The run's [manifest](../reports/opencl-style-1024-reference-init/run-manifest.json) records its exact settings. The final state NPZ is retained locally but omitted from Git because it is a generated data artifact.
