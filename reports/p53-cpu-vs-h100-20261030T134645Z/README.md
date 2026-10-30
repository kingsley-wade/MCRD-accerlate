# p53 CPU versus H100 benchmark

- CPU: cpu with [0, 1, 2, 3, 4, 5, 6, 7] pinned cores
- H100: NVIDIA H100 80GB HBM3
- Matched JAX/JAXlib: 0.12.0.1/0.12.0.1; float64
- Steps per grid: 10240; dt: 2.87224264706e-05
- Speedup is CPU median time divided by H100 median time; values above 1 mean H100 was faster.

| Grid | CPU median (s) | H100 median (s) | H100 speedup |
|---:|---:|---:|---:|
| 64² | 0.1892 | 0.1164 | 1.63x |
| 128² | 0.7290 | 0.1166 | 6.25x |
| 256² | 5.2434 | 0.1174 | 44.65x |
| 512² | 27.3653 | 0.1160 | 235.94x |
| 1024² | 113.1891 | 0.3032 | 373.29x |
| 2048² | 130.7076 | 1.1001 | 118.81x |
| 4096² | 603.6509 | 4.1021 | 147.16x |

## 4096² output parity

- u max absolute error: 5.32907e-15; relative RMS: 1.28068e-16
- v max absolute error: 7.10543e-15; relative RMS: 8.96101e-17
