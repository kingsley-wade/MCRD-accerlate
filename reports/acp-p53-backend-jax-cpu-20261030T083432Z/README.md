# Matched p53 MCRD backend benchmark

- Backend: jax-cpu (cpu)
- Dtype: float64; JAX/JAXlib: 0.12.0.1/0.12.0.1
- CPU affinity: [0, 1, 2, 3, 4, 5, 6, 7]
- Region: 25.6 x 25.6
- Grid sizes: 64, 128, 256, 512, 1024, 2048, 4096
- Steps per grid: 10240; dt: 2.87224264706e-05; simulated time: 0.29411765
- Timed repeats: 3; warmups: 1

Use benchmark.csv for steady medians. The first run includes compilation and is reported separately.
The 4096² final state is saved for CPU/H100 numerical comparison.
