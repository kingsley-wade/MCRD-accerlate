# H100 p53 MCRD fixed-domain grid study

- Physical region: 25.6 x 25.6
- Target time: 40
- Grid sizes: 64, 128, 256, 512, 1024, 2048, 4096
- Backend: JAX CUDA (NVIDIA H100 80GB HBM3), float64
- Shared time step: 2.872242647e-05; 1,392,640 steps at every grid size
- Adaptive stop check: relative RMS change < tolerance for consecutive windows; diagnostic only, all cases continue to the target time.
- The spatial comparison uses the same seeded smooth periodic six-mode initial field sampled on each grid.

Machine-readable timings and diagnostics are in p53-grid-study.csv and p53-grid-study.json.
Figures: p53-t40-4096-heatmap.png and p53-evolution-4096-t0-to-t40.png.
The compressed 4096² final field is retained locally as a raw artifact and omitted from the GitHub source repository because it is over 200 MB. Recreate it with `scripts/run_p53_grid_study.py`; the CSV, JSON, figures, and manifest are the shareable report artifacts.

This is a device-backed numerical grid study, not a CPU-versus-GPU speedup comparison.
