# Numerical audit: ACP H100 run 2026-10-30T07:03:25Z

## Run and environment

The ACP job completed all phases. Its manifest records JAX/JAXlib 0.12.0.1,
float64 for the p53 grid study, an NVIDIA H100 device, a 25.6 x 25.6 physical
region, and a final simulated time of 40.00000000000001. Each grid used
1,392,640 steps.

## p53 MCRD grid study

- All grids from 64² through 4096² were finite and reached t=40.
- Relative mass drift ranged from 5.6e-16 to 5.1e-14.
- Pairwise spatial errors decreased consistently. The reported observed orders
  for both fields are approximately 1.998 to 2.000, consistent with the
  second-order spatial stencil.
- A one-step H100/JAX float64 continuation changed 16,777,214 of 16,777,216
  cells in each field, confirming the float64 trajectory did not freeze at
  machine precision.
- The relative-change threshold was first satisfied for three consecutive
  windows near t=4.41, but later windows rose above the threshold; by t=40 the
  measured relative change was about 1.08e-5. Treat this as a temporary
  threshold crossing, not proof of convergence or a valid adaptive stop time.

## Canonical PDE smoke

- Heat equation: maximum Fourier-mode error was 3.43e-6 and the largest
  explicit stability number was 0.0328.
- Periodic Poisson: all grids reported convergence; relative residuals ranged
  from 1.08e-6 to 8.79e-6.
- Wave equation: maximum Fourier-mode error was 3.21e-6 and 128² relative
  energy drift was 1.51e-7.
- These are small-grid correctness checks, not CPU-versus-GPU speedup results.

## Transfer and checksum note

The ACP-generated SHA256 manifest included `job.log` before the shell exit trap
appended its final status lines, so its `job.log` entry is stale. All reported
scientific artifacts matched their remote checksums after SCP. The submission
script has been updated to checksum stable outputs only and to use relative
paths.

## Interpretation

This float64 p53 run is suitable to report as a fixed-time H100 grid study with
approximately second-order spatial convergence for the chosen smooth initial
field. The adaptive monitor did not establish a persistent steady state by
t=40. A separate identical-input CPU-versus-H100 timing study is still needed
for a quantitative speedup claim.
