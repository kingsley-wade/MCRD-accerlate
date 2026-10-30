# Original Structure-paper source code

This directory preserves the original, non-accelerated source-code files for the Structure paper's figure workflows (Figures 1–6). It mirrors the source paths from [`liuqx315/DPICs-MCRD-Phase-separation`](https://github.com/liuqx315/DPICs-MCRD-Phase-separation) at source commit `003d95efbe911761d90d094a9cb5f2e33e1a3a9d`.

The snapshot contains the original MATLAB `.m` files, OpenCL `.cl` helpers, Jupyter `.ipynb` notebooks, and the LPP `ReadMe.txt`. [`UPSTREAM_README.md`](UPSTREAM_README.md) preserves the source repository's project description. The simulation datasets, input workbooks, generated plots, and other non-code assets are not copied here; obtain those from the [upstream repository](https://github.com/liuqx315/DPICs-MCRD-Phase-separation) or the [Figshare archive](https://doi.org/10.6084/m9.figshare.26057656).

## Paper

Cheng Li, Man-Ting Guo, Xiaoqing He, Quan-Xing Liu, and Zhi Qi, “Modeling phase separation of biomolecular condensates with data-driven mass-conserving reaction-diffusion systems,” *Structure* (2025). [https://doi.org/10.1016/j.str.2025.05.018](https://doi.org/10.1016/j.str.2025.05.018)

The source files are preserved as a reference snapshot and retain their original notices. The upstream repository does not include a general license file, so these files are not relicensed under this package's MIT license. In particular, the OpenCL spatial helper retains its 2017 copyright notice for Johan Van de Koppel.

## What this accelerated project covers

The Python package in `src/mcrd_accel/` is a selective migration of the DNA-p53 Fig. 2b reaction-diffusion solver, not an accelerated rewrite of all Figure 1–6 workflows. Its MCRD solver has a NumPy reference backend and JAX CPU/CUDA execution paths. The LPP module is a Python/NumPy implementation; it is not GPU accelerated. MATLAB analysis scripts and the other figure-specific OpenCL notebooks remain original reference code and have not been ported here.

For the Fig. 2b solver's equation, parameter, boundary, and update differences, see [`docs/numerical-methods.md`](../../docs/numerical-methods.md) and the [OpenCL migration guide](../../docs/opencl-migration-guide.md). The source and measured migration evidence are summarized in the [runtime comparison](../../docs/runtime-comparison.md).
