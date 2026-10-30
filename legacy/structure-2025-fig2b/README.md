# Original Structure paper Fig. 2b experiment

This directory preserves the original, non-accelerated OpenCL/Jupyter implementation of the DNA-p53 MCRD experiment that this package migrates. The notebook and OpenCL helper were copied without modification from the source repository at commit `003d95efbe911761d90d094a9cb5f2e33e1a3a9d`:

- `Fig2_b_sim.ipynb` — Fig. 2b simulation notebook.
- `SpatialFunctions_iPy.cl` — OpenCL spatial operators loaded by the notebook.

The notebook expects `SpatialFunctions_iPy.cl` in its working directory. It was written for an interactive Jupyter environment with PyOpenCL and an OpenCL device; it is retained as historical reference code and is not part of the accelerated Python package or its automated tests.

## Paper and source

Cheng Li, Man-Ting Guo, Xiaoqing He, Quan-Xing Liu, and Zhi Qi, “Modeling phase separation of biomolecular condensates with data-driven mass-conserving reaction-diffusion systems,” *Structure* (2025). [https://doi.org/10.1016/j.str.2025.05.018](https://doi.org/10.1016/j.str.2025.05.018)

- Original paper-code repository: [liuqx315/DPICs-MCRD-Phase-separation](https://github.com/liuqx315/DPICs-MCRD-Phase-separation/tree/main/fig2/Fig2b_code)
- Figshare archive: [10.6084/m9.figshare.26057656](https://doi.org/10.6084/m9.figshare.26057656)

The OpenCL helper retains its original 2017 copyright notice for Johan Van de Koppel. The upstream repository does not provide a general license file; these reference files are not relicensed under this project's MIT license. The accelerated package and its newly written migration code remain covered by the root project license.

## Relation to the migration

The accelerated implementation follows the executable Fig. 2b reaction law and parameter values. It uses synchronous out-of-place updates and periodic array operations, while the original OpenCL kernel updates shared fields in place and uses ghost-cell boundaries. See [`docs/numerical-methods.md`](../../docs/numerical-methods.md), the [OpenCL migration guide](../../docs/opencl-migration-guide.md), and the [runtime comparison](../../docs/runtime-comparison.md) for the documented differences and evidence limits.
