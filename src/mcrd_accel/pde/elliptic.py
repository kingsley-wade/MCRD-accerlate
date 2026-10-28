"""Periodic Poisson equation using matrix-free conjugate gradients."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..backends import JAX_BACKENDS, get_jax_device
from ..config import Backend
from ..mcrd.operators import periodic_laplacian
from .config import GridConfig
from .operators import periodic_laplacian_jax, require_field_shape


@dataclass
class PoissonResult:
    solution: object
    residual_norm: float
    relative_residual: float
    iterations: int | None
    converged: bool
    rhs_mean: float
    runtime_seconds: float


def solve_periodic_poisson(
    grid: GridConfig,
    rhs: NDArray[np.floating],
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> PoissonResult:
    """Solve ``-Laplace(u) = rhs`` with a zero-mean gauge.

    The periodic problem is solvable only for a zero-mean right-hand side. A
    non-compatible right-hand side is rejected instead of silently projected.
    """
    require_field_shape(rhs, grid.shape, "rhs")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")
    rhs = np.asarray(rhs, dtype=grid.numpy_dtype)
    rhs_mean = float(np.mean(rhs, dtype=np.float64))
    rhs_norm = float(np.linalg.norm(rhs.ravel()))
    compatibility_tol = tolerance * max(1.0, rhs_norm)
    if abs(rhs_mean) > compatibility_tol:
        raise ValueError(f"periodic Poisson rhs must have zero mean; mean is {rhs_mean:.6g}")
    rhs = rhs - rhs_mean

    start = time.perf_counter()
    if grid.backend is Backend.NUMPY:
        try:
            from scipy.sparse.linalg import LinearOperator, cg
        except ImportError as exc:
            raise ImportError(
                "NumPy periodic Poisson requires SciPy; install mcrd-accel[analysis]"
            ) from exc

        size = int(np.prod(grid.shape))

        def matvec(vector):
            field = vector.reshape(grid.shape)
            return (-periodic_laplacian(field, grid.dx, grid.dy)).ravel()

        operator = LinearOperator((size, size), matvec=matvec, dtype=grid.numpy_dtype)
        iterations = 0

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            operator,
            rhs.ravel(),
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
            callback=count_iteration,
        )
        solution = solution.reshape(grid.shape)
        solution -= np.mean(solution, dtype=np.float64)
        residual = -periodic_laplacian(solution, grid.dx, grid.dy) - rhs
        solver_converged = info == 0
        runtime = time.perf_counter() - start
    elif grid.backend in JAX_BACKENDS:
        import jax
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import cg as jax_cg

        device = get_jax_device(grid.backend)
        rhs_device = jax.device_put(jnp.asarray(rhs), device)

        def matvec(field):
            return -periodic_laplacian_jax(field, grid.dx, grid.dy)

        with jax.default_device(device):
            solution, _ = jax_cg(
                matvec, rhs_device, tol=tolerance, atol=0.0, maxiter=max_iterations
            )
        solution = solution - jnp.mean(solution)
        residual = matvec(solution) - rhs_device
        solution.block_until_ready()
        residual.block_until_ready()
        iterations = None
        solver_converged = True
        runtime = time.perf_counter() - start
    else:
        raise NotImplementedError(f"Poisson backend {grid.backend.value!r} is unavailable")

    residual_norm = float(np.linalg.norm(np.asarray(residual).ravel()))
    relative_residual = residual_norm / max(rhs_norm, np.finfo(grid.numpy_dtype).tiny)
    residual_tolerance = max(10.0 * tolerance, 10.0 * np.finfo(grid.numpy_dtype).eps)
    converged = solver_converged and relative_residual <= residual_tolerance
    return PoissonResult(
        solution=solution,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        iterations=iterations,
        converged=converged,
        rhs_mean=rhs_mean,
        runtime_seconds=runtime,
    )
