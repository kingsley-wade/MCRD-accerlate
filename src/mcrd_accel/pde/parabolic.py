"""Periodic heat equation baseline: u_t = alpha * Laplacian(u)."""

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
class TimePDEResult:
    final_state: object
    snapshots: list[object]
    snapshot_steps: list[int]
    runtime_seconds: float
    initial_mass: float
    final_mass: float
    diagnostics: dict


def run_heat(
    grid: GridConfig,
    *,
    alpha: float,
    dt: float,
    n_steps: int,
    initial_state: NDArray[np.floating] | None = None,
    snapshot_stride: int = 0,
) -> TimePDEResult:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    if not np.isfinite(dt) or dt <= 0 or n_steps < 0 or snapshot_stride < 0:
        raise ValueError("dt must be positive; n_steps and snapshot_stride must be non-negative")
    stability_number = dt * alpha * (grid.dx**-2 + grid.dy**-2)
    if stability_number > 0.5 * (1.0 + 1e-12):
        raise ValueError(
            "explicit heat step violates the 2D stability limit: "
            f"dt*alpha*(dx^-2+dy^-2)={stability_number:.6g} > 0.5"
        )

    if initial_state is None:
        rng = np.random.default_rng(grid.seed)
        initial_state = rng.standard_normal(grid.shape).astype(grid.numpy_dtype)
    else:
        require_field_shape(initial_state, grid.shape, "initial_state")
        initial_state = np.asarray(initial_state, dtype=grid.numpy_dtype)
    mass_0 = float(np.sum(initial_state, dtype=np.float64))

    if grid.backend is Backend.NUMPY:
        state = initial_state.copy()
        snapshots: list[object] = []
        steps: list[int] = []
        start = time.perf_counter()
        for step_idx in range(1, n_steps + 1):
            state = state + (dt * alpha) * periodic_laplacian(state, grid.dx, grid.dy)
            if snapshot_stride and step_idx % snapshot_stride == 0:
                snapshots.append(state.copy())
                steps.append(step_idx)
        runtime = time.perf_counter() - start
    elif grid.backend in JAX_BACKENDS:
        import jax
        import jax.numpy as jnp

        device = get_jax_device(grid.backend)
        state = jax.device_put(jnp.asarray(initial_state), device)

        def body(current, _):
            next_state = current + (dt * alpha) * periodic_laplacian_jax(current, grid.dx, grid.dy)
            return next_state, None

        snapshots = []
        steps = []
        chunk = snapshot_stride if snapshot_stride else max(1, n_steps)
        start = time.perf_counter()
        done = 0
        while done < n_steps:
            n_chunk = min(chunk, n_steps - done)
            with jax.default_device(device):
                state, _ = jax.lax.scan(body, state, None, length=n_chunk)
            state.block_until_ready()
            done += n_chunk
            if snapshot_stride and done % snapshot_stride == 0:
                snapshots.append(state)
                steps.append(done)
        runtime = time.perf_counter() - start
    else:
        raise NotImplementedError(f"Heat equation backend {grid.backend.value!r} is unavailable")

    mass_f = float(np.asarray(state).sum(dtype=np.float64))
    return TimePDEResult(
        final_state=state,
        snapshots=snapshots,
        snapshot_steps=steps,
        runtime_seconds=runtime,
        initial_mass=mass_0,
        final_mass=mass_f,
        diagnostics={
            "dt": dt,
            "n_steps": n_steps,
            "alpha": alpha,
            "stability_number": stability_number,
        },
    )
