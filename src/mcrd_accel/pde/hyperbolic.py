"""Periodic linear wave equation using the standard leapfrog scheme."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from ..backends import JAX_BACKENDS, get_jax_device
from ..config import Backend
from ..mcrd.operators import periodic_laplacian
from .config import GridConfig
from .operators import periodic_laplacian_jax, require_field_shape
from .parabolic import TimePDEResult


def run_wave(
    grid: GridConfig,
    *,
    wave_speed: float,
    dt: float,
    n_steps: int,
    initial_displacement: NDArray[np.floating],
    initial_velocity: NDArray[np.floating] | None = None,
    snapshot_stride: int = 0,
) -> TimePDEResult:
    """Solve ``u_tt = wave_speed**2 * Laplace(u)`` with periodic boundaries."""
    if not np.isfinite(wave_speed) or wave_speed <= 0:
        raise ValueError("wave_speed must be positive and finite")
    if not np.isfinite(dt) or dt <= 0 or n_steps < 0 or snapshot_stride < 0:
        raise ValueError("dt must be positive; n_steps and snapshot_stride must be non-negative")
    require_field_shape(initial_displacement, grid.shape, "initial_displacement")
    if initial_velocity is None:
        initial_velocity = np.zeros(grid.shape, dtype=grid.numpy_dtype)
    else:
        require_field_shape(initial_velocity, grid.shape, "initial_velocity")

    u0 = np.asarray(initial_displacement, dtype=grid.numpy_dtype)
    v0 = np.asarray(initial_velocity, dtype=grid.numpy_dtype)
    cfl = wave_speed * dt * np.sqrt(grid.dx**-2 + grid.dy**-2)
    if cfl > 1.0 * (1.0 + 1e-12):
        raise ValueError(f"wave step violates the 2D leapfrog CFL limit: c*dt*sqrt(dx^-2+dy^-2)={cfl:.6g} > 1")
    initial_energy = _wave_energy(u0, v0, grid, wave_speed)
    mass_0 = float(np.sum(u0, dtype=np.float64))

    if grid.backend is Backend.NUMPY:
        previous = u0.copy()
        current = u0 + dt * v0 + 0.5 * dt**2 * wave_speed**2 * periodic_laplacian(u0, grid.dx, grid.dy)
        snapshots: list[object] = []
        steps: list[int] = []
        if snapshot_stride and n_steps >= 1 and 1 % snapshot_stride == 0:
            snapshots.append(current.copy())
            steps.append(1)
        start = time.perf_counter()
        for step_idx in range(2, n_steps + 1):
            next_state = (
                2.0 * current - previous
                + dt**2 * wave_speed**2 * periodic_laplacian(current, grid.dx, grid.dy)
            )
            previous, current = current, next_state
            if snapshot_stride and step_idx % snapshot_stride == 0:
                snapshots.append(current.copy())
                steps.append(step_idx)
        runtime = time.perf_counter() - start
        final_velocity = v0 if n_steps == 0 else (current - previous) / dt
        if n_steps == 0:
            current = u0.copy()
    elif grid.backend in JAX_BACKENDS:
        import jax
        import jax.numpy as jnp

        device = get_jax_device(grid.backend)
        u0_device = jax.device_put(jnp.asarray(u0), device)
        v0_device = jax.device_put(jnp.asarray(v0), device)
        previous = u0_device
        current = u0_device + dt * v0_device + 0.5 * dt**2 * wave_speed**2 * periodic_laplacian_jax(u0_device, grid.dx, grid.dy)
        snapshots = []
        steps = []
        if snapshot_stride and n_steps >= 1 and 1 % snapshot_stride == 0:
            snapshots.append(current)
            steps.append(1)

        def body(carry, _):
            prev, cur = carry
            nxt = 2.0 * cur - prev + dt**2 * wave_speed**2 * periodic_laplacian_jax(cur, grid.dx, grid.dy)
            return (cur, nxt), None

        start = time.perf_counter()
        completed = 1 if n_steps else 0
        carry = (previous, current)
        chunk = snapshot_stride if snapshot_stride else max(1, n_steps)
        while completed < n_steps:
            remaining = n_steps - completed
            if snapshot_stride:
                next_snapshot = snapshot_stride - completed % snapshot_stride
                n_chunk = min(next_snapshot, remaining)
            else:
                n_chunk = min(chunk, remaining)
            with jax.default_device(device):
                carry, _ = jax.lax.scan(body, carry, None, length=n_chunk)
            carry[1].block_until_ready()
            completed += n_chunk
            previous, current = carry
            if snapshot_stride and completed % snapshot_stride == 0:
                snapshots.append(current)
                steps.append(completed)
        if n_steps == 0:
            current, final_velocity = u0_device, v0_device
        else:
            previous, current = carry
            final_velocity = (current - previous) / dt
        runtime = time.perf_counter() - start
    else:
        raise NotImplementedError(f"Wave equation backend {grid.backend.value!r} is unavailable")

    final_energy = _wave_energy(current, final_velocity, grid, wave_speed)
    mass_f = float(np.asarray(current).sum(dtype=np.float64))
    return TimePDEResult(
        final_state=current,
        snapshots=snapshots,
        snapshot_steps=steps,
        runtime_seconds=runtime,
        initial_mass=mass_0,
        final_mass=mass_f,
        diagnostics={
            "dt": dt,
            "n_steps": n_steps,
            "wave_speed": wave_speed,
            "cfl_number": cfl,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "relative_energy_drift": (final_energy - initial_energy) / max(abs(initial_energy), np.finfo(grid.numpy_dtype).tiny),
            "final_velocity": final_velocity,
        },
    )


def _wave_energy(u, velocity, grid: GridConfig, wave_speed: float) -> float:
    if grid.backend in JAX_BACKENDS:
        import jax.numpy as jnp

        gradient_x = (jnp.roll(u, -1, axis=1) - u) / grid.dx
        gradient_y = (jnp.roll(u, -1, axis=0) - u) / grid.dy
        return float(0.5 * jnp.sum(velocity**2 + wave_speed**2 * (gradient_x**2 + gradient_y**2)))
    gradient_x = (np.roll(u, -1, axis=1) - u) / grid.dx
    gradient_y = (np.roll(u, -1, axis=0) - u) / grid.dy
    return float(0.5 * np.sum(velocity**2 + wave_speed**2 * (gradient_x**2 + gradient_y**2), dtype=np.float64))
