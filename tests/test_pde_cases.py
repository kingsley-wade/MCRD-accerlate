from __future__ import annotations

import builtins

import numpy as np
import pytest

from mcrd_accel.config import Backend, DType
from mcrd_accel.backends import get_jax_device
from mcrd_accel.pde import GridConfig, run_heat, run_wave, solve_periodic_poisson
from mcrd_accel.pde.operators import periodic_laplacian_jax
from mcrd_accel.mcrd.operators import periodic_laplacian


def _grid(backend: Backend = Backend.NUMPY, dtype: DType = DType.FLOAT64) -> GridConfig:
    return GridConfig(Nx=16, Ny=12, dx=0.25, dy=0.5, backend=backend, dtype=dtype)


@pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.JAX_CPU])
def test_heat_fourier_mode_matches_explicit_discrete_evolution_and_conserves_mass(backend):
    grid = _grid(backend, DType.FLOAT32 if backend is Backend.JAX_CPU else DType.FLOAT64)
    x = np.arange(grid.Nx) * grid.dx
    mode = 0.75 + np.broadcast_to(np.cos(2 * np.pi * x / (grid.Nx * grid.dx)), grid.shape).copy()
    dt, alpha, steps = 0.001, 0.2, 5
    eigenvalue = -4 * np.sin(np.pi / grid.Nx) ** 2 / grid.dx**2
    expected = 0.75 + (mode - 0.75) * (1 + dt * alpha * eigenvalue) ** steps

    result = run_heat(grid, alpha=alpha, dt=dt, n_steps=steps, initial_state=mode)

    np.testing.assert_allclose(np.asarray(result.final_state), expected, rtol=2e-5, atol=2e-6)
    assert abs(result.final_mass - result.initial_mass) < 2e-5


def test_heat_rejects_explicit_euler_step_outside_stability_limit():
    grid = _grid()
    with pytest.raises(ValueError, match="stability"):
        run_heat(grid, alpha=1.0, dt=0.5, n_steps=1)


@pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.JAX_CPU])
def test_periodic_poisson_recovers_discrete_manufactured_solution(backend):
    grid = _grid(backend, DType.FLOAT32 if backend is Backend.JAX_CPU else DType.FLOAT64)
    x = np.arange(grid.Nx) * grid.dx
    y = np.arange(grid.Ny) * grid.dy
    xx, yy = np.meshgrid(x, y)
    exact = np.cos(2 * np.pi * xx / (grid.Nx * grid.dx)) * np.sin(
        2 * np.pi * yy / (grid.Ny * grid.dy)
    )
    rhs = -periodic_laplacian(exact, grid.dx, grid.dy)

    result = solve_periodic_poisson(grid, rhs, tolerance=1e-6, max_iterations=1000)

    assert result.converged
    assert result.relative_residual < 2e-5
    if backend is Backend.JAX_CPU:
        assert result.iterations is None
    else:
        assert result.iterations is not None and result.iterations > 0
    np.testing.assert_allclose(np.asarray(result.solution), exact, rtol=2e-4, atol=2e-4)
    assert abs(float(np.mean(result.solution))) < 2e-6


def test_periodic_poisson_rejects_incompatible_nonzero_mean_rhs():
    with pytest.raises(ValueError, match="zero mean"):
        solve_periodic_poisson(_grid(), np.ones((12, 16)))


def test_pde_package_import_does_not_require_optional_scipy(monkeypatch):
    original_import = builtins.__import__

    def import_without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("SciPy is blocked for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_scipy)
    from mcrd_accel.pde.elliptic import solve_periodic_poisson as _solve

    with pytest.raises(ImportError, match="requires SciPy"):
        _solve(_grid(), np.zeros((12, 16)))


def test_periodic_poisson_reports_nonconvergence_when_iteration_budget_is_too_small():
    grid = _grid()
    rhs = np.zeros(grid.shape)
    rhs[0, 0], rhs[0, 1] = 1.0, -1.0
    result = solve_periodic_poisson(grid, rhs, tolerance=1e-12, max_iterations=1)
    assert not result.converged


@pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.JAX_CPU])
def test_wave_fourier_mode_matches_leapfrog_recurrence(backend):
    grid = _grid(backend, DType.FLOAT32 if backend is Backend.JAX_CPU else DType.FLOAT64)
    x = np.arange(grid.Nx) * grid.dx
    mode = 0.75 + np.broadcast_to(np.cos(2 * np.pi * x / (grid.Nx * grid.dx)), grid.shape).copy()
    speed, dt, steps = 0.4, 0.1, 4
    eigenvalue = -4 * np.sin(np.pi / grid.Nx) ** 2 / grid.dx**2
    cosine = 1 + 0.5 * (dt * speed) ** 2 * eigenvalue
    omega_dt = np.arccos(cosine)
    expected = 0.75 + (mode - 0.75) * np.cos(steps * omega_dt)

    result = run_wave(
        grid, wave_speed=speed, dt=dt, n_steps=steps, initial_displacement=mode, snapshot_stride=2
    )

    np.testing.assert_allclose(np.asarray(result.final_state), expected, rtol=2e-5, atol=2e-5)
    assert result.snapshot_steps == [2, 4]
    assert abs(result.final_mass - result.initial_mass) < 2e-5


@pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.JAX_CPU])
def test_wave_zero_steps_returns_initial_displacement_and_velocity(backend):
    grid = _grid(backend, DType.FLOAT32 if backend is Backend.JAX_CPU else DType.FLOAT64)
    displacement = np.arange(np.prod(grid.shape), dtype=grid.numpy_dtype).reshape(grid.shape)
    velocity = np.full(grid.shape, 0.25, dtype=grid.numpy_dtype)

    result = run_wave(
        grid,
        wave_speed=0.1,
        dt=0.01,
        n_steps=0,
        initial_displacement=displacement,
        initial_velocity=velocity,
    )

    np.testing.assert_array_equal(np.asarray(result.final_state), displacement)
    np.testing.assert_array_equal(np.asarray(result.diagnostics["final_velocity"]), velocity)
    assert result.snapshot_steps == []


def test_wave_jax_snapshot_steps_hit_stride_boundaries():
    grid = _grid(Backend.JAX_CPU, DType.FLOAT32)
    initial = np.zeros(grid.shape, dtype=np.float32)
    result = run_wave(
        grid,
        wave_speed=0.1,
        dt=0.01,
        n_steps=8,
        initial_displacement=initial,
        snapshot_stride=3,
    )
    assert result.snapshot_steps == [3, 6]


def test_wave_rejects_step_outside_cfl_limit():
    with pytest.raises(ValueError, match="CFL"):
        run_wave(
            _grid(), wave_speed=1.0, dt=1.0, n_steps=1, initial_displacement=np.zeros((12, 16))
        )


def test_jax_and_numpy_periodic_laplacian_agree():
    grid = _grid()
    field = np.random.default_rng(9).standard_normal(grid.shape)
    import jax.numpy as jnp

    actual = periodic_laplacian_jax(jnp.asarray(field), grid.dx, grid.dy)
    np.testing.assert_allclose(
        np.asarray(actual), periodic_laplacian(field, grid.dx, grid.dy), rtol=1e-5, atol=1e-5
    )


def test_jax_cpu_backend_is_explicitly_placed_on_cpu():
    device = get_jax_device(Backend.JAX_CPU)
    assert device.platform == "cpu"
    result = run_heat(_grid(Backend.JAX_CPU, DType.FLOAT32), alpha=0.1, dt=0.001, n_steps=1)
    assert result.final_state.devices() == {device}


def test_jax_cuda_does_not_silently_fall_back_to_cpu():
    try:
        device = get_jax_device(Backend.JAX_CUDA)
    except RuntimeError as exc:
        assert "no 'gpu' device" in str(exc)
    else:
        assert device.platform != "cpu"
