"""Tests for representative conservative PDE reaction networks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mcrd_accel.config import Backend, DType, SolverConfig, ValidationConfig
from mcrd_accel.mcrd import ReactionModel, State, run
from mcrd_accel.mcrd.validation import (
    run_conservation_check,
    run_spatial_convergence,
    run_stability_sweep,
    run_temporal_convergence,
)


def _config(**overrides):
    values = dict(
        kd=0.36, k1=1.35, b=60.0 / 6.7, k2=2.0, rho=9.0,
        cv=0.1072, beta=19.58, Du=0.0085, Dv=0.17,
        Nx=16, Ny=16, dx=0.2, dy=0.2, dt=1e-4, n_steps=4,
    )
    values.update(overrides)
    return SolverConfig(**values)


def _models(config):
    return (
        ReactionModel.linear_exchange(k_forward=0.8, k_reverse=0.3),
        ReactionModel.nonlinear_autocatalysis(rate=0.4),
        ReactionModel.p53_mcrd(config),
    )


@pytest.mark.parametrize("model_index", range(3))
def test_each_model_has_stoichiometric_conservation(model_index):
    config = _config()
    model = _models(config)[model_index]

    c = np.asarray(model.conserved_weights)
    S = np.asarray(model.stoichiometry)
    assert c @ S == pytest.approx(0.0)


def test_jax_step_function_is_reused_across_identical_runs():
    pytest.importorskip("jax")
    from mcrd_accel.mcrd.jax_solver import _compiled_jax_step

    config = _config()
    model = ReactionModel.linear_exchange()
    first = _compiled_jax_step(model, config.Du, config.Dv, config.dx, config.dy, config.dt)
    second = _compiled_jax_step(model, config.Du, config.Dv, config.dx, config.dy, config.dt)

    assert first is second


def test_jax_cuda_backend_reaches_device_validation_in_mcrd_dispatch():
    """CUDA must be routed to JAX; lack of a GPU is a device error, not an unsupported backend."""
    config = _config(backend=Backend.JAX_CUDA, n_steps=0)
    with pytest.raises(RuntimeError, match="no 'gpu' device"):
        run(config, model=ReactionModel.linear_exchange())


def test_linear_exchange_rate_matches_reference_formula():
    model = ReactionModel.linear_exchange(k_forward=0.8, k_reverse=0.3)
    u = np.array([[0.2, 0.4]])
    v = np.array([[0.7, 0.1]])

    np.testing.assert_allclose(model.rate_numpy(u, v), 0.8 * u - 0.3 * v)


def test_nonlinear_autocatalysis_rate_matches_reference_formula():
    model = ReactionModel.nonlinear_autocatalysis(rate=0.4)
    u = np.array([[0.2, 0.4]])
    v = np.array([[0.7, 0.1]])

    np.testing.assert_allclose(model.rate_numpy(u, v), 0.4 * u * v)


def test_linear_exchange_temporal_error_is_first_order():
    model = ReactionModel.linear_exchange(k_forward=0.8, k_reverse=0.0)
    initial = State(u=np.full((4, 4), 0.7), v=np.full((4, 4), 0.2))
    final_time = 0.5

    def error(dt):
        cfg = _config(
            Nx=4, Ny=4, Du=0.1, Dv=0.1, dt=dt,
            n_steps=round(final_time / dt), dtype=DType.FLOAT64,
        )
        state = run(cfg, initial_state=initial, model=model).final_state
        exact_u = 0.7 * np.exp(-0.8 * final_time)
        exact_v = 0.9 - exact_u
        return np.sqrt(np.mean((state.u - exact_u) ** 2 + (state.v - exact_v) ** 2))

    coarse_error = error(0.02)
    fine_error = error(0.01)
    assert 1.8 < coarse_error / fine_error < 2.2


@pytest.mark.parametrize("model_index", range(3))
def test_numpy_and_jax_model_steps_match(model_index):
    pytest.importorskip("jax")
    config = _config(n_steps=2)
    model = _models(config)[model_index]
    rng = np.random.default_rng(12)
    state = State(
        u=rng.uniform(0.2, 0.7, (16, 16)).astype(np.float32),
        v=rng.uniform(0.2, 0.7, (16, 16)).astype(np.float32),
    )

    expected_result = run(config, initial_state=state, model=model)
    actual_result = run(replace(config, backend=Backend.JAX_CPU), initial_state=state, model=model)
    expected = expected_result.final_state
    actual = actual_result.final_state

    np.testing.assert_allclose(actual.u, expected.u, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(actual.v, expected.v, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(actual_result.metadata["final_mass"], actual_result.metadata["initial_mass"], rtol=2e-6)


def test_validation_reports_selected_reaction_model():
    config = _config()
    model = ReactionModel.linear_exchange(k_forward=0.8, k_reverse=0.3)
    report = run_conservation_check(config, checkpoint_stride=2, model=model)

    assert report.metadata["reaction_model"]["name"] == "linear_exchange"
    assert report.max_drift < 2e-6


def test_stability_validation_accepts_selected_model():
    config = _config(n_steps=4)
    model = ReactionModel.nonlinear_autocatalysis(rate=0.2)
    validation = ValidationConfig(solver=config, grid_sizes=[16], time_steps=[config.dt])

    report = run_stability_sweep(validation, model=model)

    assert report.config["reaction_model"]["name"] == "nonlinear_autocatalysis"
    assert report.cases[0].status == "stable"


def test_spatial_and_temporal_validation_accept_selected_model():
    config = _config(n_steps=2, dt=1e-5)
    model = ReactionModel.nonlinear_autocatalysis(rate=0.2)
    validation = ValidationConfig(
        solver=config,
        grid_sizes=[8, 16],
        time_steps=[1e-5, 5e-6],
    )

    spatial = run_spatial_convergence(validation, model=model)
    temporal = run_temporal_convergence(validation, model=model)

    assert spatial.config["reaction_model"]["name"] == "nonlinear_autocatalysis"
    assert temporal.config["reaction_model"]["name"] == "nonlinear_autocatalysis"
    assert all(case.status == "completed" for case in spatial.cases)
    assert all(case.status == "completed" for case in temporal.cases)
