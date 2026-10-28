"""Tests for MCRD solver stepping and output policy.

This module also covers configuration validation (task 2.1) since the solver
is the primary consumer of :class:`SolverConfig`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcrd_accel.config import (
    Backend,
    ConfigurationError,
    DType,
    Integrator,
    LPPConfig,
    ResourceProfile,
    SolverConfig,
    ValidationConfig,
)
from mcrd_accel.mcrd import JAXState, State, make_initial_state, run, run_jax, step


# ---- SolverConfig construction --------------------------------------------


def _base_kwargs(**overrides):
    base = dict(
        kd=0.36,
        k1=1.35,
        b=60.0 / 6.7,
        k2=2.0,
        rho=9.0,
        cv=0.1072,
        beta=19.58,
        Du=0.0085,
        Dv=0.17,
        Nx=64,
        Ny=64,
        dx=0.2,
        dy=0.2,
        dt=0.02,
        n_steps=100,
    )
    base.update(overrides)
    return base


def test_solver_config_accepts_valid_fig2b_profile():
    cfg = SolverConfig.fig2b()
    assert cfg.Nx == 256 and cfg.Ny == 256
    assert math.isclose(cfg.b, 60.0 / 6.7)
    assert math.isclose(cfg.rho, 9.0)
    assert math.isclose(cfg.beta, 19.58)
    assert cfg.integrator is Integrator.EXPLICIT_EULER
    assert cfg.backend is Backend.NUMPY


def test_solver_config_fig2b_overrides_propagate():
    cfg = SolverConfig.fig2b(Nx=128, dt=0.01, n_steps=500)
    assert cfg.Nx == 128
    assert math.isclose(cfg.dt, 0.01)
    assert cfg.n_steps == 500


def test_solver_config_rejects_nonpositive_grid():
    for bad in (0, -1):
        with pytest.raises(ConfigurationError, match="Nx"):
            SolverConfig(**_base_kwargs(Nx=bad))
        with pytest.raises(ConfigurationError, match="Ny"):
            SolverConfig(**_base_kwargs(Ny=bad))


def test_solver_config_rejects_nonpositive_spacing():
    for bad in (0.0, -0.1):
        with pytest.raises(ConfigurationError, match="dx"):
            SolverConfig(**_base_kwargs(dx=bad))
        with pytest.raises(ConfigurationError, match="dy"):
            SolverConfig(**_base_kwargs(dy=bad))


def test_solver_config_rejects_nonpositive_dt():
    for bad in (0.0, -0.01):
        with pytest.raises(ConfigurationError, match="dt"):
            SolverConfig(**_base_kwargs(dt=bad))


def test_solver_config_rejects_nonfinite_parameters():
    for name in ("kd", "k1", "b", "k2", "rho", "cv", "beta", "Du", "Dv"):
        for bad in (float("nan"), float("inf"), -1.0, 0.0):
            with pytest.raises(ConfigurationError, match=name):
                SolverConfig(**_base_kwargs(**{name: bad}))


def test_solver_config_rejects_negative_steps():
    with pytest.raises(ConfigurationError, match="n_steps"):
        SolverConfig(**_base_kwargs(n_steps=-1))


def test_solver_config_allows_zero_steps():
    cfg = SolverConfig(**_base_kwargs(n_steps=0))
    assert cfg.n_steps == 0
    assert cfg.final_time == 0.0


def test_solver_config_rejects_negative_snapshot_stride():
    with pytest.raises(ConfigurationError, match="snapshot_stride"):
        SolverConfig(**_base_kwargs(snapshot_stride=-1))


def test_solver_config_derived_quantities():
    cfg = SolverConfig(**_base_kwargs(Nx=128, Ny=64, dx=0.25, dy=0.5, dt=0.01, n_steps=100))
    assert math.isclose(cfg.Lx, 32.0)
    assert math.isclose(cfg.Ly, 32.0)
    assert math.isclose(cfg.final_time, 1.0)


# ---- Serialization round-trip ---------------------------------------------


def test_solver_config_round_trip_through_dict():
    original = SolverConfig.fig2b(Nx=128, dt=0.01, n_steps=500, dtype=DType.FLOAT64)
    data = original.to_dict()
    restored = SolverConfig.from_dict(data)
    assert restored == original


def test_solver_config_to_dict_is_json_serializable():
    import json

    cfg = SolverConfig.fig2b()
    data = cfg.to_dict()
    # Must not raise; enums must be serialized to strings.
    text = json.dumps(data)
    assert "explicit_euler" in text
    assert "float32" in text


# ---- ValidationConfig ------------------------------------------------------


def test_validation_config_default_series():
    vc = ValidationConfig(solver=SolverConfig.fig2b())
    assert vc.grid_sizes == [64, 128, 256, 512, 1024, 2048, 4096]
    assert len(vc.time_steps) >= 4


def test_validation_config_rejects_empty_grid_series():
    with pytest.raises(ConfigurationError, match="grid_sizes"):
        ValidationConfig(solver=SolverConfig.fig2b(), grid_sizes=[])


def test_validation_config_rejects_empty_time_series():
    with pytest.raises(ConfigurationError, match="time_steps"):
        ValidationConfig(solver=SolverConfig.fig2b(), time_steps=[])


def test_validation_config_rejects_nonpositive_grid_entries():
    with pytest.raises(ConfigurationError, match="grid_sizes"):
        ValidationConfig(solver=SolverConfig.fig2b(), grid_sizes=[64, 0, 256])


def test_validation_config_rejects_nonpositive_dt_entries():
    with pytest.raises(ConfigurationError, match="time_steps"):
        ValidationConfig(solver=SolverConfig.fig2b(), time_steps=[0.02, 0.0, 0.005])


# ---- LPPConfig ------------------------------------------------------------


def test_lpp_config_rejects_maxL_le_minL():
    with pytest.raises(ConfigurationError, match="maxL"):
        LPPConfig(minL=10.0, maxL=5.0)


def test_lpp_config_rejects_invalid_edge_policy():
    with pytest.raises(ConfigurationError, match="edge_policy"):
        LPPConfig(edge_policy="bogus")


def test_lpp_config_rejects_peak_ratio_above_one():
    with pytest.raises(ConfigurationError, match="peak_ratio"):
        LPPConfig(peak_ratio=1.5)


# ---- ResourceProfile ------------------------------------------------------


def test_resource_profile_rejects_nonpositive_budget():
    with pytest.raises(ConfigurationError, match="max_resident_bytes"):
        ResourceProfile(max_resident_bytes=0)
    with pytest.raises(ConfigurationError, match="max_steps"):
        ResourceProfile(max_steps=-1)


# ---- Immutability ---------------------------------------------------------


def test_solver_config_is_immutable():
    cfg = SolverConfig.fig2b()
    with pytest.raises(AttributeError):
        cfg.kd = 1.0  # type: ignore[misc]


# ---- Solver stepping and output policy ------------------------------------


def test_step_is_out_of_place():
    """step() must not modify the input state."""
    cfg = SolverConfig(**_base_kwargs(n_steps=1))
    state = make_initial_state(cfg)
    u_before = state.u.copy()
    v_before = state.v.copy()
    _ = step(state, cfg)
    assert np.array_equal(state.u, u_before)
    assert np.array_equal(state.v, v_before)


def test_step_produces_finite_state():
    """A single step from a reasonable initial state should remain finite."""
    cfg = SolverConfig(**_base_kwargs(n_steps=1))
    state = make_initial_state(cfg)
    new_state = step(state, cfg)
    assert np.all(np.isfinite(new_state.u))
    assert np.all(np.isfinite(new_state.v))


def test_run_deterministic_with_same_seed():
    """Repeated runs with the same seed must produce identical results."""
    cfg = SolverConfig(**_base_kwargs(n_steps=50, seed=42))
    result1 = run(cfg)
    result2 = run(cfg)
    assert np.array_equal(result1.final_state.u, result2.final_state.u)
    assert np.array_equal(result1.final_state.v, result2.final_state.v)


def test_run_different_seeds_differ():
    """Different seeds should produce different trajectories."""
    cfg1 = SolverConfig(**_base_kwargs(n_steps=50, seed=1))
    cfg2 = SolverConfig(**_base_kwargs(n_steps=50, seed=2))
    result1 = run(cfg1)
    result2 = run(cfg2)
    assert not np.array_equal(result1.final_state.u, result2.final_state.u)


def test_jax_one_step_matches_numpy_for_anisotropic_grid():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    cfg = SolverConfig(**_base_kwargs(
        backend=Backend.JAX_CPU, Nx=8, Ny=8, dx=0.2, dy=0.4, dt=1e-5, n_steps=1,
    ))
    yy, xx = np.mgrid[:8, :8]
    state = State(
        u=(0.5 + 0.01 * np.sin(xx + 2 * yy)).astype(np.float32),
        v=(25.0 + 0.1 * np.cos(2 * xx - yy)).astype(np.float32),
    )

    expected = step(state, cfg)
    actual = run_jax(cfg, JAXState(jnp.asarray(state.u), jnp.asarray(state.v))).final_state

    np.testing.assert_allclose(np.asarray(actual.u), expected.u, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual.v), expected.v, rtol=2e-5, atol=2e-6)


def test_jax_random_initial_state_matches_numpy_ranges():
    pytest.importorskip("jax")
    cfg = SolverConfig(**_base_kwargs(backend=Backend.JAX_CPU, Nx=16, Ny=16, n_steps=0, seed=10))

    state = run_jax(cfg).final_state
    u, v = np.asarray(state.u), np.asarray(state.v)

    assert u.min() >= 0.25 and u.max() <= 0.75
    assert v.min() >= 24.5 and v.max() <= 25.5
    assert abs(float(u.mean()) - 0.5) < 0.04
    assert abs(float(v.mean()) - 25.0) < 0.08


def test_run_dispatches_jax_and_preserves_checkpoint_and_snapshot_policy():
    pytest.importorskip("jax")
    cfg = SolverConfig(**_base_kwargs(
        backend=Backend.JAX_CPU, Nx=8, Ny=8, n_steps=4, snapshot_stride=2,
    ))

    result = run(cfg, checkpoint_stride=2)

    assert [checkpoint.step for checkpoint in result.checkpoints] == [0, 2, 4]
    assert result.snapshot_times == [0.04, 0.08]
    assert len(result.snapshots) == 2


def test_run_returns_final_state_and_checkpoints():
    """run() should return a SimulationResult with final state and checkpoints."""
    cfg = SolverConfig(**_base_kwargs(n_steps=100))
    result = run(cfg, checkpoint_stride=25)
    assert result.final_state is not None
    assert len(result.checkpoints) > 0
    # Checkpoints at step 0, 25, 50, 75, 100
    assert result.checkpoints[0].step == 0
    assert result.checkpoints[-1].step == 100


def test_run_checkpoints_record_mass_drift():
    """Checkpoints should record conserved mass and drift."""
    cfg = SolverConfig(**_base_kwargs(n_steps=100))
    result = run(cfg, checkpoint_stride=50)
    for cp in result.checkpoints:
        assert math.isfinite(cp.mass)
        assert math.isfinite(cp.mass_drift)
        assert cp.all_finite is True


def test_run_snapshots_at_configured_stride():
    """run() should save snapshots at the configured stride."""
    cfg = SolverConfig(**_base_kwargs(n_steps=100, snapshot_stride=25))
    result = run(cfg, checkpoint_stride=0)
    # Snapshots at step 25, 50, 75, 100
    assert len(result.snapshots) == 4
    assert len(result.snapshot_times) == 4
    assert math.isclose(result.snapshot_times[0], 25 * cfg.dt)


def test_run_no_snapshots_when_stride_zero():
    """snapshot_stride=0 should disable snapshots."""
    cfg = SolverConfig(**_base_kwargs(n_steps=100, snapshot_stride=0))
    result = run(cfg, checkpoint_stride=0)
    assert len(result.snapshots) == 0


def test_run_metadata_records_configuration():
    """SimulationResult.metadata should record key configuration."""
    cfg = SolverConfig(**_base_kwargs(n_steps=50, seed=123))
    result = run(cfg)
    assert result.metadata["n_steps"] == 50
    assert result.metadata["seed"] == 123
    assert "initial_mass" in result.metadata
    assert "final_mass" in result.metadata


def test_state_rejects_shape_mismatch():
    """State should reject u and v with different shapes."""
    u = np.ones((10, 10))
    v = np.ones((10, 20))
    with pytest.raises(ValueError, match="same shape"):
        State(u=u, v=v)


def test_state_rejects_non_2d():
    """State should reject non-2D arrays."""
    u = np.ones((10,))
    v = np.ones((10,))
    with pytest.raises(ValueError, match="2D"):
        State(u=u, v=v)
