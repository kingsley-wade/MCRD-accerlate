"""Tests for conservation, convergence, and stability validation."""

from __future__ import annotations

import numpy as np
import pytest

from mcrd_accel.config import SolverConfig, ValidationConfig
from mcrd_accel.mcrd.validation import (
    ConservationReport,
    ConvergenceReport,
    StabilityReport,
    run_conservation_check,
    run_spatial_convergence,
    run_stability_sweep,
    run_temporal_convergence,
    _make_spatial_initial_state,
)


def _base_config(**overrides):
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
        Nx=32,
        Ny=32,
        dx=0.2,
        dy=0.2,
        dt=0.02,
        n_steps=100,
    )
    base.update(overrides)
    return SolverConfig(**base)


# ---- Conservation check ---------------------------------------------------


def test_conservation_check_reports_drift():
    config = _base_config()
    report = run_conservation_check(config, checkpoint_stride=50)

    assert isinstance(report, ConservationReport)
    assert report.initial_mass > 0
    assert report.final_mass > 0
    assert report.max_drift >= 0
    assert report.mean_drift >= 0
    assert report.all_finite is True
    assert report.n_checkpoints > 0


def test_conservation_check_metadata_includes_backend():
    config = _base_config()
    report = run_conservation_check(config, checkpoint_stride=50)

    assert "backend" in report.metadata
    assert "dtype" in report.metadata
    assert report.metadata["backend"] == "numpy"


def test_conservation_check_drift_below_tolerance():
    """For a short run with explicit Euler, drift should be very small."""
    config = _base_config(n_steps=200)
    report = run_conservation_check(config, checkpoint_stride=50)

    # Explicit Euler with out-of-place update should conserve mass to float32 precision
    assert report.max_drift < 1e-5


# ---- Spatial convergence --------------------------------------------------


def test_spatial_convergence_runs_all_grid_sizes():
    config = _base_config()
    validation_config = ValidationConfig(
        solver=config,
        grid_sizes=[32, 64],
        time_steps=[0.02],
    )

    report = run_spatial_convergence(validation_config)

    assert isinstance(report, ConvergenceReport)
    assert report.study_type == "spatial"
    assert len(report.cases) == 2
    assert report.cases[0].grid_size == 32
    assert report.cases[1].grid_size == 64


def test_spatial_convergence_reference_case_is_finest():
    config = _base_config(dt=0.002, n_steps=100)
    validation_config = ValidationConfig(
        solver=config,
        grid_sizes=[32, 64, 128],
        time_steps=[0.002],
    )

    report = run_spatial_convergence(validation_config)

    assert report.reference_case is not None
    assert report.reference_case.grid_size == 128


def test_spatial_initial_conditions_are_nested_samples():
    config = _base_config(Nx=8, Ny=8, seed=37)

    coarse = _make_spatial_initial_state(config, 8)
    fine = _make_spatial_initial_state(config, 16)

    np.testing.assert_allclose(coarse.u, fine.u[::2, ::2])
    np.testing.assert_allclose(coarse.v, fine.v[::2, ::2])


# ---- Temporal convergence -------------------------------------------------


def test_temporal_convergence_runs_all_dt_values():
    config = _base_config()
    validation_config = ValidationConfig(
        solver=config,
        grid_sizes=[32],
        time_steps=[0.02, 0.01, 0.005],
    )

    report = run_temporal_convergence(validation_config)

    assert isinstance(report, ConvergenceReport)
    assert report.study_type == "temporal"
    assert len(report.cases) == 3
    assert report.cases[0].dt == 0.02
    assert report.cases[1].dt == 0.01
    assert report.cases[2].dt == 0.005


# ---- Stability sweep ------------------------------------------------------


def test_stability_sweep_classifies_cases():
    config = _base_config()
    validation_config = ValidationConfig(
        solver=config,
        grid_sizes=[32],
        time_steps=[0.02, 0.01],
    )

    report = run_stability_sweep(validation_config)

    assert isinstance(report, StabilityReport)
    assert len(report.cases) == 2
    for case in report.cases:
        assert case.status in {"stable", "unstable", "nonfinite", "bound_violation", "inconclusive", "failed"}


def test_stability_sweep_estimates_interval():
    config = _base_config()
    validation_config = ValidationConfig(
        solver=config,
        grid_sizes=[32],
        time_steps=[0.02, 0.01, 0.005],
    )

    report = run_stability_sweep(validation_config)

    # If any cases are stable, should have an interval
    stable_cases = [c for c in report.cases if c.status == "stable"]
    if stable_cases:
        assert report.estimated_stable_interval is not None
        assert len(report.estimated_stable_interval) == 2
        first_failure = next((c for c in report.cases if c.status != "stable"), None)
        if first_failure is not None:
            assert report.estimated_stable_interval[1] < first_failure.dt
