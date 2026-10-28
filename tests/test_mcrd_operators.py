"""Tests for periodic MCRD operators and reaction equations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcrd_accel.mcrd.equations import (
    conserved_total,
    reaction_contributions,
    reaction_term,
    relative_mass_drift,
)
from mcrd_accel.mcrd.operators import periodic_laplacian


# ---- Periodic Laplacian ---------------------------------------------------


def test_laplacian_of_constant_field_is_zero():
    field = np.full((64, 64), 3.14, dtype=np.float32)
    lap = periodic_laplacian(field, dx=0.2, dy=0.2)
    assert np.allclose(lap, 0.0, atol=1e-6)


def test_laplacian_sum_is_zero_within_tolerance():
    """The discrete periodic Laplacian must sum to zero (conservation property)."""
    rng = np.random.default_rng(42)
    field = rng.standard_normal((128, 128)).astype(np.float32)
    lap = periodic_laplacian(field, dx=0.2, dy=0.2)
    lap_sum = np.sum(lap)
    # Tolerance: n_elements * eps(float32) ≈ 128*128 * 1.2e-7 ≈ 2e-3
    tol = field.size * np.finfo(np.float32).eps * 10
    assert abs(lap_sum) < tol, f"Laplacian sum {lap_sum} exceeds tolerance {tol}"


def test_laplacian_sum_is_zero_float64():
    """Float64 should have tighter tolerance."""
    rng = np.random.default_rng(42)
    field = rng.standard_normal((128, 128)).astype(np.float64)
    lap = periodic_laplacian(field, dx=0.2, dy=0.2)
    lap_sum = np.sum(lap)
    tol = field.size * np.finfo(np.float64).eps * 10
    assert abs(lap_sum) < tol


def test_laplacian_of_sinusoid_matches_analytical():
    """Laplacian of sin(kx*x + ky*y) should be -(kx² + ky²) * sin(kx*x + ky*y)."""
    Nx, Ny = 128, 128
    dx = dy = 0.1
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y)

    kx = 2 * np.pi / (Nx * dx)  # one period across domain
    ky = 2 * np.pi / (Ny * dy)
    field = np.sin(kx * X + ky * Y).astype(np.float64)

    lap = periodic_laplacian(field, dx, dy)
    expected = -(kx**2 + ky**2) * field

    # Finite-difference error is O(dx²); for dx=0.1, expect ~1% error
    rel_error = np.linalg.norm(lap - expected) / np.linalg.norm(expected)
    assert rel_error < 0.02, f"Relative error {rel_error} exceeds 2%"


def test_laplacian_rejects_non_2d_field():
    field = np.ones((10,))
    with pytest.raises(ValueError, match="2D"):
        periodic_laplacian(field, dx=0.2, dy=0.2)


def test_laplacian_rejects_nonpositive_spacing():
    field = np.ones((10, 10))
    with pytest.raises(ValueError, match="positive"):
        periodic_laplacian(field, dx=0.0, dy=0.2)
    with pytest.raises(ValueError, match="positive"):
        periodic_laplacian(field, dx=0.2, dy=-0.1)


def test_laplacian_supports_non_square_cells():
    """Non-square cells (dx != dy) should produce correct Laplacian."""
    field = np.ones((10, 10))
    lap = periodic_laplacian(field, dx=0.2, dy=0.3)
    # Constant field → Laplacian is zero
    assert np.allclose(lap, 0.0, atol=1e-12)

    # Quadratic field: f(x,y) = x² + y² → ∇²f = 2 + 2 = 4
    dx, dy = 0.1, 0.2
    Ny, Nx = 20, 30
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y)
    field_q = (X ** 2 + Y ** 2).astype(np.float64)
    lap_q = periodic_laplacian(field_q, dx=dx, dy=dy)
    # Interior points (away from periodic boundary) should have ∇²f ≈ 4
    assert np.allclose(lap_q[2:-2, 2:-2], 4.0, atol=1e-10)


def test_laplacian_is_out_of_place():
    """Laplacian must not modify the input array."""
    field = np.ones((10, 10), dtype=np.float32)
    field_copy = field.copy()
    _ = periodic_laplacian(field, dx=0.2, dy=0.2)
    assert np.array_equal(field, field_copy)


# ---- Reaction term --------------------------------------------------------


def test_reaction_contributions_cancel_pointwise():
    """du_dt_reaction + dv_dt_reaction = 0 pointwise (mass conservation)."""
    rng = np.random.default_rng(42)
    u = rng.uniform(0.1, 5.0, (64, 64)).astype(np.float32)
    v = rng.uniform(10.0, 30.0, (64, 64)).astype(np.float32)

    du_dt, dv_dt = reaction_contributions(
        u, v, kd=0.36, k1=1.35, b=60.0 / 6.7, k2=2.0, rho=9.0, cv=0.1072, beta=19.58
    )

    total = du_dt + dv_dt
    assert np.allclose(total, 0.0, atol=1e-5)


def test_reaction_contributions_match_fig2b_opencl_kernel_expression():
    u = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float64)
    v = np.array([[24.5, 25.0], [25.5, 26.0]], dtype=np.float64)
    kd, k1, b, k2, rho, beta, cv = 0.36, 1.35, 60.0 / 6.7, 2.0, 9.0, 19.58, 0.016 * 6.7

    expected_r = -kd * u + (k1 * u + b) * v - rho * k2 * beta * u / (k2 + u + v)
    actual_u, actual_v = reaction_contributions(u, v, kd, k1, b, k2, rho, cv, beta)

    np.testing.assert_allclose(actual_u, cv * expected_r)
    np.testing.assert_allclose(actual_v, -cv * expected_r)


def test_reaction_term_shape_matches_input():
    u = np.ones((32, 48), dtype=np.float32)
    v = np.ones((32, 48), dtype=np.float32)
    R = reaction_term(u, v, kd=0.36, k1=1.35, b=8.95, k2=2.0, rho=9.0, beta=19.58)
    assert R.shape == u.shape


def test_reaction_term_rejects_shape_mismatch():
    u = np.ones((32, 48))
    v = np.ones((48, 32))
    with pytest.raises(ValueError, match="same shape"):
        reaction_term(u, v, kd=0.36, k1=1.35, b=8.95, k2=2.0, rho=9.0, beta=19.58)


def test_reaction_term_finite_for_positive_inputs():
    """Reaction term should be finite for physically meaningful inputs."""
    rng = np.random.default_rng(42)
    u = rng.uniform(0.01, 10.0, (64, 64)).astype(np.float32)
    v = rng.uniform(0.01, 50.0, (64, 64)).astype(np.float32)
    R = reaction_term(u, v, kd=0.36, k1=1.35, b=8.95, k2=2.0, rho=9.0, beta=19.58)
    assert np.all(np.isfinite(R))


# ---- Conserved total and mass drift ---------------------------------------


def test_conserved_total_computes_sum():
    u = np.ones((10, 10), dtype=np.float32)
    v = np.full((10, 10), 2.0, dtype=np.float32)
    total = conserved_total(u, v)
    assert math.isclose(total, 300.0)


def test_conserved_total_rejects_shape_mismatch():
    u = np.ones((10, 10))
    v = np.ones((10, 20))
    with pytest.raises(ValueError, match="same shape"):
        conserved_total(u, v)


def test_relative_mass_drift_zero_for_no_change():
    drift = relative_mass_drift(100.0, 100.0)
    assert math.isclose(drift, 0.0)


def test_relative_mass_drift_small_for_tiny_change():
    drift = relative_mass_drift(100.0 + 1e-6, 100.0)
    assert abs(drift) < 1e-5


def test_relative_mass_drift_handles_zero_initial_mass():
    """When mass_0 = 0, denominator falls back to dtype tiny."""
    drift = relative_mass_drift(1e-10, 0.0, dtype=np.float32)
    tiny = np.finfo(np.float32).tiny
    expected = 1e-10 / tiny
    assert math.isclose(drift, expected, rel_tol=1e-5)
