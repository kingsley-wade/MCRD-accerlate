"""Tests for LPP spectral aggregation and peak metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mcrd_accel.lpp.peaks import (
    LPPWindowMetrics,
    compute_peak_metrics,
    direction_to_bin,
    wavelength_to_bin,
)
from mcrd_accel.lpp.spectrum import (
    BinMaps,
    aggregate_spectrum,
    compute_bin_maps,
    compute_power_spectrum,
    compute_power_spectrum_batch,
)


# ---- Bin map computation --------------------------------------------------


def test_bin_maps_shape():
    bin_maps = compute_bin_maps(w=32, minL=2.0, maxL=16.0, pad_factor=1)
    assert bin_maps.radial.shape == (32, 32)
    assert bin_maps.angular.shape == (32, 32)


def test_bin_maps_dc_excluded():
    """DC component (zero frequency) should have bin index -1."""
    bin_maps = compute_bin_maps(w=32, minL=2.0, maxL=16.0, pad_factor=1)
    # DC is at the center after fftshift
    center = 32 // 2
    assert bin_maps.radial[center, center] == -1
    assert bin_maps.angular[center, center] == -1


def test_bin_maps_wavelength_range():
    """Wavelengths outside [minL, maxL] should have radial bin -1."""
    bin_maps = compute_bin_maps(w=64, minL=4.0, maxL=16.0, n_radial=13, pad_factor=1)
    # Very high frequency (small wavelength < minL) should be -1
    # Very low frequency (large wavelength > maxL) should be -1
    # Check corners (high frequency)
    assert bin_maps.radial[0, 0] == -1  # corner = highest frequency


def test_bin_maps_rejects_invalid_params():
    with pytest.raises(ValueError, match="w must be positive"):
        compute_bin_maps(w=0, minL=2.0, maxL=16.0)
    with pytest.raises(ValueError, match="minL and maxL must be positive"):
        compute_bin_maps(w=32, minL=0.0, maxL=16.0)
    with pytest.raises(ValueError, match="maxL must exceed minL"):
        compute_bin_maps(w=32, minL=10.0, maxL=5.0)


# ---- Power spectrum -------------------------------------------------------


def test_power_spectrum_shape():
    window = np.random.rand(32, 32).astype(np.float32)
    power = compute_power_spectrum(window, pad_factor=1)
    assert power.shape == (32, 32)


def test_power_spectrum_rejects_non_2d():
    window = np.random.rand(32)
    with pytest.raises(ValueError, match="2D"):
        compute_power_spectrum(window)


def test_power_spectrum_of_sinusoid_peaks_at_expected_frequency():
    """A sinusoid with known wavelength should peak at the corresponding frequency."""
    w = 64
    wavelength = 8.0  # pixels
    frequency = 1.0 / wavelength  # cycles per pixel

    # Create a horizontal sinusoid
    x = np.arange(w)
    y = np.arange(w)
    X, Y = np.meshgrid(x, y)
    window = np.sin(2 * np.pi * frequency * X).astype(np.float64)

    power = compute_power_spectrum(window, pad_factor=1)

    # Find peak (excluding DC at center)
    center = w // 2
    power_no_dc = power.copy()
    power_no_dc[center, center] = 0

    peak_idx = np.unravel_index(np.argmax(power_no_dc), power_no_dc.shape)

    # Peak should be at frequency bin corresponding to wavelength=8
    # After fftshift, frequency bins are at indices corresponding to freq = (idx - w/2) / w
    # For wavelength=8, frequency=1/8=0.125 cycles/pixel
    # Index = w/2 + frequency*w = 32 + 0.125*64 = 32 + 8 = 40 (or 32 - 8 = 24 for negative freq)
    expected_idx_pos = (center, int(center + frequency * w))
    expected_idx_neg = (center, int(center - frequency * w))

    # Peak should be at one of these two symmetric positions
    assert peak_idx == expected_idx_pos or peak_idx == expected_idx_neg, (
        f"Peak at {peak_idx}, expected {expected_idx_pos} or {expected_idx_neg}"
    )


# ---- Batch power spectrum -------------------------------------------------


def test_power_spectrum_batch_shape():
    windows = np.random.rand(10, 32, 32).astype(np.float32)
    power = compute_power_spectrum_batch(windows, pad_factor=1)
    assert power.shape == (10, 32, 32)


def test_power_spectrum_batch_matches_individual():
    """Batch processing should match individual processing."""
    windows = np.random.rand(5, 32, 32).astype(np.float64)
    power_batch = compute_power_spectrum_batch(windows, pad_factor=1)

    for i in range(5):
        power_individual = compute_power_spectrum(windows[i], pad_factor=1)
        assert np.allclose(power_batch[i], power_individual, atol=1e-10)


def test_power_spectrum_batch_rejects_non_3d():
    windows = np.random.rand(32, 32)
    with pytest.raises(ValueError, match="3D"):
        compute_power_spectrum_batch(windows)


# ---- Spectral aggregation -------------------------------------------------


def test_aggregate_spectrum_shapes():
    w = 32
    bin_maps = compute_bin_maps(w=w, minL=2.0, maxL=16.0, n_radial=15, n_angular=36, pad_factor=1)
    power = np.random.rand(w, w).astype(np.float32)
    radial, angular = aggregate_spectrum(power, bin_maps)
    assert radial.shape == (15,)
    assert angular.shape == (36,)


def test_aggregate_spectrum_batch():
    w = 32
    bin_maps = compute_bin_maps(w=w, minL=2.0, maxL=16.0, n_radial=15, n_angular=36, pad_factor=1)
    power = np.random.rand(10, w, w).astype(np.float32)
    radial, angular = aggregate_spectrum(power, bin_maps)
    assert radial.shape == (10, 15)
    assert angular.shape == (10, 36)


def test_aggregate_spectrum_sinusoid_identifies_wavelength():
    """A sinusoid with known wavelength should have peak radial power at the corresponding bin."""
    w = 64
    wavelength = 8.0
    frequency = 1.0 / wavelength

    # Create sinusoid
    x = np.arange(w)
    y = np.arange(w)
    X, Y = np.meshgrid(x, y)
    window = np.sin(2 * np.pi * frequency * X).astype(np.float64)

    power = compute_power_spectrum(window, pad_factor=1)
    bin_maps = compute_bin_maps(w=w, minL=4.0, maxL=16.0, n_radial=13, pad_factor=1)

    radial_power, _ = aggregate_spectrum(power, bin_maps)

    # Find peak radial bin
    peak_bin = np.argmax(radial_power)

    # Bin 0 = smallest wavelength (4.0), bin 12 = largest (16.0)
    # wavelength=8.0 should be around bin (8-4)/(16-4) * 12 = 4/12 * 12 = 4
    expected_bin = int((wavelength - 4.0) / (16.0 - 4.0) * 12)
    # Allow ±1 bin tolerance due to discretization
    assert abs(peak_bin - expected_bin) <= 1, f"Peak at bin {peak_bin}, expected ~{expected_bin}"


def test_aggregate_spectrum_sinusoid_identifies_orientation():
    """A horizontal sinusoid should have peak angular power at 0° (or 180°, folded to 0°)."""
    w = 64
    wavelength = 8.0
    frequency = 1.0 / wavelength

    # Horizontal sinusoid: varies along x, constant along y
    # Wavevector points in x-direction, angle = 0°
    x = np.arange(w)
    y = np.arange(w)
    X, Y = np.meshgrid(x, y)
    window = np.sin(2 * np.pi * frequency * X).astype(np.float64)

    power = compute_power_spectrum(window, pad_factor=1)
    bin_maps = compute_bin_maps(w=w, minL=4.0, maxL=16.0, n_radial=13, n_angular=36, pad_factor=1)

    _, angular_power = aggregate_spectrum(power, bin_maps)

    # Peak should be at bin 0 (angle 0°) or bin 18 (angle 180° = 0° for director)
    peak_bin = np.argmax(angular_power)
    # For a horizontal sinusoid, the wavevector is along x, so angle = 0° or 180°
    # Director folds to [0, 180°), so both map to bin 0
    assert peak_bin == 0 or peak_bin == 18, f"Peak at bin {peak_bin}, expected 0 or 18"


def test_aggregate_spectrum_rejects_non_2d_or_3d():
    power = np.random.rand(32)
    bin_maps = compute_bin_maps(w=32, minL=2.0, maxL=16.0, pad_factor=1)
    with pytest.raises(ValueError, match="2D or 3D"):
        aggregate_spectrum(power, bin_maps)


# ---- Peak metrics ---------------------------------------------------------


def test_peak_metrics_sinusoid_identifies_wavelength_and_direction():
    """A sinusoid with known wavelength and orientation should be identified."""
    w = 64
    wavelength = 8.0
    frequency = 1.0 / wavelength

    # Horizontal sinusoid
    x = np.arange(w)
    y = np.arange(w)
    X, Y = np.meshgrid(x, y)
    window = np.sin(2 * np.pi * frequency * X).astype(np.float64)

    power = compute_power_spectrum(window, pad_factor=1)
    bin_maps = compute_bin_maps(w=w, minL=4.0, maxL=16.0, n_radial=13, n_angular=36, pad_factor=1)
    radial_power, angular_power = aggregate_spectrum(power, bin_maps)

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=4.0, maxL=16.0)

    assert metrics.valid, "Sinusoid should have valid peak"
    assert metrics.within_range, "Wavelength should be within range"
    # Wavelength should be close to 8.0 (within ±1 bin ≈ 1 pixel)
    assert abs(metrics.wavelength - 8.0) <= 1.5, f"Wavelength {metrics.wavelength} not close to 8.0"
    # Direction should be close to 0° (horizontal)
    assert metrics.direction < 0.2 or metrics.direction > np.pi - 0.2, (
        f"Direction {metrics.direction} not close to 0 or π"
    )
    # Uniqueness should be high for a pure sinusoid
    assert metrics.uniqueness > 2.0, f"Uniqueness {metrics.uniqueness} too low"


def test_peak_direction_centroid_wraps_at_director_boundary():
    bin_maps = compute_bin_maps(w=64, minL=4.0, maxL=16.0, n_radial=13, n_angular=36, pad_factor=1)
    radial_power = np.zeros(13, dtype=np.float64)
    angular_power = np.zeros(36, dtype=np.float64)
    radial_power[4] = 10.0
    radial_power[3] = 1.0
    angular_power[0] = 10.0
    angular_power[-1] = 9.0

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=4.0, maxL=16.0)

    assert metrics.valid
    assert min(metrics.direction, np.pi - metrics.direction) < 0.1


def test_peak_metrics_flat_spectrum_invalid():
    """A flat (noise) spectrum should have low uniqueness and be invalid."""
    # Create flat power spectrum
    radial_power = np.ones(13, dtype=np.float64)
    angular_power = np.ones(36, dtype=np.float64)
    bin_maps = compute_bin_maps(w=64, minL=4.0, maxL=16.0, n_radial=13, n_angular=36, pad_factor=1)

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=4.0, maxL=16.0, peak_ratio=1.5)

    assert not metrics.valid, "Flat spectrum should be invalid"
    assert np.isnan(metrics.wavelength)
    assert np.isnan(metrics.direction)


def test_peak_metrics_zero_power_invalid():
    """Zero power spectrum should be invalid."""
    radial_power = np.zeros(13, dtype=np.float64)
    angular_power = np.zeros(36, dtype=np.float64)
    bin_maps = compute_bin_maps(w=64, minL=4.0, maxL=16.0, n_radial=13, n_angular=36, pad_factor=1)

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=4.0, maxL=16.0)

    assert not metrics.valid
    assert np.isnan(metrics.wavelength)


def test_peak_metrics_batch():
    """Batch processing should return arrays of metrics."""
    # Create batch of spectra
    batch_size = 5
    radial_power = np.random.rand(batch_size, 13).astype(np.float64)
    angular_power = np.random.rand(batch_size, 36).astype(np.float64)
    bin_maps = compute_bin_maps(w=64, minL=4.0, maxL=16.0, n_radial=13, n_angular=36, pad_factor=1)

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=4.0, maxL=16.0)

    assert metrics.wavelength.shape == (batch_size,)
    assert metrics.direction.shape == (batch_size,)
    assert metrics.valid.shape == (batch_size,)


def test_peak_metrics_powmax_powmean():
    """powmax and powmean should be computed correctly."""
    radial_power = np.array([1.0, 5.0, 2.0], dtype=np.float64)
    angular_power = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    bin_maps = compute_bin_maps(w=32, minL=2.0, maxL=16.0, n_radial=3, n_angular=3, pad_factor=1)

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=2.0, maxL=16.0, peak_ratio=0.0)

    # powmax = max of radial and angular max
    assert metrics.powmax == 5.0
    # powmean = mean of non-zero values
    assert metrics.powmean > 0


# ---- Bin conversion helpers -----------------------------------------------


def test_wavelength_to_bin():
    bin_idx = wavelength_to_bin(wavelength=8.0, minL=4.0, maxL=16.0, n_radial=13)
    # (8-4)/(16-4) * 12 = 4/12 * 12 = 4
    assert bin_idx == 4


def test_wavelength_to_bin_clipping():
    # Below minL should clip to 0
    assert wavelength_to_bin(wavelength=2.0, minL=4.0, maxL=16.0, n_radial=13) == 0
    # Above maxL should clip to n-1
    assert wavelength_to_bin(wavelength=20.0, minL=4.0, maxL=16.0, n_radial=13) == 12


def test_direction_to_bin():
    # 0° -> bin 0
    assert direction_to_bin(0.0, n_angular=36) == 0
    # 90° (π/2) -> bin 18
    assert direction_to_bin(np.pi / 2, n_angular=36) == 18
    # 180° (π) -> bin 35 (clipped)
    assert direction_to_bin(np.pi - 0.01, n_angular=36) == 35


# ---- Tied maxima rejection ------------------------------------------------


def test_peak_metrics_rejects_tied_radial_maxima():
    """If multiple radial bins share the max value, the window should be invalid."""
    radial_power = np.array([1.0, 5.0, 5.0], dtype=np.float64)
    angular_power = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    bin_maps = compute_bin_maps(w=32, minL=2.0, maxL=16.0, n_radial=3, n_angular=3, pad_factor=1)

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=2.0, maxL=16.0, peak_ratio=0.0)

    assert not metrics.valid
    assert np.isnan(metrics.wavelength)


def test_peak_metrics_rejects_tied_angular_maxima():
    """If multiple angular bins share the max value, the window should be invalid."""
    radial_power = np.array([1.0, 5.0, 2.0], dtype=np.float64)
    angular_power = np.array([3.0, 1.0, 3.0], dtype=np.float64)
    bin_maps = compute_bin_maps(w=32, minL=2.0, maxL=16.0, n_radial=3, n_angular=3, pad_factor=1)

    metrics = compute_peak_metrics(radial_power, angular_power, bin_maps, minL=2.0, maxL=16.0, peak_ratio=0.0)

    assert not metrics.valid
    assert np.isnan(metrics.wavelength)
