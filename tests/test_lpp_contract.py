"""Tests for LPP bounded overlapping-window extraction and merge."""

from __future__ import annotations

import numpy as np
import pytest

from mcrd_accel.config import LPPConfig
from mcrd_accel.lpp.merge import LPPResult, apply_edge_policy, merge_lpp_windows
from mcrd_accel.lpp.peaks import LPPWindowMetrics, compute_peak_metrics
from mcrd_accel.lpp.pipeline import analyze
from mcrd_accel.lpp.spectrum import BinMaps, compute_bin_maps
from mcrd_accel.lpp.windows import (
    WindowGrid,
    compute_window_grid,
    extract_all_windows,
    extract_window,
    extract_windows_batched,
)


# ---- Window grid computation ----------------------------------------------


def test_window_grid_exact_fit():
    """Image size is exactly w + k*dL for some integer k."""
    grid = compute_window_grid(image_shape=(100, 100), w=20, dL=20)
    assert grid.n_rows == 5  # (100 - 20) // 20 + 1 = 5
    assert grid.n_cols == 5
    assert grid.n_windows == 25


def test_window_grid_with_remainder():
    """Image has leftover pixels that don't fit another window."""
    grid = compute_window_grid(image_shape=(100, 100), w=20, dL=15)
    # (100 - 20) // 15 + 1 = 80 // 15 + 1 = 5 + 1 = 6
    assert grid.n_rows == 6
    assert grid.n_cols == 6
    assert grid.n_windows == 36


def test_window_grid_image_smaller_than_w():
    """Image is smaller than window size: no windows extracted."""
    grid = compute_window_grid(image_shape=(10, 10), w=20, dL=5)
    assert grid.n_rows == 0
    assert grid.n_cols == 0
    assert grid.n_windows == 0


def test_window_grid_w_equals_image_size():
    """Window size equals image size: exactly one window."""
    grid = compute_window_grid(image_shape=(32, 32), w=32, dL=8)
    assert grid.n_rows == 1
    assert grid.n_cols == 1
    assert grid.n_windows == 1


def test_window_grid_rejects_nonpositive_w():
    with pytest.raises(ValueError, match="w must be positive"):
        compute_window_grid((100, 100), w=0, dL=10)
    with pytest.raises(ValueError, match="w must be positive"):
        compute_window_grid((100, 100), w=-5, dL=10)


def test_window_grid_rejects_nonpositive_dL():
    with pytest.raises(ValueError, match="dL must be positive"):
        compute_window_grid((100, 100), w=20, dL=0)


# ---- Single window extraction ---------------------------------------------


def test_extract_window_at_origin():
    """Window at (0, 0) should be the top-left corner."""
    image = np.arange(100).reshape(10, 10).astype(np.float32)
    window = extract_window(image, row=0, col=0, w=3, dL=2)
    expected = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]], dtype=np.float32)
    assert np.array_equal(window, expected)


def test_extract_window_at_offset():
    """Window at (row, col) should be at pixel (row*dL, col*dL)."""
    image = np.arange(100).reshape(10, 10).astype(np.float32)
    window = extract_window(image, row=2, col=3, w=4, dL=2)
    # Top-left at (4, 6)
    expected = image[4:8, 6:10]
    assert np.array_equal(window, expected)


def test_extract_window_shape():
    image = np.ones((50, 60), dtype=np.float32)
    window = extract_window(image, row=1, col=1, w=10, dL=5)
    assert window.shape == (10, 10)


# ---- Batched extraction ---------------------------------------------------


def test_batched_extraction_yields_all_windows():
    """Batched extraction should yield every window exactly once."""
    image = np.random.rand(64, 64).astype(np.float32)
    w, dL = 16, 8

    grid = compute_window_grid(image.shape, w, dL)
    all_coords = []
    all_windows = []

    for batch, coords in extract_windows_batched(image, w, dL, batch_size=10):
        assert batch.shape[0] <= 10
        assert batch.shape[1:] == (w, w)
        assert len(coords) == batch.shape[0]
        all_coords.extend(coords)
        all_windows.append(batch)

    assert len(all_coords) == grid.n_windows
    # All coordinates should be unique
    assert len(set(all_coords)) == grid.n_windows


def test_batched_extraction_matches_all_at_once():
    """Batched extraction should produce the same windows as extract_all_windows."""
    image = np.random.rand(50, 50).astype(np.float32)
    w, dL = 12, 6

    all_windows, all_coords = extract_all_windows(image, w, dL)

    batched_windows = []
    batched_coords = []
    for batch, coords in extract_windows_batched(image, w, dL, batch_size=7):
        batched_windows.append(batch)
        batched_coords.extend(coords)

    batched_all = np.concatenate(batched_windows, axis=0)

    assert np.array_equal(all_windows, batched_all)
    assert all_coords == batched_coords


def test_batched_extraction_different_batch_sizes():
    """Different batch sizes should produce the same total set of windows."""
    image = np.random.rand(40, 40).astype(np.float32)
    w, dL = 10, 5

    results = {}
    for batch_size in [1, 3, 7, 16, 100]:
        coords = []
        for _, batch_coords in extract_windows_batched(image, w, dL, batch_size=batch_size):
            coords.extend(batch_coords)
        results[batch_size] = coords

    # All should have the same coordinates
    for batch_size in [3, 7, 16, 100]:
        assert results[batch_size] == results[1]


def test_batched_extraction_empty_image():
    """Image smaller than w should yield no batches."""
    image = np.ones((10, 10), dtype=np.float32)
    batches = list(extract_windows_batched(image, w=20, dL=5, batch_size=10))
    assert len(batches) == 0


def test_batched_extraction_single_window():
    """Image exactly w×w should yield one batch with one window."""
    image = np.ones((16, 16), dtype=np.float32)
    batches = list(extract_windows_batched(image, w=16, dL=8, batch_size=10))
    assert len(batches) == 1
    batch, coords = batches[0]
    assert batch.shape == (1, 16, 16)
    assert coords == [(0, 0)]


# ---- All-at-once extraction -----------------------------------------------


def test_extract_all_windows_shape():
    image = np.random.rand(32, 32).astype(np.float32)
    windows, coords = extract_all_windows(image, w=8, dL=8)
    grid = compute_window_grid(image.shape, 8, 8)
    assert windows.shape == (grid.n_windows, 8, 8)
    assert len(coords) == grid.n_windows


def test_extract_all_windows_empty():
    image = np.ones((5, 5), dtype=np.float32)
    windows, coords = extract_all_windows(image, w=10, dL=5)
    assert windows.shape == (0, 10, 10)
    assert coords == []


# ---- LPP merge ------------------------------------------------------------


def test_merge_lpp_windows_basic():
    """Merge should produce fields on the image grid."""
    # Create a simple grid: 2x2 windows of size 16x16 with stride 8 on a 32x32 image
    grid = compute_window_grid((32, 32), w=16, dL=8)
    assert grid.n_windows == 9  # 3x3 grid

    # Create fake metrics: all windows valid with same wavelength/direction
    n_windows = grid.n_windows
    metrics = LPPWindowMetrics(
        wavelength=np.full(n_windows, 8.0, dtype=np.float64),
        direction=np.full(n_windows, 0.5, dtype=np.float64),
        powmax=np.full(n_windows, 100.0, dtype=np.float64),
        powmean=np.full(n_windows, 10.0, dtype=np.float64),
        uniqueness=np.full(n_windows, 5.0, dtype=np.float64),
        within_range=np.ones(n_windows, dtype=bool),
        valid=np.ones(n_windows, dtype=bool),
    )

    result = merge_lpp_windows(grid, metrics, image_shape=(32, 32), min_power=1.0, min_patch_size=1)

    assert result.wavelength.shape == (32, 32)
    assert result.direction.shape == (32, 32)
    assert result.validity.shape == (32, 32)

    # All covered pixels should have wavelength ≈ 8.0
    covered = result.validity
    assert np.all(np.isclose(result.wavelength[covered], 8.0))


def test_merge_lpp_windows_invalid_windows_skipped():
    """Invalid windows should not contribute to the merge."""
    grid = compute_window_grid((32, 32), w=16, dL=16)  # 2x2 = 4 windows
    n_windows = grid.n_windows

    # Only first window valid
    metrics = LPPWindowMetrics(
        wavelength=np.array([8.0, np.nan, np.nan, np.nan]),
        direction=np.array([0.5, np.nan, np.nan, np.nan]),
        powmax=np.array([100.0, 0.0, 0.0, 0.0]),
        powmean=np.array([10.0, 0.0, 0.0, 0.0]),
        uniqueness=np.array([5.0, 0.0, 0.0, 0.0]),
        within_range=np.array([True, False, False, False]),
        valid=np.array([True, False, False, False]),
    )

    result = merge_lpp_windows(grid, metrics, image_shape=(32, 32), min_power=1.0, min_patch_size=1)

    # Only the top-left 16x16 region should be valid
    assert result.validity[0, 0]
    assert not result.validity[20, 20]  # Outside first window


def test_merge_lpp_windows_doubled_angle_averaging():
    """Direction averaging should use doubled-angle for directors."""
    grid = compute_window_grid((32, 32), w=32, dL=32)  # 1 window
    n_windows = grid.n_windows

    # Single window with direction = π/4 (45°)
    metrics = LPPWindowMetrics(
        wavelength=np.array([8.0]),
        direction=np.array([np.pi / 4]),
        powmax=np.array([100.0]),
        powmean=np.array([10.0]),
        uniqueness=np.array([5.0]),
        within_range=np.array([True]),
        valid=np.array([True]),
    )

    result = merge_lpp_windows(grid, metrics, image_shape=(32, 32), min_power=1.0, min_patch_size=1)

    # Direction should be close to π/4
    valid_pixels = result.validity
    assert np.all(np.isclose(result.direction[valid_pixels], np.pi / 4, atol=0.01))


# ---- Edge policy ----------------------------------------------------------


def test_apply_edge_policy_trim():
    result = LPPResult(
        wavelength=np.ones((32, 32), dtype=np.float64) * 8.0,
        direction=np.ones((32, 32), dtype=np.float64) * 0.5,
        validity=np.ones((32, 32), dtype=bool),
        metadata={},
    )
    grid = compute_window_grid((32, 32), w=16, dL=8)

    trimmed = apply_edge_policy(result, grid, edge_policy="trim")

    # Pixels outside the last window should be NaN
    # Last window top-left at (16, 16), bottom-right at (32, 32)
    # So all pixels should be covered in this case
    assert trimmed.wavelength.shape == (32, 32)


def test_apply_edge_policy_nan():
    result = LPPResult(
        wavelength=np.ones((32, 32), dtype=np.float64) * 8.0,
        direction=np.ones((32, 32), dtype=np.float64) * 0.5,
        validity=np.ones((32, 32), dtype=bool),
        metadata={},
    )
    grid = compute_window_grid((32, 32), w=16, dL=8)

    nanned = apply_edge_policy(result, grid, edge_policy="nan")

    # Edge pixels (within w=16 of boundary) should be NaN
    assert np.isnan(nanned.wavelength[0, 0])
    assert np.isnan(nanned.wavelength[31, 31])


# ---- Public analyze API ---------------------------------------------------


def test_analyze_basic():
    """analyze() should return an LPPResult."""
    image = np.random.rand(64, 64).astype(np.float32)
    config = LPPConfig(w=16, dL=8, minL=4.0, maxL=16.0, batch_size=16)

    result = analyze(image, config)

    assert isinstance(result, LPPResult)
    assert result.wavelength.shape == (64, 64)
    assert result.direction.shape == (64, 64)
    assert result.validity.shape == (64, 64)


def test_analyze_image_too_small():
    """Image smaller than window should return empty result."""
    image = np.random.rand(10, 10).astype(np.float32)
    config = LPPConfig(w=16, dL=8, minL=4.0, maxL=16.0)

    result = analyze(image, config)

    assert result.metadata["n_windows"] == 0
    assert np.all(np.isnan(result.wavelength))


def test_analyze_rejects_non_2d():
    image = np.random.rand(10, 10, 3)
    config = LPPConfig(w=8, dL=4, minL=2.0, maxL=8.0)

    with pytest.raises(ValueError, match="2D"):
        analyze(image, config)


def test_analyze_rejects_unavailable_backend():
    image = np.ones((16, 16), dtype=np.float32)
    config = LPPConfig(w=8, dL=4, minL=2.0, maxL=8.0)

    with pytest.raises(ValueError, match="backend 'jax' is unavailable"):
        analyze(image, config, backend="jax")


def test_component_filter_fails_clearly_when_scipy_is_missing(monkeypatch):
    import builtins

    from mcrd_accel.lpp.merge import _connected_component_filter

    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy.ndimage":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    with pytest.raises(RuntimeError, match="requires SciPy"):
        _connected_component_filter(np.ones((4, 4), dtype=np.intp), 2)
