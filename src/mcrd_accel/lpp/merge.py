"""Scatter accumulation and director averaging for LPP overlap merge.

Instead of materializing the MATLAB-style 3D merge tensor (image_height x image_width x n_overlapping_windows),
we use sum/count accumulators that are updated incrementally as each window is processed.

Direction averaging uses doubled-angle representation to handle the director (orientation without sign).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .peaks import LPPWindowMetrics
from .windows import WindowGrid


@dataclass
class LPPResult:
    """Merged LPP result on the original image grid.

    Attributes
    ----------
    wavelength:
        Merged wavelength field, shape (image_height, image_width). NaN where invalid.
    direction:
        Merged direction field (radians [0, pi)), shape (image_height, image_width). NaN where invalid.
    validity:
        Boolean mask of valid pixels, shape (image_height, image_width).
    metadata:
        Dictionary with processing metadata.
    per_window:
        Dict of per-window arrays (powmax, powmean, uniqueness, within_range, tile)
        for windows that contributed to the merge.
    """

    wavelength: NDArray[np.floating]
    direction: NDArray[np.floating]
    validity: NDArray[np.bool_]
    metadata: dict
    per_window: dict = field(default_factory=dict)


def merge_lpp_windows(
    grid: WindowGrid,
    metrics: LPPWindowMetrics,
    image_shape: tuple[int, int],
    min_power: float = 1e-6,
    min_patch_size: int = 4,
) -> LPPResult:
    """Merge overlapping window metrics into image-space fields.

    Uses scatter accumulation (sum/count) to avoid materializing a 3D tensor.
    Direction averaging uses doubled-angle representation for directors.
    Filters by within_range and applies connected-component patch filtering.

    Parameters
    ----------
    grid:
        Window grid descriptor.
    metrics:
        Per-window metrics (wavelength, direction, validity, etc.).
    image_shape:
        (height, width) of the original image.
    min_power:
        Minimum power threshold for validity.
    min_patch_size:
        Minimum connected-component patch size (pixels) for validity.

    Returns
    -------
    LPPResult
        Merged fields on the image grid.
    """
    h, w_img = image_shape
    w = grid.w
    dL = grid.dL

    wavelength_sum = np.zeros((h, w_img), dtype=np.float64)
    wavelength_count = np.zeros((h, w_img), dtype=np.intp)

    sin_sum = np.zeros((h, w_img), dtype=np.float64)
    cos_sum = np.zeros((h, w_img), dtype=np.float64)
    angle_count = np.zeros((h, w_img), dtype=np.intp)

    per_window_powmax = []
    per_window_powmean = []
    per_window_uniqueness = []
    per_window_within_range = []
    per_window_tile = []

    for idx in range(grid.n_windows):
        row = idx // grid.n_cols
        col = idx % grid.n_cols

        if not metrics.valid[idx]:
            continue
        if not metrics.within_range[idx]:
            continue
        if metrics.powmax[idx] < min_power:
            continue

        wl = metrics.wavelength[idx]
        dir_rad = metrics.direction[idx]

        y0 = row * dL
        x0 = col * dL
        y1 = y0 + w
        x1 = x0 + w

        wavelength_sum[y0:y1, x0:x1] += wl
        wavelength_count[y0:y1, x0:x1] += 1

        sin_sum[y0:y1, x0:x1] += np.sin(2.0 * dir_rad)
        cos_sum[y0:y1, x0:x1] += np.cos(2.0 * dir_rad)
        angle_count[y0:y1, x0:x1] += 1

        per_window_powmax.append(float(metrics.powmax[idx]))
        per_window_powmean.append(float(metrics.powmean[idx]))
        per_window_uniqueness.append(float(metrics.uniqueness[idx]))
        per_window_within_range.append(bool(metrics.within_range[idx]))
        per_window_tile.append((row, col))

    wavelength = np.full((h, w_img), np.nan, dtype=np.float64)
    valid_wl = wavelength_count > 0
    wavelength[valid_wl] = wavelength_sum[valid_wl] / wavelength_count[valid_wl]

    direction = np.full((h, w_img), np.nan, dtype=np.float64)
    valid_angle = angle_count > 0
    sin_avg = sin_sum[valid_angle] / angle_count[valid_angle]
    cos_avg = cos_sum[valid_angle] / angle_count[valid_angle]
    dir_avg = np.arctan2(sin_avg, cos_avg) / 2.0
    dir_avg = np.mod(dir_avg, np.pi)
    direction[valid_angle] = dir_avg

    validity = _connected_component_filter(wavelength_count, min_patch_size)

    per_window = {
        "powmax": np.array(per_window_powmax),
        "powmean": np.array(per_window_powmean),
        "uniqueness": np.array(per_window_uniqueness),
        "within_range": np.array(per_window_within_range),
        "tile": per_window_tile,
    }

    metadata = {
        "n_windows": grid.n_windows,
        "n_valid_windows": int(np.sum(metrics.valid)),
        "n_merged_windows": len(per_window_powmax),
        "min_power": min_power,
        "min_patch_size": min_patch_size,
    }

    return LPPResult(
        wavelength=wavelength, direction=direction, validity=validity,
        metadata=metadata, per_window=per_window,
    )


def _connected_component_filter(
    overlap_count: NDArray[np.integer], min_patch_size: int
) -> NDArray[np.bool_]:
    """Filter valid pixels by connected-component patch size.

    A pixel is valid only if it belongs to a connected component of
    overlapping windows with at least min_patch_size pixels.
    """
    has_overlap = overlap_count > 0
    if min_patch_size <= 1:
        return has_overlap

    try:
        from scipy.ndimage import label
    except ImportError as exc:
        raise RuntimeError(
            "min_patch_size > 1 requires SciPy; install the 'analysis' extra "
            "or set min_patch_size=1 to disable component filtering"
        ) from exc

    labeled, n_features = label(has_overlap)
    validity = np.zeros_like(has_overlap)
    for label_idx in range(1, n_features + 1):
        component = labeled == label_idx
        if np.sum(component) >= min_patch_size:
            validity |= component
    return validity


def apply_edge_policy(
    result: LPPResult,
    grid: WindowGrid,
    edge_policy: str = "trim",
) -> LPPResult:
    """Apply edge policy to handle pixels near image boundaries.

    Parameters
    ----------
    result:
        Merged LPP result.
    grid:
        Window grid descriptor.
    edge_policy:
        One of:
        - "trim": trim half window width from valid-pattern boundary (MATLAB behavior)
        - "keep": keep as-is
        - "nan": set all edge pixels to NaN

    Returns
    -------
    LPPResult
        Result with edge policy applied.
    """
    if edge_policy == "keep":
        return result

    h, w_img = result.wavelength.shape
    w = grid.w
    dL = grid.dL

    if grid.n_rows > 0 and grid.n_cols > 0:
        y_max = (grid.n_rows - 1) * dL + w
        x_max = (grid.n_cols - 1) * dL + w
    else:
        y_max = 0
        x_max = 0

    if edge_policy == "trim":
        # MATLAB trims half window width from the valid-pattern boundary
        trim = w // 2
        valid_region = np.zeros((h, w_img), dtype=bool)
        y_start = min(trim, h)
        x_start = min(trim, w_img)
        y_end = min(max(y_max - trim, y_start), h)
        x_end = min(max(x_max - trim, x_start), w_img)
        valid_region[y_start:y_end, x_start:x_end] = True

        result.wavelength[~valid_region] = np.nan
        result.direction[~valid_region] = np.nan
        result.validity[~valid_region] = False
    elif edge_policy == "nan":
        edge_mask = np.ones((h, w_img), dtype=bool)
        margin = w // 2
        y_lo = min(margin, h)
        x_lo = min(margin, w_img)
        y_hi = min(max(h - margin, y_lo), h)
        x_hi = min(max(w_img - margin, x_lo), w_img)
        edge_mask[y_lo:y_hi, x_lo:x_hi] = False
        result.wavelength[edge_mask] = np.nan
        result.direction[edge_mask] = np.nan
        result.validity[edge_mask] = False

    return result
