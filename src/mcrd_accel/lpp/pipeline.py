"""Public LPP entry point: analyze an image and return merged wavelength/direction fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import LPPConfig
from .merge import LPPResult, apply_edge_policy, merge_lpp_windows
from .peaks import LPPWindowMetrics, compute_peak_metrics
from .spectrum import aggregate_spectrum, compute_bin_maps, compute_power_spectrum_batch
from .windows import compute_window_grid, extract_windows_batched


def analyze(
    image: NDArray[np.floating],
    config: LPPConfig,
    *,
    backend: str | None = None,
) -> LPPResult:
    """Analyze an image using the LPP pipeline.

    Parameters
    ----------
    image:
        2D grayscale image of shape (height, width).
    config:
        LPP configuration (window size, stride, wavelength range, etc.).
    backend:
        Optional backend name. Only ``None`` and ``"numpy"`` are implemented.

    Returns
    -------
    LPPResult
        Merged wavelength and direction fields on the image grid.

    Notes
    -----
    The pipeline:
    1. Compute window grid
    2. Extract windows in bounded batches
    3. For each batch: compute 2D FFT, aggregate, find peaks (incremental)
    4. Merge overlapping windows using scatter accumulation
    5. Apply edge policy
    """
    if backend not in (None, "numpy"):
        raise ValueError(
            f"LPP backend {backend!r} is unavailable; supported backend is 'numpy'"
        )

    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")

    h, w_img = image.shape

    grid = compute_window_grid(image.shape, config.w, config.dL)

    if grid.n_windows == 0:
        return LPPResult(
            wavelength=np.full((h, w_img), np.nan, dtype=np.float64),
            direction=np.full((h, w_img), np.nan, dtype=np.float64),
            validity=np.zeros((h, w_img), dtype=bool),
            metadata={"n_windows": 0, "note": "Image too small for window size"},
        )

    bin_maps = compute_bin_maps(
        w=config.w,
        minL=config.minL,
        maxL=config.maxL,
        n_angular=36,
    )

    # Process batches incrementally: compute metrics per batch, accumulate
    batch_metrics: list[LPPWindowMetrics] = []

    for batch, coords in extract_windows_batched(image, config.w, config.dL, batch_size=config.batch_size):
        power_batch = compute_power_spectrum_batch(batch)

        radial_batch = np.zeros((power_batch.shape[0], bin_maps.n_radial), dtype=power_batch.dtype)
        angular_batch = np.zeros((power_batch.shape[0], bin_maps.n_angular), dtype=power_batch.dtype)
        for i in range(power_batch.shape[0]):
            radial_batch[i], angular_batch[i] = aggregate_spectrum(power_batch[i], bin_maps)

        metrics = compute_peak_metrics(
            radial_batch, angular_batch, bin_maps,
            minL=config.minL, maxL=config.maxL,
            peak_ratio=config.peak_ratio,
        )
        batch_metrics.append(metrics)

    # Concatenate per-batch metrics into a single set
    all_metrics = _concat_metrics(batch_metrics)

    result = merge_lpp_windows(
        grid=grid,
        metrics=all_metrics,
        image_shape=image.shape,
        min_power=config.min_power,
        min_patch_size=config.min_patch_size,
    )

    result = apply_edge_policy(result, grid, edge_policy=config.edge_policy)

    result.metadata["image_shape"] = image.shape
    result.metadata["config"] = {
        "w": config.w,
        "dL": config.dL,
        "minL": config.minL,
        "maxL": config.maxL,
        "batch_size": config.batch_size,
    }

    return result


def _concat_metrics(batch_metrics: list[LPPWindowMetrics]) -> LPPWindowMetrics:
    """Concatenate per-batch LPPWindowMetrics into a single instance."""
    if len(batch_metrics) == 1:
        return batch_metrics[0]

    return LPPWindowMetrics(
        wavelength=np.concatenate([m.wavelength for m in batch_metrics]),
        direction=np.concatenate([m.direction for m in batch_metrics]),
        powmax=np.concatenate([m.powmax for m in batch_metrics]),
        powmean=np.concatenate([m.powmean for m in batch_metrics]),
        uniqueness=np.concatenate([m.uniqueness for m in batch_metrics]),
        within_range=np.concatenate([m.within_range for m in batch_metrics]),
        valid=np.concatenate([m.valid for m in batch_metrics]),
    )
