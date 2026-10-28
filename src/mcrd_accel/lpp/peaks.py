"""MATLAB-compatible peak/weighted metrics for LPP analysis.

For each window's 2D power spectrum:
1. Find dominant peak in 2D frequency space
2. Sub-bin interpolation for sub-pixel wavelength precision
3. Validate uniqueness (is there a single dominant peak?)
4. Return wavelength, direction, power, uniqueness, and validity masks
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .spectrum import BinMaps


@dataclass
class LPPWindowMetrics:
    """Per-window LPP metrics.

    Attributes
    ----------
    wavelength:
        Dominant wavelength in pixels. NaN if invalid.
    direction:
        Director angle in radians [0, π). NaN if invalid.
    powmax:
        Maximum power in the spectrum.
    powmean:
        Mean power in the spectrum.
    uniqueness:
        Ratio of max to mean power (higher = more unique peak).
    within_range:
        True if wavelength is within [minL, maxL].
    valid:
        True if window has a valid unique peak.
    """

    wavelength: NDArray[np.floating]
    direction: NDArray[np.floating]
    powmax: NDArray[np.floating]
    powmean: NDArray[np.floating]
    uniqueness: NDArray[np.floating]
    within_range: NDArray[np.bool_]
    valid: NDArray[np.bool_]


def _subbin_interpolate_2d(
    power: NDArray[np.floating],
    peak_row: int,
    peak_col: int,
    freq_spacing: float,
) -> tuple[float, float]:
    """Parabolic sub-bin interpolation of 2D spectral peak.

    Returns interpolated (kx_frac, ky_frac) fractional bin shifts.
    Only applies correction when the shift is within [-0.5, 0.5] (reliable range).
    """
    nrows, ncols = power.shape

    # Interpolate along columns (kx direction)
    kx_frac = 0.0
    if 0 < peak_col < ncols - 1:
        alpha = np.log(max(power[peak_row, peak_col - 1], 1e-30))
        beta = np.log(max(power[peak_row, peak_col], 1e-30))
        gamma = np.log(max(power[peak_row, peak_col + 1], 1e-30))
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            frac = 0.5 * (alpha - gamma) / denom
            if abs(frac) <= 0.5:
                kx_frac = frac

    # Interpolate along rows (ky direction)
    ky_frac = 0.0
    if 0 < peak_row < nrows - 1:
        alpha = np.log(max(power[peak_row - 1, peak_col], 1e-30))
        beta = np.log(max(power[peak_row, peak_col], 1e-30))
        gamma = np.log(max(power[peak_row + 1, peak_col], 1e-30))
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            frac = 0.5 * (alpha - gamma) / denom
            if abs(frac) <= 0.5:
                ky_frac = frac

    return kx_frac, ky_frac


def compute_peak_metrics_2d(
    power_batch: NDArray[np.floating],
    bin_maps: BinMaps,
    minL: float,
    maxL: float,
    peak_ratio: float = 0.5,
    pad_factor: int = 4,
) -> LPPWindowMetrics:
    """Compute LPP peak metrics from 2D power spectra using sub-bin interpolation.

    Parameters
    ----------
    power_batch:
        3D array of 2D power spectra, shape (batch_size, w_padded, w_padded).
    bin_maps:
        Precomputed bin maps (provides wavelength/angle mapping for uniqueness check).
    minL, maxL:
        Wavelength range of interest (pixels).
    peak_ratio:
        Threshold for uniqueness: max/mean power must exceed this ratio.
    pad_factor:
        FFT zero-padding factor used to generate the power spectra.

    Returns
    -------
    LPPWindowMetrics
        Per-window metrics with NaN for invalid windows.
    """
    batch_size = power_batch.shape[0]
    w_padded = power_batch.shape[1]

    # Frequency grid for the padded FFT
    freq = np.fft.fftfreq(w_padded, d=1.0)
    freq_shifted = np.fft.fftshift(freq)
    freq_spacing = freq_shifted[1] - freq_shifted[0] if len(freq_shifted) > 1 else 1.0

    # 2D frequency grids
    KX, KY = np.meshgrid(freq_shifted, freq_shifted)

    # Output arrays
    wavelength = np.full(batch_size, np.nan, dtype=np.float64)
    direction = np.full(batch_size, np.nan, dtype=np.float64)
    powmax = np.zeros(batch_size, dtype=np.float64)
    powmean = np.zeros(batch_size, dtype=np.float64)
    uniqueness = np.zeros(batch_size, dtype=np.float64)
    within_range = np.zeros(batch_size, dtype=bool)
    valid = np.zeros(batch_size, dtype=bool)

    for i in range(batch_size):
        power = power_batch[i]

        if np.sum(power) == 0:
            continue

        # Find 2D peak (exclude DC at center)
        dc_row, dc_col = w_padded // 2, w_padded // 2
        power_masked = power.copy()
        # Zero out DC neighborhood to avoid picking DC
        r_margin = max(1, w_padded // 20)
        r_lo = max(0, dc_row - r_margin)
        r_hi = min(w_padded, dc_row + r_margin + 1)
        c_lo = max(0, dc_col - r_margin)
        c_hi = min(w_padded, dc_col + r_margin + 1)
        power_masked[r_lo:r_hi, c_lo:c_hi] = 0

        peak_idx = np.unravel_index(np.argmax(power_masked), power_masked.shape)
        peak_row, peak_col = peak_idx
        peak_power = power[peak_row, peak_col]

        # Sub-bin interpolation (disabled: spectral peaks are sinc-like, not parabolic)
        kx_frac = 0.0
        ky_frac = 0.0

        # Interpolated frequency coordinates
        kx = freq_shifted[peak_col] + kx_frac * freq_spacing
        ky = freq_shifted[peak_row] + ky_frac * freq_spacing

        # Wavelength and direction from 2D peak
        K = np.sqrt(kx**2 + ky**2)
        if K <= 0:
            continue
        wl = 1.0 / K
        theta = np.mod(np.arctan2(ky, kx), np.pi)

        # Uniqueness: compare peak power to mean power in annulus around peak
        K_map = np.sqrt(KX**2 + KY**2)
        K_peak = K
        annulus_width = 2.0 * freq_spacing
        annulus_mask = (np.abs(K_map - K_peak) < annulus_width) & (K_map > 0)
        mean_annulus = np.mean(power[annulus_mask]) if np.any(annulus_mask) else 1e-30
        r_uniqueness = peak_power / max(mean_annulus, 1e-30)

        # Also compute radial/angular uniqueness from aggregated spectra
        radial_power, angular_power = _aggregate_single(power, bin_maps)
        r_max = np.max(radial_power) if np.any(radial_power > 0) else 0
        r_mean = np.mean(radial_power[radial_power > 0]) if np.any(radial_power > 0) else 1e-30
        a_max = np.max(angular_power) if np.any(angular_power > 0) else 0
        a_mean = np.mean(angular_power[angular_power > 0]) if np.any(angular_power > 0) else 1e-30
        combined_uniqueness = (r_max / max(r_mean, 1e-30) + a_max / max(a_mean, 1e-30)) / 2.0

        is_valid = combined_uniqueness >= peak_ratio

        if is_valid:
            in_range = minL <= wl <= maxL
            wavelength[i] = wl
            direction[i] = theta
            powmax[i] = peak_power
            powmean[i] = (r_mean + a_mean) / 2.0
            uniqueness[i] = combined_uniqueness
            within_range[i] = in_range
            valid[i] = True

    return LPPWindowMetrics(
        wavelength=wavelength,
        direction=direction,
        powmax=powmax,
        powmean=powmean,
        uniqueness=uniqueness,
        within_range=within_range,
        valid=valid,
    )


def _aggregate_single(
    power: NDArray[np.floating],
    bin_maps: BinMaps,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Aggregate a single 2D power spectrum into radial and angular bins."""
    radial_sum = np.zeros(bin_maps.n_radial, dtype=power.dtype)
    angular_sum = np.zeros(bin_maps.n_angular, dtype=power.dtype)
    radial_count = np.zeros(bin_maps.n_radial, dtype=np.intp)
    angular_count = np.zeros(bin_maps.n_angular, dtype=np.intp)

    valid_radial = bin_maps.radial >= 0
    radial_indices = bin_maps.radial[valid_radial]
    radial_values = power[valid_radial]
    np.add.at(radial_sum, radial_indices, radial_values)
    np.add.at(radial_count, radial_indices, 1)

    valid_angular = bin_maps.angular >= 0
    angular_indices = bin_maps.angular[valid_angular]
    angular_values = power[valid_angular]
    np.add.at(angular_sum, angular_indices, angular_values)
    np.add.at(angular_count, angular_indices, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        radial_power = np.where(radial_count > 0, radial_sum / radial_count, 0.0)
        angular_power = np.where(angular_count > 0, angular_sum / angular_count, 0.0)

    return radial_power, angular_power


def compute_peak_metrics(
    radial_power: NDArray[np.floating],
    angular_power: NDArray[np.floating],
    bin_maps: BinMaps,
    minL: float,
    maxL: float,
    peak_ratio: float = 0.5,
) -> LPPWindowMetrics:
    """Compute LPP peak metrics from radial and angular power spectra.

    Uses power-weighted centroid around the peak for sub-bin wavelength
    precision. Rejects windows with tied spectral maxima (MATLAB requires
    exactly one maximum).

    Parameters
    ----------
    radial_power:
        Radial power spectrum, shape (..., n_radial).
    angular_power:
        Angular power spectrum, shape (..., n_angular).
    bin_maps:
        Precomputed bin maps (provides wavelength/angle mapping).
    minL, maxL:
        Wavelength range of interest (pixels).
    peak_ratio:
        Threshold for uniqueness: max/mean power must exceed this ratio.

    Returns
    -------
    LPPWindowMetrics
        Per-window metrics with NaN for invalid windows.
    """
    if radial_power.ndim == 1:
        radial_power = radial_power[np.newaxis, :]
        angular_power = angular_power[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    batch_size = radial_power.shape[0]
    n_radial = radial_power.shape[1]
    n_angular = angular_power.shape[1]

    wavelength = np.full(batch_size, np.nan, dtype=np.float64)
    direction = np.full(batch_size, np.nan, dtype=np.float64)
    powmax = np.zeros(batch_size, dtype=np.float64)
    powmean = np.zeros(batch_size, dtype=np.float64)
    uniqueness = np.zeros(batch_size, dtype=np.float64)
    within_range = np.zeros(batch_size, dtype=bool)
    valid = np.zeros(batch_size, dtype=bool)

    radial_bins = np.arange(n_radial)
    wavelength_map = minL + (radial_bins / max(1, n_radial - 1)) * (maxL - minL)

    angular_bins = np.arange(n_angular)
    direction_map = (angular_bins / n_angular) * np.pi

    for i in range(batch_size):
        r_pow = radial_power[i]
        a_pow = angular_power[i]

        if np.sum(r_pow) == 0 or np.sum(a_pow) == 0:
            continue

        radial_max_idx = int(np.argmax(r_pow))
        radial_max_val = r_pow[radial_max_idx]

        # Reject tied maxima: MATLAB requires exactly one maximum
        n_tied = int(np.sum(r_pow == radial_max_val))
        if n_tied > 1:
            continue

        radial_mean_val = np.mean(r_pow[r_pow > 0]) if np.any(r_pow > 0) else 0.0

        angular_max_idx = int(np.argmax(a_pow))
        angular_max_val = a_pow[angular_max_idx]

        n_tied_angular = int(np.sum(a_pow == angular_max_val))
        if n_tied_angular > 1:
            continue

        angular_mean_val = np.mean(a_pow[a_pow > 0]) if np.any(a_pow > 0) else 0.0

        r_uniqueness = radial_max_val / radial_mean_val if radial_mean_val > 0 else 0.0
        a_uniqueness = angular_max_val / angular_mean_val if angular_mean_val > 0 else 0.0
        combined_uniqueness = (r_uniqueness + a_uniqueness) / 2.0

        is_valid = combined_uniqueness >= peak_ratio

        if is_valid:
            # Power-weighted centroid around radial peak for sub-bin precision
            half_w = 2
            lo = max(0, radial_max_idx - half_w)
            hi = min(n_radial, radial_max_idx + half_w + 1)
            neighborhood = r_pow[lo:hi].copy()
            neighborhood = np.maximum(neighborhood, 0.0)
            total_power = np.sum(neighborhood)
            if total_power > 0:
                wl_weighted = float(np.sum(wavelength_map[lo:hi] * neighborhood) / total_power)
            else:
                wl_weighted = float(wavelength_map[radial_max_idx])

            # Power-weighted centroid for direction
            a_indices = (np.arange(angular_max_idx - half_w, angular_max_idx + half_w + 1) % n_angular)
            a_neighborhood = np.maximum(a_pow[a_indices], 0.0)
            a_total = np.sum(a_neighborhood)
            if a_total > 0:
                angles = direction_map[a_indices]
                sin2 = np.sum(np.sin(2.0 * angles) * a_neighborhood)
                cos2 = np.sum(np.cos(2.0 * angles) * a_neighborhood)
                dir_weighted = float(np.mod(0.5 * np.arctan2(sin2, cos2), np.pi))
            else:
                dir_weighted = float(direction_map[angular_max_idx])

            in_range = minL <= wl_weighted <= maxL

            wavelength[i] = wl_weighted
            direction[i] = dir_weighted
            powmax[i] = max(radial_max_val, angular_max_val)
            powmean[i] = (radial_mean_val + angular_mean_val) / 2.0
            uniqueness[i] = combined_uniqueness
            within_range[i] = in_range
            valid[i] = True

    if squeeze:
        return LPPWindowMetrics(
            wavelength=wavelength[0],
            direction=direction[0],
            powmax=powmax[0],
            powmean=powmean[0],
            uniqueness=uniqueness[0],
            within_range=within_range[0],
            valid=valid[0],
        )
    else:
        return LPPWindowMetrics(
            wavelength=wavelength,
            direction=direction,
            powmax=powmax,
            powmean=powmean,
            uniqueness=uniqueness,
            within_range=within_range,
            valid=valid,
        )


def wavelength_to_bin(wavelength: float, minL: float, maxL: float, n_radial: int) -> int:
    """Convert wavelength to radial bin index (inverse of bin map).

    Parameters
    ----------
    wavelength:
        Wavelength in pixels.
    minL, maxL:
        Wavelength range.
    n_radial:
        Number of radial bins.

    Returns
    -------
    int
        Bin index, clipped to [0, n_radial-1].
    """
    if n_radial <= 1:
        return 0
    normalized = (wavelength - minL) / (maxL - minL)
    return int(np.clip(normalized * (n_radial - 1), 0, n_radial - 1))


def direction_to_bin(direction_rad: float, n_angular: int) -> int:
    """Convert direction angle to angular bin index.

    Parameters
    ----------
    direction_rad:
        Director angle in radians [0, π).
    n_angular:
        Number of angular bins.

    Returns
    -------
    int
        Bin index, clipped to [0, n_angular-1].
    """
    normalized = direction_rad / np.pi
    return int(np.clip(normalized * n_angular, 0, n_angular - 1))
