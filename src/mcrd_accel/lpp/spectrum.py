"""FFT, radial/angular bin aggregation for LPP spectral analysis.

For each window:
1. Subtract mean
2. Compute 2D FFT
3. Shift zero frequency to center
4. Compute power spectrum
5. Aggregate into radial (wavelength) and angular (direction) bins

Bin maps are precomputed once per window shape and reused across all windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BinMaps:
    """Precomputed bin indices for a given window shape.

    Attributes
    ----------
    radial:
        Radial bin index for each frequency bin, shape (w, w). -1 for bins outside [minL, maxL].
    angular:
        Angular bin index for each frequency bin, shape (w, w). -1 for DC component.
    n_radial:
        Number of radial bins.
    n_angular:
        Number of angular bins.
    """

    radial: NDArray[np.intp]
    angular: NDArray[np.intp]
    n_radial: int
    n_angular: int


def compute_bin_maps(
    w: int, minL: float, maxL: float, n_radial: int | None = None, n_angular: int = 36, pad_factor: int = 8
) -> BinMaps:
    """Precompute radial and angular bin maps for a window of size w×w.

    Parameters
    ----------
    w:
        Window size (pixels).
    minL, maxL:
        Wavelength range of interest (pixels).
    n_radial:
        Number of radial bins. If None, uses a fine-grained binning.
    n_angular:
        Number of angular bins (default 36, i.e., 5° resolution).
    pad_factor:
        FFT zero-padding factor (default 4). Must match power spectrum computation.

    Returns
    -------
    BinMaps
        Precomputed bin indices.
    """
    if w <= 0:
        raise ValueError(f"w must be positive, got {w}")
    if minL <= 0 or maxL <= 0:
        raise ValueError(f"minL and maxL must be positive")
    if maxL <= minL:
        raise ValueError(f"maxL must exceed minL")

    # FFT size after zero-padding
    w_padded = w * pad_factor

    # Frequency grid (after fftshift, zero frequency at center)
    # For a w_padded×w_padded FFT, frequencies range from -w_padded/2 to w_padded/2-1 (even w_padded)
    freq = np.fft.fftfreq(w_padded, d=1.0)  # cycles per pixel
    freq_shifted = np.fft.fftshift(freq)

    # 2D frequency grid
    KX, KY = np.meshgrid(freq_shifted, freq_shifted)

    # Radial frequency magnitude (cycles per pixel)
    K = np.sqrt(KX**2 + KY**2)

    # Wavelength (pixels) = 1 / frequency
    # Avoid division by zero at DC
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(K > 0, 1.0 / K, np.inf)

    # Radial bins: map wavelength to bin index
    # Bin 0 = smallest wavelength (highest frequency), bin n-1 = largest wavelength
    if n_radial is None:
        # Default: fine-grained binning for sub-pixel wavelength resolution
        # Use at least 4 bins per pixel of wavelength range
        n_radial = max(100, int(4 * (maxL - minL)))

    # Radial bin index: 0 for L < minL, n_radial-1 for L > maxL, -1 for DC
    radial = np.full((w_padded, w_padded), -1, dtype=np.intp)
    valid = (L >= minL) & (L <= maxL)
    # Normalize to [0, 1] then scale to [0, n_radial-1]
    L_normalized = (L[valid] - minL) / (maxL - minL)
    radial[valid] = np.clip((L_normalized * (n_radial - 1)).astype(np.intp), 0, n_radial - 1)

    # Angular bins: map angle to bin index in [0, n_angular)
    # Direction is a director (orientation without sign), so fold to [0, π)
    angle = np.arctan2(KY, KX)  # [-π, π]
    # Fold to [0, π)
    angle_director = np.mod(angle, np.pi)
    # Bin index: 0 for angle 0, n_angular-1 for angle π-ε
    angular = np.full((w_padded, w_padded), -1, dtype=np.intp)
    # Exclude DC component (K == 0)
    non_dc = K > 0
    angular[non_dc] = np.clip((angle_director[non_dc] / np.pi * n_angular).astype(np.intp), 0, n_angular - 1)

    return BinMaps(radial=radial, angular=angular, n_radial=n_radial, n_angular=n_angular)


def compute_power_spectrum(window: NDArray[np.floating], pad_factor: int = 8) -> NDArray[np.floating]:
    """Compute the power spectrum of a single window.

    Steps:
    1. Subtract mean
    2. Zero-pad to finer frequency resolution
    3. 2D FFT
    4. fftshift (zero frequency to center)
    5. Power = |FFT|²

    Parameters
    ----------
    window:
        2D array of shape (w, w).
    pad_factor:
        Factor to increase FFT size (default 4). Higher = finer frequency resolution.

    Returns
    -------
    NDArray
        Power spectrum, shape (pad_factor*w, pad_factor*w).
    """
    if window.ndim != 2:
        raise ValueError(f"window must be 2D, got shape {window.shape}")

    # Subtract mean
    centered = window - window.mean()

    # Zero-pad for finer frequency resolution
    w = window.shape[0]
    w_padded = w * pad_factor
    padded = np.zeros((w_padded, w_padded), dtype=centered.dtype)
    padded[:w, :w] = centered

    # 2D FFT and shift
    fft = np.fft.fft2(padded)
    fft_shifted = np.fft.fftshift(fft)

    # Power spectrum
    power = np.abs(fft_shifted) ** 2

    return power


def compute_power_spectrum_batch(
    windows: NDArray[np.floating],
    pad_factor: int = 8,
) -> NDArray[np.floating]:
    """Compute power spectra for a batch of windows.

    Parameters
    ----------
    windows:
        3D array of shape (batch_size, w, w).
    pad_factor:
        Factor to increase FFT size (default 4). Higher = finer frequency resolution.

    Returns
    -------
    NDArray
        Power spectra, shape (batch_size, pad_factor*w, pad_factor*w).
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must be 3D (batch, h, w), got shape {windows.shape}")

    # Subtract mean per window
    centered = windows - windows.mean(axis=(1, 2), keepdims=True)

    # Zero-pad for finer frequency resolution
    batch_size, w, _ = windows.shape
    w_padded = w * pad_factor
    padded = np.zeros((batch_size, w_padded, w_padded), dtype=centered.dtype)
    padded[:, :w, :w] = centered

    # 2D FFT over last two axes
    fft = np.fft.fft2(padded, axes=(-2, -1))
    fft_shifted = np.fft.fftshift(fft, axes=(-2, -1))

    # Power spectrum
    power = np.abs(fft_shifted) ** 2

    return power


def aggregate_spectrum(
    power: NDArray[np.floating],
    bin_maps: BinMaps,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Aggregate power spectrum into radial and angular bins.

    Parameters
    ----------
    power:
        Power spectrum of shape (w, w) or (batch, w, w).
    bin_maps:
        Precomputed bin indices.

    Returns
    -------
    tuple[NDArray, NDArray]
        (radial_power, angular_power) where:
        - radial_power has shape (..., n_radial)
        - angular_power has shape (..., n_angular)
    """
    if power.ndim == 2:
        # Single window
        radial_sum = np.zeros(bin_maps.n_radial, dtype=power.dtype)
        angular_sum = np.zeros(bin_maps.n_angular, dtype=power.dtype)
        radial_count = np.zeros(bin_maps.n_radial, dtype=np.intp)
        angular_count = np.zeros(bin_maps.n_angular, dtype=np.intp)

        # Radial aggregation
        valid_radial = bin_maps.radial >= 0
        radial_indices = bin_maps.radial[valid_radial]
        radial_values = power[valid_radial]
        np.add.at(radial_sum, radial_indices, radial_values)
        np.add.at(radial_count, radial_indices, 1)

        # Angular aggregation: restrict to wavelength band [minL, maxL]
        # Only include pixels where radial >= 0 (within wavelength range)
        # AND angular >= 0 (non-DC)
        in_band = (bin_maps.radial >= 0) & (bin_maps.angular >= 0)
        angular_indices = bin_maps.angular[in_band]
        angular_values = power[in_band]
        np.add.at(angular_sum, angular_indices, angular_values)
        np.add.at(angular_count, angular_indices, 1)

        # Average (avoid division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            radial_power = np.where(radial_count > 0, radial_sum / radial_count, 0.0)
            angular_power = np.where(angular_count > 0, angular_sum / angular_count, 0.0)

        return radial_power, angular_power

    elif power.ndim == 3:
        # Batch: process each window
        batch_size = power.shape[0]
        radial_power = np.zeros((batch_size, bin_maps.n_radial), dtype=power.dtype)
        angular_power = np.zeros((batch_size, bin_maps.n_angular), dtype=power.dtype)

        for i in range(batch_size):
            r, a = aggregate_spectrum(power[i], bin_maps)
            radial_power[i] = r
            angular_power[i] = a

        return radial_power, angular_power

    else:
        raise ValueError(f"power must be 2D or 3D, got shape {power.shape}")
