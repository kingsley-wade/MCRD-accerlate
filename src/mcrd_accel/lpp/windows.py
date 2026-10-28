"""Bounded overlapping-window extraction for LPP analysis.

Extracts w×w windows from an image with stride dL, processing in bounded batches
to avoid materializing all windows simultaneously (which could exceed memory for
large images with high overlap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class WindowGrid:
    """Describes the grid of windows extracted from an image.

    Attributes
    ----------
    image_shape:
        (height, width) of the source image.
    w:
        Window size (pixels).
    dL:
        Stride between window centers (pixels).
    n_rows:
        Number of window rows.
    n_cols:
        Number of window columns.
    n_windows:
        Total number of windows.
    """

    image_shape: tuple[int, int]
    w: int
    dL: int
    n_rows: int
    n_cols: int
    n_windows: int

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (n_rows, n_cols)."""
        return (self.n_rows, self.n_cols)


def compute_window_grid(image_shape: tuple[int, int], w: int, dL: int) -> WindowGrid:
    """Compute the window grid for a given image and window parameters.

    Parameters
    ----------
    image_shape:
        (height, width) of the source image.
    w:
        Window size (pixels).
    dL:
        Stride between window top-left corners (pixels).

    Returns
    -------
    WindowGrid
        Grid descriptor with row/column counts and total windows.

    Notes
    -----
    Windows are extracted starting from (0, 0) with stride dL. The last window
    must fit entirely within the image (no padding). If the image is smaller than
    w in either dimension, n_rows or n_cols will be 0.
    """
    h, w_img = image_shape
    if w <= 0:
        raise ValueError(f"w must be positive, got {w}")
    if dL <= 0:
        raise ValueError(f"dL must be positive, got {dL}")

    n_rows = max(0, (h - w) // dL + 1)
    n_cols = max(0, (w_img - w) // dL + 1)
    n_windows = n_rows * n_cols

    return WindowGrid(
        image_shape=image_shape,
        w=w,
        dL=dL,
        n_rows=n_rows,
        n_cols=n_cols,
        n_windows=n_windows,
    )


def extract_window(image: NDArray[np.floating], row: int, col: int, w: int, dL: int) -> NDArray[np.floating]:
    """Extract a single w×w window at grid position (row, col).

    Parameters
    ----------
    image:
        2D source image.
    row, col:
        Window grid indices (0-based).
    w:
        Window size.
    dL:
        Stride.

    Returns
    -------
    NDArray
        Window of shape (w, w).
    """
    y0 = row * dL
    x0 = col * dL
    return image[y0 : y0 + w, x0 : x0 + w]


def extract_windows_batched(
    image: NDArray[np.floating],
    w: int,
    dL: int,
    batch_size: int = 64,
) -> Iterator[tuple[NDArray[np.floating], list[tuple[int, int]]]]:
    """Extract windows in bounded batches.

    Yields batches of windows along with their (row, col) grid coordinates.
    This avoids materializing all windows simultaneously, keeping memory usage
    bounded by batch_size * w * w rather than n_windows * w * w.

    Parameters
    ----------
    image:
        2D source image of shape (height, width).
    w:
        Window size (pixels).
    dL:
        Stride (pixels).
    batch_size:
        Maximum number of windows per batch.

    Yields
    ------
    tuple[NDArray, list[tuple[int, int]]]
        (batch, coordinates) where batch has shape (batch_size, w, w) and
        coordinates is a list of (row, col) grid indices for each window.

    Examples
    --------
    >>> image = np.random.rand(128, 128)
    >>> for batch, coords in extract_windows_batched(image, w=32, dL=8, batch_size=16):
    ...     print(batch.shape, len(coords))
    (16, 32, 32) 16
    ...
    """
    grid = compute_window_grid(image.shape, w, dL)

    if grid.n_windows == 0:
        return

    # Iterate over windows in row-major order
    for start in range(0, grid.n_windows, batch_size):
        end = min(start + batch_size, grid.n_windows)
        batch_windows = []
        batch_coords = []

        for idx in range(start, end):
            row = idx // grid.n_cols
            col = idx % grid.n_cols
            window = extract_window(image, row, col, w, dL)
            batch_windows.append(window)
            batch_coords.append((row, col))

        # Stack into a single array: (batch_size, w, w)
        batch = np.stack(batch_windows, axis=0)
        yield batch, batch_coords


def extract_all_windows(
    image: NDArray[np.floating], w: int, dL: int
) -> tuple[NDArray[np.floating], list[tuple[int, int]]]:
    """Extract all windows at once (convenience wrapper, not memory-bounded).

    Use :func:`extract_windows_batched` for large images.

    Parameters
    ----------
    image:
        2D source image.
    w:
        Window size.
    dL:
        Stride.

    Returns
    -------
    tuple[NDArray, list[tuple[int, int]]]
        (all_windows, coordinates) where all_windows has shape (n_windows, w, w).
    """
    grid = compute_window_grid(image.shape, w, dL)
    all_windows = []
    all_coords = []

    for idx in range(grid.n_windows):
        row = idx // grid.n_cols
        col = idx % grid.n_cols
        window = extract_window(image, row, col, w, dL)
        all_windows.append(window)
        all_coords.append((row, col))

    if all_windows:
        return np.stack(all_windows, axis=0), all_coords
    else:
        return np.array([]).reshape(0, w, w), []
