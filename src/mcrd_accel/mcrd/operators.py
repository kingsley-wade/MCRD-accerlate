"""Periodic finite-difference operators (Laplacian, etc.).

All operators are out-of-place: they return new arrays without modifying inputs.
The periodic discrete Laplacian satisfies the conservation property: its domain
sum is zero up to floating-point reduction error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def periodic_laplacian(
    field: NDArray[np.floating], dx: float, dy: float
) -> NDArray[np.floating]:
    """Compute the 5-point periodic Laplacian of a 2D field.

    Uses the standard finite-difference stencil with periodic boundary conditions
    via ``numpy.roll``. The result is a new array; the input is not modified.

    Parameters
    ----------
    field:
        2D array of shape ``(Ny, Nx)``.
    dx, dy:
        Grid spacing in the x (column) and y (row) directions.

    Returns
    -------
    NDArray
        Laplacian of the field, same shape and dtype as input.

    Notes
    -----
    The discrete periodic Laplacian satisfies ``sum(laplacian(field)) ≈ 0`` up to
    floating-point reduction error. This is the key conservation property tested
    in :mod:`tests.test_mcrd_operators`.
    """
    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape {field.shape}")
    if dx <= 0 or dy <= 0:
        raise ValueError(f"dx and dy must be positive, got dx={dx}, dy={dy}")

    # 5-point stencil with independent x/y spacing:
    # ∇²f ≈ (left + right - 2*center) / dx² + (up + down - 2*center) / dy²
    left = np.roll(field, 1, axis=1)
    right = np.roll(field, -1, axis=1)
    up = np.roll(field, 1, axis=0)
    down = np.roll(field, -1, axis=0)

    laplacian = (left + right - 2.0 * field) / (dx * dx) + (up + down - 2.0 * field) / (dy * dy)
    return laplacian
