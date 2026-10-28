"""Periodic finite-difference operators for canonical PDE tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def periodic_laplacian_jax(field, dx: float, dy: float):
    import jax.numpy as jnp

    return (
        (jnp.roll(field, 1, axis=1) + jnp.roll(field, -1, axis=1) - 2.0 * field) / dx**2
        + (jnp.roll(field, 1, axis=0) + jnp.roll(field, -1, axis=0) - 2.0 * field) / dy**2
    )


def require_field_shape(field: NDArray[np.floating], shape: tuple[int, int], name: str) -> None:
    if field.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {field.shape}")
