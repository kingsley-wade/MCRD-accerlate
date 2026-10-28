"""Grid and backend settings shared by the canonical PDE cases."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import Backend, DType


@dataclass(frozen=True)
class GridConfig:
    Nx: int
    Ny: int
    dx: float
    dy: float
    backend: Backend = Backend.NUMPY
    dtype: DType = DType.FLOAT64
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.Nx, int) or self.Nx < 2:
            raise ValueError(f"Nx must be an integer >= 2, got {self.Nx!r}")
        if not isinstance(self.Ny, int) or self.Ny < 2:
            raise ValueError(f"Ny must be an integer >= 2, got {self.Ny!r}")
        if not math.isfinite(self.dx) or self.dx <= 0:
            raise ValueError(f"dx must be positive and finite, got {self.dx!r}")
        if not math.isfinite(self.dy) or self.dy <= 0:
            raise ValueError(f"dy must be positive and finite, got {self.dy!r}")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.Ny, self.Nx)

    @property
    def numpy_dtype(self):
        return np.float32 if self.dtype is DType.FLOAT32 else np.float64

    def to_dict(self) -> dict[str, object]:
        return {
            "Nx": self.Nx,
            "Ny": self.Ny,
            "dx": self.dx,
            "dy": self.dy,
            "backend": self.backend.value,
            "dtype": self.dtype.value,
            "seed": self.seed,
        }
