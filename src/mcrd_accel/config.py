"""Typed, immutable configurations for solver, validation, LPP, and backend selection.

All configuration objects are frozen dataclasses with explicit validation. Invalid
inputs raise :class:`ConfigurationError` before any allocation or compilation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Self


class ConfigurationError(ValueError):
    """Raised when a configuration field fails validation.

    Attributes
    ----------
    field:
        Name of the offending field (dotted path for nested configs).
    reason:
        Human-readable explanation of the failure.
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


class Integrator(str, Enum):
    """Supported time integrators."""

    EXPLICIT_EULER = "explicit_euler"


class BoundaryMode(str, Enum):
    """Supported boundary conditions."""

    PERIODIC = "periodic"


class DType(str, Enum):
    """Supported array dtypes."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class Backend(str, Enum):
    """Supported compute backends."""

    NUMPY = "numpy"
    JAX_CPU = "jax-cpu"
    JAX_METAL = "jax-metal"
    JAX_CUDA = "jax-cuda"


_PAPER_FLOAT_FIELDS = ("kd", "k1", "b", "k2", "rho", "cv", "beta", "Du", "Dv")


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for a two-field MCRD simulation.

    Parameters follow the notation of the paper notebooks (Fig. 2b default profile).
    The conserved quantity is ``u + v``; diffusion coefficients satisfy ``Du << Dv``.
    """

    # Paper parameters
    kd: float
    k1: float
    b: float
    k2: float
    rho: float
    cv: float
    beta: float
    Du: float
    Dv: float

    # Grid
    Nx: int
    Ny: int
    dx: float
    dy: float

    # Time
    dt: float
    n_steps: int

    # Method
    integrator: Integrator = Integrator.EXPLICIT_EULER
    boundary: BoundaryMode = BoundaryMode.PERIODIC

    # Execution
    dtype: DType = DType.FLOAT32
    backend: Backend = Backend.NUMPY
    seed: int = 0

    # Output
    snapshot_stride: int = 0  # 0 = no snapshots; otherwise save every N steps

    def __post_init__(self) -> None:
        _require_positive_int(self, "Nx", self.Nx)
        _require_positive_int(self, "Ny", self.Ny)
        _require_nonnegative_int(self, "n_steps", self.n_steps)
        _require_nonnegative_int(self, "snapshot_stride", self.snapshot_stride)
        for name in _PAPER_FLOAT_FIELDS:
            _require_finite_positive(self, name, getattr(self, name))
        _require_finite_positive(self, "dx", self.dx)
        _require_finite_positive(self, "dy", self.dy)
        _require_finite_positive(self, "dt", self.dt)
        _require_finite(self, "seed", float(self.seed))

    # ---- Factories -------------------------------------------------------

    @classmethod
    def fig2b(
        cls,
        Nx: int = 256,
        Ny: int = 256,
        dx: float = 0.2,
        dy: float = 0.2,
        dt: float = 0.02,
        n_steps: int = 1_500_000,
        **overrides: Any,
    ) -> Self:
        """Fig. 2b paper parameter profile.

        Provenance
        ----------
        Parameter values are taken from ``DPICs-MCRD-Phase-separation/fig2/Fig2b_code/Fig2_b_sim.ipynb``.
        The notebook defines ``tildeb = 60`` and ``b = tildeb / k3`` with ``k3 = 6.7``;
        we store ``b`` directly (``60 / 6.7``). ``rho = 5.0 * 1.8``, ``beta = 17.8 * 1.1``,
        ``Du = 0.17 * 0.05``, ``Dv = 0.17``, ``cv = 0.016 * 6.7``.

        Notes
        -----
        The paper README states that "within the codes symbol of beta and k2 should be
        switched for their value"; this profile records the values as they appear in the
        notebook source, not the swapped interpretation.
        """
        k3 = 6.7
        base = dict(
            kd=0.36,
            k1=1.35,
            b=60.0 / k3,
            k2=2.0,
            rho=5.0 * 1.8,
            cv=0.016 * k3,
            beta=17.8 * 1.1,
            Du=0.17 * 0.05,
            Dv=0.17,
            Nx=Nx,
            Ny=Ny,
            dx=dx,
            dy=dy,
            dt=dt,
            n_steps=n_steps,
        )
        base.update(overrides)
        return cls(**base)

    # ---- Serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        data = asdict(self)
        data["integrator"] = self.integrator.value
        data["boundary"] = self.boundary.value
        data["dtype"] = self.dtype.value
        data["backend"] = self.backend.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct a config from a dictionary produced by :meth:`to_dict`."""
        coerced = dict(data)
        coerced["integrator"] = Integrator(coerced["integrator"])
        coerced["boundary"] = BoundaryMode(coerced["boundary"])
        coerced["dtype"] = DType(coerced["dtype"])
        coerced["backend"] = Backend(coerced["backend"])
        return cls(**coerced)

    # ---- Derived quantities ---------------------------------------------

    @property
    def Lx(self) -> float:
        """Physical domain width."""
        return self.Nx * self.dx

    @property
    def Ly(self) -> float:
        """Physical domain height."""
        return self.Ny * self.dy

    @property
    def final_time(self) -> float:
        """Total simulated time."""
        return self.n_steps * self.dt


@dataclass(frozen=True)
class ResourceProfile:
    """Budget for a validation case.

    Used to skip cases that would exceed available memory or runtime before
    allocating large arrays.
    """

    max_resident_bytes: int = 4 * 1024**3  # 4 GiB default
    max_steps: int = 10_000_000
    checkpoint_stride: int = 1000
    allow_compile_cache_reuse: bool = True

    def __post_init__(self) -> None:
        _require_positive_int(self, "max_resident_bytes", self.max_resident_bytes)
        _require_positive_int(self, "max_steps", self.max_steps)
        _require_positive_int(self, "checkpoint_stride", self.checkpoint_stride)


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for multi-level validation studies.

    The grid and time-step series are explicit; the validator must never silently
    substitute a different series.
    """

    solver: SolverConfig
    grid_sizes: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096]
    )
    time_steps: list[float] = field(default_factory=lambda: [0.02, 0.01, 0.005, 0.0025])
    conservation_tol: float = 1e-5
    resource: ResourceProfile = field(default_factory=ResourceProfile)

    def __post_init__(self) -> None:
        if len(self.grid_sizes) == 0:
            raise ConfigurationError("grid_sizes", "must contain at least one entry")
        if len(self.time_steps) == 0:
            raise ConfigurationError("time_steps", "must contain at least one entry")
        for i, n in enumerate(self.grid_sizes):
            if not isinstance(n, int) or n <= 0:
                raise ConfigurationError(
                    f"grid_sizes[{i}]", f"expected positive int, got {n!r}"
                )
        for i, dt in enumerate(self.time_steps):
            if not math.isfinite(dt) or dt <= 0:
                raise ConfigurationError(
                    f"time_steps[{i}]", f"expected positive finite float, got {dt!r}"
                )
        _require_finite_positive(self, "conservation_tol", self.conservation_tol)


@dataclass(frozen=True)
class LPPConfig:
    """Configuration for the LPP image-analysis pipeline."""

    w: int = 64  # window size (pixels)
    dL: int = 8  # window stride
    minL: float = 2.0  # minimum wavelength
    maxL: float = 24.0  # maximum wavelength
    radial_tol: float = 1.0
    angular_tol_deg: float = 5.0
    peak_ratio: float = 0.5
    batch_size: int = 64
    min_power: float = 1e-6
    min_patch_size: int = 4
    edge_policy: str = "trim"

    def __post_init__(self) -> None:
        _require_positive_int(self, "w", self.w)
        _require_positive_int(self, "dL", self.dL)
        _require_positive_int(self, "batch_size", self.batch_size)
        _require_positive_int(self, "min_patch_size", self.min_patch_size)
        _require_finite_positive(self, "minL", self.minL)
        _require_finite_positive(self, "maxL", self.maxL)
        if self.maxL <= self.minL:
            raise ConfigurationError(
                "maxL", f"must exceed minL (minL={self.minL}, maxL={self.maxL})"
            )
        _require_finite_positive(self, "radial_tol", self.radial_tol)
        _require_finite_positive(self, "angular_tol_deg", self.angular_tol_deg)
        _require_finite_positive(self, "peak_ratio", self.peak_ratio)
        if self.peak_ratio > 1.0:
            raise ConfigurationError(
                "peak_ratio", f"must be in (0, 1], got {self.peak_ratio}"
            )
        if self.edge_policy not in {"trim", "keep", "nan"}:
            raise ConfigurationError(
                "edge_policy",
                f"must be one of 'trim', 'keep', 'nan'; got {self.edge_policy!r}",
            )


@dataclass(frozen=True)
class BackendConfig:
    """Requested backend and device capabilities."""

    backend: Backend = Backend.NUMPY
    device: str = "cpu"
    dtype: DType = DType.FLOAT32
    require_fft: bool = False

    def __post_init__(self) -> None:
        if self.backend is Backend.NUMPY and self.device not in {"cpu"}:
            raise ConfigurationError(
                "device",
                f"numpy backend only supports 'cpu'; got {self.device!r}",
            )


# ---- Validation helpers ----------------------------------------------------


def _require_positive_int(owner: object, name: str, value: object) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ConfigurationError(name, f"expected positive int, got {value!r}")


def _require_nonnegative_int(owner: object, name: str, value: object) -> None:
    if not isinstance(value, int) or value < 0:
        raise ConfigurationError(name, f"expected non-negative int, got {value!r}")


def _require_finite_positive(owner: object, name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(
            name, f"expected positive finite number, got {value!r}"
        )


def _require_finite(owner: object, name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(name, f"expected finite number, got {value!r}")


__all__ = [
    "Backend",
    "BackendConfig",
    "BoundaryMode",
    "ConfigurationError",
    "DType",
    "Integrator",
    "LPPConfig",
    "ResourceProfile",
    "SolverConfig",
    "ValidationConfig",
]
