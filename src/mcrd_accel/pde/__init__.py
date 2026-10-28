"""Canonical parabolic, elliptic, and hyperbolic PDE verification cases."""

from .config import GridConfig
from .elliptic import PoissonResult, solve_periodic_poisson
from .parabolic import TimePDEResult, run_heat
from .hyperbolic import run_wave

__all__ = [
    "GridConfig",
    "PoissonResult",
    "TimePDEResult",
    "run_heat",
    "run_wave",
    "solve_periodic_poisson",
]
