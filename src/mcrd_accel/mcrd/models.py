"""Small reaction networks used to validate the shared PDE solver path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import SolverConfig


@dataclass(frozen=True)
class ReactionModel:
    """Two-species, one-reaction model with an explicit conservation law."""

    name: str
    rate_kind: str
    stoichiometry: tuple[float, float]
    conserved_weights: tuple[float, float]
    parameters: tuple[float, ...]

    @classmethod
    def linear_exchange(cls, k_forward: float = 1.0, k_reverse: float = 0.5) -> ReactionModel:
        _require_nonnegative(k_forward, "k_forward", allow_zero=False)
        _require_nonnegative(k_reverse, "k_reverse")
        return cls(
            "linear_exchange", "linear_exchange", (-1.0, 1.0), (1.0, 1.0),
            (float(k_forward), float(k_reverse)),
        )

    @classmethod
    def nonlinear_autocatalysis(cls, rate: float = 1.0) -> ReactionModel:
        _require_nonnegative(rate, "rate", allow_zero=False)
        return cls(
            "nonlinear_autocatalysis", "nonlinear_autocatalysis", (-1.0, 1.0),
            (1.0, 1.0), (float(rate),),
        )

    @classmethod
    def p53_mcrd(cls, config: SolverConfig) -> ReactionModel:
        return cls(
            "p53_mcrd", "p53_mcrd", (1.0, -1.0), (1.0, 1.0),
            (config.kd, config.k1, config.b, config.k2, config.rho, config.cv, config.beta),
        )

    def rate_numpy(
        self, u: NDArray[np.floating], v: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        if self.rate_kind == "linear_exchange":
            k_forward, k_reverse = self.parameters
            return k_forward * u - k_reverse * v
        if self.rate_kind == "nonlinear_autocatalysis":
            (rate,) = self.parameters
            return rate * u * v
        if self.rate_kind == "p53_mcrd":
            kd, k1, b, k2, rho, cv, beta = self.parameters
            return cv * (
                -kd * u + (k1 * u + b) * v - rho * k2 * beta * u / (k2 + u + v)
            )
        raise ValueError(f"Unsupported reaction model {self.rate_kind!r}")

    def rate_jax(self, u, v):
        if self.rate_kind == "linear_exchange":
            k_forward, k_reverse = self.parameters
            return k_forward * u - k_reverse * v
        if self.rate_kind == "nonlinear_autocatalysis":
            (rate,) = self.parameters
            return rate * u * v
        if self.rate_kind == "p53_mcrd":
            kd, k1, b, k2, rho, cv, beta = self.parameters
            return cv * (
                -kd * u + (k1 * u + b) * v - rho * k2 * beta * u / (k2 + u + v)
            )
        raise ValueError(f"Unsupported reaction model {self.rate_kind!r}")

    def conserved_total(
        self, u: NDArray[np.floating], v: NDArray[np.floating]
    ) -> float:
        if u.shape != v.shape:
            raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")
        cu, cv = self.conserved_weights
        return float(np.sum(cu * u + cv * v, dtype=np.float64))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "stoichiometry": self.stoichiometry,
            "conserved_weights": self.conserved_weights,
            "parameters": self.parameters,
        }


def make_model_initial_state(config: SolverConfig, model: ReactionModel):
    """Make a deterministic positive random state for non-paper test models."""
    from .solver import State, make_initial_state

    if model.rate_kind == "p53_mcrd":
        return make_initial_state(config)

    rng = np.random.default_rng(config.seed)
    dtype = np.float32 if config.dtype.value == "float32" else np.float64
    u = rng.uniform(0.25, 0.75, (config.Ny, config.Nx)).astype(dtype)
    v = rng.uniform(0.25, 0.75, (config.Ny, config.Nx)).astype(dtype)
    return State(u=u, v=v)


def _require_nonnegative(value: float, name: str, *, allow_zero: bool = True) -> None:
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value!r}")
