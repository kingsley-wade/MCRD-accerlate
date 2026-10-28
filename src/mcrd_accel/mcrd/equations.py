"""Paper reaction term and conserved quantities for the two-field MCRD model.

The reaction term R(u,v) appears in both equations with opposite signs, ensuring
mass conservation: d(u+v)/dt = diffusion terms only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reaction_term(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    kd: float,
    k1: float,
    b: float,
    k2: float,
    rho: float,
    beta: float,
) -> NDArray[np.floating]:
    """Compute the paper reaction term R(u,v).

    R(u,v) = -kd*u + (k1*u + b)*v - rho*k2*beta*u / (k2 + u + v)

    The du/dt reaction contribution is +cv*R(u,v).
    The dv/dt reaction contribution is -cv*R(u,v).

    Parameters
    ----------
    u, v:
        Concentration fields, same shape.
    kd, k1, b, k2, rho, beta:
        Paper parameters (see :class:`mcrd_accel.config.SolverConfig`).

    Returns
    -------
    NDArray
        Reaction term R(u,v), same shape as u and v.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")

    f_uv = k1 * u + b
    g_uv = (k2 * beta) / (k2 + u + v)
    R = -kd * u + f_uv * v - rho * g_uv * u
    return R


def reaction_contributions(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    kd: float,
    k1: float,
    b: float,
    k2: float,
    rho: float,
    cv: float,
    beta: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the reaction contributions to du/dt and dv/dt.

    Returns
    -------
    tuple[NDArray, NDArray]
        (du_dt_reaction, dv_dt_reaction) where du_dt_reaction = +cv*R(u,v)
        and dv_dt_reaction = -cv*R(u,v). These cancel pointwise.
    """
    R = reaction_term(u, v, kd, k1, b, k2, rho, beta)
    du_dt_reaction = cv * R
    dv_dt_reaction = -cv * R
    return du_dt_reaction, dv_dt_reaction


def conserved_total(u: NDArray[np.floating], v: NDArray[np.floating]) -> float:
    """Compute the domain sum of u + v (the conserved quantity).

    Parameters
    ----------
    u, v:
        Concentration fields, same shape.

    Returns
    -------
    float
        Sum of u + v over the entire domain.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")
    return float(np.sum(u + v))


def relative_mass_drift(
    mass_t: float, mass_0: float, dtype: np.dtype = np.float32
) -> float:
    """Compute the relative drift in conserved mass.

    drift = (mass_t - mass_0) / max(abs(mass_0), tiny)

    where tiny is the smallest representable positive number for the dtype.

    Parameters
    ----------
    mass_t:
        Mass at time t.
    mass_0:
        Initial mass.
    dtype:
        Array dtype (used to determine the tolerance floor).

    Returns
    -------
    float
        Relative drift (dimensionless).
    """
    tiny = np.finfo(dtype).tiny
    denominator = max(abs(mass_0), tiny)
    return (mass_t - mass_0) / denominator
