"""Functional stepping and output policy for NumPy and JAX solver backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..backends import JAX_BACKENDS
from ..config import Backend, Integrator, SolverConfig
from .equations import conserved_total, reaction_contributions
from .models import ReactionModel, make_model_initial_state
from .operators import periodic_laplacian


@dataclass
class State:
    """Simulation state: two concentration fields u and v."""

    u: NDArray[np.floating]
    v: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.u.shape != self.v.shape:
            raise ValueError(f"u and v must have the same shape, got {self.u.shape} and {self.v.shape}")
        if self.u.ndim != 2:
            raise ValueError(f"u and v must be 2D, got shape {self.u.shape}")


@dataclass
class Checkpoint:
    """Diagnostic snapshot at a given step."""

    step: int
    time: float
    mass: float
    mass_drift: float
    all_finite: bool


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    final_state: State
    checkpoints: list[Checkpoint] = field(default_factory=list)
    snapshots: list[State] = field(default_factory=list)
    snapshot_times: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def step(
    state: State, config: SolverConfig, model: ReactionModel | None = None
) -> State:
    """Advance the state by one time step using explicit Euler.

    This is an out-of-place update: new arrays are returned, the input is not modified.

    Parameters
    ----------
    state:
        Current state (u, v).
    config:
        Solver configuration (parameters, grid, dt, integrator).

    Returns
    -------
    State
        New state after one time step.
    """
    if config.integrator is not Integrator.EXPLICIT_EULER:
        raise NotImplementedError(f"Integrator {config.integrator} not yet implemented")

    u, v = state.u, state.v

    if model is None:
        du_dt_reaction, dv_dt_reaction = reaction_contributions(
            u, v, config.kd, config.k1, config.b, config.k2, config.rho, config.cv, config.beta
        )
    else:
        reaction_rate = model.rate_numpy(u, v)
        du_dt_reaction = model.stoichiometry[0] * reaction_rate
        dv_dt_reaction = model.stoichiometry[1] * reaction_rate

    # Diffusion (periodic Laplacian)
    lap_u = periodic_laplacian(u, config.dx, config.dy)
    lap_v = periodic_laplacian(v, config.dx, config.dy)

    # Total time derivatives
    du_dt = du_dt_reaction + config.Du * lap_u
    dv_dt = dv_dt_reaction + config.Dv * lap_v

    # Explicit Euler update (out-of-place)
    u_new = u + config.dt * du_dt
    v_new = v + config.dt * dv_dt

    return State(u=u_new, v=v_new)


def make_initial_state(
    config: SolverConfig,
    u_mean: float = 0.5,
    u_amplitude: float = 0.25,
    v_mean: float = 25.0,
    v_amplitude: float = 0.5,
) -> State:
    """Generate a random initial state matching the paper notebooks.

    u ~ Uniform(u_mean - u_amplitude, u_mean + u_amplitude)
    v ~ Uniform(v_mean - v_amplitude, v_mean + v_amplitude)

    Parameters
    ----------
    config:
        Solver configuration (provides Nx, Ny, dtype, seed).
    u_mean, u_amplitude:
        Center and half-width of u initialization.
    v_mean, v_amplitude:
        Center and half-width of v initialization.

    Returns
    -------
    State
        Initial state with random uniform fields.
    """
    rng = np.random.default_rng(config.seed)

    dtype = np.float32 if config.dtype.value == "float32" else np.float64

    u_low = u_mean - u_amplitude
    u_high = u_mean + u_amplitude
    u = rng.uniform(u_low, u_high, (config.Ny, config.Nx)).astype(dtype)

    v_low = v_mean - v_amplitude
    v_high = v_mean + v_amplitude
    v = rng.uniform(v_low, v_high, (config.Ny, config.Nx)).astype(dtype)

    return State(u=u, v=v)


def run(
    config: SolverConfig,
    *,
    initial_state: Optional[State] = None,
    checkpoint_stride: int = 1000,
    model: ReactionModel | None = None,
) -> SimulationResult:
    """Run the MCRD simulation for the configured number of steps.

    Parameters
    ----------
    config:
        Solver configuration.
    initial_state:
        Optional initial state. If None, a random state is generated using
        :func:`make_initial_state` with default parameters.
    checkpoint_stride:
        Compute diagnostics every this many steps. Set to 0 to disable checkpoints.

    Returns
    -------
    SimulationResult
        Final state, checkpoints, snapshots (if requested), and metadata.
    """
    if initial_state is None:
        if config.backend is Backend.NUMPY:
            initial_state = (
                make_initial_state(config) if model is None else make_model_initial_state(config, model)
            )

    if config.backend in JAX_BACKENDS:
        from .jax_solver import JAXState, run_jax

        jax_state = None
        if initial_state is not None:
            expected_shape = (config.Ny, config.Nx)
            if initial_state.u.shape != expected_shape or initial_state.v.shape != expected_shape:
                raise ValueError(f"Initial state fields must have shape {expected_shape}")
            jax_state = JAXState(u=initial_state.u, v=initial_state.v)

        if jax_state is None and model is not None:
            initial_state = make_model_initial_state(config, model)
            jax_state = JAXState(u=initial_state.u, v=initial_state.v)

        result = run_jax(config, jax_state, checkpoint_stride=checkpoint_stride, model=model)
        state = State(u=np.asarray(result.final_state.u), v=np.asarray(result.final_state.v))
        checkpoints = [
            Checkpoint(step=step_idx, time=time_value, mass=mass, mass_drift=0.0, all_finite=finite)
            for step_idx, time_value, mass, finite in result.checkpoints
        ]
        mass_0 = result.initial_mass
        for checkpoint in checkpoints:
            checkpoint.mass_drift = (checkpoint.mass - mass_0) / max(abs(mass_0), np.finfo(state.u.dtype).tiny)

        snapshots = [
            State(u=np.asarray(snapshot.u), v=np.asarray(snapshot.v))
            for snapshot in result.snapshots
        ]
        metadata = {
            "n_steps": config.n_steps,
            "dt": config.dt,
            "final_time": config.final_time,
            "Nx": config.Nx,
            "Ny": config.Ny,
            "dx": config.dx,
            "dy": config.dy,
            "dtype": config.dtype.value,
            "backend": config.backend.value,
            "integrator": config.integrator.value,
            "seed": config.seed,
            "initial_mass": result.initial_mass,
            "final_mass": result.final_mass,
            "runtime_seconds": result.runtime_seconds,
        }
        if model is not None:
            metadata["reaction_model"] = model.to_dict()
        return SimulationResult(
            final_state=state,
            checkpoints=checkpoints,
            snapshots=snapshots,
            snapshot_times=result.snapshot_times,
            metadata=metadata,
        )

    if config.backend is not Backend.NUMPY:
        raise NotImplementedError(
            f"Backend {config.backend.value!r} is not implemented. "
            f"Only {Backend.NUMPY.value!r} is currently supported. "
            "See design.md for the backend adapter contract."
        )

    expected_shape = (config.Ny, config.Nx)
    if initial_state.u.shape != expected_shape:
        raise ValueError(
            f"Initial state shape {initial_state.u.shape} does not match "
            f"grid (Ny={config.Ny}, Nx={config.Nx}) → expected {expected_shape}"
        )

    state = initial_state
    mass = conserved_total if model is None else model.conserved_total
    mass_0 = mass(state.u, state.v)

    checkpoints: list[Checkpoint] = []
    snapshots: list[State] = []
    snapshot_times: list[float] = []

    # Initial checkpoint
    if checkpoint_stride > 0:
        checkpoints.append(
            Checkpoint(
                step=0,
                time=0.0,
                mass=mass_0,
                mass_drift=0.0,
                all_finite=True,
            )
        )

    # Time loop
    for step_idx in range(1, config.n_steps + 1):
        state = step(state, config, model)

        # Checkpoint diagnostics
        if checkpoint_stride > 0 and step_idx % checkpoint_stride == 0:
            mass_t = mass(state.u, state.v)
            drift = (mass_t - mass_0) / max(abs(mass_0), np.finfo(state.u.dtype).tiny)
            all_finite = bool(np.all(np.isfinite(state.u)) and np.all(np.isfinite(state.v)))
            checkpoints.append(
                Checkpoint(
                    step=step_idx,
                    time=step_idx * config.dt,
                    mass=mass_t,
                    mass_drift=drift,
                    all_finite=all_finite,
                )
            )

        # Snapshot
        if config.snapshot_stride > 0 and step_idx % config.snapshot_stride == 0:
            snapshots.append(State(u=state.u.copy(), v=state.v.copy()))
            snapshot_times.append(step_idx * config.dt)

    # Ensure final state is captured as a checkpoint
    if checkpoint_stride > 0 and config.n_steps > 0:
        last_cp_step = checkpoints[-1].step if checkpoints else 0
        if last_cp_step != config.n_steps:
            mass_t = mass(state.u, state.v)
            drift = (mass_t - mass_0) / max(abs(mass_0), np.finfo(state.u.dtype).tiny)
            all_finite = bool(np.all(np.isfinite(state.u)) and np.all(np.isfinite(state.v)))
            checkpoints.append(
                Checkpoint(
                    step=config.n_steps,
                    time=config.n_steps * config.dt,
                    mass=mass_t,
                    mass_drift=drift,
                    all_finite=all_finite,
                )
            )

    metadata = {
        "n_steps": config.n_steps,
        "dt": config.dt,
        "final_time": config.n_steps * config.dt,
        "Nx": config.Nx,
        "Ny": config.Ny,
        "dx": config.dx,
        "dy": config.dy,
        "dtype": config.dtype.value,
        "backend": config.backend.value,
        "integrator": config.integrator.value,
        "seed": config.seed,
        "initial_mass": mass_0,
        "final_mass": mass(state.u, state.v),
    }
    if model is not None:
        metadata["reaction_model"] = model.to_dict()

    return SimulationResult(
        final_state=state,
        checkpoints=checkpoints,
        snapshots=snapshots,
        snapshot_times=snapshot_times,
        metadata=metadata,
    )
