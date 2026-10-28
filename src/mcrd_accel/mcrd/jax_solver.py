"""JAX-accelerated two-species reaction-diffusion solver with compiled loop.

Uses jax.lax.scan for the time integration loop, which compiles the entire
simulation into a single GPU kernel launch (no Python loop overhead per step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import gcd
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..backends import JAX_BACKENDS, get_jax_device
from ..config import Backend, Integrator, SolverConfig
from .models import ReactionModel, make_model_initial_state


@dataclass
class JAXState:
    """JAX simulation state (arrays are JAX DeviceArray)."""

    u: object  # jax.Array
    v: object  # jax.Array


@dataclass
class JAXSimulationResult:
    """JAX simulation result."""

    final_state: JAXState
    final_mass: float
    initial_mass: float
    n_steps: int
    runtime_seconds: float
    checkpoints: list[tuple[int, float, float, bool]] = field(default_factory=list)
    snapshots: list[JAXState] = field(default_factory=list)
    snapshot_times: list[float] = field(default_factory=list)


def _make_jax_step(params, model: ReactionModel):
    """Create a JAX step function with immutable per-run parameters."""
    import jax.numpy as jnp

    def jax_step(state, _):
        u, v = state

        Du = params['Du']
        Dv = params['Dv']
        dx = params['dx']
        dy = params['dy']
        dt = params['dt']

        reaction_rate = model.rate_jax(u, v)

        lap_u = (
            (jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 2.0 * u) / (dx * dx)
            + (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) - 2.0 * u) / (dy * dy)
        )
        lap_v = (
            (jnp.roll(v, 1, axis=1) + jnp.roll(v, -1, axis=1) - 2.0 * v) / (dx * dx)
            + (jnp.roll(v, 1, axis=0) + jnp.roll(v, -1, axis=0) - 2.0 * v) / (dy * dy)
        )

        du_dt = model.stoichiometry[0] * reaction_rate + Du * lap_u
        dv_dt = model.stoichiometry[1] * reaction_rate + Dv * lap_v

        u_new = u + dt * du_dt
        v_new = v + dt * dv_dt

        return (u_new, v_new), None

    return jax_step


@lru_cache(maxsize=32)
def _compiled_jax_step(
    model: ReactionModel, Du: float, Dv: float, dx: float, dy: float, dt: float
):
    import jax

    params = {"Du": Du, "Dv": Dv, "dx": dx, "dy": dy, "dt": dt}
    return jax.jit(_make_jax_step(params, model))


def run_jax(
    config: SolverConfig,
    initial_state: Optional[JAXState] = None,
    *,
    checkpoint_stride: int = 1000,
    model: ReactionModel | None = None,
) -> JAXSimulationResult:
    """Run a reaction-diffusion model using JAX with compiled lax.scan loops.

    Parameters
    ----------
    config:
        Solver configuration (backend must select a JAX device).
    initial_state:
        Optional initial state. If None, a deterministic seeded state is generated.
    checkpoint_stride:
        Checkpoint interval; zero disables mass checkpoints.
    model:
        Reaction network. Defaults to the p53-dsDNA model.

    Returns
    -------
    JAXSimulationResult
        Final state and metadata.
    """
    import jax
    import jax.numpy as jnp
    import time

    if config.backend not in JAX_BACKENDS:
        raise ValueError(f"JAX solver requires a JAX backend, got {config.backend}")
    device = get_jax_device(config.backend)

    if model is None:
        model = ReactionModel.p53_mcrd(config)

    # Generate initial state if not provided
    if initial_state is None:
        state = make_model_initial_state(config, model)
        initial_state = JAXState(
            u=jax.device_put(jnp.asarray(state.u), device),
            v=jax.device_put(jnp.asarray(state.v), device),
        )
    elif initial_state.u.shape != (config.Ny, config.Nx) or initial_state.v.shape != (config.Ny, config.Nx):
        raise ValueError(f"Initial state fields must have shape {(config.Ny, config.Nx)}")
    else:
        initial_state = JAXState(
            u=jax.device_put(jnp.asarray(initial_state.u), device),
            v=jax.device_put(jnp.asarray(initial_state.v), device),
        )

    jit_step = _compiled_jax_step(
        model, config.Du, config.Dv, config.dx, config.dy, config.dt
    )

    # Run simulation
    initial_mass = float(jnp.sum(initial_state.u + initial_state.v))

    checkpoints: list[tuple[int, float, float, bool]] = []
    snapshots: list[JAXState] = []
    snapshot_times: list[float] = []
    if checkpoint_stride > 0:
        checkpoints.append((0, 0.0, initial_mass, bool(jnp.all(jnp.isfinite(initial_state.u)) & jnp.all(jnp.isfinite(initial_state.v)))))

    output_strides = [stride for stride in (checkpoint_stride, config.snapshot_stride) if stride > 0]
    chunk_size = gcd(*output_strides) if output_strides else max(1, config.n_steps)
    state = (initial_state.u, initial_state.v)
    start = time.time()
    completed_steps = 0
    while completed_steps < config.n_steps:
        n_chunk = min(chunk_size, config.n_steps - completed_steps)
        with jax.default_device(device):
            state, _ = jax.lax.scan(jit_step, state, None, length=n_chunk)
        state[0].block_until_ready()
        completed_steps += n_chunk

        if checkpoint_stride > 0 and completed_steps % checkpoint_stride == 0:
            mass = float(jnp.sum(state[0] + state[1]))
            finite = bool(jnp.all(jnp.isfinite(state[0])) & jnp.all(jnp.isfinite(state[1])))
            checkpoints.append((completed_steps, completed_steps * config.dt, mass, finite))
        if config.snapshot_stride > 0 and completed_steps % config.snapshot_stride == 0:
            snapshots.append(JAXState(u=state[0], v=state[1]))
            snapshot_times.append(completed_steps * config.dt)

    u_final, v_final = state
    if config.n_steps > 0 and checkpoint_stride > 0 and checkpoints[-1][0] != config.n_steps:
        mass = float(jnp.sum(u_final + v_final))
        finite = bool(jnp.all(jnp.isfinite(u_final)) & jnp.all(jnp.isfinite(v_final)))
        checkpoints.append((config.n_steps, config.n_steps * config.dt, mass, finite))
    u_final.block_until_ready()
    v_final.block_until_ready()
    runtime = time.time() - start

    final_mass = float(jnp.sum(u_final + v_final))

    return JAXSimulationResult(
        final_state=JAXState(u=u_final, v=v_final),
        final_mass=final_mass,
        initial_mass=initial_mass,
        n_steps=config.n_steps,
        runtime_seconds=runtime,
        checkpoints=checkpoints,
        snapshots=snapshots,
        snapshot_times=snapshot_times,
    )
