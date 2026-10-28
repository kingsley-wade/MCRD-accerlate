"""Conservation, convergence, and stability validation reports.

Runs controlled experiments and returns structured records rather than plotting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ResourceProfile, SolverConfig, ValidationConfig
from .equations import conserved_total
from .models import ReactionModel, make_model_initial_state
from .solver import State, run, step


@dataclass
class ConservationReport:
    """Report of mass conservation for a single run."""

    config: dict
    initial_mass: float
    final_mass: float
    max_drift: float
    mean_drift: float
    all_finite: bool
    n_checkpoints: int
    metadata: dict


@dataclass
class ConvergenceCase:
    """Single case in a convergence study."""

    grid_size: int | None
    dt: float | None
    error_norm: float
    observed_order: float | None
    conservation_drift: float
    runtime_seconds: float
    peak_memory_bytes: int | None
    status: str  # "completed", "skipped_resource_budget", "failed"


@dataclass
class ConvergenceReport:
    """Report of a multi-level convergence study."""

    study_type: str  # "spatial" or "temporal"
    cases: list[ConvergenceCase]
    reference_case: ConvergenceCase | None
    config: dict


@dataclass
class StabilityCase:
    """Single case in a stability sweep."""

    dt: float
    status: str  # "stable", "unstable", "nonfinite", "bound_violation", "inconclusive"
    first_failure_step: int | None
    first_failure_time: float | None
    conservation_drift: float
    runtime_seconds: float


@dataclass
class StabilityReport:
    """Report of a time-step stability sweep."""

    cases: list[StabilityCase]
    estimated_stable_interval: tuple[float, float] | None
    config: dict


def _estimate_memory_bytes(config: SolverConfig) -> int:
    """Estimate peak resident memory for a simulation config."""
    dtype_size = 4 if config.dtype.value == "float32" else 8
    n_cells = config.Nx * config.Ny
    # u, v, du_dt, dv_dt, lap_u, lap_v + intermediates
    n_arrays = 10
    return n_cells * dtype_size * n_arrays


def _check_resource_budget(
    config: SolverConfig, resource: ResourceProfile
) -> str | None:
    """Return a skip reason string if the case exceeds the budget, else None."""
    estimated = _estimate_memory_bytes(config)
    if estimated > resource.max_resident_bytes:
        return "skipped_resource_budget"
    if config.n_steps > resource.max_steps:
        return "skipped_resource_budget"
    return None


def _field_diff_l2(a: State, b: State) -> float:
    """L2 norm of the difference between two states, normalized by domain size."""
    du = a.u - b.u
    dv = a.v - b.v
    n_cells = a.u.size
    return float(np.sqrt(np.mean(du ** 2 + dv ** 2)))


def run_conservation_check(
    config: SolverConfig, checkpoint_stride: int = 1000,
    model: ReactionModel | None = None,
) -> ConservationReport:
    """Run a simulation and report mass conservation diagnostics."""
    result = run(config, checkpoint_stride=checkpoint_stride, model=model)

    drifts = [abs(cp.mass_drift) for cp in result.checkpoints]
    max_drift = max(drifts) if drifts else 0.0
    mean_drift = float(np.mean(drifts)) if drifts else 0.0

    all_finite = all(cp.all_finite for cp in result.checkpoints)

    metadata = {
        "backend": config.backend.value,
        "dtype": config.dtype.value,
        "integrator": config.integrator.value,
        "seed": config.seed,
    }
    metadata.update(result.metadata)
    if model is not None:
        metadata["reaction_model"] = model.to_dict()

    return ConservationReport(
        config={**config.to_dict(), **({"reaction_model": model.to_dict()} if model else {})},
        initial_mass=result.metadata["initial_mass"],
        final_mass=result.metadata["final_mass"],
        max_drift=max_drift,
        mean_drift=mean_drift,
        all_finite=all_finite,
        n_checkpoints=len(result.checkpoints),
        metadata=metadata,
    )


def _make_run_config(base: SolverConfig, **overrides: Any) -> SolverConfig:
    """Create a SolverConfig from a base, overriding specific fields."""
    params = dict(
        kd=base.kd, k1=base.k1, b=base.b, k2=base.k2,
        rho=base.rho, cv=base.cv, beta=base.beta,
        Du=base.Du, Dv=base.Dv,
        Nx=base.Nx, Ny=base.Ny, dx=base.dx, dy=base.dy,
        dt=base.dt, n_steps=base.n_steps,
        integrator=base.integrator, boundary=base.boundary,
        dtype=base.dtype, backend=base.backend, seed=base.seed,
    )
    params.update(overrides)
    return SolverConfig(**params)


def _make_spatial_initial_state(
    config: SolverConfig, N: int, model: ReactionModel | None = None
) -> State:
    """Sample one seeded, continuous periodic initial field on an N x N grid."""
    rng = np.random.default_rng(config.seed)
    modes = ((1, 0), (0, 1), (1, 1), (2, -1), (-1, 2), (2, 1))
    phases_u = rng.uniform(0.0, 2.0 * np.pi, len(modes))
    phases_v = rng.uniform(0.0, 2.0 * np.pi, len(modes))
    x = np.arange(N, dtype=np.float64) / N
    y = np.arange(N, dtype=np.float64) / N
    xx, yy = np.meshgrid(x, y)

    def field(phases: NDArray[np.floating]) -> NDArray[np.floating]:
        values = np.zeros((N, N), dtype=np.float64)
        for (kx, ky), phase in zip(modes, phases):
            values += np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phase)
        return values / len(modes)

    dtype = np.float32 if config.dtype.value == "float32" else np.float64
    if model is None or model.rate_kind == "p53_mcrd":
        u = (0.5 + 0.25 * field(phases_u)).astype(dtype)
        v = (25.0 + 0.5 * field(phases_v)).astype(dtype)
    else:
        u = (0.5 + 0.25 * field(phases_u)).astype(dtype)
        v = (0.5 + 0.25 * field(phases_v)).astype(dtype)
    return State(u=u, v=v)


def run_spatial_convergence(
    validation_config: ValidationConfig,
    model: ReactionModel | None = None,
) -> ConvergenceReport:
    """Run a spatial convergence study at fixed physical domain.

    Compares each grid level against the next finer level using L2 field
    differences. The finest level serves as the reference. Observed order
    is computed from successive error ratios.
    """
    grid_sizes = list(validation_config.grid_sizes)
    resource = validation_config.resource
    config = validation_config.solver
    Lx = config.Nx * config.dx
    Ly = config.Ny * config.dy

    final_time = config.n_steps * config.dt

    # Run all cases, storing final states for pairwise comparison
    cases: list[ConvergenceCase] = []
    final_states: list[State | None] = []

    for N in grid_sizes:
        dx_new = Lx / N
        dy_new = Ly / N
        n_steps = int(round(final_time / config.dt))

        run_config = _make_run_config(
            config, Nx=N, Ny=N, dx=dx_new, dy=dy_new, n_steps=n_steps,
        )

        skip_reason = _check_resource_budget(run_config, resource)
        if skip_reason:
            cases.append(ConvergenceCase(
                grid_size=N, dt=config.dt, error_norm=float("nan"),
                observed_order=None, conservation_drift=float("nan"),
                runtime_seconds=0.0, peak_memory_bytes=None, status=skip_reason,
            ))
            final_states.append(None)
            continue

        try:
            start = time.time()
            initial_state = _make_spatial_initial_state(config, N, model)
            result = run(run_config, initial_state=initial_state, checkpoint_stride=0, model=model)
            runtime = time.time() - start

            final_finite = bool(
                np.all(np.isfinite(result.final_state.u))
                and np.all(np.isfinite(result.final_state.v))
            )
            if not final_finite:
                cases.append(ConvergenceCase(
                    grid_size=N, dt=config.dt, error_norm=float("nan"),
                    observed_order=None, conservation_drift=float("nan"),
                    runtime_seconds=runtime,
                    peak_memory_bytes=_estimate_memory_bytes(run_config),
                    status="failed: nonfinite final state",
                ))
                final_states.append(None)
                continue

            mass_0 = result.metadata["initial_mass"]
            mass_f = result.metadata["final_mass"]
            tiny = np.finfo(result.final_state.u.dtype).tiny
            drift = abs(mass_f - mass_0) / max(abs(mass_0), tiny)

            cases.append(ConvergenceCase(
                grid_size=N, dt=config.dt, error_norm=float("nan"),
                observed_order=None, conservation_drift=drift,
                runtime_seconds=runtime,
                peak_memory_bytes=_estimate_memory_bytes(run_config),
                status="completed",
            ))
            final_states.append(result.final_state)
        except Exception as e:
            cases.append(ConvergenceCase(
                grid_size=N, dt=config.dt, error_norm=float("nan"),
                observed_order=None, conservation_drift=float("nan"),
                runtime_seconds=0.0, peak_memory_bytes=None,
                status=f"failed: {e}",
            ))
            final_states.append(None)

    # Compare every completed level to the finest completed reference.
    reference_case = None
    completed = [(i, c) for i, c in enumerate(cases) if c.status == "completed"]
    if completed:
        ref_idx = max(completed, key=lambda item: item[1].grid_size or 0)[0]
        reference_case = cases[ref_idx]
        s_ref = final_states[ref_idx]
        assert s_ref is not None
        n_ref = cases[ref_idx].grid_size
        assert n_ref is not None
        for idx, case in completed:
            n = case.grid_size
            state = final_states[idx]
            assert n is not None and state is not None
            if n == n_ref:
                error = 0.0
            elif n_ref % n == 0:
                ratio = n_ref // n
                ref_u = s_ref.u.reshape(n, ratio, n, ratio).mean(axis=(1, 3))
                ref_v = s_ref.v.reshape(n, ratio, n, ratio).mean(axis=(1, 3))
                error = _field_diff_l2(state, State(ref_u, ref_v))
            else:
                indices = np.floor(np.arange(n) * n_ref / n).astype(int)
                ref_u = s_ref.u[np.ix_(indices, indices)]
                ref_v = s_ref.v[np.ix_(indices, indices)]
                error = _field_diff_l2(state, State(ref_u, ref_v))
            cases[idx] = ConvergenceCase(**{**case.__dict__, "error_norm": error})

    ordered = sorted(
        ((i, c) for i, c in enumerate(cases) if c.status == "completed"),
        key=lambda item: item[1].grid_size or 0,
    )
    for (i0, c0), (i1, c1) in zip(ordered, ordered[1:]):
        if c0.error_norm > 0 and c1.error_norm > 0 and c0.grid_size and c1.grid_size > c0.grid_size:
            order = np.log(c0.error_norm / c1.error_norm) / np.log(c1.grid_size / c0.grid_size)
            cases[i0] = ConvergenceCase(**{**cases[i0].__dict__, "observed_order": float(order)})

    return ConvergenceReport(
        study_type="spatial",
        cases=cases,
        reference_case=reference_case,
        config={**validation_config.solver.to_dict(), **({"reaction_model": model.to_dict()} if model else {})},
    )


def run_temporal_convergence(
    validation_config: ValidationConfig,
    model: ReactionModel | None = None,
) -> ConvergenceReport:
    """Run a temporal convergence study at fixed grid.

    Compares each dt level against the next finer dt using L2 field differences
    at the same final time. Observed order is computed from successive error ratios.
    """
    config = validation_config.solver
    resource = validation_config.resource
    final_time = config.n_steps * config.dt

    cases: list[ConvergenceCase] = []
    final_states: list[State | None] = []

    for dt in validation_config.time_steps:
        n_steps = int(round(final_time / dt))
        run_config = _make_run_config(config, dt=dt, n_steps=n_steps)

        skip_reason = _check_resource_budget(run_config, resource)
        if skip_reason:
            cases.append(ConvergenceCase(
                grid_size=config.Nx, dt=dt, error_norm=float("nan"),
                observed_order=None, conservation_drift=float("nan"),
                runtime_seconds=0.0, peak_memory_bytes=None, status=skip_reason,
            ))
            final_states.append(None)
            continue

        try:
            start = time.time()
            result = run(run_config, checkpoint_stride=0, model=model)
            runtime = time.time() - start

            final_finite = bool(
                np.all(np.isfinite(result.final_state.u))
                and np.all(np.isfinite(result.final_state.v))
            )
            if not final_finite:
                cases.append(ConvergenceCase(
                    grid_size=config.Nx, dt=dt, error_norm=float("nan"),
                    observed_order=None, conservation_drift=float("nan"),
                    runtime_seconds=runtime,
                    peak_memory_bytes=_estimate_memory_bytes(run_config),
                    status="failed: nonfinite final state",
                ))
                final_states.append(None)
                continue

            mass_0 = result.metadata["initial_mass"]
            mass_f = result.metadata["final_mass"]
            tiny = np.finfo(result.final_state.u.dtype).tiny
            drift = abs(mass_f - mass_0) / max(abs(mass_0), tiny)

            cases.append(ConvergenceCase(
                grid_size=config.Nx, dt=dt, error_norm=float("nan"),
                observed_order=None, conservation_drift=drift,
                runtime_seconds=runtime,
                peak_memory_bytes=_estimate_memory_bytes(run_config),
                status="completed",
            ))
            final_states.append(result.final_state)
        except Exception as e:
            cases.append(ConvergenceCase(
                grid_size=config.Nx, dt=dt, error_norm=float("nan"),
                observed_order=None, conservation_drift=float("nan"),
                runtime_seconds=0.0, peak_memory_bytes=None,
                status=f"failed: {e}",
            ))
            final_states.append(None)

    # Compare every time step to the finest completed reference.
    completed = [(i, c) for i, c in enumerate(cases) if c.status == "completed"]
    reference_case = None
    if completed:
        ref_idx = min(completed, key=lambda item: item[1].dt or float("inf"))[0]
        reference_case = cases[ref_idx]
        s_ref = final_states[ref_idx]
        assert s_ref is not None
        for idx, case in completed:
            state = final_states[idx]
            assert state is not None and case.dt is not None
            error = 0.0 if idx == ref_idx else _field_diff_l2(state, s_ref)
            cases[idx] = ConvergenceCase(**{**case.__dict__, "error_norm": error})

    ordered = sorted(
        ((i, c) for i, c in enumerate(cases) if c.status == "completed"),
        key=lambda item: -(item[1].dt or 0.0),
    )
    for (i0, c0), (i1, c1) in zip(ordered, ordered[1:]):
        if c0.error_norm > 0 and c1.error_norm > 0 and c0.dt and c1.dt and c1.dt < c0.dt:
            order = np.log(c0.error_norm / c1.error_norm) / np.log(c0.dt / c1.dt)
            cases[i0] = ConvergenceCase(**{**cases[i0].__dict__, "observed_order": float(order)})

    return ConvergenceReport(
        study_type="temporal",
        cases=cases,
        reference_case=reference_case,
        config={**validation_config.solver.to_dict(), **({"reaction_model": model.to_dict()} if model else {})},
    )


def run_stability_sweep(
    validation_config: ValidationConfig,
    model: ReactionModel | None = None,
) -> StabilityReport:
    """Run a time-step stability sweep.

    Steps through each dt, checking finiteness and physical bounds (u >= 0, v >= 0)
    at every step. Records the first failure step/time.
    """
    config = validation_config.solver
    resource = validation_config.resource
    final_time = config.n_steps * config.dt

    cases: list[StabilityCase] = []

    for dt in validation_config.time_steps:
        n_steps = int(round(final_time / dt))
        run_config = _make_run_config(config, dt=dt, n_steps=n_steps)

        skip_reason = _check_resource_budget(run_config, resource)
        if skip_reason:
            cases.append(StabilityCase(
                dt=dt, status=skip_reason,
                first_failure_step=None, first_failure_time=None,
                conservation_drift=float("nan"), runtime_seconds=0.0,
            ))
            continue

        try:
            state = make_initial_state(run_config) if model is None else make_model_initial_state(run_config, model)
            mass = conserved_total if model is None else model.conserved_total
            mass_0 = mass(state.u, state.v)
            tiny = np.finfo(state.u.dtype).tiny

            start = time.time()
            first_failure_step: int | None = None
            first_failure_time_val: float | None = None
            status = "stable"
            max_drift = 0.0
            checkpoint_stride = max(1, n_steps // 10)

            for step_idx in range(1, n_steps + 1):
                state = step(state, run_config, model)

                # Check finiteness at every step
                if not (np.all(np.isfinite(state.u)) and np.all(np.isfinite(state.v))):
                    status = "nonfinite"
                    first_failure_step = step_idx
                    first_failure_time_val = step_idx * dt
                    break

                # Check physical bounds (concentrations should be non-negative)
                if np.any(state.u < 0) or np.any(state.v < 0):
                    status = "bound_violation"
                    first_failure_step = step_idx
                    first_failure_time_val = step_idx * dt
                    break

                # Check mass drift at checkpoints
                if step_idx % checkpoint_stride == 0:
                    mass_t = mass(state.u, state.v)
                    drift = abs((mass_t - mass_0) / max(abs(mass_0), tiny))
                    max_drift = max(max_drift, drift)

            # Final mass drift
            mass_t = mass(state.u, state.v)
            drift = abs((mass_t - mass_0) / max(abs(mass_0), tiny))
            max_drift = max(max_drift, drift)

            if status == "stable" and max_drift > validation_config.conservation_tol * 100:
                status = "unstable"

            runtime = time.time() - start

            cases.append(StabilityCase(
                dt=dt, status=status,
                first_failure_step=first_failure_step,
                first_failure_time=first_failure_time_val,
                conservation_drift=max_drift,
                runtime_seconds=runtime,
            ))
        except Exception as e:
            cases.append(StabilityCase(
                dt=dt, status="failed",
                first_failure_step=None, first_failure_time=None,
                conservation_drift=float("nan"), runtime_seconds=0.0,
            ))

    ordered_cases = sorted(cases, key=lambda case: case.dt)
    stable_prefix: list[float] = []
    for case in ordered_cases:
        if case.status != "stable":
            break
        stable_prefix.append(case.dt)
    stable_interval = (
        (stable_prefix[0], stable_prefix[-1]) if stable_prefix else None
    )

    return StabilityReport(
        cases=cases,
        estimated_stable_interval=stable_interval,
        config={**validation_config.solver.to_dict(), **({"reaction_model": model.to_dict()} if model else {})},
    )
