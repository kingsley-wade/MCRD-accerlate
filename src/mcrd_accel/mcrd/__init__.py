"""MCRD solver: equations, operators, stepping, and validation."""

from .equations import conserved_total, reaction_contributions, reaction_term, relative_mass_drift
from .jax_solver import JAXSimulationResult, JAXState, run_jax
from .models import ReactionModel
from .operators import periodic_laplacian
from .solver import Checkpoint, SimulationResult, State, make_initial_state, run, step
from .validation import (
    ConservationReport,
    ConvergenceReport,
    StabilityReport,
    run_conservation_check,
    run_spatial_convergence,
    run_stability_sweep,
    run_temporal_convergence,
)

__all__ = [
    "Checkpoint",
    "ConservationReport",
    "ConvergenceReport",
    "JAXSimulationResult",
    "JAXState",
    "ReactionModel",
    "SimulationResult",
    "StabilityReport",
    "State",
    "conserved_total",
    "make_initial_state",
    "periodic_laplacian",
    "reaction_contributions",
    "reaction_term",
    "relative_mass_drift",
    "run",
    "run_conservation_check",
    "run_jax",
    "run_spatial_convergence",
    "run_stability_sweep",
    "run_temporal_convergence",
    "step",
]
