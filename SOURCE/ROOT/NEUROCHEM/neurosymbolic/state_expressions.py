"""
Composite state expressions for neurosymbolic encoding (Appendix K.5).

STATE(X) = b + sum_m(w_m * T_m(t))

Defines named composite state variables as weighted sums of:
- S_*   : receptor saturations
- C_*   : NT concentrations
- phi_* : oscillation band amplitudes
- phi_X*S_Y : gated products (oscillation * saturation)

All evaluation functions are pure — no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateTerm:
    """A single term in a state expression (K.5.1)."""
    weight: float
    variable: str   # "S_DA_D1", "phi_theta", "phi_theta_gamma*S_NMDA", "C_DA"


@dataclass(frozen=True)
class StateDefinition:
    """Complete definition of a composite state variable (K.5.2)."""
    name: str
    terms: Tuple[StateTerm, ...]
    bias: float = 0.0
    bounding: str = "clip"  # "clip", "logistic", "affine_clip"


# ---------------------------------------------------------------------------
# Variable resolution (K.5.3)
# ---------------------------------------------------------------------------

def resolve_term(
    term: StateTerm,
    saturations: Dict[str, float],
    concentrations: Dict[str, float],
    oscillations: Dict[str, float],
) -> float:
    """
    Resolve a single term's variable to its current numeric value.

    Variable naming conventions:
    - "S_X"       → saturations["X"]
    - "C_X"       → concentrations["X"]
    - "phi_X"     → oscillations["X"]
    - "phi_X*S_Y" → oscillations["X"] * saturations["Y"]

    Missing variables resolve to 0.0.

    Returns
    -------
    float
        weight * resolved_value
    """
    var = term.variable

    # Gated product: phi_X*S_Y
    if "*" in var:
        parts = var.split("*", 1)
        left = _resolve_single(parts[0].strip(), saturations, concentrations, oscillations)
        right = _resolve_single(parts[1].strip(), saturations, concentrations, oscillations)
        return term.weight * left * right

    # Single variable
    value = _resolve_single(var, saturations, concentrations, oscillations)
    return term.weight * value


def _resolve_single(
    var: str,
    saturations: Dict[str, float],
    concentrations: Dict[str, float],
    oscillations: Dict[str, float],
) -> float:
    """Resolve a single variable name to its value."""
    if var.startswith("S_"):
        key = var[2:]
        return saturations.get(key, 0.0)
    if var.startswith("C_"):
        key = var[2:]
        return concentrations.get(key, 0.0)
    if var.startswith("phi_"):
        key = var[4:]
        return oscillations.get(key, 0.0)
    # Fallback: try all dicts
    return saturations.get(var, concentrations.get(var, oscillations.get(var, 0.0)))


# ---------------------------------------------------------------------------
# Bounding functions (K.5.4)
# ---------------------------------------------------------------------------

def _bound_clip(value: float) -> float:
    """Clip to [0, 1]."""
    return max(0.0, min(1.0, value))


def _bound_logistic(value: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Logistic sigmoid bounding to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-k * (value - x0)))


def _bound_affine_clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clip to [lo, hi] — for signed state variables."""
    return max(lo, min(hi, value))


def _apply_bounding(value: float, bounding: str) -> float:
    """Apply the specified bounding mode."""
    if bounding == "clip":
        return _bound_clip(value)
    elif bounding == "logistic":
        return _bound_logistic(value)
    elif bounding == "affine_clip":
        return _bound_affine_clip(value)
    return _bound_clip(value)


# ---------------------------------------------------------------------------
# Evaluation (K.5.5)
# ---------------------------------------------------------------------------

def evaluate_state(
    definition: StateDefinition,
    saturations: Dict[str, float],
    concentrations: Dict[str, float],
    oscillations: Dict[str, float],
) -> float:
    """
    Evaluate a single composite state expression.

    STATE(X) = bound(bias + sum(w_m * T_m))

    Parameters
    ----------
    definition : StateDefinition
        The state expression to evaluate.
    saturations : dict
        Receptor saturation values keyed by receptor ID.
    concentrations : dict
        NT concentration values keyed by NT name.
    oscillations : dict
        Oscillation band amplitudes keyed by band name.

    Returns
    -------
    float
        Evaluated state value after bounding.
    """
    raw = definition.bias
    for term in definition.terms:
        raw += resolve_term(term, saturations, concentrations, oscillations)
    return _apply_bounding(raw, definition.bounding)


def evaluate_all_states(
    definitions: List[StateDefinition],
    saturations: Dict[str, float],
    concentrations: Dict[str, float],
    oscillations: Dict[str, float],
) -> Dict[str, float]:
    """
    Evaluate all state definitions and return name→value mapping.

    Parameters
    ----------
    definitions : list of StateDefinition
        All state expressions to evaluate.
    saturations, concentrations, oscillations : dict
        Current system state.

    Returns
    -------
    dict
        {state_name: evaluated_value}
    """
    return {
        defn.name: evaluate_state(defn, saturations, concentrations, oscillations)
        for defn in definitions
    }
