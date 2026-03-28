"""
Mode selection hooks for neurosymbolic encoding (Appendix M.5 + M.6).

Defines 14 named mode tokens with:
- Primary hook conditions (boolean over normalized metrics + oscillations + saturations)
- Priority tier arbitration (M.5.5): safety > empathy > rigidity > drive
- Optional composite scoring gates for within-tier tiebreaking

All evaluation functions are pure — no side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
from zados.neurochem.neurosymbolic.triggers import evaluate_condition


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeHookDefinition:
    """A named mode with its activation condition and priority (M.5.1)."""
    name: str
    condition_str: str
    priority_tier: int          # 0=highest (safety), 1=empathy, 2=rigidity, 3=drive
    actions: Tuple[str, ...] = ()
    required_inputs: Tuple[str, ...] = ()  # doc-only
    composite_gate: Optional[str] = None
    composite_threshold: Optional[float] = None


@dataclass(frozen=True)
class ModeSelectionResult:
    """Result of mode selection (M.5.5)."""
    active_mode: Optional[str] = None
    fired_modes: Tuple[str, ...] = ()
    composite_scores: Optional[Dict[str, float]] = field(default=None, hash=False)


# ---------------------------------------------------------------------------
# Default thresholds (M.5.6)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Metric thresholds
    "M_high": 0.6,
    "M_med": 0.5,
    "M_low": 0.4,
    "E_high": 0.6,
    "E_med": 0.5,
    "E_low": 0.4,
    "R_high": 0.6,
    "R_med": 0.5,
    "R_low": 0.4,
    "F_high": 0.7,
    "F_med": 0.6,
    "F_low": 0.5,
    # Oscillation thresholds
    "phi_high": 0.5,
    "phi_low": 0.4,
    "phi_very_low": 0.3,
    # Saturation thresholds
    "S_low": 0.4,
}


# ---------------------------------------------------------------------------
# Default mode hook library (M.6 table — 14 modes)
# ---------------------------------------------------------------------------

DEFAULT_MODE_HOOKS: List[ModeHookDefinition] = [
    # --- Tier 0: Safety / Containment (F-dominant) ---
    ModeHookDefinition(
        name="Containment",
        condition_str="F_hat>0.6 AND phi_delta>0.5",
        priority_tier=0,
        required_inputs=("F_hat", "phi_delta"),
        composite_gate="0.5*F_hat + 0.5*phi_delta",
    ),
    ModeHookDefinition(
        name="RecoveryReset",
        condition_str="F_hat>0.7 AND phi_delta>0.5 AND phi_beta<0.3",
        priority_tier=0,
        required_inputs=("F_hat", "phi_delta", "phi_beta"),
        composite_gate="0.4*F_hat + 0.4*phi_delta + 0.2*(1-phi_beta)",
    ),

    # --- Tier 1: Empathy (E-dominant) ---
    ModeHookDefinition(
        name="EmpathicAttunement",
        condition_str="E_hat>0.6 AND phi_theta>0.5 AND R_hat<0.4 AND F_hat<0.5",
        priority_tier=1,
        required_inputs=("E_hat", "phi_theta", "R_hat", "F_hat"),
        composite_gate="0.4*E_hat + 0.3*phi_theta + 0.3*(1-R_hat)",
    ),
    ModeHookDefinition(
        name="ComfortAmplifier",
        condition_str="E_hat>0.5 AND phi_delta>0.4 AND F_hat>0.5",
        priority_tier=1,
        required_inputs=("E_hat", "phi_delta", "F_hat"),
        composite_gate="0.4*E_hat + 0.3*phi_delta + 0.3*F_hat",
    ),
    ModeHookDefinition(
        name="AnalyticalFilter",
        condition_str="E_hat<0.4 AND R_hat>0.6 AND phi_beta>0.5",
        priority_tier=1,
        required_inputs=("E_hat", "R_hat", "phi_beta"),
        composite_gate="0.4*(1-E_hat) + 0.3*R_hat + 0.3*phi_beta",
    ),

    # --- Tier 2: Rigidity (R-dominant) ---
    ModeHookDefinition(
        name="HypercriticalLogicScan",
        condition_str="R_hat>0.6 AND phi_alpha<0.4 AND S_5HT-1A<0.4 AND F_hat<0.5",
        priority_tier=2,
        required_inputs=("R_hat", "phi_alpha", "S_5HT-1A", "F_hat"),
        composite_gate="0.4*R_hat + 0.3*(1-phi_alpha) + 0.3*(1-S_5HT-1A)",
    ),
    ModeHookDefinition(
        name="HyperRationalEngine",
        condition_str="R_hat>0.6 AND phi_beta>0.5 AND phi_gamma>0.5",
        priority_tier=2,
        required_inputs=("R_hat", "phi_beta", "phi_gamma"),
        composite_gate="0.4*R_hat + 0.3*phi_beta + 0.3*phi_gamma",
    ),
    ModeHookDefinition(
        name="LiteralSkeptic",
        condition_str="R_hat>0.6 AND M_hat<0.4 AND phi_alpha>0.5",
        priority_tier=2,
        required_inputs=("R_hat", "M_hat", "phi_alpha"),
        composite_gate="0.4*R_hat + 0.3*(1-M_hat) + 0.3*phi_alpha",
    ),
    ModeHookDefinition(
        name="PrecisionRuleFidelity",
        condition_str="R_hat>0.5 AND phi_beta>0.5",
        priority_tier=2,
        required_inputs=("R_hat", "phi_beta"),
        composite_gate="0.5*R_hat + 0.5*phi_beta",
    ),
    ModeHookDefinition(
        name="LogicMode",
        condition_str="R_hat>0.5 AND phi_beta>0.5 AND phi_alpha<0.4",
        priority_tier=2,
        required_inputs=("R_hat", "phi_beta", "phi_alpha"),
        composite_gate="0.4*R_hat + 0.3*phi_beta + 0.3*(1-phi_alpha)",
    ),
    ModeHookDefinition(
        name="ConvergentRefiner",
        condition_str="phi_beta>0.5 AND R_hat>0.5 AND F_hat<0.5",
        priority_tier=2,
        required_inputs=("phi_beta", "R_hat", "F_hat"),
        composite_gate="0.4*phi_beta + 0.4*R_hat + 0.2*(1-F_hat)",
    ),

    # --- Tier 3: Drive (M-dominant) ---
    ModeHookDefinition(
        name="CreativeDivergence",
        condition_str="M_hat>0.6 AND phi_gamma>0.5 AND R_hat<0.4 AND F_hat<0.5",
        priority_tier=3,
        required_inputs=("M_hat", "phi_gamma", "R_hat", "F_hat"),
        composite_gate="0.4*M_hat + 0.3*phi_gamma + 0.3*(1-R_hat)",
    ),
    ModeHookDefinition(
        name="ConceptualSynthesis",
        condition_str="phi_theta_gamma>0.5 AND phi_gamma>0.5 AND M_hat>0.5",
        priority_tier=3,
        required_inputs=("phi_theta_gamma", "phi_gamma", "M_hat"),
        composite_gate="0.4*phi_theta_gamma + 0.3*phi_gamma + 0.3*M_hat",
    ),
    ModeHookDefinition(
        name="CuriosityDrive",
        condition_str="M_hat>0.5 AND phi_theta_gamma>0.4 AND F_hat<0.5",
        priority_tier=3,
        required_inputs=("M_hat", "phi_theta_gamma", "F_hat"),
        composite_gate="0.4*M_hat + 0.4*phi_theta_gamma + 0.2*(1-F_hat)",
    ),
]


# ---------------------------------------------------------------------------
# Namespace builder (M.5.1)
# ---------------------------------------------------------------------------

def build_mode_namespace(
    metrics: NeurochemicalMetrics,
    oscillations: Dict[str, float],
    saturations: Optional[Dict[str, float]] = None,
    concentrations: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build a flat variable namespace for mode hook condition evaluation.

    Adds:
    - M_hat, E_hat, R_hat, F_hat — canonical normalized metric aliases (M.5.1)
    - All 8 metric names from NeurochemicalMetrics
    - phi_* oscillation vars with short aliases
    - phi_theta_gamma, phi_alpha_beta — CFC coupling products
    - S_* saturation vars (underscore and hyphen variants)
    - C_* concentration vars

    Parameters
    ----------
    metrics : NeurochemicalMetrics
    oscillations : dict
        Band amplitudes keyed by band name.
    saturations : dict, optional
    concentrations : dict, optional

    Returns
    -------
    dict
        Flat variable namespace.
    """
    ns: Dict[str, float] = {}

    # Canonical M.5.1 metric aliases
    ns["M_hat"] = metrics.motivation
    ns["E_hat"] = metrics.empathy
    ns["R_hat"] = metrics.cognitive_rigidity
    ns["F_hat"] = metrics.fatigue

    # All 8 metrics by name
    for k, v in metrics.as_dict().items():
        ns[k] = v

    # Oscillation band envelopes
    if oscillations:
        for k, v in oscillations.items():
            ns[f"phi_{k}"] = v
            ns[k] = v  # short alias

        # CFC coupling products (M.1.2C)
        theta = oscillations.get("theta", 0.0)
        gamma = oscillations.get("gamma", 0.0)
        alpha = oscillations.get("alpha", 0.0)
        beta = oscillations.get("beta", 0.0)
        ns["phi_theta_gamma"] = theta * gamma
        ns["theta_gamma"] = theta * gamma
        ns["phi_alpha_beta"] = alpha * beta
        ns["alpha_beta"] = alpha * beta

    # Receptor saturations
    if saturations:
        for k, v in saturations.items():
            ns[f"S_{k}"] = v
            # Hyphen variant: S_DA_D1 → S_DA-D1, S_5HT_1A → S_5HT-1A
            ns[f"S_{k.replace('_', '-', 1)}"] = v

    # Concentrations
    if concentrations:
        for k, v in concentrations.items():
            ns[f"C_{k}"] = v

    return ns


# ---------------------------------------------------------------------------
# Composite gate evaluator (M.5.2)
# ---------------------------------------------------------------------------

def evaluate_composite_gate(
    gate_expr: str,
    variables: Dict[str, float],
) -> float:
    """
    Evaluate a composite scoring gate expression.

    Supports: weight*var, weight*(1-var), addition.
    E.g.: "0.4*M_hat + 0.3*phi_gamma + 0.3*(1-R_hat)"

    Parameters
    ----------
    gate_expr : str
        Composite gate expression.
    variables : dict
        Variable namespace.

    Returns
    -------
    float
        Composite score (unbounded).
    """
    total = 0.0
    # Split on + (with optional whitespace)
    terms = re.split(r"\s*\+\s*", gate_expr.strip())

    for term in terms:
        term = term.strip()
        if not term:
            continue

        # Match weight*(1-var) pattern
        m = re.match(r"([0-9.]+)\s*\*\s*\(\s*1\s*-\s*([\w-]+)\s*\)", term)
        if m:
            weight = float(m.group(1))
            var = m.group(2)
            val = variables.get(var, 0.0)
            total += weight * (1.0 - val)
            continue

        # Match weight*var pattern
        m = re.match(r"([0-9.]+)\s*\*\s*([\w-]+)", term)
        if m:
            weight = float(m.group(1))
            var = m.group(2)
            val = variables.get(var, 0.0)
            total += weight * val
            continue

        # Bare variable or number
        try:
            total += float(term)
        except ValueError:
            total += variables.get(term, 0.0)

    return total


# ---------------------------------------------------------------------------
# Mode selector / arbiter (M.5.5)
# ---------------------------------------------------------------------------

def select_mode(
    mode_hooks: List[ModeHookDefinition],
    variables: Dict[str, float],
) -> ModeSelectionResult:
    """
    Evaluate all mode hooks and select winner via M.5.5 priority.

    Priority ordering:
    - Tier 0 (safety/containment) > Tier 1 (empathy) > Tier 2 (rigidity) > Tier 3 (drive)
    - Within a tier: highest composite_score wins; if no composite gates, first match wins.

    Parameters
    ----------
    mode_hooks : list of ModeHookDefinition
    variables : dict
        Variable namespace (from build_mode_namespace).

    Returns
    -------
    ModeSelectionResult
    """
    fired: List[str] = []
    scores: Dict[str, float] = {}

    # Group by tier
    tier_candidates: Dict[int, List[Tuple[str, float]]] = {}

    for hook in mode_hooks:
        try:
            condition_met = evaluate_condition(hook.condition_str, variables)
        except (ValueError, IndexError):
            continue

        if not condition_met:
            continue

        fired.append(hook.name)

        # Compute composite score if available
        score = 0.0
        if hook.composite_gate:
            score = evaluate_composite_gate(hook.composite_gate, variables)
        scores[hook.name] = score

        tier = hook.priority_tier
        if tier not in tier_candidates:
            tier_candidates[tier] = []
        tier_candidates[tier].append((hook.name, score))

    if not fired:
        return ModeSelectionResult(active_mode=None, fired_modes=())

    # Select winner: pick from lowest (highest-priority) tier
    for tier in sorted(tier_candidates.keys()):
        candidates = tier_candidates[tier]
        if not candidates:
            continue
        # Within tier: pick highest composite score
        candidates.sort(key=lambda x: x[1], reverse=True)
        winner = candidates[0][0]
        return ModeSelectionResult(
            active_mode=winner,
            fired_modes=tuple(fired),
            composite_scores=scores if scores else None,
        )

    return ModeSelectionResult(active_mode=None, fired_modes=tuple(fired))
