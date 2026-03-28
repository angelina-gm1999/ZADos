"""
Regulatory Neurochemical Modulator (Extractor 3).

Provides temporally-smoothed (leaky-integrator) regulatory feedback that
maps evaluation vector axes to NT/receptor parameter modifications. This
replaces the instantaneous ``compute_reward_feedback()`` with gradual,
τ-controlled ramp dynamics.

Also computes oscillation envelope modulation: regulatory state shapes
oscillation band amplitudes via configurable rules.

Output format matches ``engine.apply_feedback()`` exactly — the caller
simply passes the feedback_params dict to the engine.

Usage
-----
>>> from zados.neurochem.extractors.regulatory_modulator import (
...     RegulatoryModulatorState, step_regulatory_modulator,
...     compute_oscillation_envelope, DEFAULT_REGULATORY_CONFIG,
... )
>>> state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
>>> state, feedback = step_regulatory_modulator(state, eval_vector, config, dt=0.01)
>>> engine.apply_feedback(feedback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from zados.neurochem.extractors.leaky_integrator import (
    LeakyIntegratorState,
    leaky_integrator_step,
)
from zados.neurochem.state.oscillation_state import OscillationState


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class RegulatoryPathwayConfig:
    """
    Configuration for a single regulatory feedback pathway.

    Each pathway maps an evaluation axis → NT/receptor parameter via
    a leaky integrator for temporal smoothing.

    Attributes
    ----------
    name : str
        Pathway identifier (e.g., "OXT_attunement").
    evaluation_axis : str
        Which e_k(t) drives this pathway.
    target_nt : str
        Target NT or receptor modified.
    target_param : str
        Parameter being modified: "C_baseline_delta", "u_base_multiplier",
        "K_d_multiplier".
    target_category : str
        "neurotransmitters" or "receptors" — determines placement in
        the feedback_params dict.
    coupling_weight : float
        Coupling strength ρ_k^(i).
    tau : float
        Time constant for leaky integrator (seconds).
    baseline : float
        Equilibrium value R_0 for the integrator.
    gain : float
        Input gain for the integrator.
    center : float
        Neutral point — evaluation axis value that produces zero output.
        (Only used for "C_baseline_delta" type pathways.)
    """
    name: str
    evaluation_axis: str
    target_nt: str
    target_param: str
    target_category: str = "neurotransmitters"
    coupling_weight: float = 1.0
    tau: float = 20.0
    baseline: float = 0.0
    gain: float = 0.05
    center: float = 0.5


@dataclass(frozen=True)
class RegulatoryModulatorConfig:
    """
    Full regulatory modulator configuration.

    Attributes
    ----------
    pathways : tuple of RegulatoryPathwayConfig
        All regulatory feedback pathways.
    """
    pathways: Tuple[RegulatoryPathwayConfig, ...]


# Default 4 pathways matching existing feedback system:
# OXT baseline  ← social_salience   (Attunement)
# CB1 baseline  ← novelty           (Innovation)
# NE  reuptake  ← logical_conflict  (Logic × ContradictionLoad)
# GABA_B K_d    ← urgency           (Ethics × TimelineMismatch)
#
# Note: Only 4 of 8 evaluation axes have dedicated regulatory pathways.
# The remaining 4 axes (emotional_valence, coherence, reward_alignment,
# identity_resonance) influence neurochemistry through the reactivity
# matrix (stochastic burst deltas) and the 4M/4R emotion splitter
# (modulatory and reactive pathways) rather than slow τ-smoothed
# regulatory feedback.
DEFAULT_REGULATORY_CONFIG = RegulatoryModulatorConfig(pathways=(
    RegulatoryPathwayConfig(
        name="OXT_attunement",
        evaluation_axis="social_salience",
        target_nt="OXT",
        target_param="C_baseline_delta",
        target_category="neurotransmitters",
        coupling_weight=1.0,
        tau=20.0,
        baseline=0.0,
        gain=0.05,
        center=0.5,
    ),
    RegulatoryPathwayConfig(
        name="CB1_innovation",
        evaluation_axis="novelty",
        target_nt="CB1",
        target_param="C_baseline_delta",
        target_category="neurotransmitters",
        coupling_weight=1.0,
        tau=25.0,
        baseline=0.0,
        gain=0.05,
        center=0.5,
    ),
    RegulatoryPathwayConfig(
        name="NE_logic",
        evaluation_axis="logical_conflict",
        target_nt="NE",
        target_param="u_base_multiplier",
        target_category="neurotransmitters",
        coupling_weight=1.0,
        tau=15.0,
        baseline=1.0,
        gain=0.3,
        center=0.0,
    ),
    # GABA_B receptor K_d tuning — slow, τ-smoothed regulatory pathway.
    # Distinct from GABA NT burst deltas in reactivity_matrix.py which
    # produce fast phasic concentration changes.
    RegulatoryPathwayConfig(
        name="GABA_B_ethics",
        evaluation_axis="urgency",
        target_nt="GABA_B",
        target_param="K_d_multiplier",
        target_category="receptors",
        coupling_weight=1.0,
        tau=20.0,
        baseline=1.0,
        gain=0.2,
        center=0.0,
    ),
))


# =====================================================================
# Oscillation Envelope Rules
# =====================================================================

@dataclass(frozen=True)
class OscillationEnvelopeRule:
    """
    Rule mapping regulatory integrator output to oscillation band amplitude.

    Attributes
    ----------
    pathway_name : str
        Which regulatory integrator to read.
    target_band : str
        Oscillation band to modulate ("delta", "theta", "alpha", "beta", "gamma").
    coefficient : float
        Modulation strength.
    formula : str
        "additive" — new_amp = current_amp + coefficient * integrator_value
        "multiplicative" — new_amp = current_amp * (1.0 + coefficient * integrator_value)
    """
    pathway_name: str
    target_band: str
    coefficient: float
    formula: str = "additive"


DEFAULT_ENVELOPE_RULES: Tuple[OscillationEnvelopeRule, ...] = (
    OscillationEnvelopeRule("OXT_attunement",  "theta", 0.3),   # social → theta
    OscillationEnvelopeRule("CB1_innovation",  "gamma", 0.4),   # novelty → gamma
    OscillationEnvelopeRule("NE_logic",        "beta",  0.3),   # conflict → beta
    OscillationEnvelopeRule("GABA_B_ethics",   "alpha", 0.2),   # urgency → alpha
)


# =====================================================================
# State
# =====================================================================

@dataclass
class RegulatoryModulatorState:
    """
    State container for the regulatory modulator.

    Holds one ``LeakyIntegratorState`` per pathway.

    Attributes
    ----------
    integrator_states : dict
        Maps pathway name → LeakyIntegratorState.
    """
    integrator_states: Dict[str, LeakyIntegratorState] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: RegulatoryModulatorConfig) -> RegulatoryModulatorState:
        """
        Initialize from config — each pathway gets an integrator at its baseline.

        Parameters
        ----------
        config : RegulatoryModulatorConfig
            Configuration with pathway definitions.

        Returns
        -------
        RegulatoryModulatorState
            Initialized state.
        """
        integrators = {}
        for pathway in config.pathways:
            integrators[pathway.name] = LeakyIntegratorState(
                value=pathway.baseline,
                baseline=pathway.baseline,
            )
        return cls(integrator_states=integrators)

    def as_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "integrator_states": {
                name: state.as_dict()
                for name, state in self.integrator_states.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> RegulatoryModulatorState:
        """Restore from dictionary."""
        integrators = {}
        for name, state_dict in data.get("integrator_states", {}).items():
            integrators[name] = LeakyIntegratorState.from_dict(state_dict)
        return cls(integrator_states=integrators)


# =====================================================================
# Pure functions
# =====================================================================

def _compute_pathway_input(
    evaluation_vector: Dict[str, float],
    pathway: RegulatoryPathwayConfig,
) -> float:
    """
    Compute the input signal for a single regulatory pathway.

    For baseline-delta pathways: input = (e_k - center) * coupling_weight
    For multiplier pathways: input = e_k * coupling_weight

    Parameters
    ----------
    evaluation_vector : dict
        Current E(t).
    pathway : RegulatoryPathwayConfig
        Pathway config.

    Returns
    -------
    float
        Input signal to the leaky integrator.
    """
    e_k = evaluation_vector.get(pathway.evaluation_axis, 0.0)

    if pathway.target_param == "C_baseline_delta":
        # Baseline delta: deviation from center drives the integrator
        return (e_k - pathway.center) * pathway.coupling_weight
    else:
        # Multiplier pathways: axis value directly drives
        return e_k * pathway.coupling_weight


def _integrator_output_to_feedback(
    integrator_value: float,
    pathway: RegulatoryPathwayConfig,
) -> float:
    """
    Convert integrator output to the final feedback parameter value.

    For C_baseline_delta: the integrator value IS the delta (clamped).
    For multiplier types: 1.0 + integrator_value (centered on baseline=1.0).

    Parameters
    ----------
    integrator_value : float
        Current leaky integrator value.
    pathway : RegulatoryPathwayConfig
        Pathway configuration.

    Returns
    -------
    float
        Feedback parameter value.
    """
    if pathway.target_param == "C_baseline_delta":
        # Clamp to [-gain, +gain]
        return max(-pathway.gain, min(pathway.gain, integrator_value))
    else:
        # For multiplier types, the integrator holds deviation from baseline=1.0
        # Clamp to [1-gain, 1+gain]
        return max(1.0 - pathway.gain, min(1.0 + pathway.gain, integrator_value))


def step_regulatory_modulator(
    state: RegulatoryModulatorState,
    evaluation_vector: Dict[str, float],
    config: RegulatoryModulatorConfig = DEFAULT_REGULATORY_CONFIG,
    dt: float = 0.01,
) -> Tuple[RegulatoryModulatorState, Dict]:
    """
    Step all regulatory pathways by one timestep.

    Each pathway:
    1. Extracts input from evaluation vector
    2. Steps leaky integrator
    3. Converts integrator output to feedback parameter

    Parameters
    ----------
    state : RegulatoryModulatorState
        Current state.
    evaluation_vector : dict
        Current E(t), maps axis name → float.
    config : RegulatoryModulatorConfig
        Pathway configurations.
    dt : float
        Timestep.

    Returns
    -------
    tuple of (RegulatoryModulatorState, dict)
        New state and feedback_params dict matching engine.apply_feedback() format::

            {
                "neurotransmitters": {
                    "OXT": {"C_baseline_delta": float},
                    ...
                },
                "receptors": {
                    "GABA_B": {"K_d_multiplier": float},
                    ...
                },
            }
    """
    new_integrators = {}
    feedback: Dict[str, Dict[str, Dict[str, float]]] = {
        "neurotransmitters": {},
        "receptors": {},
    }

    for pathway in config.pathways:
        # Get current integrator state (or create default)
        integrator = state.integrator_states.get(
            pathway.name,
            LeakyIntegratorState(value=pathway.baseline, baseline=pathway.baseline),
        )

        # Compute input signal from evaluation vector
        input_signal = _compute_pathway_input(evaluation_vector, pathway)

        # Step the leaky integrator
        new_integrator = leaky_integrator_step(
            integrator,
            input_signal,
            dt,
            tau=pathway.tau,
            gain=pathway.gain,
        )
        new_integrators[pathway.name] = new_integrator

        # Convert integrator output to feedback parameter
        param_value = _integrator_output_to_feedback(
            new_integrator.value, pathway,
        )

        # Place in feedback dict
        category = feedback[pathway.target_category]
        if pathway.target_nt not in category:
            category[pathway.target_nt] = {}
        category[pathway.target_nt][pathway.target_param] = param_value

    new_state = RegulatoryModulatorState(integrator_states=new_integrators)
    return new_state, feedback


def compute_oscillation_envelope(
    regulatory_state: RegulatoryModulatorState,
    current_oscillations: OscillationState,
    rules: Tuple[OscillationEnvelopeRule, ...] = DEFAULT_ENVELOPE_RULES,
) -> OscillationState:
    """
    Modulate oscillation band amplitudes from regulatory state.

    Each rule reads a regulatory integrator value and adjusts a target
    band amplitude additively or multiplicatively.

    Parameters
    ----------
    regulatory_state : RegulatoryModulatorState
        Current regulatory modulator state.
    current_oscillations : OscillationState
        Current oscillation state (will be copied, not mutated).
    rules : tuple of OscillationEnvelopeRule
        Modulation rules.

    Returns
    -------
    OscillationState
        New oscillation state with modified band amplitudes.
    """
    result = current_oscillations.copy()

    for rule in rules:
        integrator = regulatory_state.integrator_states.get(rule.pathway_name)
        if integrator is None:
            continue

        # Use absolute distance from baseline as modulation signal.
        # This ensures both positive and negative deviations from the
        # integrator's resting point produce oscillation envelope
        # modulation (e.g., both high and low social_salience shift
        # theta amplitude), matching the bidirectional design of the
        # regulatory pathways.
        mod_signal = abs(integrator.value - integrator.baseline)

        current_amp = result.get_band(rule.target_band)

        if rule.formula == "additive":
            new_amp = current_amp + rule.coefficient * mod_signal
        elif rule.formula == "multiplicative":
            new_amp = current_amp * (1.0 + rule.coefficient * mod_signal)
        else:
            raise ValueError(f"Unknown envelope formula: {rule.formula!r}")

        # set_band clamps to [0, 1]
        result.set_band(rule.target_band, new_amp)

    return result
