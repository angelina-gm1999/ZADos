"""
State-derived oscillation generator.

Derives oscillation band amplitudes from current neurochemical state,
closing the bidirectional NT ↔ oscillation loop.

Default derivation rules are based on Appendix I NT_BAND_ASSOCIATIONS:
- gamma ← DA phasic + GLU phasic + ACh phasic   (binding/integration)
- theta ← OXT tonic + 5HT tonic + DA tonic       (narrative/simulation)
- alpha ← GABA tonic + 5HT tonic + CB1 tonic      (inhibitory gating)
- beta  ← NE total + cortisol total + ACh total    (precision/arousal)
- delta ← MOR tonic + CB1 tonic + GABA tonic       (recovery/reset)
- sigma ← GABA tonic + GLU phasic - NE tonic       (sleep spindle, sleep-only)

Usage
-----
>>> from zados.neurochem.oscillations.generators import derive_oscillation_state
>>> osc = derive_oscillation_state(nt_states)
"""

from __future__ import annotations

from typing import Dict, Optional

from zados.neurochem.state import NeurotransmitterState
from zados.neurochem.state.oscillation_state import OscillationState


# Default band derivation rules.
# Structure: {band_name: {nt_name: (component, weight)}}
# component: "C_tonic", "C_phasic", or "C" (total)
DEFAULT_BAND_DERIVATION_RULES: Dict[str, Dict[str, tuple]] = {
    "gamma": {
        "DA": ("C_phasic", 0.4),
        "GLU": ("C_phasic", 0.3),
        "ACh": ("C_phasic", 0.3),
    },
    "theta": {
        "OXT": ("C_tonic", 0.4),
        "5HT": ("C_tonic", 0.3),
        "DA": ("C_tonic", 0.3),
    },
    "alpha": {
        "GABA": ("C_tonic", 0.5),
        "5HT": ("C_tonic", 0.3),
        "CB1": ("C_tonic", 0.2),
    },
    "beta": {
        "NE": ("C", 0.4),
        "cortisol": ("C", 0.3),
        "ACh": ("C", 0.3),
    },
    "delta": {
        "MOR": ("C_tonic", 0.4),
        "CB1": ("C_tonic", 0.3),
        "GABA": ("C_tonic", 0.3),
    },
    "sigma": {
        "GABA": ("C_tonic", 0.5),    # TRN GABA bursting drives spindle rhythm
        "GLU": ("C_phasic", 0.3),    # TC relay rebound bursting
        "NE": ("C_tonic", -0.3),     # NE suppresses spindles (negative weight)
    },
}


def _get_nt_component(state: NeurotransmitterState, component: str) -> float:
    """Extract a concentration component from an NT state."""
    if component == "C_tonic":
        return state.C_tonic
    elif component == "C_phasic":
        return state.C_phasic
    elif component == "C":
        return state.C
    else:
        return 0.0


def derive_oscillation_state(
    nt_states: Dict[str, NeurotransmitterState],
    band_derivation_rules: Optional[Dict[str, Dict[str, tuple]]] = None,
) -> OscillationState:
    """
    Derive oscillation band amplitudes from current NT concentrations.

    Each band amplitude is a weighted sum of NT concentration components,
    clamped to [0, 1].

    Parameters
    ----------
    nt_states : dict
        Map of NT name -> NeurotransmitterState
    band_derivation_rules : dict, optional
        Custom rules. Structure: {band: {nt_name: (component, weight)}}.
        Defaults to DEFAULT_BAND_DERIVATION_RULES.

    Returns
    -------
    OscillationState
        Derived oscillation state with all bands in [0, 1]
    """
    rules = band_derivation_rules or DEFAULT_BAND_DERIVATION_RULES

    band_values = {}
    for band_name in ["delta", "theta", "alpha", "beta", "gamma", "sigma"]:
        band_rules = rules.get(band_name, {})
        total = 0.0
        for nt_name, (component, weight) in band_rules.items():
            nt_state = nt_states.get(nt_name)
            if nt_state is not None:
                total += weight * _get_nt_component(nt_state, component)
        band_values[band_name] = max(0.0, min(1.0, total))

    return OscillationState(
        delta=band_values.get("delta", 0.0),
        theta=band_values.get("theta", 0.0),
        alpha=band_values.get("alpha", 0.0),
        beta=band_values.get("beta", 0.0),
        gamma=band_values.get("gamma", 0.0),
        sigma=band_values.get("sigma", 0.0),
    )
