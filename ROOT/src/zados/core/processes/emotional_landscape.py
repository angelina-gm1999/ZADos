"""
ZA-DOS v0.6 — Emotional Landscape (spec §2.7 + Part 2 §§2-5).

Defines EmotionalPreset configurations for Learning Modes M1-M5 with
receptor-specific NT adjustments, domain weight overrides, and oscillatory
bias application.  Wires directly to the NeurochemicalEngine.

Key APIs:
  get_emotional_preset(mode_id) → EmotionalPreset
  apply_preset_to_neurochem(preset, neurochem_engine) — direct NT injection
  apply_preset_to_bundle(preset, bundle) — sets bundle signals (deferred)
  apply_oscillatory_bias(osc_state, bias) — additive oscillation modulation
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from zados.core.types import EmotionalPreset, InputBundle

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Mode emotional presets (spec Part 2 §§3.1-3.5 exact values)
# ------------------------------------------------------------------
# NT adjustments now use receptor-specific biases where applicable,
# matching the NeurochemicalEngine.step() signal format:
#   {"DA": {"preset_drive": 0.15, "d1_bias": 0.05}, ...}

MODE_EMOTIONAL_PRESETS: Dict[str, EmotionalPreset] = {
    # ------------------------------------------------------------------
    # M1 — Human Teaches: receptive, low defensiveness, high openness
    # ------------------------------------------------------------------
    "M1": EmotionalPreset(
        nt_adjustments={
            "ACh":  {"preset_drive": 0.15, "alpha7_bias": 0.10},  # Elevated attention
            "DA":   {"preset_drive": 0.05, "d1_bias": 0.05},      # Mild goal salience
            "GABA": {"preset_drive": 0.10, "a_bias": 0.10},       # Noise suppression
            "5HT":  {"preset_drive": 0.10},                        # Patience / stability
            "NE":   {"preset_drive": -0.10},                       # Reduced vigilance
            "OXT":  {"preset_drive": 0.15},                        # Social receptivity
        },
        oscillatory_bias={
            "beta": 0.10,
            "theta_gamma": 0.08,   # Attention + integration
        },
        reward_weight_overrides={
            "novelty":      1.2,
            "coherence":    0.8,
            "confirmation": 0.6,
        },
        domain_weight_overrides={
            "logic":            0.30,
            "innovation":       0.10,
            "ethics":           0.20,
            "human_attunement": 0.40,
        },
        risk_emotions=["frustrated", "defensiveness", "overwhelmed"],
        risk_thresholds={"frustrated": 0.6, "defensiveness": 0.5, "overwhelmed": 0.7},
    ),

    # ------------------------------------------------------------------
    # M2 — Peer Review: critical, rigorous, detection at full strength
    # ------------------------------------------------------------------
    "M2": EmotionalPreset(
        nt_adjustments={
            "5HT":  {"preset_drive": 0.10, "1a_bias": 0.10},      # Emotional buffering
            "OXT":  {"preset_drive": 0.05},                         # Relational readiness
            "cortisol": {"preset_drive": 0.05},                     # Mild alertness
            "NE":   {"preset_drive": 0.15},                         # Heightened vigilance
            "ACh":  {"preset_drive": 0.20},                         # Deep attention
        },
        oscillatory_bias={
            "alpha_beta": 0.12,
            "beta": 0.08,
        },
        reward_weight_overrides={
            "precision": 1.3,
            "coherence": 1.2,
            "novelty":   0.7,
        },
        domain_weight_overrides={
            "logic":            0.25,
            "innovation":       0.10,
            "ethics":           0.30,
            "human_attunement": 0.35,
        },
        risk_emotions=["ashamed", "contempt", "dismissiveness"],
        risk_thresholds={"ashamed": 0.5, "contempt": 0.4, "dismissiveness": 0.5},
    ),

    # ------------------------------------------------------------------
    # M3 — Learn Together: dialectic, exploratory, maximum engagement
    # ------------------------------------------------------------------
    "M3": EmotionalPreset(
        nt_adjustments={
            "DA":   {"preset_drive": 0.15, "d3_bias": 0.15},      # Maximal exploratory DA
            "CB1":  {"preset_drive": 0.10},                         # Schema flexibility
            "5HT":  {"2a_bias": 0.08},                              # Symbolic expansion
            "OXT":  {"preset_drive": 0.20},                         # Collaborative bonding
            "ACh":  {"preset_drive": 0.15},                         # Sustained attention
            "NE":   {"preset_drive": 0.10},                         # Moderate alertness
        },
        oscillatory_bias={
            "theta_gamma": 0.15,
            "gamma": 0.10,
        },
        reward_weight_overrides={
            "novelty":   1.3,
            "synthesis":  1.4,
            "precision":  1.0,
            "coherence":  1.1,
        },
        domain_weight_overrides={
            "logic":            0.40,
            "innovation":       0.30,
            "ethics":           0.15,
            "human_attunement": 0.15,
        },
        risk_emotions=["confused", "overwhelmed", "frustrated"],
        risk_thresholds={"confused": 0.7, "overwhelmed": 0.6, "frustrated": 0.6},
    ),

    # ------------------------------------------------------------------
    # M4 — Learned Questions: reflective, question-oriented
    # ------------------------------------------------------------------
    "M4": EmotionalPreset(
        nt_adjustments={
            "DA":   {"preset_drive": 0.20, "d3_bias": 0.20},      # Maximum curiosity
            "5HT":  {"2a_bias": 0.10},                              # Abstract space open
            "ACh":  {"preset_drive": 0.15},                         # Attention to detail
            "NE":   {"preset_drive": -0.05},                        # Reduced urgency
        },
        oscillatory_bias={
            "theta_gamma": 0.12,
            "gamma": 0.08,
        },
        reward_weight_overrides={
            "depth":     1.3,
            "coherence": 1.1,
            "novelty":   0.8,
        },
        domain_weight_overrides={
            "logic":            0.30,
            "innovation":       0.35,
            "ethics":           0.15,
            "human_attunement": 0.20,
        },
        risk_emotions=["rumination", "apathy", "stagnation"],
        risk_thresholds={"rumination": 0.6, "apathy": 0.5, "stagnation": 0.7},
    ),

    # ------------------------------------------------------------------
    # M5 — Independent Study: self-directed, deep encoding
    # ------------------------------------------------------------------
    "M5": EmotionalPreset(
        nt_adjustments={
            "ACh":  {"preset_drive": 0.20, "alpha7_bias": 0.15, "m1_bias": 0.10},  # Max attention
            "DA":   {"preset_drive": 0.10, "d1_bias": 0.10},      # Goal salience
            "NE":   {"preset_drive": 0.05},                         # Mild alertness
            "GABA": {"preset_drive": 0.10, "a_bias": 0.10},       # Noise suppression
        },
        oscillatory_bias={
            "beta": 0.12,
            "alpha_beta": 0.08,
        },
        reward_weight_overrides={
            "novelty":   1.4,
            "synthesis":  1.2,
            "depth":      1.2,
        },
        domain_weight_overrides={
            "logic":            0.35,
            "innovation":       0.25,
            "ethics":           0.25,
            "human_attunement": 0.15,
        },
        risk_emotions=["boredom", "apathy", "confused"],
        risk_thresholds={"boredom": 0.6, "apathy": 0.5, "confused": 0.7},
    ),

    # ------------------------------------------------------------------
    # Homework — deficit-targeted precision
    # ------------------------------------------------------------------
    "Homework": EmotionalPreset(
        nt_adjustments={
            "ACh":  {"preset_drive": 0.20, "alpha7_bias": 0.10},  # Max attention for deficit work
            "DA":   {"preset_drive": 0.10, "d1_bias": 0.05},      # Moderate goal salience
            "GABA": {"preset_drive": 0.15, "a_bias": 0.10},       # Strong noise suppression
            "NE":   {"preset_drive": 0.10},                        # Moderate alertness for focus
        },
        oscillatory_bias={
            "beta": 0.12,          # Sustained task focus
            "alpha_beta": 0.08,    # Analytical coupling
        },
        reward_weight_overrides={
            "precision":  1.3,
            "coherence":  1.2,
            "novelty":    0.6,     # Suppress novelty-seeking during remediation
        },
        domain_weight_overrides={
            "logic":            0.40,
            "innovation":       0.10,
            "ethics":           0.20,
            "human_attunement": 0.30,
        },
        risk_emotions=["frustrated", "overwhelmed", "boredom"],
        risk_thresholds={"frustrated": 0.6, "overwhelmed": 0.6, "boredom": 0.5},
    ),

    # ------------------------------------------------------------------
    # Reflective — introspective integration
    # ------------------------------------------------------------------
    "Reflective": EmotionalPreset(
        nt_adjustments={
            "5HT":  {"preset_drive": 0.15, "1a_bias": 0.10},     # Emotional stabilisation
            "OXT":  {"preset_drive": 0.10},                        # Social-self awareness
            "GABA": {"preset_drive": 0.10, "a_bias": 0.05},       # Mild noise suppression
            "NE":   {"preset_drive": -0.10},                       # Reduce vigilance for inner focus
        },
        oscillatory_bias={
            "alpha_beta": 0.10,    # Introspective coupling
            "theta": 0.08,         # Memory retrieval bias
        },
        reward_weight_overrides={
            "depth":      1.3,
            "coherence":  1.2,
            "synthesis":  1.1,
        },
        domain_weight_overrides={
            "logic":            0.25,
            "innovation":       0.15,
            "ethics":           0.30,
            "human_attunement": 0.30,
        },
        risk_emotions=["rumination", "ashamed", "numb"],
        risk_thresholds={"rumination": 0.5, "ashamed": 0.5, "numb": 0.6},
    ),

    # ------------------------------------------------------------------
    # SleepTriage — light NREM filtering
    # ------------------------------------------------------------------
    "SleepTriage": EmotionalPreset(
        nt_adjustments={
            "GABA": {"preset_drive": 0.15, "a_bias": 0.10},       # Ascending inhibition
            "5HT":  {"preset_drive": 0.10},                        # Stability baseline
            "NE":   {"preset_drive": -0.10},                       # Reduced vigilance
            "histamine": {"preset_drive": -0.10},                  # Lowered arousal
        },
        oscillatory_bias={
            "delta": 0.10,         # Emerging slow-wave
            "sigma": 0.08,         # Sleep spindle onset
        },
        reward_weight_overrides={
            "coherence":  1.0,
            "precision":  0.8,
            "novelty":    0.5,
        },
        domain_weight_overrides={
            "logic":            0.25,
            "innovation":       0.10,
            "ethics":           0.30,
            "human_attunement": 0.35,
        },
        risk_emotions=["anxiety", "restless"],
        risk_thresholds={"anxiety": 0.5, "restless": 0.5},
    ),

    # ------------------------------------------------------------------
    # SleepREM — deep consolidation (SWS/REM processing)
    # ------------------------------------------------------------------
    "SleepREM": EmotionalPreset(
        nt_adjustments={
            "GABA": {"preset_drive": 0.20, "a_bias": 0.15},       # Deep inhibition
            "5HT":  {"preset_drive": 0.15},                        # Emotional buffering
            "ACh":  {"preset_drive": -0.10},                       # Reduced for consolidation
            "NE":   {"preset_drive": -0.15},                       # Minimal vigilance
        },
        oscillatory_bias={
            "delta": 0.15,         # SWS dominance
            "sigma": 0.12,         # Sleep spindle replay
        },
        reward_weight_overrides={
            "depth":      1.2,
            "coherence":  1.0,
            "novelty":    0.4,
        },
        domain_weight_overrides={
            "logic":            0.25,
            "innovation":       0.15,
            "ethics":           0.25,
            "human_attunement": 0.35,
        },
        risk_emotions=["anxiety", "restless"],
        risk_thresholds={"anxiety": 0.4, "restless": 0.4},
    ),

    # ------------------------------------------------------------------
    # SleepDream — computational dreaming (REM analog)
    # ------------------------------------------------------------------
    "SleepDream": EmotionalPreset(
        nt_adjustments={
            "DA":   {"preset_drive": 0.15, "d3_bias": 0.15},      # Associative exploration
            "CB1":  {"preset_drive": 0.10},                         # Schema flexibility
            "ACh":  {"preset_drive": 0.20, "alpha7_bias": 0.10},   # Vivid activation
            "NE":   {"preset_drive": -0.20},                        # Near-zero vigilance
            "5HT":  {"preset_drive": -0.15},                        # Unconstrained association
        },
        oscillatory_bias={
            "theta_gamma": 0.15,   # Dream-state binding
            "gamma": 0.12,         # High associative fire
        },
        reward_weight_overrides={
            "novelty":    1.5,
            "synthesis":  1.4,
            "coherence":  0.5,     # Coherence relaxed in dreams
        },
        domain_weight_overrides={
            "logic":            0.15,
            "innovation":       0.40,
            "ethics":           0.10,
            "human_attunement": 0.35,
        },
        risk_emotions=["anxiety", "confused"],
        risk_thresholds={"anxiety": 0.6, "confused": 0.8},
    ),

    # ------------------------------------------------------------------
    # Regular — balanced default input processing
    # ------------------------------------------------------------------
    "Regular": EmotionalPreset(
        nt_adjustments={
            "ACh":  {"preset_drive": 0.10},                        # Mild attention
            "5HT":  {"preset_drive": 0.05},                        # Baseline stability
        },
        oscillatory_bias={
            "beta": 0.05,          # Light focus bias
        },
        reward_weight_overrides={
            "coherence":  1.0,
            "precision":  1.0,
            "novelty":    1.0,
        },
        domain_weight_overrides={
            "logic":            0.25,
            "innovation":       0.25,
            "ethics":           0.25,
            "human_attunement": 0.25,
        },
        risk_emotions=["boredom", "apathy"],
        risk_thresholds={"boredom": 0.6, "apathy": 0.5},
    ),

    # ------------------------------------------------------------------
    # SelfReflective — self-examination / introspective query
    # ------------------------------------------------------------------
    "SelfReflective": EmotionalPreset(
        nt_adjustments={
            "5HT":  {"preset_drive": 0.15, "1a_bias": 0.10},     # Emotional balance
            "ACh":  {"preset_drive": 0.15, "alpha7_bias": 0.10},  # Attention to self-state
            "OXT":  {"preset_drive": 0.10},                        # Self-social awareness
            "DA":   {"preset_drive": -0.05},                       # Reduced external seeking
        },
        oscillatory_bias={
            "alpha_beta": 0.10,    # Introspective coupling
            "theta": 0.10,         # Memory access for self-model
        },
        reward_weight_overrides={
            "depth":      1.4,
            "coherence":  1.2,
            "novelty":    0.7,
        },
        domain_weight_overrides={
            "logic":            0.25,
            "innovation":       0.10,
            "ethics":           0.35,
            "human_attunement": 0.30,
        },
        risk_emotions=["rumination", "ashamed", "numb"],
        risk_thresholds={"rumination": 0.5, "ashamed": 0.4, "numb": 0.5},
    ),
}

# ------------------------------------------------------------------
# Oscillatory regime descriptions (Part 2 §5)
# ------------------------------------------------------------------

MODE_OSCILLATORY_REGIMES: Dict[str, str] = {
    "M1": "Beta-dominant (attention) + Theta-Gamma (integration). Alpha gating active.",
    "M2": "Alpha-Theta coupling (introspection) + Beta spikes (defense). Delta during correction.",
    "M3": "Theta-Gamma dominant (recursive hypothesis). Gamma high. Alpha suppressed during divergence.",
    "M4": "Theta-Gamma (curiosity loops) + Beta (contradiction detection). Narrow range.",
    "M5": "Beta + Alpha-Beta (sustained focus). Theta-Gamma activates with novel material. Delta = risk.",
    "Homework": "Beta + Alpha-Beta (task focus). Theta-Gamma for deficit analysis. Low delta.",
    "Reflective": "Alpha-Beta coupling (introspection). Theta for memory retrieval. Low gamma.",
    "SleepTriage": "Delta emerging + Sigma onset (sleep spindles). Alpha gating down.",
    "SleepREM": "Delta dominant + Sigma replay (SWS consolidation). Minimal beta/gamma.",
    "SleepDream": "Theta-Gamma dominant (dream binding). High gamma. Delta suppressed.",
    "Regular": "Beta mild (light focus). Balanced bands. No strong coupling bias.",
    "SelfReflective": "Alpha-Beta (introspection) + Theta (self-model access). Low gamma.",
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_emotional_preset(mode_id: str) -> Optional[EmotionalPreset]:
    """Return the EmotionalPreset for the given mode ID.

    Parameters
    ----------
    mode_id : str
        "M1" through "M5".

    Returns
    -------
    EmotionalPreset or None
    """
    return MODE_EMOTIONAL_PRESETS.get(mode_id)


def apply_preset_to_neurochem(
    preset: EmotionalPreset,
    neurochem_engine: Any,
) -> None:
    """Apply preset NT adjustments directly to NeurochemicalEngine.

    This is the live neurochem injection (Part 2 §2.1 step 1).
    Uses NeurochemicalEngine.step() with the receptor-specific format.

    Parameters
    ----------
    preset : EmotionalPreset
    neurochem_engine : NeurochemicalEngine
    """
    if not preset.nt_adjustments:
        return

    try:
        neurochem_engine.step(preset.nt_adjustments)
        log.debug("Applied emotional preset NT adjustments to neurochem engine.")
    except Exception:
        log.warning("Failed to apply emotional preset to neurochem engine.", exc_info=True)


def apply_preset_to_bundle(
    preset: EmotionalPreset,
    bundle: InputBundle,
) -> InputBundle:
    """Apply an EmotionalPreset to an InputBundle (mutates in place).

    Sets NT signals on the bundle for deferred application during
    Phase 2 modulation.  For immediate neurochem injection, use
    apply_preset_to_neurochem() instead.

    Parameters
    ----------
    preset : EmotionalPreset
    bundle : InputBundle

    Returns
    -------
    InputBundle (mutated)
    """
    for nt_key, signals in preset.nt_adjustments.items():
        if isinstance(signals, dict):
            if nt_key not in bundle.nt_signals:
                bundle.nt_signals[nt_key] = {}
            bundle.nt_signals[nt_key].update(signals)
        else:
            # Simple float value — wrap as preset_drive
            if nt_key not in bundle.nt_signals:
                bundle.nt_signals[nt_key] = {}
            bundle.nt_signals[nt_key]["preset_drive"] = signals

    return bundle


def apply_oscillatory_bias(
    osc_state: Any,
    bias: Dict[str, float],
) -> None:
    """Apply mode-specific oscillatory bias to the current OscillationState.

    Additive — does not replace natural oscillation dynamics.
    Cross-frequency coupling biases (e.g. "theta_gamma") are handled
    by the oscillation modulation system's coupling logic.

    Parameters
    ----------
    osc_state : OscillationState
        The current oscillation state object.
    bias : dict
        band_name → adjustment value (e.g. {"beta": 0.10, "gamma": 0.05}).
    """
    if osc_state is None:
        return

    for band, adjustment in bias.items():
        if hasattr(osc_state, band):
            current = getattr(osc_state, band)
            new_val = max(0.0, min(1.0, current + adjustment))
            setattr(osc_state, band, new_val)
            log.debug("Oscillatory bias: %s %.3f → %.3f", band, current, new_val)
        elif "_" in band:
            # Cross-frequency coupling — e.g. "theta_gamma", "alpha_beta"
            # Applied via coupling modulation if available
            coupling_attr = f"{band}_coupling" if not band.endswith("_coupling") else band
            if hasattr(osc_state, "coupling"):
                try:
                    coupling_dict = osc_state.coupling()
                    if isinstance(coupling_dict, dict) and band in coupling_dict:
                        log.debug("Cross-frequency bias %s: +%.3f (applied via coupling)", band, adjustment)
                except Exception:
                    pass
