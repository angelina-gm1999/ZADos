"""
ZA-DOS Core Pipeline — Phase 2: NT Modulation (post-dispatch, spec Part V revised).

Runs AFTER Phase 3 (engine dispatch).  Mode selection is NOT performed here —
the InputClassifier fork (RegularInputPipeline) already locked the mode to
"regular_input"; fine-grained reward profile selection is delegated to
RegularInputPipeline._profile_from_intent().

Revised 7-step sequence:
1. Normalize and apply bundle.nt_signals → NeurochemEngine.step()
2. Read stmm.emotion_detection (populated by E28 in Phase 3) → emotion NT
   → engine.step()  [STMM is canonical bridge, not dispatch_result directly]
3. Run ExtractorOrchestrator sub-components:
   - emotion inputs from STMM, eval vector from E23 intent
   - emotion_tracker → regulatory_modulator → burst_deltas → 4R reactive signals
4. Apply extractor modulation_signals → engine.step()
5. Neurosymbolic readout → NeurochemicalMetrics
6. Determine reward_profile_name from E23 intent category (no neurosymbolic
   select_mode — that lives in session boot and learning pipelines)
7. Compute engine priority weights from metrics

Also populates STMM cephalic_liquid_logger and cortical_reflection.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from zados.cognitive_engines.constants import normalize_nt_key
from zados.core.types import EngineDispatchResult, InputBundle, NTModulationResult, PerceptionSnapshot

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent category → reward profile mapping
# E23 IntentCategory values (lowercase) → static profile names
# ---------------------------------------------------------------------------
_INTENT_CATEGORY_TO_PROFILE: Dict[str, str] = {
    "connection":     "receptive_learning",
    "challenge":      "critical_review",
    "exploration":    "curiosity_driven",
    "discharge":      "receptive_learning",
    "pragmatic":      "regular_input",
    "symbolic":       "reflective_synthesis",
    "defensive":      "critical_review",
    "disintegration": "regular_input",
}

# Intent category → evaluation vector axes (for extractor sub-components)
_INTENT_TO_EVAL_AXIS: Dict[str, Dict[str, float]] = {
    "connection":     {"emotional_valence": 0.7, "social_salience": 0.6},
    "challenge":      {"logical_conflict": 0.6, "coherence": 0.7},
    "exploration":    {"novelty": 0.8, "reward_alignment": 0.6},
    "discharge":      {"emotional_valence": 0.8, "urgency": 0.5},
    "pragmatic":      {"coherence": 0.7, "reward_alignment": 0.6},
    "symbolic":       {"identity_resonance": 0.7, "emotional_valence": 0.5},
    "defensive":      {"urgency": 0.6, "logical_conflict": 0.5},
    "disintegration": {"urgency": 0.9, "emotional_valence": 0.3},
}

# Default eval vector when intent category is unknown
_DEFAULT_EVAL_AXIS: Dict[str, float] = {
    "coherence": 0.5,
    "reward_alignment": 0.5,
    "novelty": 0.3,
}


def run_nt_modulation(
    bundle: InputBundle,
    perception: PerceptionSnapshot,
    dispatch_result: EngineDispatchResult,
    neurochem_engine: Any,
    stmm: Any,
    osc_state: Any = None,
    extractor_state: Any = None,
) -> NTModulationResult:
    """Run NT modulation using post-dispatch data.

    Parameters
    ----------
    bundle : InputBundle
    perception : PerceptionSnapshot
        Phase 1 output — carries E23 IntentionMapResult.
    dispatch_result : EngineDispatchResult
        Phase 3 output — carries E28 EmotionalDetectionResult.
    neurochem_engine : NeurochemicalEngine
    stmm : STMMStore
        STMM is mutated: cephalic_liquid_logger + cortical_reflection.
    osc_state : OscillationState, optional
    extractor_state : ExtractorState, optional
        Persisted extractor state from session (carries leaky integrators
        across turns).
    """
    result = NTModulationResult()

    # ------------------------------------------------------------------
    # Step 1: Apply bundle NT signals
    # ------------------------------------------------------------------
    if bundle.nt_signals:
        normalized = _normalize_signal_dict(bundle.nt_signals)
        try:
            neurochem_engine.step(modulation_signals=normalized)
        except Exception:
            log.exception("NeurochemEngine.step(bundle signals) failed.")

    # ------------------------------------------------------------------
    # Step 2: Read emotion state from STMM (populated by E28 in Phase 3)
    #         → convert to NT signals and apply
    # ------------------------------------------------------------------
    _apply_stmm_emotion_signals(stmm, neurochem_engine)

    # ------------------------------------------------------------------
    # Step 3+4: Run extractor sub-components → apply modulation signals
    # Emotion inputs sourced from STMM (canonical bridge) not dispatch_result
    # ------------------------------------------------------------------
    new_extractor_state, ext_result = _run_extractor_subcomponents(
        perception=perception,
        stmm=stmm,
        osc_state=osc_state,
        extractor_state=extractor_state,
    )
    result.extractor_result = ext_result
    result.updated_extractor_state = new_extractor_state

    if ext_result is not None and ext_result.modulation_signals:
        try:
            normalized_ext = _normalize_signal_dict(ext_result.modulation_signals)
            neurochem_engine.step(modulation_signals=normalized_ext)
        except Exception:
            log.exception("NeurochemEngine.step(extractor signals) failed.")

    # ------------------------------------------------------------------
    # Step 5: Neurosymbolic readout → metrics
    # ------------------------------------------------------------------
    metrics_dict: Dict[str, float] = {}
    metrics = None
    try:
        readout = neurochem_engine.get_neurosymbolic_readout()
        if isinstance(readout, dict):
            metrics_dict = readout
        elif hasattr(readout, "as_dict"):
            metrics_dict = readout.as_dict()
            metrics = readout
        else:
            metrics_dict = dict(readout)
    except Exception:
        log.exception("get_neurosymbolic_readout() failed.")

    result.metrics = metrics
    result.metrics_dict = metrics_dict

    # ------------------------------------------------------------------
    # Step 6: Reward profile from intent category (no select_mode)
    # For regular input, the mode_token is always "regular_input".
    # Fine-grained profile selection lives in RegularInputPipeline.
    # ------------------------------------------------------------------
    mode_token = "regular_input"
    reward_profile_name = _profile_from_perception(perception)

    result.mode_token = mode_token
    result.reward_profile_name = reward_profile_name

    # ------------------------------------------------------------------
    # Step 7: Engine priority weights
    # ------------------------------------------------------------------
    try:
        from zados.neurochem.inference_matrix.nt_to_engine import (
            compute_engine_priority_weights,
        )
        epw = compute_engine_priority_weights(metrics_dict)
        result.engine_weights = epw.as_dict()
    except Exception:
        log.exception("compute_engine_priority_weights() failed.")

    # ------------------------------------------------------------------
    # Populate STMM
    # ------------------------------------------------------------------
    _populate_stmm(stmm, neurochem_engine, result, osc_state)

    return result


# ------------------------------------------------------------------
# STMM emotion → NT signals
# ------------------------------------------------------------------

def _read_stmm_emotion_inputs(stmm: Any) -> Dict[str, float]:
    """Read emotion state from stmm.emotion_detection (canonical STMM bridge).

    Returns a merged dict: user_emotion_signals overrides system_emotion_state
    so that user-detected emotions (from E28) take priority over system state.
    """
    try:
        ed = stmm.emotion_detection
        system_state = dict(getattr(ed, "system_emotion_state", {}) or {})
        user_signals = dict(getattr(ed, "user_emotion_signals", {}) or {})
        system_state.update(user_signals)   # user signals take priority
        return {k: float(v) for k, v in system_state.items() if v}
    except Exception:
        return {}


def _apply_stmm_emotion_signals(stmm: Any, neurochem_engine: Any) -> None:
    """Read emotion state from STMM and apply as NT signals."""
    emotion_profile = _read_stmm_emotion_inputs(stmm)
    if not emotion_profile:
        return
    try:
        from zados.neurochem.utils.emotion_interface import emotion_profile_to_signals
        signals = emotion_profile_to_signals(emotion_profile)
        normalized = _normalize_signal_dict(signals)
        neurochem_engine.step(modulation_signals=normalized)
    except Exception:
        log.exception("STMM emotion→NT signal application failed.")


# ------------------------------------------------------------------
# Extractor sub-components
# ------------------------------------------------------------------

def _run_extractor_subcomponents(
    perception: PerceptionSnapshot,
    stmm: Any,
    osc_state: Any,
    extractor_state: Any,
) -> tuple:
    """Run extractor sub-components with an intent-based evaluation vector.

    Reads emotion inputs from STMM (canonical bridge) so it stays consistent
    with whatever E28 wrote during Phase 3.

    Returns (updated_extractor_state, ExtractorResult | None).
    """
    try:
        from zados.neurochem.extractors.extractor_orchestrator import (
            ExtractorOrchestrator,
            ExtractorState,
        )
        from zados.neurochem.extractors.emotion_tracker import step_emotion_tracker
        from zados.neurochem.extractors.regulatory_modulator import step_regulatory_modulator
        from zados.neurochem.extractors.reactivity_matrix import (
            compute_stochastic_burst_deltas,
            burst_deltas_to_modulation_signals,
        )
        from zados.neurochem.extractors.urgency_forecast import step_urgency_forecast
        from zados.neurochem.extractors.emotion_splitter import split_emotion_effects
        from zados.neurochem.utils.emotion_interface import emotion_profile_to_signals

        # Restore or initialize extractor state
        if extractor_state is not None:
            state = extractor_state
        else:
            state = ExtractorState.initialize()

        # Build eval vector from intent category
        eval_vector = _build_intent_eval_vector(perception)

        # Emotion inputs from STMM (populated by E28 in Phase 3)
        emotion_inputs: Dict[str, float] = _read_stmm_emotion_inputs(stmm)

        dt = 0.01

        # Step emotion tracker
        from zados.neurochem.extractors.emotion_tracker import (
            DEFAULT_EMOTION_TRACKER_CONFIGS,
            get_dominant_emotion,
            get_emotion_saturations,
        )
        from zados.neurochem.extractors.regulatory_modulator import (
            DEFAULT_REGULATORY_CONFIG,
            DEFAULT_ENVELOPE_RULES,
            compute_oscillation_envelope,
        )
        from zados.neurochem.extractors.reactivity_matrix import DEFAULT_REACTIVITY_CONFIG
        from zados.neurochem.extractors.urgency_forecast import DEFAULT_URGENCY_FORECAST_CONFIG
        from zados.neurochem.extractors.emotion_splitter import DEFAULT_EMOTION_SPLIT_CONFIGS

        if state.emotion_tracker_state is not None:
            state.emotion_tracker_state = step_emotion_tracker(
                state.emotion_tracker_state,
                emotion_inputs,
                dt,
                configs=DEFAULT_EMOTION_TRACKER_CONFIGS,
            )

        emotion_saturations = (
            get_emotion_saturations(state.emotion_tracker_state)
            if state.emotion_tracker_state else {}
        )
        dominant_emotion = (
            get_dominant_emotion(state.emotion_tracker_state)
            if state.emotion_tracker_state else ("none", 0.0)
        )

        # Split emotions → 4M + 4R
        modulatory_adjustments: Dict[str, float] = {}
        reactive_profile: Dict[str, float] = {}
        if state.emotion_tracker_state is not None:
            modulatory_adjustments, reactive_profile = split_emotion_effects(
                state.emotion_tracker_state,
                DEFAULT_EMOTION_SPLIT_CONFIGS,
            )

        # Add 4M adjustments to eval vector
        adjusted_eval = dict(eval_vector)
        for axis, adj in modulatory_adjustments.items():
            adjusted_eval[axis] = max(0.0, min(1.0, adjusted_eval.get(axis, 0.5) + adj))

        # Urgency forecast
        urgency_risk = 0.0
        urgency_burst_deltas: Dict[str, float] = {}
        urgency_feedback: Dict[str, Any] = {"neurotransmitters": {}, "receptors": {}}
        if state.urgency_forecast_state is not None:
            (state.urgency_forecast_state,
             urgency_risk,
             urgency_burst_deltas,
             urgency_feedback) = step_urgency_forecast(
                state.urgency_forecast_state,
                adjusted_eval,
                dt,
                config=DEFAULT_URGENCY_FORECAST_CONFIG,
                rng=None,
            )

        # Step regulatory modulator
        feedback_params: Dict[str, Any] = {"neurotransmitters": {}, "receptors": {}}
        if state.regulatory_state is not None:
            state.regulatory_state, feedback_params = step_regulatory_modulator(
                state.regulatory_state,
                adjusted_eval,
                DEFAULT_REGULATORY_CONFIG,
                dt,
            )

        # Merge urgency feedback
        for section in ("neurotransmitters", "receptors"):
            for key, params in urgency_feedback.get(section, {}).items():
                if key not in feedback_params[section]:
                    feedback_params[section][key] = {}
                for k, v in params.items():
                    if k.endswith("_multiplier"):
                        feedback_params[section][key][k] = (
                            feedback_params[section][key].get(k, 1.0) * v
                        )
                    else:
                        feedback_params[section][key][k] = (
                            feedback_params[section][key].get(k, 0.0) + v
                        )

        # Oscillation envelope
        oscillation_update = None
        if osc_state is not None and state.regulatory_state is not None:
            oscillation_update = compute_oscillation_envelope(
                state.regulatory_state,
                osc_state,
                DEFAULT_ENVELOPE_RULES,
            )

        # Stochastic burst deltas
        burst_deltas = compute_stochastic_burst_deltas(
            adjusted_eval,
            state.prev_evaluation_vector,
            dt,
            config=DEFAULT_REACTIVITY_CONFIG,
            rng=None,
        )
        for nt_name, delta in urgency_burst_deltas.items():
            burst_deltas[nt_name] = burst_deltas.get(nt_name, 0.0) + delta

        # 4R reactive signals
        reactive_signals: Dict[str, Any] = {}
        if reactive_profile:
            reactive_signals = emotion_profile_to_signals(reactive_profile)

        # Merge all modulation signals
        from zados.neurochem.extractors.reactivity_matrix import burst_deltas_to_modulation_signals
        modulation_signals = burst_deltas_to_modulation_signals(burst_deltas)
        for nt_name, signals in reactive_signals.items():
            if nt_name not in modulation_signals:
                modulation_signals[nt_name] = {}
            for signal_key, value in signals.items():
                if signal_key in modulation_signals[nt_name]:
                    modulation_signals[nt_name][signal_key] += value
                else:
                    modulation_signals[nt_name][signal_key] = value

        state.prev_evaluation_vector = adjusted_eval

        from zados.neurochem.extractors.extractor_orchestrator import ExtractorResult
        ext_result = ExtractorResult(
            evaluation_vector=adjusted_eval,
            modulation_signals=modulation_signals,
            feedback_params=feedback_params,
            oscillation_update=oscillation_update,
            emotion_saturations=emotion_saturations,
            dominant_emotion=dominant_emotion,
            burst_deltas=burst_deltas,
            urgency_risk=urgency_risk,
        )
        return state, ext_result

    except Exception:
        log.exception("Extractor sub-component run failed.")
        return extractor_state, None


def _build_intent_eval_vector(perception: PerceptionSnapshot) -> Dict[str, float]:
    """Build evaluation vector from E23 intent category."""
    if perception is None or perception.intent_result is None:
        return dict(_DEFAULT_EVAL_AXIS)

    # Try to get intent category from IntentionMapResult
    intent_result = perception.intent_result
    category = None

    # IntentionMapResult may expose category via .dominant_intent or .intent_category
    if hasattr(intent_result, "intent_category"):
        cat = intent_result.intent_category
        category = cat.value.lower() if hasattr(cat, "value") else str(cat).lower()
    elif hasattr(intent_result, "dominant_intent"):
        category = str(intent_result.dominant_intent).lower()

    if category in _INTENT_TO_EVAL_AXIS:
        return dict(_INTENT_TO_EVAL_AXIS[category])

    # Fall back to intent_vector if available
    if perception.intent_vector:
        # Map highest-confidence intent to eval axis
        top_intent = max(perception.intent_vector, key=perception.intent_vector.get)
        for key in _INTENT_TO_EVAL_AXIS:
            if key in top_intent.lower():
                return dict(_INTENT_TO_EVAL_AXIS[key])

    return dict(_DEFAULT_EVAL_AXIS)


def _profile_from_perception(perception: PerceptionSnapshot) -> str:
    """Derive reward_profile_name from E23 intent category."""
    if perception is None or perception.intent_result is None:
        return "regular_input"

    intent_result = perception.intent_result
    category = None

    if hasattr(intent_result, "intent_category"):
        cat = intent_result.intent_category
        category = cat.value.lower() if hasattr(cat, "value") else str(cat).lower()
    elif hasattr(intent_result, "dominant_intent"):
        category = str(intent_result.dominant_intent).lower()

    return _INTENT_CATEGORY_TO_PROFILE.get(category or "", "regular_input")


# ------------------------------------------------------------------
# STMM population
# ------------------------------------------------------------------

def _populate_stmm(
    stmm: Any,
    neurochem_engine: Any,
    result: NTModulationResult,
    osc_state: Any,
) -> None:
    """Write NT snapshot, oscillations, metrics, and mode to STMM."""
    try:
        nt_snapshot: Dict[str, float] = {}
        for name in neurochem_engine.registry.neurotransmitter_names():
            nt_state = neurochem_engine.registry.get_neurotransmitter(name)
            nt_snapshot[name.lower()] = nt_state.C
        result.nt_snapshot = nt_snapshot
        stmm.cephalic_liquid_logger.nt_concentrations = nt_snapshot

        osc_dict = _osc_to_dict(osc_state, neurochem_engine)
        result.osc_snapshot = osc_dict
        stmm.cephalic_liquid_logger.oscillatory_bands = osc_dict
        stmm.cephalic_liquid_logger.neurosymbolic_metrics = result.metrics_dict
        stmm.cortical_reflection.active_mode = result.mode_token
    except Exception:
        log.exception("STMM population in Phase 2 failed.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize_signal_dict(signals: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize NT signal keys to uppercase (engine.step expects uppercase)."""
    normalized: Dict[str, Any] = {}
    for nt_key, sub_signals in signals.items():
        upper_key = normalize_nt_key(nt_key, target="upper")
        if not isinstance(sub_signals, dict):
            normalized[upper_key] = sub_signals
            continue
        if upper_key in normalized:
            for k, v in sub_signals.items():
                normalized[upper_key][k] = normalized[upper_key].get(k, 0.0) + v
        else:
            normalized[upper_key] = dict(sub_signals)
    return normalized


def _osc_to_dict(osc_state: Any, neurochem_engine: Any) -> Dict[str, float]:
    """Extract oscillation amplitudes as a flat dict."""
    _BANDS = ("delta", "theta", "alpha", "beta", "gamma", "sigma")
    if osc_state is not None:
        d = {band: getattr(osc_state, band, 0.0) for band in _BANDS}
        for coupling in ("theta_gamma_coupling", "alpha_beta_coupling", "delta_sigma_coupling"):
            if hasattr(osc_state, coupling):
                try:
                    d[coupling.replace("_coupling", "")] = getattr(osc_state, coupling)()
                except Exception:
                    pass
        return d
    try:
        osc = neurochem_engine.registry.get_oscillations()
        d = {band: getattr(osc, band, 0.0) for band in _BANDS}
        for coupling in ("theta_gamma_coupling", "alpha_beta_coupling", "delta_sigma_coupling"):
            if hasattr(osc, coupling):
                try:
                    d[coupling.replace("_coupling", "")] = getattr(osc, coupling)()
                except Exception:
                    pass
        return d
    except Exception:
        return {band: 0.0 for band in _BANDS}
