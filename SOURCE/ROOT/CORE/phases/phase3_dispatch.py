"""
ZA-DOS Core Pipeline — Phase 3: Engine Dispatch (spec Part VI).

Dispatches cognitive engines based on the intent archetype and engine
priority weights from Phase 2.  Populates STMM with execution records,
emotion detection, memory contrast, and intention analysis.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from zados.cognitive_engines.constants import ENGINE_IDS
from zados.core.dispatch_table import get_dispatch_list
from zados.core.types import EngineDispatchResult, PipelineState

log = logging.getLogger(__name__)


def run_engine_dispatch(
    state: PipelineState,
    engines: Dict[int, Any],
    nt_state_lowercase: Dict[str, float],
    memory_contrast: Any = None,
) -> EngineDispatchResult:
    """Dispatch cognitive engines for the current turn.

    Parameters
    ----------
    state : PipelineState
        Accumulated pipeline state (bundle, perception, modulation).
    engines : dict
        engine_number → engine instance.
    nt_state_lowercase : dict
        NT name (lowercase) → concentration.
    memory_contrast : MemoryContrast, optional
        If provided, runs memory contrast after engine dispatch.
    """
    result = EngineDispatchResult()
    bundle = state.bundle
    stmm = state.stmm

    # Determine which engines to dispatch
    archetype = ""
    if state.perception:
        archetype = state.perception.intent_archetype
    if not archetype:
        archetype = bundle.intent_archetype

    # Phase 2 now runs AFTER Phase 3; fall back to bundle weights if modulation
    # is not yet set (normal path for regular input).
    engine_weights = {}
    if state.modulation:
        engine_weights = state.modulation.engine_weights
    if not engine_weights:
        engine_weights = state.bundle.engine_weights

    dispatch_list = get_dispatch_list(archetype, engine_weights)

    # ------------------------------------------------------------------
    # Dispatch each engine
    # ------------------------------------------------------------------
    for eng_num in dispatch_list:
        engine = engines.get(eng_num)
        if engine is None:
            result.engines_skipped.append(eng_num)
            _record_execution(stmm, eng_num, 0.0, "engine not registered", skipped=True)
            continue

        t0 = time.perf_counter()
        try:
            engine.update_neurochem_state(nt_state_lowercase)
            eng_input = _build_engine_input(eng_num, state, engines, nt_state_lowercase)
            eng_result = engine.process(eng_input)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.engine_results[eng_num] = _result_to_dict(eng_result)
            result.engines_run.append(eng_num)
            _record_execution(stmm, eng_num, elapsed_ms, _summary(eng_result))

            # Capture E28 result specifically (emotion detection)
            if eng_num == 28:
                result.e28_result = eng_result
                _populate_emotion_detection(stmm, eng_result)

        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.exception("Engine %d (%s) failed.", eng_num, ENGINE_IDS.get(eng_num, "?"))
            result.engines_skipped.append(eng_num)
            _record_execution(stmm, eng_num, elapsed_ms, "exception", skipped=True)

    # ------------------------------------------------------------------
    # Run E28 separately if not in dispatch list but emotion profile exists
    # ------------------------------------------------------------------
    if 28 not in dispatch_list and 28 in engines and bundle.emotion_profile:
        _run_e28_standalone(engines[28], state, nt_state_lowercase, result, stmm)

    # ------------------------------------------------------------------
    # If no E28 ran, populate emotion detection from bundle
    # ------------------------------------------------------------------
    if result.e28_result is None and bundle.emotion_profile:
        _populate_emotion_from_bundle(stmm, bundle)

    # ------------------------------------------------------------------
    # Memory contrast
    # ------------------------------------------------------------------
    if memory_contrast is not None:
        _run_memory_contrast(stmm, memory_contrast, bundle)

    # ------------------------------------------------------------------
    # Populate STMM intention_analysis from perception
    # ------------------------------------------------------------------
    if state.perception:
        _populate_intention_analysis(stmm, state.perception, bundle)

    return result


# ------------------------------------------------------------------
# Engine input builders
# ------------------------------------------------------------------

def _build_engine_input(
    eng_num: int,
    state: PipelineState,
    engines: Dict[int, Any],
    nt_state: Dict[str, float],
) -> Any:
    """Build the appropriate input for each engine.

    For engines with complex dataclass inputs, we construct them lazily.
    For dict-input engines, we assemble the dict directly.
    """
    bundle = state.bundle
    perception = state.perception

    # ---- Detection cluster (E1, E2, E4, E5, E6) ----
    if eng_num == 1:
        return _build_e1_input(state)
    if eng_num == 2:
        return _build_e2_input(state, engines)
    if eng_num == 4:
        return _build_e4_input(state, engines)
    if eng_num == 5:
        return _build_e5_input(state)
    if eng_num == 6:
        return _build_e6_input(state, engines)

    # ---- Dialectic (E7, E14) ----
    if eng_num == 7:
        return _build_e7_input(state, engines)
    if eng_num == 14:
        return _build_e14_input(state, engines)

    # ---- Executive control (E3) ----
    if eng_num == 3:
        return _build_e3_input(state, nt_state)

    # ---- Knowledge substrate (E9, E10, E16) — dict-based ----
    if eng_num in (9, 10, 16):
        return {"nt_state": nt_state, "mode": state.modulation.mode_token if state.modulation else "NORMAL"}

    # ---- Evaluation (E12) ----
    if eng_num == 12:
        return _build_e12_input(state)

    # ---- Reasoning (E13, E15, E21) ----
    if eng_num == 13:
        return _build_e13_input(state)
    if eng_num == 15:
        return _build_e15_input(state)
    if eng_num == 21:
        return _build_e21_input(state)

    # ---- Metacognition (E24) ----
    if eng_num == 24:
        return _build_e24_input(state)

    # ---- Pattern analysis (E20) — dict-based ----
    if eng_num == 20:
        patterns = []
        if perception and perception.pattern_list:
            patterns = perception.pattern_list
        return {"nt_state": nt_state, "patterns": patterns}

    # ---- Emotion (E28) ----
    if eng_num == 28:
        return _build_e28_input(state)

    # Fallback: pass None (engine uses default)
    return None


# ------------------------------------------------------------------
# Individual engine input builders
# ------------------------------------------------------------------

def _build_e1_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.contradiction_detection_engine import ComparisonSet
    return ComparisonSet()


def _build_e2_input(state: PipelineState, engines: Dict[int, Any]) -> Any:
    from zados.cognitive_engines.py_engines.paradox_detection_engine import ParadoxInput
    # E2 needs E1's output (contradiction_result)
    e1_result = state.dispatch.engine_results.get(1) if state.dispatch else None
    return ParadoxInput(contradiction_result=e1_result) if e1_result else ParadoxInput()


def _build_e4_input(state: PipelineState, engines: Dict[int, Any]) -> Any:
    from zados.cognitive_engines.py_engines.fallacy_detection_engine import FallacyInput
    return FallacyInput()


def _build_e5_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.bias_detection_engine import BiasDetectionInput
    return BiasDetectionInput()


def _build_e6_input(state: PipelineState, engines: Dict[int, Any]) -> Any:
    from zados.cognitive_engines.py_engines.logic_trap_detection_engine import LogicTrapInput
    return LogicTrapInput()


def _build_e7_input(state: PipelineState, engines: Dict[int, Any]) -> Any:
    from zados.cognitive_engines.py_engines.simulated_opposition_engine import OppositionGateInput
    return OppositionGateInput(
        original_query=state.bundle.raw_text,
    )


def _build_e14_input(state: PipelineState, engines: Dict[int, Any]) -> Any:
    from zados.cognitive_engines.py_engines.socratic_reasoning_engine import SocraticInput
    return SocraticInput()


def _build_e3_input(state: PipelineState, nt_state: Dict[str, float]) -> Any:
    from zados.cognitive_engines.py_engines.soar_production_engine import SOARInput
    engine_outputs = {}
    if state.dispatch:
        for eng_num, res in state.dispatch.engine_results.items():
            engine_outputs[str(eng_num)] = res if isinstance(res, dict) else {"result": res}
    return SOARInput(
        engine_outputs=engine_outputs,
        nt_state=nt_state,
        active_mode=state.modulation.mode_token if state.modulation else "NORMAL",
    )


def _build_e12_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.logical_brain_engine import LogicalBrainInput
    return LogicalBrainInput()


def _build_e13_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.simulation_brain_engine import SimulationBrainInput
    intent_desc = []
    intent_conf = []
    if state.perception and state.perception.intent_vector:
        for k, v in state.perception.intent_vector.items():
            intent_desc.append(k)
            intent_conf.append(v)
    return SimulationBrainInput(
        intent_descriptions=tuple(intent_desc),
        intent_confidences=tuple(intent_conf),
        active_mode=state.modulation.mode_token if state.modulation else "normal",
    )


def _build_e15_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.decision_making_engine import DecisionMakingInput
    return DecisionMakingInput()


def _build_e21_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.strategic_decision_engine import StrategicDecisionInput
    return StrategicDecisionInput(
        active_mode=state.modulation.mode_token if state.modulation else "normal",
    )


def _build_e24_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.heuristic_bias_engine import HeuristicBiasInput
    return HeuristicBiasInput()


def _build_e28_input(state: PipelineState) -> Any:
    from zados.cognitive_engines.py_engines.emotional_detection_engine import EmotionalDetectionInput
    tokens = []
    if state.perception and state.perception.engine_statuses:
        pass  # tokens already set by tokenizer in Phase 1
    tokens_str = state.bundle.raw_text.split()
    return EmotionalDetectionInput(
        tokens=tuple(tokens_str),
        raw_text=state.bundle.raw_text,
    )


# ------------------------------------------------------------------
# E28 standalone run
# ------------------------------------------------------------------

def _run_e28_standalone(
    e28: Any,
    state: PipelineState,
    nt_state: Dict[str, float],
    result: EngineDispatchResult,
    stmm: Any,
) -> None:
    """Run E28 outside of the normal dispatch path."""
    t0 = time.perf_counter()
    try:
        e28.update_neurochem_state(nt_state)
        inp = _build_e28_input(state)
        e28_result = e28.process(inp)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result.e28_result = e28_result
        result.engines_run.append(28)
        _record_execution(stmm, 28, elapsed_ms, _summary(e28_result))
        _populate_emotion_detection(stmm, e28_result)
    except Exception:
        log.exception("E28 standalone run failed.")


# ------------------------------------------------------------------
# STMM population helpers
# ------------------------------------------------------------------

def _record_execution(
    stmm: Any,
    eng_num: int,
    timing_ms: float,
    summary: str,
    skipped: bool = False,
) -> None:
    """Record an EngineExecution in the STMM brain_process_tracker."""
    if stmm is None:
        return
    try:
        from zados.memory.short_term.components import EngineExecution
        stmm.brain_process_tracker.record(EngineExecution(
            engine_id=ENGINE_IDS.get(eng_num, f"E{eng_num}"),
            timing_ms=timing_ms,
            output_summary=summary[:200],
            skipped=skipped,
            skip_reason="engine unavailable" if skipped else "",
        ))
    except Exception:
        pass


def _populate_emotion_detection(stmm: Any, e28_result: Any) -> None:
    """Write E28 output to stmm.emotion_detection."""
    if stmm is None or e28_result is None:
        return
    try:
        ed = stmm.emotion_detection
        # Neurochemical profile → system_emotion_state
        if hasattr(e28_result, "neurochemical_profile"):
            ed.system_emotion_state = dict(e28_result.neurochemical_profile)
        # Detected emotions → user_emotion_signals
        if hasattr(e28_result, "detected_emotions"):
            for de in e28_result.detected_emotions:
                name = getattr(de, "emotion_name", getattr(de, "name", str(de)))
                intensity = getattr(de, "intensity", 0.0)
                ed.user_emotion_signals[name] = intensity
        # Tone vector
        if hasattr(e28_result, "tone_vector"):
            tv = e28_result.tone_vector
            ed.tone_valence = tv.get("valence", 0.0)
            ed.tone_coherence = tv.get("coherence", 0.5)
            ed.tone_warmth = tv.get("warmth", 0.0)
            ed.tone_discord = tv.get("discord", 0.0)
        # Saturation from oscillatory/phasic signals
        if hasattr(e28_result, "phasic_4r"):
            for k, v in e28_result.phasic_4r.items():
                ed.saturation_levels[k] = abs(v)
    except Exception:
        log.exception("Failed to populate emotion_detection from E28.")


def _populate_emotion_from_bundle(stmm: Any, bundle: Any) -> None:
    """Fallback: populate emotion_detection from bundle.emotion_profile."""
    if stmm is None:
        return
    try:
        ed = stmm.emotion_detection
        for k, v in bundle.emotion_profile.items():
            ed.user_emotion_signals[k] = v
            ed.system_emotion_state[k] = v
    except Exception:
        pass


def _run_memory_contrast(stmm: Any, memory_contrast: Any, bundle: Any) -> None:
    """Run two-pass MemoryContrast and populate stmm.memory_contrast.

    Pass 1 — flat MTMM + LTMM (existing behaviour).
    Pass 2 — scoped REGULAR_SCOPE (thoughts/held_blocks, overview_logs,
              general_questions, knowledge/lessons, knowledge/library).
    Results are deduplicated by entry_id.
    """
    try:
        from zados.memory.short_term.components import MemoryMatch
        from zados.memory.managers.scope_filter import REGULAR_SCOPE

        mc = stmm.memory_contrast
        seen_ids: set = set()
        current_doc = {"text": bundle.raw_text, "output": "", "content": bundle.raw_text}

        def _add_refs(refs: list) -> None:
            for ref in refs:
                # contrast.py returns "packet_id" (MTMM/LTMM) or "entry_id" (scoped)
                eid = ref.get("packet_id", ref.get("entry_id", ref.get("id", "")))
                if eid in seen_ids:
                    continue
                seen_ids.add(eid)
                mc.matched_entries.append(MemoryMatch(
                    entry_id=eid,
                    source_tier=ref.get("source", ref.get("tier", "MTMM")),
                    similarity=ref.get("similarity", 0.0),
                    content_summary=ref.get("summary", ""),
                    metadata=ref,
                ))

        # Pass 1: flat contrast
        try:
            result1 = memory_contrast.contrast(current=current_doc, query_type="context")
            mc.delta_align = getattr(result1, "similarity", 0.0)
            _add_refs(getattr(result1, "references", []))
        except Exception:
            log.exception("Memory contrast pass 1 failed.")

        # Pass 2: scoped contrast (REGULAR_SCOPE)
        try:
            result2 = memory_contrast.contrast(
                current=current_doc,
                query_type="context",
                scope_filter=REGULAR_SCOPE,
            )
            _add_refs(getattr(result2, "references", []))
        except Exception:
            log.debug("Memory contrast pass 2 (scoped) failed — likely no scope_filter support yet.")

    except Exception:
        log.exception("Memory contrast setup failed.")


def _populate_intention_analysis(stmm: Any, perception: Any, bundle: Any) -> None:
    """Write perception results to stmm.intention_analysis."""
    try:
        ia = stmm.intention_analysis
        if perception.intent_result is not None:
            ir = perception.intent_result
            ia.primary_intention = getattr(ir, "dominant_intent", "")
            ia.confidence = getattr(ir, "intent_confidence", 0.0)
            ia.primary_archetype = getattr(ir, "primary_archetype", "")
            ia.sub_intentions = list(getattr(ir, "rising_intents", []))
        elif bundle.intent_archetype:
            ia.primary_archetype = bundle.intent_archetype
    except Exception:
        log.exception("Failed to populate intention_analysis.")


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert engine result to a dict if not already."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    return {"value": str(result)}


def _summary(result: Any) -> str:
    """Brief summary string for EngineExecution.output_summary."""
    if isinstance(result, dict):
        return str(list(result.keys()))[:200]
    if hasattr(result, "__class__"):
        return result.__class__.__name__
    return str(result)[:200]
