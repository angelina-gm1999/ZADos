"""
ZA-DOS Core Pipeline — Phase 7: Post-Processing & Memory Loop (spec Part X).

9-step sequence:
1. Build MemoryPacket from full state
2. MTMM write (RawInteractionLogger)
3. E29 → compression policy
4. E17 → reward-based learning
5. E22 → contextual learning
6. E25 → recursive meta-learning
7. compute_reward_feedback() → NeurochemEngine.apply_feedback()
8. MTMM compression + indexing
9. LTMM promotion check
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from zados.core.types import PipelineState, PostProcessResult

log = logging.getLogger(__name__)


def run_postprocessing(
    state: PipelineState,
    neurochem_engine: Any,
    memory: Any,
    engines: Dict[int, Any],
    phase5_result: Any,
    session: Any = None,
    journal_writer: Any = None,
) -> PostProcessResult:
    """Execute the 9-step post-processing sequence + optional journal write.

    Parameters
    ----------
    state : PipelineState
    neurochem_engine : NeurochemicalEngine
    memory : MemoryLayer
    engines : dict   (engine_number → instance)
    phase5_result : Phase5Result
    session : SessionState, optional
    journal_writer : JournalWriter, optional
        If provided, Step 10 writes an event-log journal entry to JournalStore.
        Trigger is PERIODIC (every 5 turns), INNOVATION_FLAG when E7/E14/E19
        engine results are non-empty, or LTMM_THRESHOLD when LTMM promotion
        occurred.  Pipeline parameters (intent, reward profile, dominant
        emotion) are attached as notes for full reproducibility.
    """
    result = PostProcessResult()
    stmm = state.stmm
    bundle = state.bundle

    # ------------------------------------------------------------------
    # Step 1: Build MemoryPacket
    # ------------------------------------------------------------------
    packet = _build_memory_packet(state, phase5_result)
    result.memory_packet = packet

    # ------------------------------------------------------------------
    # Step 2: MTMM write
    # ------------------------------------------------------------------
    try:
        memory.mtmm.logger.append(packet)
    except Exception:
        log.exception("MTMM logger.append() failed.")

    # ------------------------------------------------------------------
    # Step 3: E29 → compression policy
    # ------------------------------------------------------------------
    e29 = engines.get(29)
    if e29 is not None:
        try:
            from zados.cognitive_engines.py_engines.memory_compression_engine import PacketDescriptor

            pd = PacketDescriptor(
                packet_id=packet.packet_id,
                text_length=len(bundle.raw_text),
                unique_tokens=len(set(bundle.raw_text.split())),
                total_tokens=len(bundle.raw_text.split()),
                emotional_significance=getattr(packet, "emotional_significance", 0.0),
                reward_mean=_mean_reward(phase5_result),
                has_unresolved=bool(stmm.memory_contrast.unresolved_query_matches),
                flags=list(getattr(packet, "flags", [])),
                trust_weight=getattr(packet, "trust_weight", 1.0),
            )
            e29_result = e29.process({"packets": [pd], "transition_type": "stmm_to_mtmm"})
            decisions = e29_result.get("decisions", [])
            if decisions:
                result.compression_policy = getattr(decisions[0], "policy", str(decisions[0]))
        except Exception:
            log.exception("E29 (MemoryCompression) failed.")

    # ------------------------------------------------------------------
    # Read post-reward NT snapshot for learning engines (steps 4-6).
    # Phase 5 applied reward-driven NT modulation to NeurochemEngine;
    # we propagate that updated state to E17/E22/E25 before they process.
    # ------------------------------------------------------------------
    nt_snapshot = _read_nt_snapshot(neurochem_engine)
    mode_token = ""
    if state.modulation:
        mode_token = getattr(state.modulation, "mode_token", "") or ""

    # ------------------------------------------------------------------
    # Step 4: E17 → reward-based learning
    # ------------------------------------------------------------------
    e17 = engines.get(17)
    if e17 is not None:
        try:
            if nt_snapshot:
                e17.update_neurochem_state(nt_snapshot)
            e17_input = _build_e17_input(phase5_result, mode_token, state=state, session=session)
            e17_result = e17.process(e17_input)
            result.learning_updates["e17"] = _result_to_dict(e17_result)
            _apply_e17_adjustments(session, e17_result)
        except Exception:
            log.exception("E17 (RewardBasedLearning) failed.")

    # ------------------------------------------------------------------
    # Step 5: E22 → contextual learning
    # ------------------------------------------------------------------
    e22 = engines.get(22)
    if e22 is not None:
        try:
            if nt_snapshot:
                e22.update_neurochem_state(nt_snapshot)
            e22_input = _build_e22_input(state, stmm)
            e22_result = e22.process(e22_input)
            result.learning_updates["e22"] = _result_to_dict(e22_result)
        except Exception:
            log.exception("E22 (ContextualLearning) failed.")

    # ------------------------------------------------------------------
    # Step 6: E25 → recursive meta-learning
    # ------------------------------------------------------------------
    e25 = engines.get(25)
    if e25 is not None:
        try:
            if nt_snapshot:
                e25.update_neurochem_state(nt_snapshot)
            e25_input = _build_e25_input(result)
            e25_result = e25.process(e25_input)
            result.learning_updates["e25"] = _result_to_dict(e25_result)
        except Exception:
            log.exception("E25 (RecursiveLearning) failed.")

    # ------------------------------------------------------------------
    # Step 7: Reward feedback → NeurochemEngine
    # ------------------------------------------------------------------
    try:
        from zados.reward.feedback.modulator import compute_reward_feedback

        meta_directive = stmm.reward_evaluation.meta_directive or {}
        domain_results = getattr(phase5_result, "domain_results", {})
        feedback = compute_reward_feedback(meta_directive, domain_results)
        if feedback:
            neurochem_engine.apply_feedback(feedback)
    except Exception:
        log.exception("Reward feedback application failed.")

    # ------------------------------------------------------------------
    # Step 8: MTMM compression + indexing
    # ------------------------------------------------------------------
    try:
        cp = memory.mtmm.context_processor
        cp.compress_old_entries(
            current_turn=state.turn_index,
            window=10,
        )
        importance = _compute_importance(phase5_result, stmm)
        cp.index_packet(packet, importance)
    except Exception:
        log.exception("MTMM compression/indexing failed.")

    # ------------------------------------------------------------------
    # Step 9: LTMM promotion check
    # ------------------------------------------------------------------
    ltmm_promoted = False
    try:
        from zados.memory.long_term.consolidation import MemoryConsolidationEngine

        consolidator = MemoryConsolidationEngine(memory.ltmm)
        promoted = consolidator.consolidate([packet])
        if promoted:
            ltmm_promoted = True
            log.info("LTMM promotion: %s", promoted)
    except Exception:
        log.exception("LTMM consolidation failed.")

    # ------------------------------------------------------------------
    # Step 10: Journal write (event-log for regular input turns)
    # ------------------------------------------------------------------
    if journal_writer is not None:
        try:
            _maybe_write_journal(state, stmm, journal_writer, ltmm_promoted)
        except Exception:
            log.exception("Journal write failed in Phase 7.")

    stmm.brain_process_tracker.mark_stage("postprocess_complete", True)
    return result


# ------------------------------------------------------------------
# MemoryPacket builder
# ------------------------------------------------------------------

def _build_memory_packet(state: PipelineState, phase5_result: Any) -> Any:
    """Construct a MemoryPacket from full pipeline state."""
    from zados.memory.types import MemoryPacket

    stmm = state.stmm
    bundle = state.bundle

    # Neurochemical snapshot
    nc_snapshot = dict(stmm.cephalic_liquid_logger.nt_concentrations)

    # Reward scores
    reward_scores = {}
    if phase5_result is not None:
        domain_results = getattr(phase5_result, "domain_results", {})
        if isinstance(domain_results, dict):
            for k, v in domain_results.items():
                if hasattr(v, "general_score"):
                    reward_scores[k] = v.general_score
                elif isinstance(v, (int, float)):
                    reward_scores[k] = float(v)

    # Emotion vector
    emotion_vector = dict(stmm.emotion_detection.user_emotion_signals)

    # Verbal summary
    verbal_summary = ""
    vt = stmm.cortical_reflection.verbal_reflection
    if vt:
        sentences = [s.strip() for s in vt.split(".") if s.strip()]
        if len(sentences) <= 3:
            verbal_summary = vt.strip()
        else:
            verbal_summary = ". ".join(sentences[:2] + [sentences[-1]]) + "."

    answer = ""
    if state.answer:
        answer = state.answer.final_answer

    return MemoryPacket(
        timestamp=state.timestamp,
        turn_index=state.turn_index,
        user_message=bundle.raw_text,
        system_response=answer,
        intention=stmm.intention_analysis.primary_intention,
        emotion_vector=emotion_vector,
        neurochemical_snapshot=nc_snapshot,
        reward_scores=reward_scores,
        flags=list(stmm.reward_evaluation.flags),
        trust_weight=1.0,
        emotional_significance=_emotional_significance(stmm),
        verbal_summary=verbal_summary,
        verbal_emotion_labels=list(stmm.cortical_reflection.verbal_emotion_labels),
        time_context=dict(getattr(bundle, "time_context", {})),
    )


# ------------------------------------------------------------------
# Learning engine input builders
# ------------------------------------------------------------------

def _build_e17_input(
    phase5_result: Any,
    mode_token: str = "",
    state: Any = None,
    session: Any = None,
) -> Any:
    """Build RewardLearningInput from Phase5Result.

    Normalises domain keys ("human_attunement" → "attunement") and maps
    mode_token → OperationalMode so E17's learning-rate multipliers activate.

    Exposes the active reward profile's domain weights as learnable parameters
    so E17 can compute prediction errors and produce adjustments.  Any weights
    already in session.learned_domain_weights override the static baseline.
    """
    from zados.cognitive_engines.py_engines.reward_based_learning_engine import RewardLearningInput
    from zados.cognitive_engines.py_engines.contradiction_detection_engine import OperationalMode

    # Profile domain key → E17 canonical domain name
    _DOMAIN_ALIASES: Dict[str, str] = {"human_attunement": "attunement"}

    # --- Reward signals (one float per domain) ---
    reward_signals: Dict[str, float] = {}
    domain_results = getattr(phase5_result, "domain_results", {})
    if isinstance(domain_results, dict):
        for k, v in domain_results.items():
            canonical = _DOMAIN_ALIASES.get(k, k)
            if hasattr(v, "general_score"):
                reward_signals[canonical] = v.general_score
            elif isinstance(v, (int, float)):
                reward_signals[canonical] = float(v)

    # --- Mode mapping ---
    _MODE_MAP: Dict[str, OperationalMode] = {
        "LearningMode_M1":         OperationalMode.LEARNING,
        "LearningMode_M2":         OperationalMode.LEARNING,
        "LearningMode_M3":         OperationalMode.LEARNING,
        "LearningMode_M4":         OperationalMode.LEARNING,
        "LearningMode_M5":         OperationalMode.LEARNING,
        "SleepMode_Dream":         OperationalMode.REM_DREAM,
        "SleepMode_REM":           OperationalMode.REM_NORMAL,
        "SleepMode_Triage":        OperationalMode.REFLECTIVE,
        "MetaLearning_Homework":   OperationalMode.LEARNING,
        "MetaLearning_Reflective": OperationalMode.REFLECTIVE,
    }
    active_mode = _MODE_MAP.get(mode_token, OperationalMode.NORMAL)

    # --- Learnable parameters: reward profile domain weights ---
    # Profile key "human_attunement" normalised to param_id "attunement_weight"
    # so the parameter_id space stays consistent with E17's DomainType enum.
    _PROFILE_KEY_TO_PARAM: Dict[str, str] = {
        "logic":            "logic_weight",
        "ethics":           "ethics_weight",
        "innovation":       "innovation_weight",
        "human_attunement": "attunement_weight",
    }

    parameter_values:  Dict[str, float] = {}
    parameter_domains: Dict[str, str]   = {}

    # Step 1: seed from active static reward profile
    profile_name = ""
    if state is not None and getattr(state, "modulation", None) is not None:
        profile_name = getattr(state.modulation, "reward_profile_name", "") or ""
    try:
        from zados.reward.profile.static_profiles import PROFILE_REGISTRY
        profile = PROFILE_REGISTRY.get(profile_name)
        if profile is None and profile_name:
            log.debug("Profile '%s' not found in PROFILE_REGISTRY; using defaults.", profile_name)
        if profile is not None:
            for domain_key, param_id in _PROFILE_KEY_TO_PARAM.items():
                val = profile.domain_weights.get(domain_key)
                if val is not None:
                    e17_domain = _DOMAIN_ALIASES.get(domain_key, domain_key)
                    parameter_values[param_id]  = float(val)
                    parameter_domains[param_id] = e17_domain
    except Exception:
        pass

    # Step 2: apply session-accumulated learned overrides on top of static baseline
    if session is not None:
        for pid, val in getattr(session, "learned_domain_weights", {}).items():
            if pid in parameter_values:
                parameter_values[pid] = float(val)

    return RewardLearningInput(
        reward_signals=reward_signals,
        active_mode=active_mode,
        parameter_values=parameter_values,
        parameter_domains=parameter_domains,
    )


def _build_e22_input(state: PipelineState, stmm: Any) -> Any:
    """Build ContextInput from pipeline state."""
    from zados.cognitive_engines.py_engines.contextual_learning_engine import ContextInput

    return ContextInput(
        topic=stmm.intention_analysis.primary_intention,
        emotion_state=dict(stmm.emotion_detection.user_emotion_signals),
        intent=stmm.intention_analysis.primary_archetype,
        raw_text=state.bundle.raw_text,
    )


def _build_e25_input(result: PostProcessResult) -> Any:
    """Build RecursiveLearningInput from E17 results."""
    from zados.cognitive_engines.py_engines.recursive_learning_engine import (
        MetaMetrics,
        RecursiveLearningInput,
    )

    e17_data = result.learning_updates.get("e17", {})
    metrics = MetaMetrics()

    if isinstance(e17_data, dict):
        pe_list = e17_data.get("prediction_errors", [])
        if pe_list and isinstance(pe_list, list):
            deltas = []
            for pe in pe_list:
                if hasattr(pe, "magnitude"):
                    deltas.append(pe.magnitude)
                elif isinstance(pe, dict):
                    deltas.append(abs(pe.get("delta", 0.0)))
            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                conv_ratio = float(e17_data.get("convergence_ratio", 0.0))
                metrics = MetaMetrics(
                    mean_abs_delta=mean_delta,
                    convergence_ratio=conv_ratio,
                )

    return RecursiveLearningInput(e17_metrics=metrics)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mean_reward(phase5_result: Any) -> float:
    """Mean of domain general scores."""
    if phase5_result is None:
        return 0.0
    dr = getattr(phase5_result, "domain_results", {})
    if not dr:
        return 0.0
    scores = []
    for v in dr.values():
        if hasattr(v, "general_score"):
            scores.append(v.general_score)
        elif isinstance(v, (int, float)):
            scores.append(float(v))
    return sum(scores) / len(scores) if scores else 0.0


def _emotional_significance(stmm: Any) -> float:
    """Compute emotional significance from saturation levels."""
    sat = stmm.emotion_detection.saturation_levels
    if not sat:
        ue = stmm.emotion_detection.user_emotion_signals
        return max(ue.values(), default=0.0) if ue else 0.0
    return max(sat.values(), default=0.0)


def _compute_importance(phase5_result: Any, stmm: Any) -> float:
    """Simple importance score for indexing."""
    reward_score = 0.0
    if phase5_result:
        reward_score = getattr(phase5_result, "composite_score",
                               stmm.reward_evaluation.composite_score)
    emo_sig = _emotional_significance(stmm)
    return min((reward_score + emo_sig) / 2.0, 1.0)


def _result_to_dict(result: Any) -> Any:
    """Convert engine result to dict-safe form."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    return result


def _read_nt_snapshot(neurochem_engine: Any) -> Dict[str, float]:
    """Read current NT concentrations as lowercase-keyed dict from the engine.

    Called at the start of the Phase 7 learning block so that E17/E22/E25
    receive the NT state *after* Phase 5 reward modulation has been applied.
    """
    snapshot: Dict[str, float] = {}
    try:
        for name in neurochem_engine.registry.neurotransmitter_names():
            nt = neurochem_engine.registry.get_neurotransmitter(name)
            snapshot[name.lower()] = nt.C
    except Exception:
        pass
    return snapshot


def _maybe_write_journal(
    state: PipelineState,
    stmm: Any,
    writer: Any,
    ltmm_promoted: bool,
) -> None:
    """Determine journal trigger and write an event-log entry.

    Trigger priority (first match wins):
      1. LTMM_THRESHOLD  — LTMM promotion occurred this turn
      2. INNOVATION_FLAG — E7 (simulated opposition), E14 (socratic), or
                           E19 (pattern) engine returned non-empty results
      3. PERIODIC        — every 5 turns (turn_index % 5 == 0)
      (no match)         — skip journal write this turn

    Pipeline parameters that shaped the answer (intent_category,
    reward_profile_name, dominant_emotion) are attached as notes so
    the journal entry is fully traceable to the processing context.
    """
    from zados.memory.long_term.journal.entry import JournalContext, JournalTrigger

    turn_index = getattr(state, "turn_index", 0)

    # Determine trigger
    if ltmm_promoted:
        trigger = JournalTrigger.LTMM_THRESHOLD
        trigger_source = "ltmm_consolidation"
    else:
        # Check for innovation/exploration engine activity (E7, E14, E19)
        engine_results: Dict[int, Any] = {}
        if state.dispatch:
            engine_results = state.dispatch.engine_results
        innovation_active = any(
            bool(engine_results.get(eid))
            for eid in (7, 14, 19)
        )
        if innovation_active:
            trigger = JournalTrigger.INNOVATION_FLAG
            trigger_source = "engine_dispatch"
        elif turn_index % 5 == 0:
            trigger = JournalTrigger.PERIODIC
            trigger_source = "phase7_postprocess"
        else:
            return  # No trigger — skip this turn

    # Build pipeline-parameter notes (the parameters that defined the answer)
    notes: List[str] = []
    if state.modulation:
        intent = getattr(state.modulation, "reward_profile_name", "")
        if intent:
            notes.append(f"reward_profile:{intent}")
        ext = getattr(state.modulation, "extractor_result", None)
        if ext:
            dom_emo = getattr(ext, "dominant_emotion", None)
            if dom_emo and isinstance(dom_emo, tuple) and len(dom_emo) >= 2:
                notes.append(f"dominant_emotion:{dom_emo[0]}:{dom_emo[1]:.2f}")
    if state.perception and state.perception.intent_result:
        ir = state.perception.intent_result
        cat = getattr(ir, "intent_category", None) or getattr(ir, "dominant_intent", None)
        if cat:
            cat_str = cat.value.lower() if hasattr(cat, "value") else str(cat).lower()
            notes.append(f"intent:{cat_str}")

    # Append temporal context flags so journal entries carry time metadata
    if state.bundle:
        tc_flags = getattr(state.bundle, "time_context", {}).get("flags", [])
        notes.extend(tc_flags)

    session_id = ""
    if state.bundle:
        session_id = getattr(state.bundle, "session_id", "")

    ctx = JournalContext(
        trigger=trigger,
        trigger_source=trigger_source,
        stmm=stmm,
        notes=notes,
        turn_range=(turn_index, turn_index),
        session_id=session_id,
    )
    writer.write(ctx)


def _apply_e17_adjustments(session: Any, e17_result: Any) -> None:
    """Write E17 parameter adjustments back to session.learned_domain_weights.

    Domain weights are clamped to [0.0, 1.0].  Parameters whose
    convergence_status is CONSOLIDATED are skipped — E17 has frozen them.

    Parameters
    ----------
    session : SessionState (or None)
    e17_result : RewardLearningResult
        Raw result from E17.process() (not the dict-converted version).
    """
    if session is None:
        return
    ldw = getattr(session, "learned_domain_weights", None)
    if ldw is None:
        return

    adjustments = getattr(e17_result, "adjustments", [])
    for adj in adjustments:
        pid   = getattr(adj, "parameter_id", None)
        new_v = getattr(adj, "new_value", None)
        if pid is None or new_v is None:
            continue
        # Skip consolidated parameters (E17 has permanently frozen them)
        cs = getattr(adj, "convergence_status", "")
        if getattr(cs, "value", cs) == "consolidated":
            continue
        ldw[pid] = max(0.0, min(1.0, float(new_v)))
