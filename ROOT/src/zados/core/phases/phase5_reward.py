"""
ZA-DOS Core Pipeline — Phase 5: Reward Evaluation (spec Part VIII).

Two-pathway reward evaluation + NT signal application to NeurochemEngine.
This is THE critical wiring step — the pipeline applies Phase 5 NT signals
(both tonic and phasic) to the NeurochemicalEngine, which the monolithic
LLMInterpretationLayer.run() does NOT do.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from zados.cognitive_engines.constants import normalize_nt_key
from zados.core.types import PipelineState, RewardEvaluationResult

log = logging.getLogger(__name__)


def run_reward_evaluation(
    state: PipelineState,
    stmm: Any,
    neurochem_engine: Any,
    phase5_evaluator: Any,
    input_bundle_dict: Dict[str, Any],
) -> RewardEvaluationResult:
    """Run Phase 5 two-pathway reward evaluation.

    1. Phase5Evaluator.evaluate() → Phase5Result
    2. Apply tonic NT signals to NeurochemEngine
    3. Apply phasic pathway (modulation_signals + feedback_params)

    Parameters
    ----------
    state : PipelineState
    stmm : STMMStore
    neurochem_engine : NeurochemicalEngine
    phase5_evaluator : Phase5Evaluator
    input_bundle_dict : dict
    """
    result = RewardEvaluationResult()

    vt_output = ""
    if state.thinking and not state.thinking.skipped:
        vt_output = state.thinking.thinking_trace

    # ------------------------------------------------------------------
    # 1. Run Phase5Evaluator (tonic + phasic pathways internally)
    # ------------------------------------------------------------------
    try:
        phase5_result = phase5_evaluator.evaluate(
            vt_output, stmm, input_bundle=input_bundle_dict,
        )
        result.phase5_result = phase5_result
    except Exception:
        log.exception("Phase5Evaluator.evaluate() failed.")
        # Return empty result — downstream uses defaults
        from zados.LLM_interpretation.phase5_evaluator import Phase5Result
        result.phase5_result = Phase5Result()
        stmm.brain_process_tracker.mark_stage("phase5_complete", True)
        return result

    # ------------------------------------------------------------------
    # 2. Apply TONIC NT signals to NeurochemEngine
    # ------------------------------------------------------------------
    nt_signals = getattr(phase5_result, "nt_signals", None)
    if nt_signals:
        try:
            normalized = _normalize_signals(nt_signals)
            neurochem_engine.step(modulation_signals=normalized)
            result.tonic_applied = True
            log.debug("Phase 5 tonic NT signals applied: %s", list(normalized.keys()))
        except Exception:
            log.exception("Failed to apply Phase 5 tonic NT signals.")

    # ------------------------------------------------------------------
    # 3. Apply PHASIC pathway (ExtractorOrchestrator output)
    # ------------------------------------------------------------------
    extractor_result = getattr(phase5_result, "extractor_result", None)
    if extractor_result is not None:
        # 3a. Modulation signals
        mod_signals = getattr(extractor_result, "modulation_signals", None)
        if mod_signals:
            try:
                normalized_phasic = _normalize_signals(mod_signals)
                neurochem_engine.step(modulation_signals=normalized_phasic)
                log.debug("Phase 5 phasic modulation signals applied.")
            except Exception:
                log.exception("Failed to apply phasic modulation signals.")

        # 3b. Feedback params (receptor/NT baseline adjustments)
        feedback_params = getattr(extractor_result, "feedback_params", None)
        if feedback_params:
            try:
                neurochem_engine.apply_feedback(feedback_params)
                result.phasic_applied = True
                log.debug("Phase 5 phasic feedback params applied.")
            except Exception:
                log.exception("Failed to apply phasic feedback params.")

    stmm.brain_process_tracker.mark_stage("phase5_complete", True)
    return result


def _normalize_signals(signals: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Normalize NT signal keys to uppercase for NeurochemEngine.step()."""
    normalized: Dict[str, Dict[str, float]] = {}
    for nt_key, sub in signals.items():
        upper_key = normalize_nt_key(nt_key, target="upper")
        if isinstance(sub, dict):
            if upper_key in normalized:
                for k, v in sub.items():
                    normalized[upper_key][k] = normalized[upper_key].get(k, 0.0) + float(v)
            else:
                normalized[upper_key] = {k: float(v) for k, v in sub.items()}
        elif isinstance(sub, (int, float)):
            # Scalar signal — wrap in emotion_drive
            normalized.setdefault(upper_key, {})["emotion_drive"] = float(sub)
    return normalized
