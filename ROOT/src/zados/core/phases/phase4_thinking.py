"""
ZA-DOS Core Pipeline — Phase 4: Thinking Blocks / VT (spec Part VII, LLM Pass 1).

Generates the Verbalized Thinking (VT) monologue via ``call_llama_with_retry``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from zados.LLM_interpretation.constants import (
    FALLBACK_VT,
    URG_HIGH,
    URG_SKIP_VT,
    VT_OUTPUT_MAX,
    VT_TEMPERATURE,
)
from zados.LLM_interpretation.ollama import LLMCallError, call_llama_with_retry
from zados.LLM_interpretation.prompt_builder import VTPromptBuilder
from zados.core.types import PipelineState, ThinkingResult

log = logging.getLogger(__name__)


def run_thinking_pass(
    state: PipelineState,
    stmm: Any,
    input_bundle_dict: Dict[str, Any],
) -> ThinkingResult:
    """Generate the VT monologue (LLM pass 1).

    Parameters
    ----------
    state : PipelineState
        Accumulated pipeline state.
    stmm : STMMStore
        Populated STMM (read by VTPromptBuilder).
    input_bundle_dict : dict
        Bundle context dict for VTPromptBuilder.

    Returns
    -------
    ThinkingResult
    """
    # Urgency gate: skip VT if urgency >= URG_SKIP_VT
    prior_urgency = input_bundle_dict.get("prior_urgency_risk", 0.0)
    if prior_urgency >= URG_SKIP_VT:
        stmm.brain_process_tracker.mark_stage("vt_skipped_urgency", True)
        return ThinkingResult(
            thinking_trace="",
            skipped=True,
            skip_reason=f"urgency={prior_urgency:.2f} >= {URG_SKIP_VT}",
        )

    # Budget reduction at elevated urgency
    vt_budget = VT_OUTPUT_MAX
    if prior_urgency >= URG_HIGH:
        vt_budget = int(VT_OUTPUT_MAX * 0.70)

    # Build VT prompt
    builder = VTPromptBuilder()
    vt_prompt = builder.build(stmm, input_bundle=input_bundle_dict)
    vt_messages: List[Dict[str, str]] = [{"role": "user", "content": vt_prompt}]

    # Call LLM
    try:
        vt_result = call_llama_with_retry(
            vt_messages,
            max_tokens=vt_budget,
            temperature=VT_TEMPERATURE,
        )
        vt_text = vt_result.get("content", "")
    except LLMCallError:
        log.warning("VT LLM call failed; using fallback.")
        vt_text = FALLBACK_VT

    # Write to STMM
    stmm.cortical_reflection.verbal_reflection = vt_text
    stmm.cortical_reflection.verbal_emotion_labels = _top_emotions(stmm)
    stmm.brain_process_tracker.mark_stage("vt_complete", True)

    return ThinkingResult(thinking_trace=vt_text)


def _top_emotions(stmm: Any) -> List[str]:
    """Top-5 emotions from system_emotion_state where value > 0.15."""
    state = stmm.emotion_detection.system_emotion_state
    if not state:
        return []
    filtered = [(k, v) for k, v in state.items() if v > 0.15]
    return [k for k, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:5]]
