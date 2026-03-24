"""
ZA-DOS Core Pipeline — Phase 6: Final Answer / RG (spec Part IX, LLM Pass 2).

Generates the user-facing response via ``call_llama_with_retry``.
Handles meta_directive gates (suppress/abstain) and tool calls (web_search).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from zados.LLM_interpretation.constants import (
    CSS_SEVERE,
    FALLBACK_RESPONSE,
    RG_OUTPUT_MAX,
    RG_OUTPUT_SEV,
    RG_OUTPUT_URG,
    RG_TEMPERATURE,
    URG_HIGH,
)
from zados.LLM_interpretation.ollama import LLMCallError, call_llama_with_retry
from zados.LLM_interpretation.prompt_builder import RGPromptBuilder
from zados.LLM_interpretation.tools import SEARCH_TOOLS, _execute_search
from zados.core.types import AnswerResult, PipelineState

log = logging.getLogger(__name__)


def run_answer_pass(
    state: PipelineState,
    stmm: Any,
    phase5_result: Any,
    input_bundle_dict: Dict[str, Any],
) -> AnswerResult:
    """Generate the user-facing response (LLM pass 2).

    Parameters
    ----------
    state : PipelineState
    stmm : STMMStore
    phase5_result : Phase5Result
    input_bundle_dict : dict
    """
    meta = stmm.reward_evaluation.meta_directive or {}

    # ------------------------------------------------------------------
    # Gate 1: suppress → empty response, no LLM call
    # ------------------------------------------------------------------
    if meta.get("suppress", False):
        stmm.brain_process_tracker.mark_stage("rg_suppressed", True)
        return AnswerResult(final_answer="", directive_applied="suppress")

    # ------------------------------------------------------------------
    # Gate 2: abstain → short ack, no LLM call
    # ------------------------------------------------------------------
    if meta.get("abstain", False):
        response = _abstain_response(stmm)
        stmm.add_system_response(response)
        stmm.brain_process_tracker.mark_stage("rg_complete", True)
        return AnswerResult(final_answer=response, directive_applied="abstain")

    # ------------------------------------------------------------------
    # Urgency-only response (VT was skipped)
    # ------------------------------------------------------------------
    if state.thinking and state.thinking.skipped:
        return _urgency_response(stmm, phase5_result, input_bundle_dict)

    # ------------------------------------------------------------------
    # Normal RG path
    # ------------------------------------------------------------------
    vt_output = state.thinking.thinking_trace if state.thinking else ""
    extractor_result = getattr(phase5_result, "extractor_result", None)
    selected_mode = getattr(phase5_result, "selected_mode", None)

    # Token budget
    rg_tokens = _compute_rg_budget(stmm, phase5_result)

    # Build RG messages
    builder = RGPromptBuilder()
    rg_messages = builder.build(
        stmm,
        vt_output,
        extractor_result=extractor_result,
        input_bundle=input_bundle_dict,
        selected_mode=selected_mode,
    )

    # Tool eligibility
    tools_arg = SEARCH_TOOLS if _search_eligible(stmm) else None

    # Call LLM
    try:
        rg_result = call_llama_with_retry(
            rg_messages,
            max_tokens=rg_tokens,
            temperature=RG_TEMPERATURE,
            tools=tools_arg,
        )
        response = _handle_tool_calls(rg_result, rg_messages)
    except LLMCallError:
        log.warning("RG LLM call failed; using fallback.")
        response = FALLBACK_RESPONSE

    stmm.add_system_response(response)
    stmm.brain_process_tracker.mark_stage("rg_complete", True)

    return AnswerResult(final_answer=response, directive_applied="allow")


# ------------------------------------------------------------------
# Budget computation
# ------------------------------------------------------------------

def _compute_rg_budget(stmm: Any, phase5_result: Any) -> int:
    """Determine RG token budget based on urgency and CSS."""
    urgency = getattr(phase5_result, "urgency_risk", 0.0)
    if urgency >= URG_HIGH:
        return RG_OUTPUT_URG

    sat = stmm.emotion_detection.saturation_levels
    css = max(sat.values(), default=0.0) if sat else 0.0
    if css >= CSS_SEVERE:
        return RG_OUTPUT_SEV

    return RG_OUTPUT_MAX


# ------------------------------------------------------------------
# Urgency response (VT skipped)
# ------------------------------------------------------------------

def _urgency_response(
    stmm: Any,
    phase5_result: Any,
    input_bundle_dict: Dict[str, Any],
) -> AnswerResult:
    """Brief RG when urgency > URG_SKIP_VT and VT was skipped."""
    builder = RGPromptBuilder()
    rg_messages = builder.build(
        stmm,
        vt_output="",
        extractor_result=None,
        input_bundle=input_bundle_dict,
        selected_mode="Containment",
    )

    try:
        rg_result = call_llama_with_retry(
            rg_messages,
            max_tokens=RG_OUTPUT_URG,
            temperature=RG_TEMPERATURE,
        )
        response = rg_result.get("content", FALLBACK_RESPONSE)
    except LLMCallError:
        response = FALLBACK_RESPONSE

    stmm.add_system_response(response)
    stmm.brain_process_tracker.mark_stage("rg_complete", True)
    return AnswerResult(final_answer=response, directive_applied="allow")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _abstain_response(stmm: Any) -> str:
    """Short programmatic ack when abstain=True."""
    user_msg = stmm.active_message_buffer.latest_user()
    if user_msg:
        return "I've noted your message. I'm not in a position to respond to that fully right now."
    return "Acknowledged."


def _search_eligible(stmm: Any) -> bool:
    """True if information_seeking intent and < 2 memory matches."""
    ia = stmm.intention_analysis
    mc = stmm.memory_contrast
    return (
        ia.primary_intention == "information_seeking"
        and len(mc.matched_entries) < 2
    )


def _handle_tool_calls(
    response: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> str:
    """Execute first web_search tool call if present."""
    tool_calls = response.get("tool_calls")
    if not tool_calls:
        return response.get("content", "")

    tool_call = tool_calls[0]
    fn = tool_call.get("function", {})
    fn_name = fn.get("name", "")
    fn_args = fn.get("arguments", {})

    if fn_name == "web_search":
        query = fn_args.get("query", "") if isinstance(fn_args, dict) else ""
        search_result = _execute_search(query)

        updated_messages = list(messages) + [
            {"role": "tool", "content": search_result, "name": "web_search"},
        ]
        try:
            final = call_llama_with_retry(
                updated_messages,
                max_tokens=RG_OUTPUT_MAX,
                temperature=RG_TEMPERATURE,
                tools=None,
            )
            return final.get("content", "")
        except LLMCallError:
            return response.get("content", "")

    return response.get("content", "")
