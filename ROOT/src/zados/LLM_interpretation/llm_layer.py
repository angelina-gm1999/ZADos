"""
ZA-DOS LLM Interpretation Layer — main entry point (v0.5).

LLMInterpretationLayer.run(stmm, input_bundle) is the single public interface.

v0.5 Pipeline:
    1.  Gate check           — suppress / abstain / allow  (from prev meta_directive)
    2.  Urgency gate         — if prior urgency > URG_SKIP_VT, skip VT entirely
    3.  Verbalized Thinking  — internal monologue generation (VT / Phase 4)
    4.  Phase 5 evaluation   — two-pathway reward eval of thinking trace
        4a. Tonic:  SynthesisEngine + NeurochemicalAdapter → sustained NT mod
        4b. Phasic: ExtractorOrchestrator.step() → stochastic burst deltas
        4c. Mode re-selection after NT update
    5.  Response Generation  — user-facing response (RG / Phase 6)
        System Components A/B/C + VT as assistant message + history

The layer is READ-ONLY with respect to all upstream STMM fields *except*:
    stmm.cortical_reflection.verbal_reflection
    stmm.cortical_reflection.verbal_emotion_labels
    stmm.cortical_reflection.active_mode        (Phase 5 mode re-selection)
    stmm.reward_evaluation.meta_directive        (Phase 5 update)
    stmm.reward_evaluation.composite_score       (Phase 5 update)
    stmm.cephalic_liquid_logger.extractor_state  (Phase 5 NT signals)
    stmm.active_message_buffer                   (via stmm.add_system_response)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from zados.LLM_interpretation.constants import (
    CSS_SEVERE,
    FALLBACK_RESPONSE,
    FALLBACK_VT,
    RG_OUTPUT_MAX,
    RG_OUTPUT_SEV,
    RG_OUTPUT_URG,
    RG_TEMPERATURE,
    URG_ELEVATED,
    URG_HIGH,
    URG_SKIP_VT,
    VT_OUTPUT_MAX,
    VT_TEMPERATURE,
)
from zados.LLM_interpretation.ollama import LLMCallError, call_llama_with_retry
from zados.LLM_interpretation.phase5_evaluator import Phase5Evaluator, Phase5Result
from zados.LLM_interpretation.prompt_builder import RGPromptBuilder, VTPromptBuilder
from zados.LLM_interpretation.tools import SEARCH_TOOLS, _execute_search


class LLMInterpretationLayer:
    """
    Final stage of the ZA-DOS processing pipeline (v0.5).

    Reads a fully-computed STMMStore, runs VT → Phase 5 → RG, and returns
    the user-facing response string.

    Usage
    -----
    layer = LLMInterpretationLayer()
    response = layer.run(stmm, input_bundle={...})

    Parameters
    ----------
    synthesis_engine : SynthesisEngine, optional
        Tonic pathway engine for Phase 5.
    nt_adapter : NeurochemicalAdapter, optional
        NT signal adapter for Phase 5.
    orchestrator : ExtractorOrchestrator, optional
        Phasic pathway orchestrator for Phase 5.
    domain_evaluators : dict, optional
        Map of domain_name → RewardDomain for Phase 5 domain evaluation.
    """

    def __init__(
        self,
        synthesis_engine=None,
        nt_adapter=None,
        orchestrator=None,
        domain_evaluators: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._vt_builder = VTPromptBuilder()
        self._rg_builder = RGPromptBuilder()
        self._phase5     = Phase5Evaluator(
            synthesis_engine=synthesis_engine,
            nt_adapter=nt_adapter,
            orchestrator=orchestrator,
            domain_evaluators=domain_evaluators,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        stmm,
        input_bundle: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Gate-check → Urgency gate → VT → Phase 5 → RG → response.

        Parameters
        ----------
        stmm : STMMStore
            Fully-computed short-term memory state.
        input_bundle : dict, optional
            Pipeline context carrying:
                extractor_state          — ExtractorState or dict
                prior_urgency_risk       — float [0, 1]
                emotion_profile          — dict of emotion_id → strength
                current_oscillations     — OscillationState or None
                mission_briefing         — str
                active_reward_profile_name — str

        Returns
        -------
        str
            ""            if suppress=True  (no LLM calls made)
            short ack     if abstain=True   (no VT call made)
            urgency resp  if urgency > URG_SKIP_VT  (brief RG, no VT)
            full response otherwise
        """
        bundle = input_bundle or {}
        meta   = stmm.reward_evaluation.meta_directive or {}

        # ---- 1. Gate check (prev meta_directive) -----------------------
        if meta.get("suppress", False):
            stmm.brain_process_tracker.mark_stage("llm_suppressed", True)
            return ""

        if meta.get("abstain", False):
            response = self._generate_abstain_response(stmm)
            stmm.add_system_response(response)
            return response

        # ---- 2. Urgency gate (v0.5) ------------------------------------
        prior_urgency = bundle.get("prior_urgency_risk", 0.0)
        if prior_urgency >= URG_SKIP_VT:
            stmm.brain_process_tracker.mark_stage("vt_skipped_urgency", True)
            response = self._generate_urgency_response(stmm, bundle)
            stmm.add_system_response(response)
            return response

        # ---- 3. Verbalized Thinking (VT / Phase 4) ---------------------
        vt_output = self._run_vt(stmm, bundle, prior_urgency)

        # Write VT outputs to cortical reflection (permitted writes)
        stmm.cortical_reflection.verbal_reflection     = vt_output
        stmm.cortical_reflection.verbal_emotion_labels = self._top_emotions(stmm)
        stmm.brain_process_tracker.mark_stage("vt_complete", True)

        # ---- 4. Phase 5 — Two-pathway reward evaluation (v0.5) ---------
        phase5_result = self._run_phase5(vt_output, stmm, bundle)
        stmm.brain_process_tracker.mark_stage("phase5_complete", True)

        # ---- 5. Response Generation (RG / Phase 6) ----------------------
        response = self._run_rg(stmm, vt_output, phase5_result, bundle)
        stmm.brain_process_tracker.mark_stage("rg_complete", True)

        stmm.add_system_response(response)
        return response

    # ------------------------------------------------------------------
    # Phase 4 — Verbalized Thinking
    # ------------------------------------------------------------------

    def _run_vt(
        self,
        stmm,
        bundle: Dict[str, Any],
        prior_urgency: float,
    ) -> str:
        """
        Generate VT monologue.  Budget reduced by 30% if urgency >= URG_HIGH.
        """
        vt_budget = VT_OUTPUT_MAX
        if prior_urgency >= URG_HIGH:
            vt_budget = int(VT_OUTPUT_MAX * 0.70)

        vt_prompt   = self._vt_builder.build(stmm, input_bundle=bundle)
        vt_messages = [{"role": "user", "content": vt_prompt}]

        try:
            vt_result = call_llama_with_retry(
                vt_messages,
                max_tokens=vt_budget,
                temperature=VT_TEMPERATURE,
            )
            return vt_result.get("content", "")
        except LLMCallError:
            return FALLBACK_VT

    # ------------------------------------------------------------------
    # Phase 5 — Two-pathway reward evaluation
    # ------------------------------------------------------------------

    def _run_phase5(
        self,
        vt_output: str,
        stmm,
        bundle: Dict[str, Any],
    ) -> Phase5Result:
        """
        Delegate to Phase5Evaluator.  Returns Phase5Result.
        Silently returns an empty result on any error.
        """
        try:
            return self._phase5.evaluate(vt_output, stmm, input_bundle=bundle)
        except Exception:
            return Phase5Result()

    # ------------------------------------------------------------------
    # Phase 6 — Response Generation
    # ------------------------------------------------------------------

    def _run_rg(
        self,
        stmm,
        vt_output: str,
        phase5_result: Phase5Result,
        bundle: Dict[str, Any],
    ) -> str:
        """
        Build RG prompt, call LLM, handle tool calls.
        Token budget is gated by CSS and urgency.
        """
        # Determine RG token budget
        rg_tokens = self._compute_rg_budget(stmm, phase5_result)

        # Build RG messages
        rg_messages = self._rg_builder.build(
            stmm,
            vt_output,
            extractor_result=phase5_result.extractor_result,
            input_bundle=bundle,
            selected_mode=phase5_result.selected_mode,
        )

        # Tool eligibility
        tools_arg = SEARCH_TOOLS if self._search_eligible(stmm) else None

        try:
            rg_result = call_llama_with_retry(
                rg_messages,
                max_tokens=rg_tokens,
                temperature=RG_TEMPERATURE,
                tools=tools_arg,
            )
            return self._handle_tool_calls(rg_result, rg_messages)
        except LLMCallError:
            return FALLBACK_RESPONSE

    def _compute_rg_budget(self, stmm, phase5_result: Phase5Result) -> int:
        """
        Determine RG token budget based on CSS and urgency.

        Priority (highest first):
            1. urgency >= URG_HIGH  → RG_OUTPUT_URG  (250)
            2. CSS >= CSS_SEVERE    → RG_OUTPUT_SEV  (300)
            3. default              → RG_OUTPUT_MAX  (800)
        """
        # Urgency check
        urgency = phase5_result.urgency_risk
        if urgency >= URG_HIGH:
            return RG_OUTPUT_URG

        # CSS check
        sat = stmm.emotion_detection.saturation_levels
        css = max(sat.values(), default=0.0) if sat else 0.0
        if css >= CSS_SEVERE:
            return RG_OUTPUT_SEV

        return RG_OUTPUT_MAX

    # ------------------------------------------------------------------
    # Urgency response (v0.5 — Gate 3.2 triggered, VT skipped)
    # ------------------------------------------------------------------

    def _generate_urgency_response(
        self,
        stmm,
        bundle: Dict[str, Any],
    ) -> str:
        """
        Brief RG response when urgency > URG_SKIP_VT.
        VT is skipped entirely.  Uses a minimal RG prompt with urgency
        conditioning and reduced token budget.
        """
        # Store empty VT
        stmm.cortical_reflection.verbal_reflection     = ""
        stmm.cortical_reflection.verbal_emotion_labels = self._top_emotions(stmm)

        # Build minimal RG messages (no VT, no Phase 5)
        rg_messages = self._rg_builder.build(
            stmm,
            vt_output="",
            extractor_result=None,
            input_bundle=bundle,
            selected_mode="Containment",
        )

        try:
            rg_result = call_llama_with_retry(
                rg_messages,
                max_tokens=RG_OUTPUT_URG,
                temperature=RG_TEMPERATURE,
            )
            return rg_result.get("content", FALLBACK_RESPONSE)
        except LLMCallError:
            return FALLBACK_RESPONSE

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _generate_abstain_response(self, stmm) -> str:
        """Short programmatic acknowledgement when abstain=True."""
        user_msg = stmm.active_message_buffer.latest_user()
        if user_msg:
            return (
                "I've noted your message. "
                "I'm not in a position to respond to that fully right now."
            )
        return "Acknowledged."

    # ------------------------------------------------------------------
    # Emotion helpers
    # ------------------------------------------------------------------

    def _top_emotions(self, stmm) -> List[str]:
        """
        Return the top-5 emotion names from system_emotion_state
        where value > 0.15, ordered descending.
        """
        state = stmm.emotion_detection.system_emotion_state
        if not state:
            return []
        filtered = [(k, v) for k, v in state.items() if v > 0.15]
        return [k for k, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:5]]

    # ------------------------------------------------------------------
    # Search eligibility
    # ------------------------------------------------------------------

    def _search_eligible(self, stmm) -> bool:
        """
        True if:
          - primary_intention == "information_seeking"
          - AND fewer than 2 matched memory contrast entries
        """
        ia = stmm.intention_analysis
        mc = stmm.memory_contrast
        return (
            ia.primary_intention == "information_seeking"
            and len(mc.matched_entries) < 2
        )

    # ------------------------------------------------------------------
    # Tool call handling
    # ------------------------------------------------------------------

    def _handle_tool_calls(
        self,
        response: Dict[str, Any],
        messages: List[Dict[str, str]],
    ) -> str:
        """
        If the RG response contains tool_calls, execute the first
        web_search call, append the result, and make one follow-up
        RG call (no tools this time to avoid recursion).

        Falls back to the original content on any error.
        """
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            return response.get("content", "")

        tool_call = tool_calls[0]
        fn        = tool_call.get("function", {})
        fn_name   = fn.get("name", "")
        fn_args   = fn.get("arguments", {})

        if fn_name == "web_search":
            query = fn_args.get("query", "") if isinstance(fn_args, dict) else ""
            search_result = _execute_search(query)

            updated_messages = list(messages) + [
                {
                    "role":    "tool",
                    "content": search_result,
                    "name":    "web_search",
                }
            ]
            try:
                final = call_llama_with_retry(
                    updated_messages,
                    max_tokens=RG_OUTPUT_MAX,
                    temperature=RG_TEMPERATURE,
                    tools=None,  # no recursive tool calling
                )
                return final.get("content", "")
            except LLMCallError:
                return response.get("content", "")

        # Unknown tool — return original content
        return response.get("content", "")

    # ------------------------------------------------------------------
    # Verbal summary helper (consumed by Memory Exit Compressor)
    # ------------------------------------------------------------------

    def _generate_verbal_summary(self, vt_text: str) -> str:
        """
        Extractive compression of VT monologue to ~100 words.
        Selects first 2 sentences + last sentence.

        Called externally by MemoryExitCompressor, not by run().
        """
        if not vt_text.strip():
            return ""

        sentences = [s.strip() for s in vt_text.split(".") if s.strip()]
        if len(sentences) <= 3:
            return vt_text.strip()

        selected = sentences[:2] + [sentences[-1]]
        return ". ".join(selected) + "."
