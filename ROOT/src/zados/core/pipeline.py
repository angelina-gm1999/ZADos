"""
ZA-DOS Core Pipeline — AnswerPipeline (spec §2.1).

Single-turn orchestrator that sequences Phases 0-7 and returns a
PipelineResult.  This is the core of Pipeline 2.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from zados.core.phases.phase0_reception import validate_bundle
from zados.core.phases.phase1_perception import run_perception
from zados.core.phases.phase2_modulation import run_nt_modulation
from zados.core.phases.phase3_dispatch import run_engine_dispatch
from zados.core.phases.phase4_thinking import run_thinking_pass
from zados.core.phases.phase5_reward import run_reward_evaluation
from zados.core.phases.phase6_answer import run_answer_pass
from zados.core.phases.phase7_postprocess import run_postprocessing
from zados.core.thinking_blocks import ThinkingBlockBuilder
from zados.core.time_context import get_time_context
from zados.core.types import (
    InputBundle,
    PipelineResult,
    PipelineState,
    SessionState,
)
from zados.LLM_interpretation.phase5_evaluator import Phase5Evaluator
from zados.memory.short_term.store import STMMStore

log = logging.getLogger(__name__)


class AnswerPipeline:
    """Single-turn answer pipeline: Phase 0 → Phase 7.

    Usage
    -----
    >>> pipeline = AnswerPipeline(neurochem_engine, memory, engines)
    >>> result = pipeline.process_turn(bundle, session)
    >>> print(result.final_answer)

    Parameters
    ----------
    neurochem_engine : NeurochemicalEngine
    memory : MemoryLayer
    engines : dict
        engine_number → engine instance (all 29 engines).
    tokenizer : Tokenizer, optional
    semantic_expander : SemanticExpander, optional
    synthesis_engine : SynthesisEngine, optional
    nt_adapter : NeurochemicalAdapter, optional
    orchestrator : ExtractorOrchestrator, optional
    domain_evaluators : dict, optional
    """

    def __init__(
        self,
        neurochem_engine: Any,
        memory: Any,
        engines: Dict[int, Any],
        tokenizer: Any = None,
        semantic_expander: Any = None,
        synthesis_engine: Any = None,
        nt_adapter: Any = None,
        orchestrator: Any = None,
        domain_evaluators: Optional[Dict[str, Any]] = None,
        hardcoded_store: Any = None,
        journal_writer: Any = None,
    ) -> None:
        self.engine = neurochem_engine
        self.memory = memory
        self.engines = engines
        self.tokenizer = tokenizer
        self.semantic_expander = semantic_expander

        self._phase5_evaluator = Phase5Evaluator(
            synthesis_engine=synthesis_engine,
            nt_adapter=nt_adapter,
            orchestrator=orchestrator,
            domain_evaluators=domain_evaluators,
        )

        # Identity alignment checker (optional — no-ops gracefully when absent)
        self._hardcoded_store = hardcoded_store
        self._alignment_checker = None
        if hardcoded_store is not None:
            try:
                from zados.memory.long_term.identity.alignment import IdentityAlignmentChecker
                correlation_store = None
                try:
                    identity_ns = getattr(memory, "identity", None)
                    if identity_ns is not None:
                        correlation_store = getattr(identity_ns, "correlation", None)
                except Exception:
                    pass
                self._alignment_checker = IdentityAlignmentChecker(
                    hardcoded_store,
                    correlation_store=correlation_store,
                )
            except Exception:
                log.warning("IdentityAlignmentChecker could not be initialized.")

        self._thinking_builder = ThinkingBlockBuilder()
        self._journal_writer = journal_writer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> PipelineResult:
        """Execute Phases 0-7 for one conversational turn.

        Parameters
        ----------
        bundle : InputBundle
            Populated input from Pipeline 1 (or session builder).
        session : SessionState
            Persistent session state.

        Returns
        -------
        PipelineResult
        """
        # Stamp temporal context on the bundle (idempotent — only if not already set)
        if not bundle.time_context:
            tc = get_time_context(session_start=session.session_start_time)
            bundle.time_context = tc.to_dict()

        # Initialise STMM and pipeline state
        stmm = STMMStore()
        stmm.add_user_message(bundle.raw_text)

        state = PipelineState(
            bundle=bundle,
            stmm=stmm,
            turn_index=session.turn_count,
            timestamp=time.time(),
        )

        # ==============================================================
        # Phase 0 — Input validation
        # ==============================================================
        validate_bundle(bundle)
        stmm.brain_process_tracker.mark_stage("phase0_validated", True)

        # ==============================================================
        # Phase 1 — Perception
        # ==============================================================
        nt_snapshot_lc = self._get_nt_snapshot_lowercase()

        state.perception = run_perception(
            bundle,
            self.engines,
            nt_snapshot_lc,
            tokenizer=self.tokenizer,
            semantic_expander=self.semantic_expander,
            stmm=stmm,
        )
        stmm.brain_process_tracker.mark_stage("phase1_perception", True)

        # ==============================================================
        # Phase 3 — Engine Dispatch (runs BEFORE Phase 2)
        # Engine weights from bundle (set by RegularInputPipeline.EngineToolkit)
        # are used as the fallback when modulation is not yet set.
        # ==============================================================
        state.dispatch = run_engine_dispatch(
            state,
            self.engines,
            nt_snapshot_lc,
            memory_contrast=self.memory.contrast,
        )
        stmm.brain_process_tracker.mark_stage("phase3_dispatch", True)

        # ==============================================================
        # Phase 2 — NT Modulation (post-dispatch, intent-based)
        # Uses E28 emotion results + E23 intent from dispatch/perception.
        # ==============================================================
        state.modulation = run_nt_modulation(
            bundle,
            state.perception,
            state.dispatch,
            self.engine,
            stmm,
            osc_state=bundle.osc_state or session.osc_state,
            extractor_state=bundle.extractor_state or session.extractor_state,
        )
        stmm.brain_process_tracker.mark_stage("phase2_modulation", True)

        # Persist updated extractor state to session
        if getattr(state.modulation, "updated_extractor_state", None) is not None:
            session.extractor_state = state.modulation.updated_extractor_state

        # Update NT snapshot after modulation
        if state.modulation and state.modulation.nt_snapshot:
            nt_snapshot_lc = state.modulation.nt_snapshot

        # ==============================================================
        # ThinkingContext — compressed context for LLM thinking pass
        # ==============================================================
        thinking_context = self._build_thinking_context(state, stmm, session)

        # Identity alignment check
        if self._alignment_checker is not None:
            try:
                alignment_result = self._alignment_checker.check(thinking_context)
                thinking_context.alignment_result = alignment_result
                thinking_context.personality_prompts = alignment_result.personality_prompts
            except Exception:
                log.exception("Identity alignment check failed.")
        elif self._hardcoded_store is not None:
            # Store available but checker wasn't initialized — extract personality only
            try:
                thinking_context.personality_prompts = [
                    e.content for e in self._hardcoded_store.get_by_category("personality")
                ]
            except Exception:
                pass

        # ==============================================================
        # Build input_bundle_dict (bridge to LLM layer, enriched with context)
        # ==============================================================
        bundle_dict = self._build_bundle_dict(bundle, state, session, thinking_context)

        # ==============================================================
        # Phase 4 — Thinking Blocks (VT / LLM Pass 1)
        # ==============================================================
        state.thinking = run_thinking_pass(state, stmm, bundle_dict)
        # (mark_stage happens inside run_thinking_pass)

        # ==============================================================
        # Phase 5 — Reward Evaluation (both pathways + engine apply)
        # ==============================================================
        state.reward = run_reward_evaluation(
            state,
            stmm,
            self.engine,
            self._phase5_evaluator,
            bundle_dict,
        )
        # (mark_stage happens inside run_reward_evaluation)

        # ==============================================================
        # Phase 6 — Final Answer (RG / LLM Pass 2)
        # ==============================================================
        phase5_result = state.reward.phase5_result if state.reward else None

        state.answer = run_answer_pass(
            state,
            stmm,
            phase5_result,
            bundle_dict,
        )
        # (mark_stage happens inside run_answer_pass)

        # ==============================================================
        # Phase 7 — Post-processing & Memory Loop
        # ==============================================================
        state.postprocess = run_postprocessing(
            state,
            self.engine,
            self.memory,
            self.engines,
            phase5_result,
            session=session,
            journal_writer=self._journal_writer,
        )
        # (mark_stage happens inside run_postprocessing)

        # ==============================================================
        # Update session state
        # ==============================================================
        session.turn_count += 1
        session.last_interaction_timestamp = state.timestamp
        # extractor_state already saved after Phase 2; also check Phase 5 result
        if phase5_result is not None:
            er = getattr(phase5_result, "extractor_result", None)
            if er is not None and hasattr(er, "state"):
                session.extractor_state = er.state

        return PipelineResult(
            final_answer=state.answer.final_answer if state.answer else "",
            state=state,
            directive=state.answer.directive_applied if state.answer else "allow",
            phase5_result=phase5_result,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_nt_snapshot_lowercase(self) -> Dict[str, float]:
        """Read current NT concentrations as lowercase-keyed dict."""
        snapshot: Dict[str, float] = {}
        try:
            for name in self.engine.registry.neurotransmitter_names():
                nt = self.engine.registry.get_neurotransmitter(name)
                snapshot[name.lower()] = nt.C
        except Exception:
            log.warning("Failed to read NT snapshot from engine.")
        return snapshot

    def _build_thinking_context(
        self,
        state: PipelineState,
        stmm: Any,
        session: SessionState,
    ) -> Any:
        """Build ThinkingContext after Phase 3 + Phase 2 complete."""
        try:
            ltmm = getattr(self.memory, "ltmm", None) or getattr(self.memory, "long_term", None)
            return self._thinking_builder.build(state, stmm, session, ltmm=ltmm)
        except Exception:
            log.exception("ThinkingBlockBuilder.build() failed; returning empty context.")
            from zados.core.thinking_blocks.types import ThinkingContext
            return ThinkingContext()

    def _build_bundle_dict(
        self,
        bundle: InputBundle,
        state: PipelineState,
        session: SessionState,
        thinking_context: Any = None,
    ) -> Dict[str, Any]:
        """Build the dict that VTPromptBuilder / RGPromptBuilder expect."""
        d: Dict[str, Any] = {
            "mission_briefing": bundle.mission_briefing or session.mission_briefing or "",
            "active_reward_profile_name": (
                state.modulation.reward_profile_name
                if state.modulation
                else session.reward_profile_name
            ),
        }

        # Urgency risk from extractor result (Phase 2) or session state
        prior_urgency = 0.0
        if state.modulation and getattr(state.modulation, "extractor_result", None):
            prior_urgency = getattr(state.modulation.extractor_result, "urgency_risk", 0.0)
        elif bundle.extractor_state and hasattr(bundle.extractor_state, "urgency_risk"):
            prior_urgency = bundle.extractor_state.urgency_risk
        elif session.extractor_state and hasattr(session.extractor_state, "urgency_risk"):
            prior_urgency = session.extractor_state.urgency_risk
        d["prior_urgency_risk"] = prior_urgency

        # Emotion profile
        d["emotion_profile"] = bundle.emotion_profile

        # Oscillation state
        d["current_oscillations"] = bundle.osc_state or session.osc_state

        # Extractor state (updated from Phase 2)
        d["extractor_state"] = (
            getattr(state.modulation, "updated_extractor_state", None)
            or bundle.extractor_state
            or session.extractor_state
        )

        # ThinkingContext — full compressed context for LLM pass
        d["thinking_context"] = thinking_context

        # Operational context flags — pipeline origin, mode overrides, signal markers.
        # Forwarded to VT/RG prompt builders so the LLM is conditioned on special
        # contexts (dream mode, learning reframe, emphasis directives, etc.).
        d["context_flags"] = dict(bundle.context_flags)

        return d
