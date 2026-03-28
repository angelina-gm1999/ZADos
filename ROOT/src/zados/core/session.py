"""
ZA-DOS Core Pipeline — SessionOrchestrator (spec Part II).

Manages session lifecycle:
  - Boot sequence (Branch A/B/C classification)
  - Per-turn processing (builds InputBundle, delegates to AnswerPipeline)
  - Drift detection via MemoryContrast
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from zados.core.mode_profiles import profile_for_mode
from zados.core.pipeline import AnswerPipeline
from zados.core.types import InputBundle, PipelineResult, SessionState

log = logging.getLogger(__name__)

# Time-delta thresholds for branch classification (spec §2.2)
BRANCH_A_THRESHOLD = 5.0       # < 5s → Branch A (rapid continuation)
BRANCH_B_THRESHOLD = 600.0     # < 10min → Branch B (normal)
# >= 10min → Branch C (cold start / context recapture)


class SessionOrchestrator:
    """Top-level session lifecycle manager.

    Usage
    -----
    >>> orch = SessionOrchestrator(neurochem_engine, memory, engines)
    >>> orch.open_session()
    >>> response = orch.process_turn("Hello, how are you?")

    Parameters
    ----------
    neurochem_engine : NeurochemicalEngine
    memory : MemoryLayer
    engines : dict  (engine_number → engine instance)
    tokenizer : Tokenizer, optional
    semantic_expander : SemanticExpander, optional
    synthesis_engine, nt_adapter, orchestrator, domain_evaluators
        Passed through to Phase5Evaluator via AnswerPipeline.
    hardcoded_store : HardcodedStore, optional
        Identity hardcoded store — passed to AnswerPipeline for IdentityAlignmentChecker.
    journal_writer : JournalWriter, optional
        Cognitive journal writer — passed to AnswerPipeline for Phase 7 journal writes.
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

        self.pipeline = AnswerPipeline(
            neurochem_engine=neurochem_engine,
            memory=memory,
            engines=engines,
            tokenizer=tokenizer,
            semantic_expander=semantic_expander,
            synthesis_engine=synthesis_engine,
            nt_adapter=nt_adapter,
            orchestrator=orchestrator,
            domain_evaluators=domain_evaluators,
            hardcoded_store=hardcoded_store,
            journal_writer=journal_writer,
        )

        self.session: Optional[SessionState] = None

    # ------------------------------------------------------------------
    # Boot sequence (spec §2.2, steps B.1-B.9)
    # ------------------------------------------------------------------

    def open_session(self, previous_session: Optional[SessionState] = None) -> SessionState:
        """Boot the session and return the initialised SessionState.

        Parameters
        ----------
        previous_session : SessionState, optional
            If provided, calculates time delta and classifies branch.
        """
        now = time.time()
        session = SessionState()
        session.last_interaction_timestamp = now

        # B.1: Time delta classification
        if previous_session is not None:
            elapsed = now - (previous_session.last_interaction_timestamp or 0.0)
            session.branch = _classify_branch(elapsed)
            session.extractor_state = previous_session.extractor_state
            session.osc_state = previous_session.osc_state
        else:
            session.branch = "C"  # Cold start

        # B.3: Read current NT state (already decayed via engine pharmacodynamics)
        # B.4: Neurosymbolic readout → metrics
        try:
            metrics_dict = self.engine.get_neurosymbolic_readout()
            if not isinstance(metrics_dict, dict) and hasattr(metrics_dict, "as_dict"):
                metrics_dict = metrics_dict.as_dict()
        except Exception:
            metrics_dict = {}

        # B.5: Mode selection → initial mode token
        initial_mode = "Normal"
        try:
            from zados.neurochem.neurosymbolic.mode_hooks import (
                DEFAULT_MODE_HOOKS,
                build_mode_namespace,
                select_mode,
            )
            from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics

            metrics_obj = NeurochemicalMetrics(**{
                k: metrics_dict.get(k, 0.5)
                for k in ("motivation", "empathy", "cognitive_rigidity", "fatigue",
                           "precision", "openness", "anxiety", "social_engagement")
            })
            osc_dict = self._get_osc_dict(session.osc_state)
            variables = build_mode_namespace(metrics_obj, osc_dict)
            mode_result = select_mode(DEFAULT_MODE_HOOKS, variables)
            initial_mode = mode_result.active_mode or "Normal"
        except Exception:
            log.exception("Session boot mode selection failed.")

        session.initial_mode = initial_mode
        session.reward_profile_name = profile_for_mode(initial_mode)

        # B.6: MTMM prior context search
        if session.branch == "C":
            try:
                cp = self.memory.mtmm.context_processor
                prior = cp.search("session_context", limit=5)
                if prior:
                    session.mission_briefing = prior[0] if len(prior) == 1 else prior
            except Exception:
                log.debug("MTMM context search for session boot returned nothing.")

        # B.7: Context Prompt exchange — mission briefing collected from user
        # at session open.  Call set_mission_briefing(text) after open_session()
        # returns to populate this field before the first turn.

        # B.8: E23 initial intent (not needed until first turn)
        # B.9: MemoryContrast drift monitor (activated below)

        # Knowledge bootstrap — seed foundational knowledge into LTMM stores
        # and AtomSpace (E9) before the first turn.
        try:
            from zados.bootstrap import KnowledgeBootstrap
            atomspace = self.engines.get(9) if self.engines else None
            KnowledgeBootstrap.run(self.memory, atomspace_engine=atomspace)
        except Exception:
            log.exception("Knowledge bootstrap failed — session continues without seed.")

        self.session = session
        log.info(
            "Session opened: branch=%s, mode=%s, profile=%s",
            session.branch, initial_mode, session.reward_profile_name,
        )
        return session

    # ------------------------------------------------------------------
    # Per-turn processing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # B.7 — Mission briefing (spec §2.2 CP.1-CP.4)
    # ------------------------------------------------------------------

    def set_mission_briefing(self, briefing: str) -> None:
        """Store the user's session-level starter prompt (B.7).

        Called after open_session() with the user's description of what they
        want to talk about / the context for this session.  This is carried
        in InputBundle.mission_briefing every turn and included verbatim in
        ThinkingContext.mission_briefing so the LLM maintains background
        awareness throughout the session.

        Parameters
        ----------
        briefing : str
            Free-text description from the user (e.g. "I want to discuss
            the ethics of AI governance — keep this as the main thread").
        """
        if self.session is None:
            self.open_session()
        self.session.mission_briefing = briefing
        log.info("Session %s: mission briefing set (%d chars)", self.session.session_id, len(briefing))

    def process_turn(self, raw_text: str) -> str:
        """Build an InputBundle and run the full pipeline.

        Parameters
        ----------
        raw_text : str
            User message text.

        Returns
        -------
        str
            User-facing response.
        """
        if self.session is None:
            self.open_session()

        bundle = self._build_input_bundle(raw_text)
        result = self.pipeline.process_turn(bundle, self.session)
        self._update_session_from_result(result)
        return result.final_answer

    def process_turn_full(self, raw_text: str) -> PipelineResult:
        """Like process_turn but returns the full PipelineResult."""
        if self.session is None:
            self.open_session()

        bundle = self._build_input_bundle(raw_text)
        result = self.pipeline.process_turn(bundle, self.session)
        self._update_session_from_result(result)
        return result

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(self) -> bool:
        """Run MemoryContrast drift detection.

        Returns True if drift exceeds threshold (re-run E23 recommended).
        """
        if self.session is None:
            return False

        try:
            contrast = self.memory.contrast
            result = contrast.contrast(
                current={"text": "", "content": "drift_check"},
                query_type="context",
            )
            divergence = getattr(result, "divergence", 0.0)
            return divergence > 0.5  # threshold for significant drift
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Session close
    # ------------------------------------------------------------------

    def close_session(self) -> Dict[str, Any]:
        """Close the current session, trigger consolidation and overview write."""
        if self.session is None:
            return {}

        summary: Dict[str, Any] = {"session_id": self.session.session_id}

        # 1. Write OverviewLogEntry
        try:
            self.memory.manager.write_session_overview(self.session)
            summary["overview_written"] = True
        except Exception:
            log.debug("close_session: failed to write session overview.")
            summary["overview_written"] = False

        # 2. Consolidate MTMM → LTMM
        try:
            self.memory.manager.consolidate()
            summary["consolidated"] = True
        except Exception:
            log.debug("close_session: failed to consolidate MTMM → LTMM.")
            summary["consolidated"] = False

        # 3. Increment stagnation counters
        try:
            self.memory.manager.tick_unsolved()
            summary["tick_unsolved"] = True
        except Exception:
            log.debug("close_session: failed to tick unsolved counters.")
            summary["tick_unsolved"] = False

        # 4. Flush STMM → MTMM
        try:
            self.memory.end_cycle()
            summary["end_cycle"] = True
        except Exception:
            log.debug("close_session: failed to end memory cycle (STMM → MTMM).")
            summary["end_cycle"] = False

        # 5. Persist cognitive engine data (AtomSpace → CognitoolsDataStore)
        try:
            self._persist_cognitools()
            summary["cognitools_persisted"] = True
        except Exception:
            log.debug("close_session: failed to persist cognitools data.")
            summary["cognitools_persisted"] = False

        log.info("Session %s closed.", self.session.session_id)

        # 5. Store reference for next open and clear current session
        self._previous_session = self.session
        self.session = None

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_session_from_result(self, result: PipelineResult) -> None:
        """Persist turn-level state back to the session after pipeline completes."""
        if self.session is None or result is None:
            return
        state = result.state
        if state is None:
            return
        # Save reward_profile_name from Phase 2 modulation
        if state.modulation and state.modulation.reward_profile_name:
            self.session.reward_profile_name = state.modulation.reward_profile_name
        # Save updated extractor state (also handled in pipeline.py but kept here for safety)
        if state.modulation and getattr(state.modulation, "updated_extractor_state", None):
            self.session.extractor_state = state.modulation.updated_extractor_state

    def _build_input_bundle(self, raw_text: str) -> InputBundle:
        """Construct an InputBundle from raw text + session state."""
        session = self.session
        bundle = InputBundle(
            raw_text=raw_text,
            active_mode=session.initial_mode if session else "",
            mission_briefing=session.mission_briefing if session else None,
            osc_state=session.osc_state if session else None,
            extractor_state=session.extractor_state if session else None,
        )

        # Carry forward emotion profile from E28 of previous turn if available
        # (empty on first turn — Phase 3 will handle it)
        return bundle

    def _persist_cognitools(self) -> None:
        """Persist cognitive engine data to CognitoolsDataStore at session end."""
        cognitools_store = None
        try:
            knowledge = getattr(self.memory, "knowledge", None)
            if knowledge is not None:
                cognitools_store = getattr(knowledge, "cognitools_data", None)
        except Exception:
            return

        if cognitools_store is None:
            return

        # Persist AtomSpace (E9)
        e9 = self.engines.get(9)
        if e9 is not None and hasattr(e9, "persist_to_store"):
            try:
                e9.persist_to_store(cognitools_store)
                log.debug("AtomSpace (E9) state persisted to CognitoolsDataStore.")
            except Exception:
                log.debug("AtomSpace persistence failed.", exc_info=True)

    def _get_osc_dict(self, osc_state: Any) -> Dict[str, float]:
        """Extract oscillation amplitudes as a flat dict."""
        if osc_state is not None:
            return {
                band: getattr(osc_state, band, 0.0)
                for band in ("delta", "theta", "alpha", "beta", "gamma")
            }
        try:
            osc = self.engine.registry.get_oscillations()
            return {
                band: getattr(osc, band, 0.0)
                for band in ("delta", "theta", "alpha", "beta", "gamma")
            }
        except Exception:
            return {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _classify_branch(elapsed_seconds: float) -> str:
    """Classify the session branch from time delta."""
    if elapsed_seconds < BRANCH_A_THRESHOLD:
        return "A"
    if elapsed_seconds < BRANCH_B_THRESHOLD:
        return "B"
    return "C"
