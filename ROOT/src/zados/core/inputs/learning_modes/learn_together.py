"""
ZA-DOS v0.6 — M3: Learn Together (spec §3.3.3 + Part 2 §3.3 + Part 4 §4).

Full dialectic mode.  Maximum engine budget (18), all clusters
active.  Emphasis on synthesis, collaborative exploration, and
deep engagement with material.

Neurochem wiring (Part 2 §3.3):
  - Preset: maximal DA-D3 (exploration), CB1 (schema flexibility),
    5-HT2A (symbolic expansion), high OXT (collaborative bonding)
  - Full ExtractorOrchestrator stochastic pathway active
  - BOTH tonic/deterministic AND phasic/stochastic pathways run
  - Curiosity-Courage-Discovery-Joy cycle tracked across turns

Part 4 §4 additions:
  - Human challenge logic: ZA-DOS actively checks human claims
    against established knowledge via E1/E4 contradiction/fallacy
  - Dialectic output: when a claim contradicts knowledge, ZA-DOS
    presents the contradiction as a learning signal
  - contrast_challenges populated on LearningModeResult
  - Held thinking block check on emotion spikes
  - Unlimited question and expansion depth

Engine budget: 18 (T1+T2).
Risk emotions: confused, overwhelmed, frustrated.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from zados.core.inputs.learning_modes.base import LearningModePipeline
from zados.core.processes.subject_classifier import classify_subject_from_text
from zados.core.types import (
    EngineTier,
    InputBundle,
    LearningModeResult,
    SessionState,
    UnsolvedQuestion,
)

log = logging.getLogger(__name__)

# M3 emotional cycle states
_CYCLE_EXPLORING = "exploring"
_CYCLE_PIVOTING = "pivoting"
_CYCLE_CONSOLIDATING = "consolidating"

# Session state key for M3 cycle tracking
_M3_CYCLE_KEY = "m3_emotional_cycle"


class LearnTogetherPipeline(LearningModePipeline):
    """M3 — Learn Together: full dialectic, collaborative learning."""

    mode_id = "M3"
    mode_number = 3

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> LearningModeResult:
        """Process a turn in M3 (Learn Together) mode.

        Stages (Part 4 §4):
          0. Setup — preset, engine resolve, drift check
          1. Scoped memory contrast (identity + thoughts)
          2. Engine dispatch (all clusters active, max budget)
          3. VT thinking + held-thinking-block check
          4. M3-specific: human challenge logic, stochastic pathway,
             emotional cycle tracking
          5. LTMM write (scoped)
          6. Unsolved question extraction (unlimited per turn)
          7. Response generation (dialectic output)
          8. NT feedback + homeostatic + learning record

        Parameters
        ----------
        bundle : InputBundle
        session : SessionState

        Returns
        -------
        LearningModeResult
        """
        subject = classify_subject_from_text(bundle.raw_text)

        # ---- Stage 0: Setup ----
        self._apply_emotional_preset(bundle)
        self._resolve_engines(bundle, subject)
        self._check_drift(bundle)

        risks = self._check_risk_emotions(bundle)
        if risks:
            log.info("M3 risk emotions triggered (pre-pipeline): %s", risks)

        # ---- Stage 1: Scoped memory contrast ----
        if self._pipeline_scope is not None:
            bundle._pipeline_read_scope = self._pipeline_scope.read_scope  # type: ignore[attr-defined]

        # ---- Stage 2: Engine dispatch ----
        result = self._pipeline.process_turn(bundle, session)

        # ---- Stage 3: VT thinking + held block check ----
        feedback = self._run_feedback_loop(bundle, result, session)
        emotion_profile = feedback.get("emotion_profile", {})

        held_block_ids = self._check_held_thinking_block(
            emotion_profile=emotion_profile,
            thinking_trace=_get_thinking_trace(result),
            bundle=bundle,
            session=session,
        )

        # ---- Stage 4: M3-specific processing ----
        # 4a: Human challenge logic (Part 4 §4.2)
        challenges = self._check_human_claims(bundle, result, feedback)

        # 4b: Stochastic pathway (Part 2 §3.3)
        self._run_stochastic_pathway(result, feedback)

        # 4c: Emotional cycle tracking
        self._track_emotional_cycle(feedback, session)

        # ---- Stage 5-7: LTMM write, response ----
        # (handled by pipeline)

        # ---- Stage 8: Record + extract ----
        self._record_learning(session, result, subject.value)
        unsolved = self._extract_unsolved(result, subject.value)

        return LearningModeResult(
            mode_number=self.mode_number,
            pipeline_result=result,
            unsolved_questions=unsolved,
            held_thinking_blocks=held_block_ids,
            contrast_challenges=challenges,
        )

    def _run_stochastic_pathway(
        self,
        result: Any,
        feedback: Dict[str, Any],
    ) -> None:
        """Run the full ExtractorOrchestrator stochastic pathway (Part 2 §3.3).

        M3 is the primary mode that uses BOTH:
          Pathway A: Tonic/Deterministic (SynthesisEngine + NeurochemicalAdapter)
          Pathway B: Phasic/Stochastic (full 9-step ExtractorOrchestrator)

        Pathway A runs through the normal Phase 5 infrastructure.
        Pathway B is wired here explicitly.
        """
        if self._extractor is None or self._neurochem is None:
            return

        # Build domain results from Phase 5 if available
        domain_results = self._extract_domain_results(result)
        emotion_profile = feedback.get("emotion_profile", {})

        if not domain_results:
            return

        try:
            osc_state = self._neurochem.get_oscillation_state()

            # Pathway B: Full 9-step ExtractorOrchestrator
            extractor_result = self._extractor.step(
                domain_results=domain_results,
                emotion_inputs=emotion_profile,
                current_oscillations=osc_state,
                dt=0.01,
            )

            # Apply stochastic modulation signals
            mod_signals = getattr(extractor_result, "modulation_signals", None)
            if mod_signals:
                self._neurochem.step(mod_signals)

            # Apply feedback params
            feedback_params = getattr(extractor_result, "feedback_params", None)
            if feedback_params:
                self._neurochem.apply_feedback(feedback_params)

            log.debug("M3: ExtractorOrchestrator stochastic pathway completed.")

        except Exception:
            log.debug("M3 stochastic pathway execution failed.", exc_info=True)

    def _track_emotional_cycle(
        self,
        feedback: Dict[str, Any],
        session: SessionState,
    ) -> None:
        """Track the curiosity-courage-discovery-joy cycle (Part 2 §3.3).

        Cycle states:
          exploring    → curious dominant, boost simulated opposition
          pivoting     → frustrated, strategy shift, NE pivot signal
          consolidating → joy/excited, discovery flag, reinforce approach

        State is stored on the session's metadata (if available) or
        tracked internally.
        """
        dominant = feedback.get("dominant_emotion", ("neutral", 0.0))
        dominant_name = dominant[0] if isinstance(dominant, tuple) else str(dominant)

        # Get current cycle state
        session_meta = getattr(session, "metadata", None)
        if session_meta is None or not isinstance(session_meta, dict):
            # Use a simple dict on session if metadata not available
            if not hasattr(session, "_m3_cycle"):
                session._m3_cycle = _CYCLE_EXPLORING  # type: ignore[attr-defined]
            cycle_state = session._m3_cycle  # type: ignore[attr-defined]
        else:
            cycle_state = session_meta.get(_M3_CYCLE_KEY, _CYCLE_EXPLORING)

        new_state = cycle_state

        if dominant_name == "curious" and cycle_state == _CYCLE_EXPLORING:
            # Boost DA-D3 further for novelty seeking
            if self._neurochem is not None:
                try:
                    self._neurochem.step({"DA": {"novelty_boost": 0.1}})
                except Exception:
                    pass
            log.debug("M3 cycle: exploring → curious → boosting novelty.")

        elif dominant_name == "frustrated" and cycle_state == _CYCLE_EXPLORING:
            # Strategy shift — change approach
            new_state = _CYCLE_PIVOTING
            # Temporary NE pivot signal for analytical recalibration
            if self._neurochem is not None:
                try:
                    self._neurochem.step({"NE": {"pivot_signal": 0.2}})
                except Exception:
                    pass
            log.info("M3 cycle: exploring → pivoting (frustration-triggered).")

        elif dominant_name in ("joy", "excited") and cycle_state in (
            _CYCLE_EXPLORING, _CYCLE_PIVOTING
        ):
            # Discovery!  Record strongly to learning log
            new_state = _CYCLE_CONSOLIDATING
            log.info("M3 cycle: %s → consolidating (discovery moment).", cycle_state)

        elif cycle_state == _CYCLE_CONSOLIDATING:
            # After consolidation, return to exploration
            new_state = _CYCLE_EXPLORING
            log.debug("M3 cycle: consolidating → exploring (cycle restart).")

        # Store updated cycle state
        if session_meta is not None and isinstance(session_meta, dict):
            session_meta[_M3_CYCLE_KEY] = new_state
        elif hasattr(session, "_m3_cycle"):
            session._m3_cycle = new_state  # type: ignore[attr-defined]

    def _extract_domain_results(self, result: Any) -> Dict[str, Any]:
        """Extract domain evaluation results from Phase 5 output.

        Returns dict suitable for ExtractorOrchestrator.step().
        """
        domain_results: Dict[str, Any] = {}

        if result.state and result.state.reward and result.state.reward.phase5_result:
            p5 = result.state.reward.phase5_result
            # Phase5Result may have domain_results or per-domain scores
            dr = getattr(p5, "domain_results", None)
            if isinstance(dr, dict):
                domain_results = dr
            else:
                # Reconstruct from available attributes
                for domain in ("logic", "ethics", "innovation", "human_attunement"):
                    score = getattr(p5, f"{domain}_score", None)
                    if score is not None:
                        domain_results[domain] = {"score": score}

        return domain_results

    def _check_human_claims(
        self,
        bundle: InputBundle,
        result: Any,
        feedback: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Human challenge logic (Part 4 §4.2).

        ZA-DOS actively checks human claims against established knowledge.
        When a claim contradicts known facts (via E1 contradiction detection
        or E4 fallacy detection), the contradiction is recorded as a
        contrast_challenge to be presented in the dialectic response.

        This is NOT adversarial — it's collaborative verification.
        ZA-DOS presents the contradiction as: "I learned X, but you said Y.
        Can you help me understand which is correct?"

        Returns
        -------
        List[Dict[str, Any]]
            Each dict: {"claim", "known_fact", "source", "confidence", "engine"}
        """
        challenges: List[Dict[str, Any]] = []

        engine_results: Dict[int, Dict[str, Any]] = {}
        if result is not None and hasattr(result, "state") and result.state:
            if result.state.dispatch:
                engine_results = result.state.dispatch.engine_results

        # E1 — Contradiction detection: human's claim vs. established knowledge
        e1 = engine_results.get(1, {})
        contradictions = e1.get("contradictions", [])
        if isinstance(contradictions, list):
            for c in contradictions:
                if not isinstance(c, dict):
                    continue
                challenges.append({
                    "claim": c.get("claim", bundle.raw_text[:100]),
                    "known_fact": c.get("existing_content", ""),
                    "source": c.get("source", "knowledge/lessons"),
                    "confidence": c.get("confidence", 0.5),
                    "engine": "E1_contradiction",
                })

        # E4 — Fallacy detection: if human's reasoning contains fallacy
        e4 = engine_results.get(4, {})
        fallacies = e4.get("fallacies", [])
        if isinstance(fallacies, list):
            for f in fallacies:
                if not isinstance(f, dict):
                    continue
                challenges.append({
                    "claim": f.get("statement", bundle.raw_text[:100]),
                    "known_fact": f.get("fallacy_type", ""),
                    "source": "logical_analysis",
                    "confidence": f.get("confidence", 0.5),
                    "engine": "E4_fallacy",
                })

        if challenges:
            log.info(
                "M3: Human challenge logic found %d contradiction(s) to discuss.",
                len(challenges),
            )

        return challenges

    def _extract_unsolved(self, result: Any, subject: str) -> List[UnsolvedQuestion]:
        """Extract unsolved questions from engine results.

        Looks for questions flagged by E26 (uncertainty patterns)
        or raised during dialectic exchange (E7/E14).
        """
        questions: List[UnsolvedQuestion] = []

        if result.state and result.state.dispatch:
            engine_results = result.state.dispatch.engine_results

            # E26 — uncertainty patterns
            e26 = engine_results.get(26, {})
            uncertainties = e26.get("unresolved", [])
            for u in uncertainties:
                text = u if isinstance(u, str) else str(u)
                q = self._unsolved_buffer.add(
                    question_text=text,
                    source_mode="M3",
                    source_context=subject,
                    urgency_score=0.6,
                    tags=[subject, "dialectic"],
                )
                questions.append(q)

        return questions


def _get_thinking_trace(result: Any) -> str:
    """Extract thinking trace from a PipelineResult safely."""
    try:
        if result is not None and hasattr(result, "state") and result.state:
            if result.state.thinking:
                return result.state.thinking.thinking_trace or ""
    except Exception:
        pass
    return ""
