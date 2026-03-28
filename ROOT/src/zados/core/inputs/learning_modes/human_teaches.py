"""
ZA-DOS v0.6 — M1: Human Teaches (spec §3.3.1 + Part 2 §3.1 + Part 4 §2).

Receptive learning mode.  Detection engines are reframed to
OperationalMode.LEARNING so that contradictions are treated as
learning signals rather than adversarial attacks.

Neurochem wiring (Part 2 §3.1):
  - Preset: high ACh (encoding), mild DA-D1, GABA noise suppression,
    high OXT (social receptivity), low NE (reduced vigilance)
  - Confusion (>0.5) → temporary adversarial override for 1 turn
  - Overwhelm → E27/CRH detection → budget throttle
  - Joy (understanding clicks) → reward recording via E17

Part 4 §2 additions:
  - Stage 4: Clarifying-question generation (max 2 per turn)
  - Stage 3: Held thinking block check on emotion spike
  - Stage 5/6: scope-aware LTMM writes (knowledge/lessons, thoughts/)
  - T1* LEARNING reframe for Contradiction, Paradox, Socratic engines

Engine budget: 14 (T1+T2).
Risk emotions: frustrated, defensiveness, overwhelmed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from zados.core.inputs.learning_modes.base import LearningModePipeline
from zados.core.processes.subject_classifier import classify_subject_from_text
from zados.core.types import (
    InputBundle,
    LearningModeResult,
    SessionState,
)

log = logging.getLogger(__name__)


class HumanTeachesPipeline(LearningModePipeline):
    """M1 — Human Teaches: receptive, absorption-focused learning."""

    mode_id = "M1"
    mode_number = 1

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> LearningModeResult:
        """Process a turn in M1 (Human Teaches) mode.

        Stages (Part 4 §2):
          0. Setup — preset, engine resolve, LEARNING reframe, drift
          1. Scoped memory contrast (knowledge/lessons, library)
          2. Engine dispatch via AnswerPipeline
          3. VT thinking + held-thinking-block check
          4. M1-specific: question generation, emotional transitions
          5. LTMM write (scoped: knowledge/lessons, thoughts/)
          6. Question extraction
          7. Response generation
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

        # Set detection engines to LEARNING mode via context flags
        # (Part 4 §2.1: T1* LEARNING — comprehension, not adversarial)
        bundle.context_flags["operational_mode"] = True
        bundle.context_flags["learning_reframe"] = True

        self._check_drift(bundle)

        # Pre-pipeline risk check
        risks = self._check_risk_emotions(bundle)
        if risks:
            log.info("M1 risk emotions triggered (pre-pipeline): %s", risks)

        # ---- Stage 1: Scoped memory contrast ----
        # (handled internally by AnswerPipeline via scope on bundle)
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

        # ---- Stage 4: M1-specific transitions + question gen ----
        suggest_mode = self._handle_m1_transitions(
            bundle, result, feedback, session
        )
        questions = self._generate_clarifying_questions(
            bundle, result, feedback
        )

        # ---- Stage 5-7: LTMM write, response (handled by pipeline) ----

        # ---- Stage 8: Record learning ----
        self._record_learning(session, result, subject.value)

        return LearningModeResult(
            mode_number=self.mode_number,
            pipeline_result=result,
            suggest_mode_change=suggest_mode,
            held_thinking_blocks=held_block_ids,
            unsolved_questions=questions,
        )

    def _handle_m1_transitions(
        self,
        bundle: InputBundle,
        result: Any,
        feedback: Dict[str, Any],
        session: SessionState,
    ) -> str:
        """Handle M1-specific emotional transitions (Part 2 §3.1).

        - Confused (>0.5): temporary adversarial mode for contradiction
          detection (1 turn override).
        - Overwhelmed: E27/CRH detection → budget throttle.
        - Joy (understanding clicks): record positive outcome to E17.

        Returns
        -------
        str
            Suggested mode change (empty string if none).
        """
        emotion_profile = feedback.get("emotion_profile", {})
        homeostatic_result = feedback.get("homeostatic_result")

        # CONFUSION → temporary adversarial override
        if emotion_profile.get("confused", 0.0) > 0.5:
            log.info(
                "M1: Confusion detected (%.2f) — enabling adversarial "
                "contradiction detection for this turn.",
                emotion_profile["confused"],
            )
            # Override: remove learning reframe for contradiction detection
            # This allows E1 to run in full NORMAL mode temporarily
            bundle.context_flags["learning_reframe"] = False
            bundle.context_flags["confusion_override"] = True

        # OVERWHELMED → budget throttle via E27/CRH detection
        if homeostatic_result is not None:
            has_crh_violation = _check_violation(homeostatic_result, "CRH", "elevated")
            if has_crh_violation:
                log.info(
                    "M1: CRH elevated — throttling engine budget by 4."
                )
                # Reduce engine weights to simulate budget throttle
                for key in list(bundle.engine_weights.keys()):
                    if bundle.engine_weights[key] <= 0.5:
                        bundle.engine_weights[key] = 0.0

        # Overwhelm risk emotion
        if emotion_profile.get("overwhelmed", 0.0) > 0.7:
            log.info("M1: Overwhelm detected — consider reducing load.")

        # JOY → record positive outcome (understanding clicks)
        if emotion_profile.get("joy", 0.0) > 0.5 or emotion_profile.get("excited", 0.0) > 0.5:
            log.info(
                "M1: Joy/excitement detected — recording positive learning outcome."
            )
            if self._e17 is not None:
                try:
                    self._e17.record_positive_outcome(
                        context="m1_understanding",
                        strength=emotion_profile.get("joy", 0.0),
                    )
                except Exception:
                    log.debug("E17 positive outcome recording failed.", exc_info=True)

        # Bidirectional loop closure (Part 2 §3.1)
        eval_results = feedback.get("eval_results", {})
        if eval_results:
            self._step8_feedback_to_neurochem(eval_results)

        return ""  # M1 doesn't suggest mode changes

    def _generate_clarifying_questions(
        self,
        bundle: InputBundle,
        result: Any,
        feedback: Dict[str, Any],
    ) -> List[Any]:
        """Generate clarifying questions (Part 4 §2.3, max 2 per turn).

        M1 generates questions when:
          - Confusion is detected (>0.3) — seek clarification
          - Novel concepts found — explore depth
          - Engine results indicate gaps

        These are stored as unsolved questions in thoughts/general_questions
        and optionally surfaced in the response.

        Returns
        -------
        List[UnsolvedQuestion]
        """
        questions: List[Any] = []
        max_q = self._config.max_questions_per_turn  # M1 = 2

        if max_q <= 0:
            return questions

        emotion_profile = feedback.get("emotion_profile", {})
        confusion_level = emotion_profile.get("confused", 0.0)

        # Extract engine results for gap detection
        engine_results: Dict[int, Dict[str, Any]] = {}
        if result is not None and hasattr(result, "state") and result.state:
            if result.state.dispatch:
                engine_results = result.state.dispatch.engine_results

        # Confusion-based questions
        if confusion_level > 0.3 and len(questions) < max_q:
            from zados.core.types import UnsolvedQuestion
            q = UnsolvedQuestion(
                question_text=f"Clarification needed on: {bundle.raw_text[:100]}",
                source_mode="M1",
                source_context=f"confusion={confusion_level:.2f}",
                urgency_score=min(1.0, confusion_level + 0.2),
                scope_tag="general",
            )
            questions.append(q)
            self._unsolved_buffer.add(q)

        # Novelty-based questions (from E19 pattern identification)
        e19 = engine_results.get(19, {})
        novel_patterns = [
            p for p in e19.get("patterns", [])
            if isinstance(p, dict) and p.get("status") == "CANDIDATE"
        ]
        if novel_patterns and len(questions) < max_q:
            from zados.core.types import UnsolvedQuestion
            pattern_label = novel_patterns[0].get("label", "unknown pattern")
            q = UnsolvedQuestion(
                question_text=f"Explore novel pattern: {pattern_label}",
                source_mode="M1",
                source_context="novelty_detection",
                urgency_score=0.6,
                scope_tag="general",
            )
            questions.append(q)
            self._unsolved_buffer.add(q)

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


def _check_violation(homeostatic_result: Any, nt_key: str, violation_type: str) -> bool:
    """Check if a homeostatic result contains a specific NT violation."""
    try:
        violations = getattr(homeostatic_result, "violations", [])
        if isinstance(violations, list):
            for v in violations:
                if isinstance(v, dict):
                    if v.get("nt") == nt_key and v.get("type") == violation_type:
                        return True
                elif hasattr(v, "nt") and hasattr(v, "type"):
                    if v.nt == nt_key and v.type == violation_type:
                        return True
        elif isinstance(violations, dict):
            nt_v = violations.get(nt_key, {})
            if isinstance(nt_v, dict) and nt_v.get("type") == violation_type:
                return True
    except Exception:
        pass
    return False
