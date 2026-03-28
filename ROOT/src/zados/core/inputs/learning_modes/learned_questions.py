"""
ZA-DOS v0.6 — M4: Learned Questions (spec §3.3.4 + Part 2 §3.4 + Part 4 §5).

Reflective, question-driven mode.  Interacts with the UnsolvedBuffer
to surface and explore questions that emerged during previous learning
sessions.

Neurochem wiring (Part 2 §3.4):
  - Preset: maximum DA-D3 (curiosity drive), 5-HT2A (abstract space),
    ACh (attention to detail), slightly reduced NE (less urgency)
  - Question quality influenced by NT state / NeurochemicalMetrics:
      openness > 0.7  → exploratory questions ("What if...")
      precision > 0.7 → targeted questions ("Why does X contradict Y?")
      anxiety > 0.5   → clarifying questions ("Can you explain X?")

Part 4 §5 additions:
  - Sub-mode routing: automatic/prompted/clustered selection
  - Dream threshold: questions with stagnation_count >= 5 flagged
    as dream_candidates for Dream Mode offline processing
  - Question clustering: group related unsolved questions
  - Held thinking block check on emotion spikes
  - Question surfacing reads from thoughts/unsolved_buffer

Engine budget: 12 (T1+T2).
Risk emotions: rumination, apathy, stagnation.
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
    UnsolvedQuestion,
)

log = logging.getLogger(__name__)

# Dream threshold — questions with stagnation above this are flagged
# for Dream Mode offline processing (Part 4 §5.2).
_DREAM_STAGNATION_THRESHOLD = 5


class LearnedQuestionsPipeline(LearningModePipeline):
    """M4 — Learned Questions: reflective, question-oriented learning."""

    mode_id = "M4"
    mode_number = 4

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> LearningModeResult:
        """Process a turn in M4 (Learned Questions) mode.

        Stages (Part 4 §5):
          0. Setup — preset, engine resolve, drift check
          1. Scoped memory contrast (knowledge + unsolved_buffer reads)
          2. Engine dispatch
          3. VT thinking + held-thinking-block check
          4. M4-specific: sub-mode routing, question style, NT-based
             question selection, dream threshold flagging
          5. LTMM write (scoped: knowledge/lessons, knowledge_maps,
             thoughts/unsolved_buffer)
          6. Question extraction (max 1 focused question per turn)
          7. Response generation (abbreviated depth)
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

        # ---- Stage 1: Scoped memory contrast ----
        if self._pipeline_scope is not None:
            bundle._pipeline_read_scope = self._pipeline_scope.read_scope  # type: ignore[attr-defined]

        # ---- Sub-mode routing (Part 4 §5.1) ----
        # Select question via sub-mode: automatic / prompted / clustered
        selected_question = self._submode_route_question(bundle, session)

        # Determine question style from NT state (Part 2 §3.4)
        question_style = self._determine_question_style()
        log.info("M4: Question style determined as '%s'.", question_style)
        bundle.context_flags["question_style"] = True
        bundle.context_flags[f"style_{question_style}"] = True

        self._check_drift(bundle)

        risks = self._check_risk_emotions(bundle)
        if risks:
            log.info("M4 risk emotions triggered (pre-pipeline): %s", risks)

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

        # ---- Stage 4: Dream threshold flagging ----
        dream_candidates = self._flag_dream_candidates()

        # ---- Stage 5-7: LTMM write, response ----
        # (handled by pipeline, abbreviated depth per config)

        # ---- Stage 8: Record + update buffer ----
        self._record_learning(session, result, subject.value)

        unsolved: List[UnsolvedQuestion] = []
        if selected_question is not None:
            self._unsolved_buffer.mark_attempted(
                selected_question.question_id,
                partial_answer=getattr(result, "final_answer", "")[:200],
            )
            unsolved.append(selected_question)

        return LearningModeResult(
            mode_number=self.mode_number,
            pipeline_result=result,
            unsolved_questions=unsolved,
            held_thinking_blocks=held_block_ids,
            dream_candidates=dream_candidates,
        )

    def _submode_route_question(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> Optional[UnsolvedQuestion]:
        """Sub-mode routing (Part 4 §5.1).

        Three sub-modes:
          - Automatic: user says "next" / empty → buffer.select_next()
          - Prompted: user provides specific question → skip buffer
          - Clustered: group related questions by subject/domain

        Returns the selected question or None (if user provided their own).
        """
        user_text = bundle.raw_text.strip().lower()
        is_auto_prompt = not user_text or user_text in (
            "next", "continue", "next question", "go on",
        )

        if is_auto_prompt:
            # Automatic sub-mode: priority-weighted selection from buffer
            selected = self._unsolved_buffer.select_next()
            if selected is not None:
                bundle.raw_text = selected.question_text
                log.info(
                    "M4 auto-selected question: %s (priority=%.2f, stagnation=%d)",
                    selected.question_id, selected.urgency_score,
                    getattr(selected, "resolution_attempts", 0),
                )
            return selected
        else:
            # Prompted sub-mode: user has their own question
            log.debug("M4: Prompted sub-mode — user provided question directly.")
            return None

    def _flag_dream_candidates(self) -> List[str]:
        """Flag questions with stagnation >= threshold as dream candidates.

        Part 4 §5.2: Questions that have been attempted multiple times
        without resolution are flagged for Dream Mode offline processing
        where the NT dynamics and associative network can produce novel
        connections.

        Returns list of question IDs flagged.
        """
        candidates: List[str] = []

        try:
            all_questions = self._unsolved_buffer.get_all()
        except (AttributeError, TypeError):
            return candidates

        for q in all_questions:
            attempts = getattr(q, "resolution_attempts", 0)
            if attempts >= _DREAM_STAGNATION_THRESHOLD and not q.resolved:
                candidates.append(q.question_id)
                log.info(
                    "M4: Question %s flagged for Dream Mode "
                    "(stagnation=%d >= %d).",
                    q.question_id, attempts, _DREAM_STAGNATION_THRESHOLD,
                )

        return candidates

    def _determine_question_style(self) -> str:
        """Determine question style from NeurochemicalMetrics (Part 2 §3.4).

        High openness  (>0.7) → exploratory: "What if...", "How does X relate to Y?"
        High precision (>0.7) → targeted:    "Why does X contradict Y?"
        High anxiety   (>0.5) → clarifying:  "Can you explain X more simply?"

        Returns
        -------
        str
            One of "exploratory", "targeted", "clarifying", "balanced".
        """
        metrics = self._step5_compute_metrics()
        if metrics is None:
            return "balanced"

        metrics_dict = self._get_metrics_dict(metrics)

        openness = metrics_dict.get("openness", 0.5)
        precision = metrics_dict.get("precision", 0.5)
        anxiety = metrics_dict.get("anxiety", 0.0)

        # Priority: anxiety check first (high anxiety overrides other states)
        if anxiety > 0.5:
            return "clarifying"
        elif openness > 0.7:
            return "exploratory"
        elif precision > 0.7:
            return "targeted"

        return "balanced"


def _get_thinking_trace(result: Any) -> str:
    """Extract thinking trace from a PipelineResult safely."""
    try:
        if result is not None and hasattr(result, "state") and result.state:
            if result.state.thinking:
                return result.state.thinking.thinking_trace or ""
    except Exception:
        pass
    return ""
