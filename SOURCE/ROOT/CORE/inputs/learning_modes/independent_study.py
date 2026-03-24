"""
ZA-DOS v0.6 — M5: Independent Study (spec §3.3.5 + Part 2 §3.5 + Part 4 §6).

Self-directed, exploratory learning mode.  Interacts with the
UnsolvedBuffer and emphasises novel discovery and deep encoding.

Neurochem wiring (Part 2 §3.5):
  - Preset: max ACh-alpha7/M1 (attention), DA-D1 (goal salience),
    mild NE (alertness), GABA-A (noise suppression)
  - CRITICAL: E28 (Emotional Detection) is OFF in M5 — no human
    input to read emotions from.  Boredom/apathy are detected
    directly from NT dynamics:
      Boredom: DA-D3 dropping + low CB1 + low openness
      Apathy:  global DA collapse + GABA-B rise (fatigue + low motivation)
  - StudyAction responses: switch_material, study_break, mode_switch

Part 4 §6 additions:
  - Response suppression: M5 is autonomous, no response generated
  - E28 OFF flag: no human input → no emotion detection
  - LibraryIngestor stub: material chunking + AtomSpace linking
  - Scoped reads across 5 knowledge folders
  - Scoped writes: knowledge/lessons, library, notebook, academic_questions

Engine budget: 14 (T1+T2).
Risk emotions: boredom, apathy, confused.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from zados.core.inputs.learning_modes.base import LearningModePipeline
from zados.core.processes.emotional_landscape import get_emotional_preset
from zados.core.processes.subject_classifier import classify_subject_from_text
from zados.core.types import (
    InputBundle,
    LearningModeResult,
    SessionState,
    StudyAction,
    UnsolvedQuestion,
)

log = logging.getLogger(__name__)


class IndependentStudyPipeline(LearningModePipeline):
    """M5 — Independent Study: self-directed, exploratory learning."""

    mode_id = "M5"
    mode_number = 5

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> LearningModeResult:
        """Process a turn in M5 (Independent Study) mode.

        Stages (Part 4 §6):
          0. Setup — preset, engine resolve, E28 OFF flag, drift check
          1. Scoped memory contrast (5 knowledge folders)
          2. Engine dispatch
          3. VT thinking (no held-block: E28 OFF means no emotion profile)
          4. M5-specific: NT-based boredom/apathy detection, study risk
          5. LTMM write (scoped: knowledge/lessons, library, notebook)
          6. Unsolved question harvesting
          7. Response SUPPRESSED (autonomous mode)
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

        # E28 OFF in M5 — no human input to detect emotions from
        bundle.context_flags["e28_disabled"] = True
        bundle.context_flags["autonomous_mode"] = True

        self._check_drift(bundle)

        risks = self._check_risk_emotions(bundle)
        if risks:
            log.info("M5 risk emotions triggered (pre-pipeline): %s", risks)

        # ---- Stage 1: Scoped memory contrast ----
        if self._pipeline_scope is not None:
            bundle._pipeline_read_scope = self._pipeline_scope.read_scope  # type: ignore[attr-defined]

        # ---- Stage 2: Engine dispatch ----
        result = self._pipeline.process_turn(bundle, session)

        # ---- Stage 3: VT thinking ----
        # NOTE: E28 OFF → no emotion profile from human input.
        # Held-thinking-block check skipped (no emotion detection).
        # Feedback loop uses whatever emotion data is available
        # but the primary risk detection is _detect_study_risk_states().
        feedback = self._run_feedback_loop(bundle, result, session)

        # ---- Stage 4: M5-specific: NT-based risk detection ----
        study_action = None
        risk_state = self._detect_study_risk_states()
        if risk_state is not None:
            study_action = self._handle_study_risk(risk_state)
            log.info(
                "M5: Study risk '%s' detected — action: %s",
                risk_state, study_action.action if study_action else "none",
            )

        # ---- Stage 7: Response SUPPRESSED (Part 4 §6) ----
        # M5 autonomous mode — no response generated for human.
        # Internal processing still occurs but generate_response=False.
        if not self._config.generate_response:
            if hasattr(result, "response"):
                result.response = ""
            if hasattr(result, "final_answer"):
                result.final_answer = ""
            log.debug("M5: Response suppressed (autonomous mode).")

        # ---- Stage 8: Record + harvest ----
        self._record_learning(session, result, subject.value)
        unsolved = self._harvest_unsolved(result, subject.value)

        return LearningModeResult(
            mode_number=self.mode_number,
            pipeline_result=result,
            unsolved_questions=unsolved,
            study_action=study_action,
        )

    # ------------------------------------------------------------------
    # M5-specific: NT-based risk detection (Part 2 §3.5)
    # ------------------------------------------------------------------

    def _detect_study_risk_states(self) -> Optional[str]:
        """Detect boredom/apathy from NT dynamics without E28.

        E28 (Emotional Detection) is OFF in M5 because there is no
        human input to read emotions from.  Instead, we detect
        boredom and apathy directly from the neurochemical state:

        Boredom detection:
          DA-D3 receptor saturation dropping (novelty fading)
          + CB1 saturation dropping (schema flexibility declining)
          + low openness metric

        Apathy detection:
          High fatigue metric + low motivation metric
          (reflects global DA collapse + GABA-B rise)

        Returns
        -------
        str or None
            "boredom", "apathy", or None if no risk detected.
        """
        preset = get_emotional_preset(self.mode_id)
        if preset is None:
            return None

        # Compute metrics from current NT state
        metrics = self._step5_compute_metrics()
        if metrics is None:
            return None

        metrics_dict = self._get_metrics_dict(metrics)

        # Boredom detection: DA-D3 dropping + low novelty signal
        da_d3_sat = self._get_receptor_saturation("DA", "D3")
        cb1_sat = self._get_receptor_saturation("CB1", "CB1")
        openness = metrics_dict.get("openness", 0.5)

        boredom_score = (
            (1.0 - da_d3_sat) * 0.4
            + (1.0 - cb1_sat) * 0.3
            + (1.0 - openness) * 0.3
        )

        boredom_threshold = preset.risk_thresholds.get("boredom", 0.6)
        if boredom_score > boredom_threshold:
            log.info(
                "M5: Boredom detected — score=%.3f (threshold=%.2f), "
                "DA-D3_sat=%.3f, CB1_sat=%.3f, openness=%.3f",
                boredom_score, boredom_threshold, da_d3_sat, cb1_sat, openness,
            )
            return "boredom"

        # Apathy detection: global DA collapse + GABA-B rise
        fatigue = metrics_dict.get("fatigue", 0.0)
        motivation = metrics_dict.get("motivation", 0.5)
        apathy_threshold = preset.risk_thresholds.get("apathy", 0.5)

        if fatigue > 0.7 and motivation < 0.3:
            log.info(
                "M5: Apathy detected — fatigue=%.3f, motivation=%.3f "
                "(threshold=%.2f)",
                fatigue, motivation, apathy_threshold,
            )
            return "apathy"

        return None

    def _handle_study_risk(self, risk_state: str) -> Optional[StudyAction]:
        """Respond to boredom/apathy during independent study (Part 2 §3.5).

        Boredom → switch material (novelty depleted for current topic)
        Apathy  → study break (let NT state recover via pharmacodynamic decay)

        Parameters
        ----------
        risk_state : str
            "boredom" or "apathy".

        Returns
        -------
        StudyAction or None
        """
        if risk_state == "boredom":
            return StudyAction(
                action="switch_material",
                reason="novelty_depleted",
            )
        elif risk_state == "apathy":
            return StudyAction(
                action="study_break",
                reason="fatigue_high_motivation_low",
                duration_minutes=5,
            )
        return None

    # ------------------------------------------------------------------
    # Unsolved question harvesting
    # ------------------------------------------------------------------

    def _harvest_unsolved(self, result: Any, subject: str) -> List[UnsolvedQuestion]:
        """Extract unsolved questions from independent study results.

        Looks for uncertainty patterns (E26) and novel patterns (E19)
        that warrant further exploration.
        """
        questions: List[UnsolvedQuestion] = []

        if result.state and result.state.dispatch:
            engine_results = result.state.dispatch.engine_results

            # E26 — uncertainty patterns
            e26 = engine_results.get(26, {})
            for u in e26.get("unresolved", []):
                text = u if isinstance(u, str) else str(u)
                q = self._unsolved_buffer.add(
                    question_text=text,
                    source_mode="M5",
                    source_context=subject,
                    urgency_score=0.5,
                    tags=[subject, "independent_study"],
                )
                questions.append(q)

            # E19 — novel patterns that need deeper exploration
            e19 = engine_results.get(19, {})
            novel_patterns = [
                p for p in e19.get("patterns", [])
                if isinstance(p, dict) and p.get("status") == "CANDIDATE"
            ]
            for p in novel_patterns[:3]:  # cap at 3 questions per turn
                pattern_desc = p.get("description", str(p))
                q = self._unsolved_buffer.add(
                    question_text=f"Explore pattern: {pattern_desc}",
                    source_mode="M5",
                    source_context=subject,
                    urgency_score=0.4,
                    tags=[subject, "pattern_exploration"],
                )
                questions.append(q)

        return questions
