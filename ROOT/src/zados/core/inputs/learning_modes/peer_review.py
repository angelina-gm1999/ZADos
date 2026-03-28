"""
ZA-DOS v0.6 — M2: Peer Review (spec §3.3.2 + Part 2 §3.2 + Part 4 §3).

Critical analysis mode.  Detection engines at full strength.
Emphasis on finding flaws, inconsistencies, and logical gaps.

Neurochem wiring (Part 2 §3.2):
  - Preset: high NE (vigilance), high ACh (deep attention),
    5-HT1A buffering, mild cortisol alertness
  - Regret pathway (>0.4): promote retroactive alignment to T1,
    negative RPE via E17, correction tag in learning log
  - Validation pathway (valued+proud >0.5): positive RPE,
    reflective identity update
  - Shame spiral detection: cortisol elevated 3+ turns →
    containment + suggest M1

Part 4 §3 additions:
  - Two-pass memory contrast: Pass A (general LTMM) + Pass B
    (retroactive against own prior outputs)
  - Stage 5b: CoreMemoryUpdateGate — corrections staged as
    PendingCoreMemoryUpdate, NOT applied mid-conversation
  - Relief tracking: relief after correction → positive RPE
  - Held thinking block check on identity-relevant emotions

Engine budget: 16 (T1+T2).
Risk emotions: ashamed, contempt, dismissiveness.
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
    PendingCoreMemoryUpdate,
    SessionState,
)

log = logging.getLogger(__name__)

# Shame spiral threshold — elevated cortisol for N consecutive turns
_SHAME_SPIRAL_THRESHOLD = 3


class PeerReviewPipeline(LearningModePipeline):
    """M2 — Peer Review: critical, rigorous analytical learning."""

    mode_id = "M2"
    mode_number = 2

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._shame_spiral_counter: int = 0

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> LearningModeResult:
        """Process a turn in M2 (Peer Review) mode.

        Stages (Part 4 §3):
          0. Setup — preset, engine resolve, drift check
          1. Two-pass memory contrast (Pass A general + Pass B retroactive)
          2. Engine dispatch via AnswerPipeline
          3. VT thinking + held-thinking-block check
          4. M2-specific: regret/validation/shame transitions
          5. LTMM write + CoreMemoryUpdateGate (staged, not applied)
          6. (no question extraction in M2)
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
        self._check_drift(bundle)

        risks = self._check_risk_emotions(bundle)
        if risks:
            log.info("M2 risk emotions triggered (pre-pipeline): %s", risks)

        # ---- Stage 1: Scoped memory contrast (two-pass) ----
        # Pass A: general LTMM via scope (identity/core, conclusions, journal)
        if self._pipeline_scope is not None:
            bundle._pipeline_read_scope = self._pipeline_scope.read_scope  # type: ignore[attr-defined]

        # Mark bundle for retroactive contrast (Pass B) — downstream
        # MemoryContrast will check own prior outputs if this flag is set
        if self._config.use_retroactive_contrast:
            bundle.context_flags["retroactive_contrast"] = True

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

        # ---- Stage 4: M2-specific transitions ----
        suggest_mode = self._handle_m2_transitions(
            bundle, result, feedback, session
        )

        # ---- Stage 5b: Core Memory Update Gate ----
        pending_updates = self._check_core_memory_corrections(
            bundle, result, feedback, session,
        )

        # ---- Stage 5: Relief tracking (post-correction positive RPE) ----
        self._check_relief_after_correction(emotion_profile)

        # ---- Stage 8: Record learning ----
        self._record_learning(session, result, subject.value)

        return LearningModeResult(
            mode_number=self.mode_number,
            pipeline_result=result,
            suggest_mode_change=suggest_mode,
            held_thinking_blocks=held_block_ids,
            pending_core_updates=pending_updates,
        )

    def _handle_m2_transitions(
        self,
        bundle: InputBundle,
        result: Any,
        feedback: Dict[str, Any],
        session: SessionState,
    ) -> str:
        """Handle M2-specific emotional transitions (Part 2 §3.2).

        Three pathways:
          1. Regret (>0.4): retroactive alignment promotion, negative RPE
          2. Validation (valued+proud >0.5): positive RPE, identity update
          3. Shame spiral (cortisol elevated 3+ turns): containment + M1

        Returns
        -------
        str
            Suggested mode change (empty string if none).
        """
        emotion_profile = feedback.get("emotion_profile", {})
        homeostatic_result = feedback.get("homeostatic_result")
        metrics = feedback.get("metrics")

        # ──────────────────────────────────────────────
        # REGRET PATHWAY (Part 2 §3.2)
        # ──────────────────────────────────────────────
        regret_strength = emotion_profile.get("regret", 0.0)
        if regret_strength > 0.4:
            log.info(
                "M2: Regret pathway activated (%.2f) — promoting retroactive "
                "alignment, recording negative RPE.",
                regret_strength,
            )
            # Promote retroactive alignment engine to T1 if available
            # (this is a conceptual promotion — reflected in engine weights)
            for key in bundle.engine_weights:
                if "retroactive" in key.lower() or "alignment" in key.lower():
                    bundle.engine_weights[key] = 1.0

            # Record negative prediction error via E17
            if self._e17 is not None:
                try:
                    self._e17.record_negative_prediction_error(
                        context="m2_regret_correction",
                        magnitude=regret_strength,
                    )
                except AttributeError:
                    # E17 may not have this specific method — try generic
                    try:
                        self._e17.process({"rpe": -regret_strength})
                    except Exception:
                        pass
                except Exception:
                    log.debug("E17 negative RPE recording failed.", exc_info=True)

        # ──────────────────────────────────────────────
        # VALIDATION PATHWAY (Part 2 §3.2)
        # ──────────────────────────────────────────────
        valued_score = emotion_profile.get("valued", 0.0)
        proud_score = emotion_profile.get("proud", 0.0)
        if valued_score + proud_score > 0.5:
            log.info(
                "M2: Validation pathway activated (valued=%.2f + proud=%.2f) "
                "— recording positive validation.",
                valued_score, proud_score,
            )
            # Record positive validation via E17
            if self._e17 is not None:
                try:
                    self._e17.record_positive_outcome(
                        context="m2_peer_validation",
                        strength=valued_score + proud_score,
                    )
                except Exception:
                    log.debug("E17 positive validation recording failed.", exc_info=True)

            # Reset shame spiral counter on positive feedback
            self._shame_spiral_counter = 0

        # ──────────────────────────────────────────────
        # SHAME SPIRAL DETECTION (Part 2 §3.2)
        # ──────────────────────────────────────────────
        ashamed_score = emotion_profile.get("ashamed", 0.0)
        cortisol_elevated = False
        if homeostatic_result is not None:
            cortisol_elevated = _check_cortisol_elevated(homeostatic_result)

        # Also check via metrics
        if metrics is not None:
            anxiety = getattr(metrics, "anxiety", 0.0)
            if anxiety > 0.7:
                cortisol_elevated = True

        # Track shame spiral
        if cortisol_elevated or ashamed_score > 0.4:
            self._shame_spiral_counter += 1
            log.info(
                "M2: Shame spiral counter incremented to %d.",
                self._shame_spiral_counter,
            )
        else:
            # Decay counter if no stress this turn
            self._shame_spiral_counter = max(0, self._shame_spiral_counter - 1)

        # Shame spiral intervention
        if self._shame_spiral_counter >= _SHAME_SPIRAL_THRESHOLD:
            log.warning(
                "M2: Shame spiral detected (%d consecutive elevated turns) "
                "— recommending switch to M1 (Human Teaches).",
                self._shame_spiral_counter,
            )
            # Homeostatic containment if available
            if self._e27 is not None:
                try:
                    self._e27.force_containment()
                except AttributeError:
                    # E27 may not have force_containment — try generic containment
                    try:
                        self._e27.process({"containment_request": True})
                    except Exception:
                        pass
                except Exception:
                    log.debug("E27 containment request failed.", exc_info=True)

            self._shame_spiral_counter = 0
            return "M1"  # Suggest softer interaction mode

        return ""  # No mode change

    def _check_core_memory_corrections(
        self,
        bundle: InputBundle,
        result: Any,
        feedback: Dict[str, Any],
        session: SessionState,
    ) -> List[PendingCoreMemoryUpdate]:
        """Stage 5b: Check if peer review flagged core memory corrections.

        Part 4 §3.2 — corrections to identity/core entries are staged
        as PendingCoreMemoryUpdate, NOT applied mid-conversation.  The
        CoreMemoryUpdateGate engine validates them later during Homework
        or Reflective mode.

        Returns list of pending updates (also stored on session).
        """
        pending: List[PendingCoreMemoryUpdate] = []

        # Check engine results for contradiction detections against identity
        engine_results: Dict[int, Dict[str, Any]] = {}
        if result is not None and hasattr(result, "state") and result.state:
            if result.state.dispatch:
                engine_results = result.state.dispatch.engine_results

        # E1 — Contradiction detection results
        e1 = engine_results.get(1, {})
        contradictions = e1.get("contradictions", [])
        if isinstance(contradictions, list):
            for c in contradictions:
                if not isinstance(c, dict):
                    continue
                # Only flag contradictions against identity content
                target = c.get("target_source", "")
                if "identity" not in target.lower() and "core" not in target.lower():
                    continue

                emotion_profile = feedback.get("emotion_profile", {})
                update = PendingCoreMemoryUpdate(
                    core_memory_key=c.get("target_id", ""),
                    current_value=c.get("existing_content", ""),
                    proposed_value=c.get("correction_content", bundle.raw_text[:200]),
                    correction_source=f"m2_peer_review_e1",
                    correction_session_id=getattr(session, "session_id", ""),
                    emotion_snapshot={
                        k: v for k, v in emotion_profile.items() if v > 0.1
                    },
                    confidence=c.get("confidence", 0.5),
                )
                pending.append(update)
                log.info(
                    "M2: Core memory correction staged (update_id=%s, key=%s)",
                    update.update_id, update.core_memory_key,
                )

        return pending

    def _check_relief_after_correction(
        self, emotion_profile: Dict[str, float],
    ) -> None:
        """Track relief after correction — positive RPE signal.

        Part 4 §3.2: When ZA-DOS feels relief after accepting a correction
        (regret → relief transition), record positive learning outcome.
        """
        relief = emotion_profile.get("relief", 0.0)
        if relief > 0.3 and self._e17 is not None:
            try:
                self._e17.record_positive_outcome(
                    context="m2_correction_relief",
                    strength=relief,
                )
                log.info("M2: Relief after correction (%.2f) → positive RPE.", relief)
            except Exception:
                log.debug("E17 relief recording failed.", exc_info=True)


def _get_thinking_trace(result: Any) -> str:
    """Extract thinking trace from a PipelineResult safely."""
    try:
        if result is not None and hasattr(result, "state") and result.state:
            if result.state.thinking:
                return result.state.thinking.thinking_trace or ""
    except Exception:
        pass
    return ""


def _check_cortisol_elevated(homeostatic_result: Any) -> bool:
    """Check if homeostatic result indicates elevated cortisol/CRH."""
    try:
        violations = getattr(homeostatic_result, "violations", [])
        if isinstance(violations, list):
            for v in violations:
                if isinstance(v, dict):
                    nt = v.get("nt", "").lower()
                    if nt in ("cor", "crh", "cortisol"):
                        return True
                elif hasattr(v, "nt"):
                    if v.nt.lower() in ("cor", "crh", "cortisol"):
                        return True
        elif isinstance(violations, dict):
            for key in ("cor", "crh", "cortisol", "COR", "CRH"):
                if key in violations:
                    return True
    except Exception:
        pass

    # Also check bound flags if available
    try:
        bound_flags = getattr(homeostatic_result, "bound_flags", {})
        if isinstance(bound_flags, dict):
            for key in ("cor", "crh", "cortisol"):
                if bound_flags.get(key, {}).get("elevated", False):
                    return True
    except Exception:
        pass

    return False
