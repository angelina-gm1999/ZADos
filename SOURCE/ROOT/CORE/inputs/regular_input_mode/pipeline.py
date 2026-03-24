"""
ZA-DOS v0.6 — Regular Input Pipeline (spec §3.2).

Wraps AnswerPipeline with:
  1. E23 intent classification → PipelineDepthConfig
  2. EngineTier → engine_weights application
  3. Drift detection via ContextAnchorManager
  4. Delegation to AnswerPipeline.process_turn()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from zados.core.inputs.regular_input_mode.intent_adapter import adapt_intent_to_depth
from zados.core.processes.context_anchor import ContextAnchorManager
from zados.core.processes.engine_toolkit import EngineToolkit
from zados.core.processes.subject_classifier import classify_subject_from_text
from zados.core.types import (
    InputBundle,
    PipelineResult,
    SessionState,
    SubjectCategory,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent category → reward profile
# Fine-grained selection for regular input based on E23 intent + mission_briefing.
# ---------------------------------------------------------------------------

_INTENT_TO_PROFILE: dict = {
    "connection":     "receptive_learning",
    "challenge":      "critical_review",
    "exploration":    "curiosity_driven",
    "discharge":      "receptive_learning",
    "pragmatic":      "regular_input",
    "symbolic":       "reflective_synthesis",
    "defensive":      "critical_review",
    "disintegration": "regular_input",
}

# Mission briefing keywords that override the intent-derived profile
_BRIEFING_KEYWORD_OVERRIDES: list = [
    (["study", "learn", "homework", "course"], "receptive_learning"),
    (["review", "critique", "evaluate", "assess"], "critical_review"),
    (["explore", "curious", "wonder", "hypothetical"], "curiosity_driven"),
    (["reflect", "think about", "introspect", "self"], "reflective_synthesis"),
    (["creative", "write", "story", "imagination"], "curiosity_driven"),
]


def _profile_from_intent(intent_result: Any, mission_briefing: str) -> str:
    """Derive the fine-grained reward profile from E23 + mission briefing.

    Mission briefing keywords take precedence over intent category when
    there is a strong keyword match, so the user's session context acts as
    a persistent background orientation.
    """
    # Mission briefing keyword override
    if mission_briefing:
        briefing_lower = mission_briefing.lower()
        for keywords, profile in _BRIEFING_KEYWORD_OVERRIDES:
            if any(kw in briefing_lower for kw in keywords):
                return profile

    # Intent category mapping
    if intent_result is None:
        return "regular_input"

    category = None
    if hasattr(intent_result, "intent_category"):
        cat = intent_result.intent_category
        category = cat.value.lower() if hasattr(cat, "value") else str(cat).lower()
    elif hasattr(intent_result, "dominant_intent"):
        category = str(intent_result.dominant_intent).lower()

    return _INTENT_TO_PROFILE.get(category or "", "regular_input")


class RegularInputPipeline:
    """Standard input processing — wraps AnswerPipeline with intent-depth tuning.

    Parameters
    ----------
    answer_pipeline : AnswerPipeline
        The v0.5 pipeline to delegate to.
    context_manager : ContextAnchorManager, optional
        For drift detection.
    engines : dict, optional
        engine_number → engine instance (needed for E23 access).
    general_question_store : GeneralQuestionStore, optional
        LTMM store for non-academic questions.  Low-confidence answers
        are written here for later revisiting.
    """

    # Low-confidence threshold — answers below this trigger a question write
    _LOW_CONFIDENCE_THRESHOLD: float = 0.4

    def __init__(
        self,
        answer_pipeline: Any,
        context_manager: Optional[ContextAnchorManager] = None,
        engines: Optional[Dict[int, Any]] = None,
        general_question_store: Any = None,
    ) -> None:
        self._pipeline = answer_pipeline
        self._context = context_manager or ContextAnchorManager()
        self._engines = engines or {}
        self._toolkit = EngineToolkit()
        self._general_question_store = general_question_store

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> PipelineResult:
        """Process a regular input turn.

        Steps:
          1. Run E23 (if available) to get intent classification.
          2. Adapt intent → PipelineDepthConfig.
          3. Classify subject → SubjectCategory.
          4. Resolve engine tiers → engine_weights on bundle.
          5. Check drift.
          6. Delegate to AnswerPipeline.process_turn().

        Parameters
        ----------
        bundle : InputBundle
        session : SessionState

        Returns
        -------
        PipelineResult
        """
        # Step 1: Run E23 for intent classification (if available)
        intent_result = self._run_e23(bundle.raw_text)

        # Step 1b: Determine reward profile from intent + mission briefing
        mission_briefing = str(
            getattr(bundle, "mission_briefing", None)
            or getattr(session, "mission_briefing", None)
            or ""
        )
        reward_profile = _profile_from_intent(intent_result, mission_briefing)
        session.reward_profile_name = reward_profile
        log.debug("Regular pipeline reward_profile: %s", reward_profile)

        # Step 2: Adapt intent to depth config
        depth_config = adapt_intent_to_depth(intent_result)
        log.debug(
            "Regular pipeline depth: style=%s, engine_cap=%d, thinking_budget=%d",
            depth_config.phase6_response_style,
            depth_config.phase3_engine_count_cap,
            depth_config.phase4_thinking_token_budget,
        )

        # Step 3: Classify subject
        subject = classify_subject_from_text(bundle.raw_text)

        # Step 4: Resolve engine tiers and apply as weights
        tiers = self._toolkit.resolve("regular", subject)
        weights = self._toolkit.tiers_to_weights_by_id(tiers)
        bundle.engine_weights.update(weights)

        # Step 5: Check drift (if anchor is active)
        if self._context.active_anchor is not None:
            if self._context.has_drifted(bundle.raw_text):
                log.info("Context drift detected in regular mode — re-anchoring.")
                self._context.create_anchor(
                    raw_text=bundle.raw_text,
                    subject_hint=subject.value,
                    intent_prior=getattr(intent_result, "dominant_intent", ""),
                )

        # Step 6: Delegate to AnswerPipeline
        result = self._pipeline.process_turn(bundle, session)

        # Step 7: Extract low-confidence answers as questions for later revisiting
        self._extract_low_confidence_questions(result, bundle, session)

        return result

    def _run_e23(self, raw_text: str) -> Any:
        """Run E23 IntentionMapEngine if available.

        Returns IntentionMapResult or None.
        """
        e23 = self._engines.get(23)
        if e23 is None:
            return None

        try:
            result = e23.process({"text": raw_text})
            return result
        except Exception:
            log.debug("E23 intent classification failed in regular pipeline.")
            return None

    def _extract_low_confidence_questions(
        self,
        result: PipelineResult,
        bundle: InputBundle,
        session: SessionState,
    ) -> None:
        """Write low-confidence answers to GeneralQuestionStore for revisiting.

        When the pipeline produces an answer with confidence below the
        threshold, the user's question is captured as a GeneralQuestion
        so it can be revisited in M4 (Learned Questions) or during
        self-reflective processing.
        """
        if self._general_question_store is None:
            return

        # Extract confidence from result
        confidence = getattr(result, "confidence", None)
        if confidence is None:
            state = getattr(result, "state", None)
            confidence = getattr(state, "answer_confidence", None) if state else None
        if confidence is None or confidence >= self._LOW_CONFIDENCE_THRESHOLD:
            return

        # The input text is likely a question — capture it
        raw = bundle.raw_text.strip()
        if not raw:
            return

        try:
            from zados.core.tags import T
            from zados.memory.long_term.thoughts.types import GeneralQuestion

            gq = GeneralQuestion(
                formulation=raw[:500],
                source="low_confidence_answer",
                domain_hint=None,
                priority=max(0.3, 1.0 - confidence),  # lower confidence → higher priority
                tags=[
                    T.pipeline("regular_input"),
                    T.mode("normal"),
                    T.origin("general"),
                ],
            )
            self._general_question_store.write(gq)
            log.debug(
                "Low-confidence answer (%.2f) → GeneralQuestion %s",
                confidence, gq.question_id,
            )
        except Exception:
            log.debug(
                "GeneralQuestionStore write failed in regular pipeline.",
                exc_info=True,
            )
