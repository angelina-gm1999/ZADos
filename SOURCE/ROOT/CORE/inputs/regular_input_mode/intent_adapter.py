"""
ZA-DOS v0.6 — Intent Adapter (spec §2.8 bridge).

Adapts an IntentionMapResult (from E23) into a PipelineDepthConfig
that tunes how deeply each pipeline phase processes the input.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from zados.core.processes.intent_pipeline_optimizer import (
    DEFAULT_PIPELINE_PROFILE,
    get_pipeline_profile,
)
from zados.core.types import PipelineDepthConfig

log = logging.getLogger(__name__)

# Intent categories that E23 may report (mapped to pipeline profiles)
_INTENT_CATEGORY_MAP = {
    "social":       "connection",
    "emotional":    "connection",
    "connection":   "connection",
    "adversarial":  "challenge",
    "testing":      "challenge",
    "challenge":    "challenge",
    "curious":      "exploration",
    "inquiry":      "exploration",
    "exploration":  "exploration",
    "venting":      "discharge",
    "emotional_release": "discharge",
    "discharge":    "discharge",
    "practical":    "pragmatic",
    "task":         "pragmatic",
    "pragmatic":    "pragmatic",
    "abstract":     "symbolic",
    "metaphorical": "symbolic",
    "symbolic":     "symbolic",
    "guarded":      "defensive",
    "protective":   "defensive",
    "defensive":    "defensive",
    "crisis":       "disintegration",
    "destabilised": "disintegration",
    "disintegration": "disintegration",
}

# Containment profile — forced when disintegration is detected
_CONTAINMENT_PROFILE = PipelineDepthConfig(
    perception_depth=0.5,
    semiotics_depth=0.5,
    emotion_detection_sensitivity=1.0,
    phase1_depth=0.5,
    phase3_engine_count_cap=8,
    phase4_thinking_token_budget=128,
    phase5_reward_thoroughness=0.3,
    phase6_response_style="containment",
)


def adapt_intent_to_depth(
    intent_result: Any,
) -> PipelineDepthConfig:
    """Convert an IntentionMapResult to a PipelineDepthConfig.

    Parameters
    ----------
    intent_result : IntentionMapResult
        From E23.  Expected attributes:
        - dominant_intent : str
        - intent_confidence : float
        - primary_archetype : str

    Returns
    -------
    PipelineDepthConfig
    """
    if intent_result is None:
        return DEFAULT_PIPELINE_PROFILE

    # Extract dominant intent
    dominant = getattr(intent_result, "dominant_intent", "")
    confidence = getattr(intent_result, "intent_confidence", 0.5)

    if not dominant:
        return DEFAULT_PIPELINE_PROFILE

    # Normalise to pipeline profile category
    intent_lower = dominant.lower().strip()
    category = _INTENT_CATEGORY_MAP.get(intent_lower, "")

    # Disintegration alert — force containment regardless of confidence
    if category == "disintegration":
        log.warning("Disintegration intent detected (confidence=%.2f) — forcing containment profile.", confidence)
        return _CONTAINMENT_PROFILE

    if not category:
        log.debug("Unknown intent category '%s', using default profile.", dominant)
        return DEFAULT_PIPELINE_PROFILE

    profile = get_pipeline_profile(category)

    # Modulate by confidence — low confidence pulls toward default
    if confidence < 0.4:
        log.debug("Low intent confidence (%.2f), blending toward default.", confidence)
        default = DEFAULT_PIPELINE_PROFILE
        profile = PipelineDepthConfig(
            perception_depth=_blend(profile.perception_depth, default.perception_depth, confidence),
            semiotics_depth=_blend(profile.semiotics_depth, default.semiotics_depth, confidence),
            emotion_detection_sensitivity=_blend(
                profile.emotion_detection_sensitivity,
                default.emotion_detection_sensitivity,
                confidence,
            ),
            phase1_depth=_blend(profile.phase1_depth, default.phase1_depth, confidence),
            phase3_engine_count_cap=profile.phase3_engine_count_cap,
            phase4_thinking_token_budget=profile.phase4_thinking_token_budget,
            phase5_reward_thoroughness=_blend(
                profile.phase5_reward_thoroughness,
                default.phase5_reward_thoroughness,
                confidence,
            ),
            phase6_response_style=profile.phase6_response_style,
        )

    return profile


def _blend(a: float, b: float, weight: float) -> float:
    """Linearly blend two values.  weight=1.0 → all a, weight=0.0 → all b."""
    return a * weight + b * (1.0 - weight)
