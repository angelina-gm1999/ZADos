"""
ZA-DOS v0.6 — Intent Pipeline Optimizer (spec §2.8).

Maps 8 intent categories (from E23 IntentionMapResult) to
PipelineDepthConfig profiles that tune how deeply each pipeline
phase processes the input.
"""
from __future__ import annotations

from typing import Dict, Optional

from zados.core.types import PipelineDepthConfig

# ------------------------------------------------------------------
# 8 intent categories → PipelineDepthConfig
# ------------------------------------------------------------------

INTENT_PIPELINE_PROFILES: Dict[str, PipelineDepthConfig] = {
    # Connection — social / emotional intent
    "connection": PipelineDepthConfig(
        perception_depth=0.6,
        semiotics_depth=0.7,
        emotion_detection_sensitivity=0.9,
        phase1_depth=0.6,
        phase3_engine_count_cap=14,
        phase4_thinking_token_budget=256,
        phase5_reward_thoroughness=0.5,
        phase6_response_style="warm",
    ),

    # Challenge — testing / adversarial intent
    "challenge": PipelineDepthConfig(
        perception_depth=0.9,
        semiotics_depth=0.8,
        emotion_detection_sensitivity=0.7,
        phase1_depth=0.9,
        phase3_engine_count_cap=22,
        phase4_thinking_token_budget=768,
        phase5_reward_thoroughness=0.9,
        phase6_response_style="precise",
    ),

    # Exploration — curiosity / inquiry intent
    "exploration": PipelineDepthConfig(
        perception_depth=0.8,
        semiotics_depth=0.7,
        emotion_detection_sensitivity=0.5,
        phase1_depth=0.8,
        phase3_engine_count_cap=20,
        phase4_thinking_token_budget=512,
        phase5_reward_thoroughness=0.7,
        phase6_response_style="elaborate",
    ),

    # Discharge — venting / emotional release intent
    "discharge": PipelineDepthConfig(
        perception_depth=0.5,
        semiotics_depth=0.8,
        emotion_detection_sensitivity=1.0,
        phase1_depth=0.5,
        phase3_engine_count_cap=10,
        phase4_thinking_token_budget=128,
        phase5_reward_thoroughness=0.4,
        phase6_response_style="empathic",
    ),

    # Pragmatic — practical / task-oriented intent
    "pragmatic": PipelineDepthConfig(
        perception_depth=0.7,
        semiotics_depth=0.4,
        emotion_detection_sensitivity=0.3,
        phase1_depth=0.7,
        phase3_engine_count_cap=16,
        phase4_thinking_token_budget=384,
        phase5_reward_thoroughness=0.6,
        phase6_response_style="concise",
    ),

    # Symbolic — abstract / metaphorical intent
    "symbolic": PipelineDepthConfig(
        perception_depth=0.8,
        semiotics_depth=1.0,
        emotion_detection_sensitivity=0.6,
        phase1_depth=0.8,
        phase3_engine_count_cap=18,
        phase4_thinking_token_budget=640,
        phase5_reward_thoroughness=0.7,
        phase6_response_style="elaborate",
    ),

    # Defensive — guarded / self-protective intent
    "defensive": PipelineDepthConfig(
        perception_depth=0.7,
        semiotics_depth=0.6,
        emotion_detection_sensitivity=0.8,
        phase1_depth=0.7,
        phase3_engine_count_cap=14,
        phase4_thinking_token_budget=256,
        phase5_reward_thoroughness=0.6,
        phase6_response_style="careful",
    ),

    # Disintegration — crisis / destabilisation signal
    "disintegration": PipelineDepthConfig(
        perception_depth=0.5,
        semiotics_depth=0.5,
        emotion_detection_sensitivity=1.0,
        phase1_depth=0.5,
        phase3_engine_count_cap=8,
        phase4_thinking_token_budget=128,
        phase5_reward_thoroughness=0.3,
        phase6_response_style="containment",
    ),
}

# Default profile for unknown intents
DEFAULT_PIPELINE_PROFILE = PipelineDepthConfig()


def get_pipeline_profile(intent_category: str) -> PipelineDepthConfig:
    """Return the PipelineDepthConfig for the given intent category.

    Parameters
    ----------
    intent_category : str
        One of: connection, challenge, exploration, discharge,
        pragmatic, symbolic, defensive, disintegration.

    Returns
    -------
    PipelineDepthConfig
    """
    return INTENT_PIPELINE_PROFILES.get(
        intent_category.lower(),
        DEFAULT_PIPELINE_PROFILE,
    )
