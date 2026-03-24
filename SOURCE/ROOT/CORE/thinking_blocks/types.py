"""
ThinkingContext — structured data block passed to VTPromptBuilder.

Contains everything the LLM thinking pass needs in a single, readable
object.  All fields are plain Python types (str, dict, list) so the
prompt builder can serialize them without special handling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationTurn:
    """One recent MTMM conversation turn."""
    role: str = ""           # "user" | "assistant"
    text: str = ""
    turn_index: int = 0


@dataclass
class HeldBlock:
    """A held thinking block retrieved from LTMM thoughts/held_blocks."""
    block_id: str = ""
    content: str = ""
    trigger_summary: str = ""   # what caused the thought to be held
    tags: List[str] = field(default_factory=list)


@dataclass
class MemoryCrossNote:
    """Alignment/divergence note between an engine flag and a memory match."""
    engine_id: str = ""
    flag_type: str = ""      # e.g. "contradiction", "bias", "pattern"
    flag_detail: str = ""
    memory_match_id: str = ""
    memory_summary: str = ""
    relation: str = ""       # "confirms" | "diverges" | "extends"


@dataclass
class ThinkingContext:
    """Full compressed context for the LLM thinking pass (Phase 4).

    Built by ThinkingBlockBuilder after Phase 3 completes.

    Fields
    ------
    mission_briefing : str
        Session-level starter prompt set by user at session open.
    engine_flags : dict
        Key engine outputs compressed to salient flags.
        Keys: engine short-name; values: summary dicts.
    memory_matches : list of dict
        Top memory contrast matches (from both flat + scoped passes).
    cross_contrast_notes : list of MemoryCrossNote
        Engine flag × memory match alignment/divergence notes.
    recent_turns : list of ConversationTurn
        Last 2 MTMM conversation turns (user + assistant).
    held_blocks : list of HeldBlock
        Unreviewed held thinking blocks from LTMM.
    reward_profile_name : str
        Active reward profile for this turn.
    intent_category : str
        E23 intent category (lowercase).
    dominant_emotion : tuple (str, float)
        From E28 or extractor result.
    nt_snapshot : dict
        Current NT concentrations (lowercase keys).
    alignment_result : Any
        AlignmentResult from IdentityAlignmentChecker (populated later).
    personality_prompts : list of str
        Personality/tone prompt fragments (populated later).
    """
    mission_briefing: str = ""
    engine_flags: Dict[str, Any] = field(default_factory=dict)
    memory_matches: List[Dict[str, Any]] = field(default_factory=list)
    cross_contrast_notes: List[MemoryCrossNote] = field(default_factory=list)
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    held_blocks: List[HeldBlock] = field(default_factory=list)
    reward_profile_name: str = "regular_input"
    intent_category: str = ""
    dominant_emotion: tuple = ("none", 0.0)
    nt_snapshot: Dict[str, float] = field(default_factory=dict)
    alignment_result: Any = None
    personality_prompts: List[str] = field(default_factory=list)
