"""
Thoughts namespace — dataclass definitions.

HeldThinkingBlock — emotionally-interrupted thought fragments.
OverviewLogEntry  — brief per-session cognitive overview.
GeneralQuestion   — non-academic question queue.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class HeldThinkingBlock:
    """
    Thought fragment saved when emotion detection crosses threshold
    during the pre-response thinking phase.

    Direct LTMM write (not staged through STMM/MTMM) due to emotional
    significance.  Threshold: emotion intensity > 0.6 on any single
    emotion from the 46-taxonomy, OR any identity-relevant emotion.
    """
    block_id:            str            = field(default_factory=lambda: str(uuid.uuid4()))
    thought_fragment:    str            = ""
    emotion_tag:         str            = ""  # emotion name from 46-taxonomy
    emotion_trigger_type: str           = ""
    nt_snapshot:         Dict[str, float] = field(default_factory=dict)  # 4 metrics
    context_summary:     str            = ""
    pipeline_phase:      str            = ""  # "phase4_thinking" | "verbalized_thinking"
    source_turn_ref:     str            = ""  # turn/packet_id this occurred within
    session_id:          str            = ""
    timestamp:           datetime       = field(default_factory=datetime.utcnow)
    tags:                List[str]      = field(default_factory=list)
    reviewed:            bool           = False

    def to_search_text(self) -> str:
        parts = [self.thought_fragment, self.context_summary, self.emotion_tag]
        parts.extend(self.tags)
        return " ".join(parts)


@dataclass
class OverviewLogEntry:
    """Brief cognitive overview generated at session end."""
    log_id:            str            = field(default_factory=lambda: str(uuid.uuid4()))
    session_id:        str            = ""
    summary:           str            = ""  # ~200 words
    mode_sequence:     List[str]      = field(default_factory=list)
    subject_tags:      List[str]      = field(default_factory=list)
    dominant_emotions: List[str]      = field(default_factory=list)
    nt_arc:            Dict[str, List[float]] = field(default_factory=dict)
    open_threads:      List[str]      = field(default_factory=list)
    timestamp:         datetime       = field(default_factory=datetime.utcnow)

    def to_search_text(self) -> str:
        parts = [self.summary]
        parts.extend(self.subject_tags)
        parts.extend(self.dominant_emotions)
        parts.extend(self.open_threads)
        return " ".join(parts)


@dataclass
class GeneralQuestion:
    """
    Non-academic question in the active question queue.

    General = about anything including identity, relational, existential.
    For domain-specific knowledge gaps, see AcademicQuestion.
    """
    question_id:     str            = field(default_factory=lambda: str(uuid.uuid4()))
    formulation:     str            = ""
    source:          str            = ""  # "self_generated" | "user_triggered" | "engine_flagged"
    domain_hint:     Optional[str]  = None
    priority:        float          = 0.5  # [0, 1]
    stagnation_count: int           = 0
    resolved:        bool           = False
    resolution_note: str            = ""
    created_at:      datetime       = field(default_factory=datetime.utcnow)
    last_checked:    datetime       = field(default_factory=datetime.utcnow)
    tags:            List[str]      = field(default_factory=list)

    def to_search_text(self) -> str:
        parts = [self.formulation, self.resolution_note]
        if self.domain_hint:
            parts.append(self.domain_hint)
        parts.extend(self.tags)
        return " ".join(parts)
