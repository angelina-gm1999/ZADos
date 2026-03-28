"""
Journal Entry dataclasses and enumerations.

JournalTrigger  — five conditions that cause a journal write
ReviewStatus    — lifecycle of a reflection prompt
JournalEntry    — the full artifact stored in JournalStore
JournalContext  — input payload pipelines pass to JournalWriter.write()
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JournalTrigger(str, Enum):
    PERIODIC        = "periodic"          # every N turns
    LTMM_THRESHOLD  = "ltmm_threshold"   # MTMM→LTMM compressor decided relevance
    REM_COMPLETE    = "rem_complete"      # dreaming / consolidation pipeline finished
    INNOVATION_FLAG = "innovation_flag"  # innovation module flagged something
    DEV             = "dev"               # developer-triggered via interface


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"   # prompts generated, not yet revisited
    IN_REVIEW  = "in_review"    # system is actively processing these prompts
    RESOLVED   = "resolved"     # prompts answered / integrated into learned buffer


# ---------------------------------------------------------------------------
# Engine annotations — structured output from E18 / E19 / E20
# ---------------------------------------------------------------------------

@dataclass
class EngineAnnotations:
    """
    Structured metadata produced by cognitive engines
    running on the journal entry's text at write time.
    """
    # E18 — Data Analysis: entity-relation triples extracted from prose + VT
    entities:        List[str]              = field(default_factory=list)
    relations:       List[Tuple[str, str, str]] = field(default_factory=list)  # (subj, pred, obj)
    co_occurrences:  List[Tuple[str, str]]  = field(default_factory=list)      # (entity_a, entity_b)

    # E19 — Pattern Identification: patterns found in current entry text
    identified_patterns:    List[str]       = field(default_factory=list)
    pattern_types:          List[str]       = field(default_factory=list)      # PatternType values

    # E20 — Pattern Comparison: how this entry relates to past journal entries
    cross_session_patterns: List[str]       = field(default_factory=list)      # recurring across entries
    parallel_concepts:      List[str]       = field(default_factory=list)      # conceptual comparisons
    novelty_flags:          List[str]       = field(default_factory=list)      # patterns with no prior match


# ---------------------------------------------------------------------------
# Journal Entry
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    """
    The full artifact written to JournalStore.

    Written by JournalWriter; read by identity pipeline, LTMM retrieval,
    and future journal review cycles.
    """
    # Identity
    entry_id:       str      = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:      datetime = field(default_factory=datetime.utcnow)
    session_id:     str      = ""
    turn_range:     Tuple[int, int] = (0, 0)   # (first_turn, last_turn) this covers

    # Trigger
    trigger:        JournalTrigger = JournalTrigger.PERIODIC
    trigger_source: str            = ""        # which module fired the trigger

    # LLM-generated content
    prose:               str       = ""        # reflective monologue (150-400 words)
    reflection_prompts:  List[str] = field(default_factory=list)  # open questions, unanswered

    # Source material
    vt_source:      str = ""    # VT monologue that seeded this entry (from cortical_reflection)

    # Cognitive engine annotations
    annotations: EngineAnnotations = field(default_factory=EngineAnnotations)

    # State snapshots at time of writing
    emotion_snapshot:    Dict[str, float] = field(default_factory=dict)  # system_emotion_state
    nt_snapshot:         Dict[str, float] = field(default_factory=dict)  # nt_concentrations
    reward_snapshot:     Dict[str, float] = field(default_factory=dict)  # per_domain_weighted_scores
    tone_snapshot:       Dict[str, float] = field(default_factory=dict)  # valence/warmth/discord/coherence

    # Lifecycle
    review_status:      ReviewStatus = ReviewStatus.UNREVIEWED
    linked_entry_ids:   List[str]    = field(default_factory=list)  # related past entries
    tags:               List[str]    = field(default_factory=list)  # auto-generated retrieval tags

    # Pipeline notes (passed in by calling pipeline, stored verbatim)
    pipeline_notes:     List[str]    = field(default_factory=list)

    def mark_in_review(self) -> None:
        self.review_status = ReviewStatus.IN_REVIEW

    def resolve(self) -> None:
        self.review_status = ReviewStatus.RESOLVED

    def link(self, other_entry_id: str) -> None:
        if other_entry_id not in self.linked_entry_ids:
            self.linked_entry_ids.append(other_entry_id)

    def to_search_text(self) -> str:
        """Flat text representation for semantic search indexing."""
        parts = [self.prose] + self.reflection_prompts + self.tags
        if self.annotations.identified_patterns:
            parts += self.annotations.identified_patterns
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Journal Context — input payload for JournalWriter.write()
# ---------------------------------------------------------------------------

@dataclass
class JournalContext:
    """
    Everything a pipeline needs to pass to JournalWriter.write().

    The writer reads stmm for all snapshots; `notes` is pipeline-specific
    context the calling pipeline wants permanently attached to the entry.
    """
    trigger:        JournalTrigger
    trigger_source: str
    stmm:           Any                    # STMMStore — typed as Any to avoid circular import
    notes:          List[str]              = field(default_factory=list)
    turn_range:     Tuple[int, int]        = (0, 0)
    session_id:     str                    = ""
