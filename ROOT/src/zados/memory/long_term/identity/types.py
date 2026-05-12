"""
Identity namespace — dataclass definitions.

CoreMemory, UpdateRecord, PendingUpdate — peer-review-gated self-model.
IdentityConclusion — AI-derived values / lessons / self-insights.
IdentityJournalEntry — reflective journaling scoped to identity.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ── Core Memories ────────────────────────────────────────────────────────

@dataclass
class UpdateRecord:
    """Preserves a single historical version of a CoreMemory."""
    previous_content: str
    updated_at:       datetime = field(default_factory=datetime.utcnow)
    peer_review_ref:  str = ""  # LearningLogEntry ID that validated the update


@dataclass
class CoreMemory:
    """
    Immutable-by-default identity memory.

    Can NEVER be deleted.  Updates require peer-review validation and
    preserve all previous versions via the ``update_history`` chain.
    """
    memory_id:      str            = field(default_factory=lambda: str(uuid.uuid4()))
    content:        str            = ""
    memory_type:    str            = ""  # "experience" | "relationship" | "self_model" | "event"
    tags:           List[str]      = field(default_factory=list)
    created_at:     datetime       = field(default_factory=datetime.utcnow)
    updated_at:     datetime       = field(default_factory=datetime.utcnow)
    version:        int            = 1
    update_history: List[UpdateRecord] = field(default_factory=list)


@dataclass
class PendingUpdate:
    """Staged update to a CoreMemory awaiting peer-review validation."""
    update_id:        str            = field(default_factory=lambda: str(uuid.uuid4()))
    target_memory_id: str            = ""
    proposed_content: str            = ""
    reason:           str            = ""  # AI-generated justification
    created_at:       datetime       = field(default_factory=datetime.utcnow)
    status:           str            = "pending"  # "pending" | "approved" | "rejected"
    peer_review_ref:  Optional[str]  = None       # set when validated


# ── Identity Conclusions ─────────────────────────────────────────────────

@dataclass
class IdentityConclusion:
    """AI-derived value, lesson, or self-insight."""
    conclusion_id:      str            = field(default_factory=lambda: str(uuid.uuid4()))
    content:            str            = ""
    conclusion_type:    str            = ""  # "value" | "lesson" | "self_insight" | "boundary"
    source_refs:        List[str]      = field(default_factory=list)
    confidence:         float          = 0.5
    tags:               List[str]      = field(default_factory=list)
    created_at:         datetime       = field(default_factory=datetime.utcnow)
    last_reinforced:    datetime       = field(default_factory=datetime.utcnow)
    reinforcement_count: int           = 0


# ── Identity Journal ─────────────────────────────────────────────────────

# ── Identity Correlations ────────────────────────────────────────────

class CorrelationRelationType(str, Enum):
    """How a developmental identity element relates to a hardcoded entry."""
    INSTANTIATES = "instantiates"   # developmental item is a concrete instance of hardcoded principle
    EXTENDS      = "extends"        # developmental item builds on hardcoded principle
    SUPPORTS     = "supports"       # developmental item provides evidence for hardcoded principle
    DEEPENS      = "deepens"        # developmental item deepens understanding of hardcoded principle
    TENSIONS     = "tensions_with"  # developmental item has productive tension with hardcoded principle
    QUESTIONS    = "questions"      # developmental item raises questions about hardcoded principle


@dataclass
class IdentityCorrelation:
    """
    Maps a relation between a fixed (hardcoded) identity entry and a
    developmental identity element (conclusion, core memory, or journal entry).

    ZADOS cannot modify or delete hardcoded entries.  It can only create,
    update, or remove correlations that describe how its developmental
    identity relates to its fixed foundations.
    """
    correlation_id:     str            = field(default_factory=lambda: str(uuid.uuid4()))
    hardcoded_entry_id: str            = ""
    developmental_id:   str            = ""
    developmental_type: str            = ""  # "conclusion" | "core_memory" | "journal_entry"
    relation_type:      str            = ""  # CorrelationRelationType value
    description:        str            = ""  # AI-generated description of the relation
    confidence:         float          = 0.5
    tags:               List[str]      = field(default_factory=list)
    created_at:         datetime       = field(default_factory=datetime.utcnow)
    last_validated:     datetime       = field(default_factory=datetime.utcnow)
    validation_count:   int            = 0

    def to_search_text(self) -> str:
        parts = [self.description, self.relation_type,
                 self.developmental_type, self.hardcoded_entry_id]
        parts.extend(self.tags)
        return " ".join(parts)


class IdentityJournalEntryType(str, Enum):
    REGULAR    = "regular"     # standard reflective journal entry
    REFLECTION = "reflection"  # triggered by reflective mode pipeline
    COMMENT    = "comment"     # annotation on a previous entry


@dataclass
class IdentityJournalEntry:
    """Reflective journal entry scoped to identity development."""
    entry_id:        str            = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type:      IdentityJournalEntryType = IdentityJournalEntryType.REGULAR
    content:         str            = ""
    parent_entry_id: Optional[str]  = None   # for COMMENT type
    nt_snapshot:     Dict[str, float] = field(default_factory=dict)  # 4 metrics
    emotion_tags:    List[str]      = field(default_factory=list)
    source_pipeline: str            = ""  # "reflective_mode" | "peer_review" | "rem"
    tags:            List[str]      = field(default_factory=list)
    timestamp:       datetime       = field(default_factory=datetime.utcnow)

    def to_search_text(self) -> str:
        """Concatenate searchable fields for TF-IDF indexing."""
        parts = [self.content, self.source_pipeline]
        parts.extend(self.tags)
        parts.extend(self.emotion_tags)
        return " ".join(parts)
