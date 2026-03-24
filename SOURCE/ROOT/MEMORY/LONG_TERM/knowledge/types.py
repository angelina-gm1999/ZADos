"""
Knowledge namespace — dataclass definitions.

LessonEntry      — validated academic insight.
KnowledgeNode    — graph node for KnowledgeMap.
KnowledgeLink    — graph edge for KnowledgeMap.
KnowledgeMap     — human-readable semantic graph.
NotebookEntry    — academic journaling.
AcademicQuestion — domain-specific knowledge gap.
LibraryEntry     — ingested reference material.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class LessonEntry:
    """Validated academic insight produced by a learning mode."""
    lesson_id:          str            = field(default_factory=lambda: str(uuid.uuid4()))
    content:            str            = ""
    subject_category:   str            = ""  # from SubjectCategory enum
    source_mode:        str            = ""  # M1-M5
    source_refs:        List[str]      = field(default_factory=list)
    confidence:         float          = 0.5
    validation_status:  str            = "pending"  # "validated" | "pending" | "contradicted"
    cross_links:        List[str]      = field(default_factory=list)
    knowledge_map_refs: List[str]      = field(default_factory=list)
    tags:               List[str]      = field(default_factory=list)
    created_at:         datetime       = field(default_factory=datetime.utcnow)
    last_reinforced:    datetime       = field(default_factory=datetime.utcnow)
    reinforcement_count: int           = 0

    def to_search_text(self) -> str:
        parts = [self.content, self.subject_category, self.source_mode]
        parts.extend(self.tags)
        return " ".join(parts)


@dataclass
class KnowledgeNode:
    """Single node in a KnowledgeMap graph."""
    node_id:    str   = field(default_factory=lambda: str(uuid.uuid4()))
    label:      str   = ""
    node_type:  str   = ""  # "concept" | "principle" | "fact" | "open_question"
    confidence: float = 0.5


@dataclass
class KnowledgeLink:
    """Directed edge in a KnowledgeMap graph."""
    link_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    source_node: str   = ""
    target_node: str   = ""
    relation:    str   = ""  # "supports" | "contradicts" | "extends" | "requires" | "exemplifies"
    weight:      float = 1.0


@dataclass
class KnowledgeMap:
    """
    Human-readable semantic graph for a subject domain.

    Distinct from AtomSpace (live probabilistic inference substrate).
    They can be linked via ``atomspace_ref`` but are independently maintained.
    """
    map_id:               str                 = field(default_factory=lambda: str(uuid.uuid4()))
    title:                str                 = ""
    subject_category:     str                 = ""
    description:          str                 = ""
    nodes:                List[KnowledgeNode] = field(default_factory=list)
    links:                List[KnowledgeLink] = field(default_factory=list)
    contributing_lessons: List[str]           = field(default_factory=list)
    atomspace_ref:        Optional[str]       = None
    last_updated:         datetime            = field(default_factory=datetime.utcnow)
    tags:                 List[str]           = field(default_factory=list)

    def to_search_text(self) -> str:
        parts = [self.title, self.description, self.subject_category]
        parts.extend(n.label for n in self.nodes)
        parts.extend(self.tags)
        return " ".join(parts)


@dataclass
class NotebookEntry:
    """Academic journaling entry about knowledge-domain learning."""
    note_id:            str            = field(default_factory=lambda: str(uuid.uuid4()))
    content:            str            = ""
    subject_category:   str            = ""
    source_mode:        str            = ""  # M1-M5 or "homework"
    related_lessons:    List[str]      = field(default_factory=list)
    related_questions:  List[str]      = field(default_factory=list)
    nt_snapshot:        Dict[str, float] = field(default_factory=dict)  # 4 metrics
    timestamp:          datetime       = field(default_factory=datetime.utcnow)
    tags:               List[str]      = field(default_factory=list)

    def to_search_text(self) -> str:
        parts = [self.content, self.subject_category, self.source_mode]
        parts.extend(self.tags)
        return " ".join(parts)


@dataclass
class AcademicQuestion:
    """
    Domain-specific knowledge gap, analogous to GeneralQuestion
    but scoped to a subject category.
    """
    question_id:     str            = field(default_factory=lambda: str(uuid.uuid4()))
    formulation:     str            = ""
    source:          str            = ""  # "self_generated" | "user_triggered" | "engine_flagged"
    subject_category: str           = ""
    domain:          str            = ""
    priority:        float          = 0.5
    stagnation_count: int           = 0
    resolved:        bool           = False
    resolution_note: str            = ""
    created_at:      datetime       = field(default_factory=datetime.utcnow)
    last_checked:    datetime       = field(default_factory=datetime.utcnow)
    tags:            List[str]      = field(default_factory=list)

    def to_search_text(self) -> str:
        parts = [self.formulation, self.subject_category, self.domain,
                 self.resolution_note]
        parts.extend(self.tags)
        return " ".join(parts)


@dataclass
class LibraryEntry:
    """Ingested reference material (book, document, article)."""
    entry_id:   str            = field(default_factory=lambda: str(uuid.uuid4()))
    title:      str            = ""
    content:    str            = ""
    source_type: str           = ""  # "book" | "article" | "document" | "upload"
    domain:     str            = ""
    tags:       List[str]      = field(default_factory=list)
    nt_snapshot: Dict[str, float] = field(default_factory=dict)  # 4 metrics
    timestamp:  datetime       = field(default_factory=datetime.utcnow)

    def to_search_text(self) -> str:
        parts = [self.title, self.content, self.domain]
        parts.extend(self.tags)
        return " ".join(parts)
