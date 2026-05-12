"""
LTMM Namespace facades.

Three dataclass facades expose the namespaced stores as cohesive units:
    IdentityNamespace  — identity/
    ThoughtsNamespace  — thoughts/
    KnowledgeNamespace — knowledge/

``build_namespaces()`` is the factory that constructs all three.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from zados.memory.long_term.identity.hardcoded.store import HardcodedStore
from zados.memory.long_term.identity.core_memories.store import CoreMemoryStore
from zados.memory.long_term.identity.core_memories.pending.queue import PendingUpdateQueue
from zados.memory.long_term.identity.development.conclusions import IdentityConclusionStore
from zados.memory.long_term.identity.development.identity_journal.store import IdentityJournalStore
from zados.memory.long_term.identity.correlation.store import IdentityCorrelationStore

from zados.memory.long_term.thoughts.overview_logs.store import OverviewLogStore
from zados.memory.long_term.thoughts.held_thinking_blocks.store import HeldThinkingBlockStore
from zados.memory.long_term.thoughts.unsolved_buffer.store import UnsolvedBufferStore
from zados.memory.long_term.thoughts.general_questions.store import GeneralQuestionStore

from zados.memory.long_term.knowledge.library.store import LibraryStore
from zados.memory.long_term.knowledge.lessons.store import LessonStore
from zados.memory.long_term.knowledge.academic_buffer.store import AcademicBufferStore
from zados.memory.long_term.knowledge.academic_questions.store import AcademicQuestionStore
from zados.memory.long_term.knowledge.knowledge_maps.store import KnowledgeMapStore
from zados.memory.long_term.knowledge.cognitools_data.store import CognitoolsDataStore
from zados.memory.long_term.knowledge.notebook.store import NotebookStore


@dataclass
class IdentityNamespace:
    """Facade for identity/ sub-stores."""
    hardcoded:    HardcodedStore              = field(default_factory=HardcodedStore)
    core:         CoreMemoryStore             = field(default_factory=CoreMemoryStore)
    pending:      PendingUpdateQueue          = field(default_factory=PendingUpdateQueue)
    conclusions:  IdentityConclusionStore     = field(default_factory=IdentityConclusionStore)
    journal:      IdentityJournalStore        = field(default_factory=IdentityJournalStore)
    correlation:  IdentityCorrelationStore    = field(default_factory=IdentityCorrelationStore)


@dataclass
class ThoughtsNamespace:
    """Facade for thoughts/ sub-stores."""
    overview_logs:   OverviewLogStore          = field(default_factory=OverviewLogStore)
    held_blocks:     HeldThinkingBlockStore    = field(default_factory=HeldThinkingBlockStore)
    unsolved_buffer: UnsolvedBufferStore       = field(default_factory=UnsolvedBufferStore)
    general_questions: GeneralQuestionStore    = field(default_factory=GeneralQuestionStore)


@dataclass
class KnowledgeNamespace:
    """Facade for knowledge/ sub-stores."""
    library:            LibraryStore            = field(default_factory=LibraryStore)
    lessons:            LessonStore             = field(default_factory=LessonStore)
    academic_buffer:    AcademicBufferStore      = field(default_factory=AcademicBufferStore)
    academic_questions: AcademicQuestionStore    = field(default_factory=AcademicQuestionStore)
    knowledge_maps:     KnowledgeMapStore        = field(default_factory=KnowledgeMapStore)
    cognitools_data:    CognitoolsDataStore      = field(default_factory=CognitoolsDataStore)
    notebook:           NotebookStore            = field(default_factory=NotebookStore)


def build_namespaces() -> tuple:
    """
    Construct all three namespace facades.

    Returns (identity, thoughts, knowledge).
    """
    return (
        IdentityNamespace(),
        ThoughtsNamespace(),
        KnowledgeNamespace(),
    )
