"""
Knowledge namespace — stores for what ZA-DOS knows.

Sub-stores
----------
library             Ingested reference material.
lessons             Validated academic insights.
academic_buffer     Unsolved academic concepts (mirrors UnsolvedConceptsBuffer API).
academic_questions  Domain-specific knowledge gap questions.
knowledge_maps      Human-readable semantic graphs.
cognitools_data     Per-engine persistent state (KV, no search).
notebook            Academic journaling.
"""
from zados.memory.long_term.knowledge.library.store import LibraryStore
from zados.memory.long_term.knowledge.lessons.store import LessonStore
from zados.memory.long_term.knowledge.academic_buffer.store import AcademicBufferStore
from zados.memory.long_term.knowledge.academic_questions.store import AcademicQuestionStore
from zados.memory.long_term.knowledge.knowledge_maps.store import KnowledgeMapStore
from zados.memory.long_term.knowledge.cognitools_data.store import CognitoolsDataStore
from zados.memory.long_term.knowledge.notebook.store import NotebookStore

__all__ = [
    "LibraryStore",
    "LessonStore",
    "AcademicBufferStore",
    "AcademicQuestionStore",
    "KnowledgeMapStore",
    "CognitoolsDataStore",
    "NotebookStore",
]
