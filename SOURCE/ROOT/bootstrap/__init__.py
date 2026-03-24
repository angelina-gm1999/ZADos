"""
ZADOS Knowledge Bootstrap
=========================

Pre-populates in-memory stores with foundational knowledge before the first
user interaction.  Seeds grow incrementally as documents are processed into
the seed files under ``seeds/``.

Workflow
--------
1. Drop a document into ``ROOT/knowledge_sources/books/``.
2. Process it (Claude extracts knowledge → adds to seed files).
3. On next session open, ``KnowledgeBootstrap.run()`` loads all seeds into:
   - AtomSpaceEngine (E9) — cognitive concept ontology
   - KnowledgeMapStore  — human-readable semantic graphs
   - LessonStore        — pre-validated foundational lessons
   - LibraryStore       — reference material entries
"""

from zados.bootstrap.knowledge_bootstrap import KnowledgeBootstrap

__all__ = ["KnowledgeBootstrap"]
