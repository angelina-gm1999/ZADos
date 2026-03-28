"""
ZADOS Memory Layer — public surface.

Three temporal tiers + specialized logs + lifecycle manager.

Quickstart:
    from zados.memory import MemoryLayer
    ml = MemoryLayer()
    ml.stmm.add_user_message("Hello")
    # ... engines populate ml.stmm fields ...
    packet = ml.manager.on_cycle_end(ml.stmm)
"""
from __future__ import annotations

from zados.memory.types import (
    CompressionLevel,
    MemoryPacket,
    MemoryTier,
    SpeakerID,
)
from zados.memory.short_term import STMMStore, MemoryExitCompressor
from zados.memory.mid_term import MTMMStore
from zados.memory.long_term import (
    LTMMStore,
    LTMMEntry,
    Granularity,
    MemoryConsolidationEngine,
    MemoryRelevanceHeuristicsEngine,
    FractalPatternComparator,
)
from zados.memory.long_term.specialized_logs import SpecializedLogs
from zados.memory.long_term.namespaces import (
    IdentityNamespace,
    ThoughtsNamespace,
    KnowledgeNamespace,
    build_namespaces,
)
from zados.memory.managers import MemoryContrast, MemoryImplementationManager
from zados.memory.long_term.retrieval_router import RetrievalRouter, RetrievalContext
from zados.memory.long_term.journal.store import JournalStore


class MemoryLayer:
    """
    Convenience facade that wires all three tiers together.

    Attributes:
        stmm      — active working memory (one per processing cycle)
        mtmm      — session-scoped memory
        ltmm      — persistent cross-session memory (flat, legacy)
        identity  — namespaced identity stores
        thoughts  — namespaced thoughts stores
        knowledge — namespaced knowledge stores
        manager   — lifecycle enforcer + LTMM gatekeeper
        contrast  — MemoryContrastPort impl for Logic submodules
        router    — query-type-based retrieval router
    """

    def __init__(self) -> None:
        self.stmm    = STMMStore()
        self.mtmm    = MTMMStore()
        self.ltmm    = LTMMStore()

        # Namespaced LTMM sub-stores (additive — ltmm flat store preserved)
        self.identity, self.thoughts, self.knowledge = build_namespaces()

        # Cognitive journal (shared across pipelines)
        self.journal_store = JournalStore()

        self.manager = MemoryImplementationManager(
            self.mtmm,
            self.ltmm,
            overview_log_store=self.thoughts.overview_logs,
        )
        _ns = {
            "identity":  self.identity,
            "thoughts":  self.thoughts,
            "knowledge": self.knowledge,
        }
        self.contrast = MemoryContrast(
            self.mtmm,
            self.ltmm,
            namespaces=_ns,
        )
        self.router = RetrievalRouter(self.ltmm, namespaces=_ns)

    # Convenience passthrough for the most common call
    def end_cycle(self) -> MemoryPacket:
        """Compress STMM → MTMM and return the produced packet."""
        return self.manager.on_cycle_end(self.stmm)


__all__ = [
    # Tier stores
    "STMMStore",
    "MTMMStore",
    "LTMMStore",
    "LTMMEntry",
    # Types
    "MemoryPacket",
    "MemoryTier",
    "CompressionLevel",
    "SpeakerID",
    "Granularity",
    # Engines
    "MemoryExitCompressor",
    "MemoryConsolidationEngine",
    "MemoryRelevanceHeuristicsEngine",
    "FractalPatternComparator",
    # Managers
    "MemoryContrast",
    "MemoryImplementationManager",
    "SpecializedLogs",
    # Namespaces
    "IdentityNamespace",
    "ThoughtsNamespace",
    "KnowledgeNamespace",
    # Journal
    "JournalStore",
    # Facade
    "MemoryLayer",
]
