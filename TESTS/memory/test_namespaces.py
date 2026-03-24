"""Tests for LTMM namespace facades and MemoryLayer integration."""
import pytest
from zados.memory import MemoryLayer
from zados.memory.long_term.namespaces import (
    IdentityNamespace, ThoughtsNamespace, KnowledgeNamespace, build_namespaces,
)
from zados.memory.long_term.identity.hardcoded.store import HardcodedStore
from zados.memory.long_term.identity.core_memories.store import CoreMemoryStore
from zados.memory.long_term.identity.core_memories.pending.queue import PendingUpdateQueue
from zados.memory.long_term.identity.development.conclusions import IdentityConclusionStore
from zados.memory.long_term.identity.development.identity_journal.store import IdentityJournalStore
from zados.memory.long_term.thoughts.overview_logs.store import OverviewLogStore
from zados.memory.long_term.thoughts.held_thinking_blocks.store import HeldThinkingBlockStore
from zados.memory.long_term.thoughts.general_questions.store import GeneralQuestionStore
from zados.memory.long_term.knowledge.library.store import LibraryStore
from zados.memory.long_term.knowledge.lessons.store import LessonStore
from zados.memory.long_term.knowledge.knowledge_maps.store import KnowledgeMapStore
from zados.memory.long_term.knowledge.cognitools_data.store import CognitoolsDataStore
from zados.memory.long_term.knowledge.notebook.store import NotebookStore


class TestBuildNamespaces:
    def test_returns_three_namespaces(self):
        identity, thoughts, knowledge = build_namespaces()
        assert isinstance(identity, IdentityNamespace)
        assert isinstance(thoughts, ThoughtsNamespace)
        assert isinstance(knowledge, KnowledgeNamespace)

    def test_identity_stores(self):
        ns = IdentityNamespace()
        assert isinstance(ns.hardcoded, HardcodedStore)
        assert isinstance(ns.core, CoreMemoryStore)
        assert isinstance(ns.pending, PendingUpdateQueue)
        assert isinstance(ns.conclusions, IdentityConclusionStore)
        assert isinstance(ns.journal, IdentityJournalStore)

    def test_thoughts_stores(self):
        ns = ThoughtsNamespace()
        assert isinstance(ns.overview_logs, OverviewLogStore)
        assert isinstance(ns.held_blocks, HeldThinkingBlockStore)
        assert isinstance(ns.general_questions, GeneralQuestionStore)

    def test_knowledge_stores(self):
        ns = KnowledgeNamespace()
        assert isinstance(ns.library, LibraryStore)
        assert isinstance(ns.lessons, LessonStore)
        assert isinstance(ns.knowledge_maps, KnowledgeMapStore)
        assert isinstance(ns.cognitools_data, CognitoolsDataStore)
        assert isinstance(ns.notebook, NotebookStore)

    def test_each_call_creates_fresh_instances(self):
        id1, th1, kn1 = build_namespaces()
        id2, th2, kn2 = build_namespaces()
        assert id1.core is not id2.core
        assert th1.overview_logs is not th2.overview_logs
        assert kn1.library is not kn2.library


class TestMemoryLayerNamespaces:
    def test_memory_layer_has_namespaces(self):
        ml = MemoryLayer()
        assert isinstance(ml.identity, IdentityNamespace)
        assert isinstance(ml.thoughts, ThoughtsNamespace)
        assert isinstance(ml.knowledge, KnowledgeNamespace)

    def test_legacy_ltmm_still_works(self):
        ml = MemoryLayer()
        assert ml.ltmm is not None
        assert len(ml.ltmm) == 0

    def test_namespaces_independent_of_flat_ltmm(self):
        ml = MemoryLayer()
        # Writing to flat store doesn't affect namespaced stores
        from zados.memory.long_term.store import LTMMEntry
        from zados.memory.types import MemoryPacket
        pkt = MemoryPacket(user_message="test", system_response="response")
        ml.ltmm.write(LTMMEntry(packet=pkt))
        assert len(ml.ltmm) == 1
        assert len(ml.identity.core) == 0
        assert len(ml.thoughts.overview_logs) == 0
        assert len(ml.knowledge.library) == 0

    def test_end_cycle_still_works(self):
        ml = MemoryLayer()
        ml.stmm.add_user_message("hello")
        packet = ml.end_cycle()
        assert packet is not None
