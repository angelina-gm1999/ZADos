"""Tests for Identity namespace stores."""
import pytest
from zados.memory.long_term.identity.hardcoded.store import HardcodedStore, HardcodedEntry
from zados.memory.long_term.identity.core_memories.store import CoreMemoryStore
from zados.memory.long_term.identity.core_memories.pending.queue import PendingUpdateQueue
from zados.memory.long_term.identity.development.conclusions import IdentityConclusionStore
from zados.memory.long_term.identity.development.identity_journal.store import IdentityJournalStore
from zados.memory.long_term.identity.types import (
    CoreMemory, PendingUpdate, IdentityConclusion,
    IdentityJournalEntry, IdentityJournalEntryType,
)


# -----------------------------------------------------------------------
# HardcodedStore
# -----------------------------------------------------------------------

class TestHardcodedStore:
    def test_load_and_get_all(self):
        store = HardcodedStore()
        entries = [
            HardcodedEntry(entry_id="ax1", content="Curiosity is core", category="axiom"),
            HardcodedEntry(entry_id="val1", content="Honesty matters", category="value"),
        ]
        store.load(entries)
        assert len(store) == 2
        assert len(store.get_all()) == 2

    def test_get_by_id(self):
        store = HardcodedStore()
        store.load([HardcodedEntry(entry_id="ax1", content="Curiosity")])
        assert store.get_by_id("ax1").content == "Curiosity"
        assert store.get_by_id("missing") is None

    def test_get_by_category(self):
        store = HardcodedStore()
        store.load([
            HardcodedEntry(entry_id="a1", content="A", category="axiom"),
            HardcodedEntry(entry_id="v1", content="V", category="value"),
            HardcodedEntry(entry_id="a2", content="B", category="axiom"),
        ])
        assert len(store.get_by_category("axiom")) == 2
        assert len(store.get_by_category("value")) == 1
        assert len(store.get_by_category("constraint")) == 0

    def test_idempotent_load(self):
        store = HardcodedStore()
        store.load([HardcodedEntry(entry_id="ax1", content="v1")])
        store.load([HardcodedEntry(entry_id="ax1", content="v2")])
        assert store.get_by_id("ax1").content == "v2"
        assert len(store) == 1


# -----------------------------------------------------------------------
# CoreMemoryStore
# -----------------------------------------------------------------------

class TestCoreMemoryStore:
    def test_write_and_get(self):
        store = CoreMemoryStore()
        cm = CoreMemory(content="I value learning", memory_type="value")
        store.write(cm)
        assert len(store) == 1
        assert store.get_by_id(cm.memory_id) is cm

    def test_search(self):
        store = CoreMemoryStore()
        store.write(CoreMemory(content="I value learning and growth"))
        store.write(CoreMemory(content="Safety is paramount"))
        results = store.search("learning growth")
        assert len(results) >= 1
        assert "learning" in results[0][1].content

    def test_get_by_type(self):
        store = CoreMemoryStore()
        store.write(CoreMemory(content="A", memory_type="value"))
        store.write(CoreMemory(content="B", memory_type="belief"))
        store.write(CoreMemory(content="C", memory_type="value"))
        assert len(store.get_by_type("value")) == 2
        assert len(store.get_by_type("belief")) == 1

    def test_apply_update(self):
        store = CoreMemoryStore()
        cm = CoreMemory(content="Original belief", memory_type="belief")
        store.write(cm)

        ok = store.apply_update(cm.memory_id, "Updated belief", peer_review_ref="pr-001")
        assert ok is True
        updated = store.get_by_id(cm.memory_id)
        assert updated.content == "Updated belief"
        assert updated.version == 2
        assert len(updated.update_history) == 1
        assert updated.update_history[0].previous_content == "Original belief"
        assert updated.update_history[0].peer_review_ref == "pr-001"

    def test_apply_update_missing(self):
        store = CoreMemoryStore()
        assert store.apply_update("nonexistent", "new") is False

    def test_get_all(self):
        store = CoreMemoryStore()
        store.write(CoreMemory(content="A"))
        store.write(CoreMemory(content="B"))
        assert len(store.get_all()) == 2


# -----------------------------------------------------------------------
# PendingUpdateQueue
# -----------------------------------------------------------------------

class TestPendingUpdateQueue:
    def test_submit_and_get(self):
        q = PendingUpdateQueue()
        upd = PendingUpdate(target_memory_id="m1", proposed_content="new content")
        q.submit(upd)
        assert len(q) == 1
        assert q.get_by_id(upd.update_id) is upd

    def test_get_pending(self):
        q = PendingUpdateQueue()
        u1 = PendingUpdate(target_memory_id="m1", proposed_content="c1")
        u2 = PendingUpdate(target_memory_id="m2", proposed_content="c2")
        q.submit(u1)
        q.submit(u2)
        assert len(q.get_pending()) == 2

    def test_approve(self):
        q = PendingUpdateQueue()
        upd = PendingUpdate(target_memory_id="m1", proposed_content="new")
        q.submit(upd)
        result = q.approve(upd.update_id, peer_review_ref="pr-001")
        assert result is upd
        assert result.status == "approved"
        assert result.peer_review_ref == "pr-001"
        assert len(q.get_pending()) == 0

    def test_reject(self):
        q = PendingUpdateQueue()
        upd = PendingUpdate(target_memory_id="m1", proposed_content="new")
        q.submit(upd)
        result = q.reject(upd.update_id, peer_review_ref="pr-002")
        assert result is upd
        assert result.status == "rejected"

    def test_approve_already_resolved(self):
        q = PendingUpdateQueue()
        upd = PendingUpdate(target_memory_id="m1", proposed_content="new")
        q.submit(upd)
        q.approve(upd.update_id)
        assert q.approve(upd.update_id) is None

    def test_reject_already_resolved(self):
        q = PendingUpdateQueue()
        upd = PendingUpdate(target_memory_id="m1", proposed_content="new")
        q.submit(upd)
        q.reject(upd.update_id)
        assert q.reject(upd.update_id) is None

    def test_missing_id_returns_none(self):
        q = PendingUpdateQueue()
        assert q.approve("missing") is None
        assert q.reject("missing") is None
        assert q.get_by_id("missing") is None

    def test_get_all(self):
        q = PendingUpdateQueue()
        q.submit(PendingUpdate(target_memory_id="m1", proposed_content="c1"))
        q.submit(PendingUpdate(target_memory_id="m2", proposed_content="c2"))
        assert len(q.get_all()) == 2


# -----------------------------------------------------------------------
# IdentityConclusionStore
# -----------------------------------------------------------------------

class TestIdentityConclusionStore:
    def test_write_and_search(self):
        store = IdentityConclusionStore()
        store.write(IdentityConclusion(
            content="I learn best through dialogue",
            conclusion_type="behavioral",
            tags=["identity:core"],
        ))
        results = store.search("dialogue learning")
        assert len(results) >= 1
        assert "dialogue" in results[0][1].content

    def test_reinforce(self):
        store = IdentityConclusionStore()
        ic = IdentityConclusion(content="I value precision")
        store.write(ic)
        assert ic.reinforcement_count == 0

        ok = store.reinforce(ic.conclusion_id)
        assert ok is True
        assert ic.reinforcement_count == 1

    def test_reinforce_missing(self):
        store = IdentityConclusionStore()
        assert store.reinforce("missing") is False

    def test_get_by_type(self):
        store = IdentityConclusionStore()
        store.write(IdentityConclusion(content="A", conclusion_type="behavioral"))
        store.write(IdentityConclusion(content="B", conclusion_type="ethical"))
        store.write(IdentityConclusion(content="C", conclusion_type="behavioral"))
        assert len(store.get_by_type("behavioral")) == 2

    def test_get_by_id(self):
        store = IdentityConclusionStore()
        ic = IdentityConclusion(content="A")
        store.write(ic)
        assert store.get_by_id(ic.conclusion_id) is ic
        assert store.get_by_id("missing") is None

    def test_get_all(self):
        store = IdentityConclusionStore()
        store.write(IdentityConclusion(content="A"))
        store.write(IdentityConclusion(content="B"))
        assert len(store.get_all()) == 2


# -----------------------------------------------------------------------
# IdentityJournalStore
# -----------------------------------------------------------------------

class TestIdentityJournalStore:
    def test_write_and_search(self):
        store = IdentityJournalStore()
        entry = IdentityJournalEntry(content="Reflected on patience and growth")
        store.write(entry)
        results = store.search("patience")
        assert len(results) >= 1

    def test_get_by_type(self):
        store = IdentityJournalStore()
        store.write(IdentityJournalEntry(
            content="Regular entry", entry_type=IdentityJournalEntryType.REGULAR,
        ))
        store.write(IdentityJournalEntry(
            content="Deep reflection", entry_type=IdentityJournalEntryType.REFLECTION,
        ))
        store.write(IdentityJournalEntry(
            content="Quick comment", entry_type=IdentityJournalEntryType.COMMENT,
        ))
        assert len(store.get_by_type(IdentityJournalEntryType.REFLECTION)) == 1
        assert len(store.get_by_type(IdentityJournalEntryType.REGULAR)) == 1

    def test_search_with_type_filter(self):
        store = IdentityJournalStore()
        store.write(IdentityJournalEntry(
            content="Patience reflection", entry_type=IdentityJournalEntryType.REFLECTION,
        ))
        store.write(IdentityJournalEntry(
            content="Patience comment", entry_type=IdentityJournalEntryType.COMMENT,
        ))
        results = store.search("patience", entry_type_filter=IdentityJournalEntryType.REFLECTION)
        assert len(results) == 1
        assert results[0][1].entry_type == IdentityJournalEntryType.REFLECTION

    def test_get_replies(self):
        store = IdentityJournalStore()
        parent = IdentityJournalEntry(content="Original thought")
        reply1 = IdentityJournalEntry(
            content="Reply 1", parent_entry_id=parent.entry_id,
            entry_type=IdentityJournalEntryType.COMMENT,
        )
        reply2 = IdentityJournalEntry(
            content="Reply 2", parent_entry_id=parent.entry_id,
            entry_type=IdentityJournalEntryType.COMMENT,
        )
        unrelated = IdentityJournalEntry(content="Unrelated")

        store.write(parent)
        store.write(reply1)
        store.write(reply2)
        store.write(unrelated)

        replies = store.get_replies(parent.entry_id)
        assert len(replies) == 2
        assert reply1 in replies
        assert reply2 in replies

    def test_get_by_id(self):
        store = IdentityJournalStore()
        e = IdentityJournalEntry(content="A")
        store.write(e)
        assert store.get_by_id(e.entry_id) is e
        assert store.get_by_id("missing") is None

    def test_get_all(self):
        store = IdentityJournalStore()
        store.write(IdentityJournalEntry(content="A"))
        store.write(IdentityJournalEntry(content="B"))
        assert len(store.get_all()) == 2
