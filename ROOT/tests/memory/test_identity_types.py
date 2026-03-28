"""Tests for identity namespace dataclasses."""
import pytest
from zados.memory.long_term.identity.types import (
    CoreMemory, UpdateRecord, PendingUpdate,
    IdentityConclusion, IdentityJournalEntry, IdentityJournalEntryType,
)


class TestCoreMemory:
    def test_defaults(self):
        cm = CoreMemory()
        assert cm.memory_id  # uuid generated
        assert cm.version == 1
        assert cm.update_history == []

    def test_custom_fields(self):
        cm = CoreMemory(content="I value honesty", memory_type="self_model",
                        tags=["identity:core"])
        assert cm.content == "I value honesty"
        assert cm.memory_type == "self_model"

    def test_update_history_is_independent(self):
        a = CoreMemory()
        b = CoreMemory()
        a.update_history.append(UpdateRecord(previous_content="old"))
        assert len(b.update_history) == 0


class TestPendingUpdate:
    def test_defaults(self):
        pu = PendingUpdate(target_memory_id="abc", proposed_content="new")
        assert pu.status == "pending"
        assert pu.peer_review_ref is None


class TestIdentityConclusion:
    def test_defaults(self):
        ic = IdentityConclusion(content="I learn best by doing",
                                conclusion_type="lesson")
        assert ic.confidence == 0.5
        assert ic.reinforcement_count == 0


class TestIdentityJournalEntry:
    def test_defaults(self):
        e = IdentityJournalEntry(content="Today I reflected on trust")
        assert e.entry_type == IdentityJournalEntryType.REGULAR
        assert e.parent_entry_id is None

    def test_comment_type(self):
        e = IdentityJournalEntry(
            entry_type=IdentityJournalEntryType.COMMENT,
            parent_entry_id="parent-1",
            content="Annotation",
        )
        assert e.entry_type == IdentityJournalEntryType.COMMENT

    def test_to_search_text(self):
        e = IdentityJournalEntry(content="trust", tags=["identity:core"],
                                 source_pipeline="reflective_mode")
        text = e.to_search_text()
        assert "trust" in text
        assert "identity:core" in text

    def test_nt_snapshot_independence(self):
        a = IdentityJournalEntry()
        b = IdentityJournalEntry()
        a.nt_snapshot["motivation"] = 0.8
        assert "motivation" not in b.nt_snapshot
