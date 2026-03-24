"""Tests for Thoughts namespace stores."""
import pytest
from zados.memory.long_term.thoughts.overview_logs.store import OverviewLogStore
from zados.memory.long_term.thoughts.held_thinking_blocks.store import HeldThinkingBlockStore
from zados.memory.long_term.thoughts.unsolved_buffer.store import UnsolvedBufferStore
from zados.memory.long_term.thoughts.general_questions.store import GeneralQuestionStore
from zados.memory.long_term.thoughts.types import (
    OverviewLogEntry, HeldThinkingBlock, GeneralQuestion,
)
from zados.memory.long_term.specialized_logs import UnsolvedConceptsBuffer, UnsolvedConceptEntry


# -----------------------------------------------------------------------
# OverviewLogStore
# -----------------------------------------------------------------------

class TestOverviewLogStore:
    def test_write_and_search(self):
        store = OverviewLogStore()
        store.write(OverviewLogEntry(
            session_id="s1", summary="Explored ethical dilemmas deeply",
        ))
        results = store.search("ethical dilemma")
        assert len(results) >= 1
        assert "ethical" in results[0][1].summary

    def test_get_by_id(self):
        store = OverviewLogStore()
        entry = OverviewLogEntry(session_id="s1", summary="Test session")
        store.write(entry)
        assert store.get_by_id(entry.log_id) is entry
        assert store.get_by_id("missing") is None

    def test_get_by_session(self):
        store = OverviewLogStore()
        e1 = OverviewLogEntry(session_id="s1", summary="First session")
        e2 = OverviewLogEntry(session_id="s2", summary="Second session")
        store.write(e1)
        store.write(e2)
        assert store.get_by_session("s1") is e1
        assert store.get_by_session("s3") is None

    def test_get_all(self):
        store = OverviewLogStore()
        store.write(OverviewLogEntry(session_id="s1", summary="A"))
        store.write(OverviewLogEntry(session_id="s2", summary="B"))
        assert len(store.get_all()) == 2

    def test_len(self):
        store = OverviewLogStore()
        assert len(store) == 0
        store.write(OverviewLogEntry(session_id="s1", summary="X"))
        assert len(store) == 1


# -----------------------------------------------------------------------
# HeldThinkingBlockStore
# -----------------------------------------------------------------------

class TestHeldThinkingBlockStore:
    def test_write_and_search(self):
        store = HeldThinkingBlockStore()
        store.write(HeldThinkingBlock(
            thought_fragment="what if trust is fragile",
            emotion_tag="anxiety",
        ))
        results = store.search("trust fragile")
        assert len(results) >= 1

    def test_get_unreviewed(self):
        store = HeldThinkingBlockStore()
        b1 = HeldThinkingBlock(thought_fragment="A", reviewed=False)
        b2 = HeldThinkingBlock(thought_fragment="B", reviewed=True)
        store.write(b1)
        store.write(b2)
        unreviewed = store.get_unreviewed()
        assert len(unreviewed) == 1
        assert unreviewed[0] is b1

    def test_mark_reviewed(self):
        store = HeldThinkingBlockStore()
        block = HeldThinkingBlock(thought_fragment="deep thought")
        store.write(block)
        assert not block.reviewed

        ok = store.mark_reviewed(block.block_id)
        assert ok is True
        assert block.reviewed is True
        assert len(store.get_unreviewed()) == 0

    def test_mark_reviewed_missing(self):
        store = HeldThinkingBlockStore()
        assert store.mark_reviewed("missing") is False

    def test_get_by_id(self):
        store = HeldThinkingBlockStore()
        block = HeldThinkingBlock(thought_fragment="T")
        store.write(block)
        assert store.get_by_id(block.block_id) is block
        assert store.get_by_id("missing") is None

    def test_get_all(self):
        store = HeldThinkingBlockStore()
        store.write(HeldThinkingBlock(thought_fragment="A"))
        store.write(HeldThinkingBlock(thought_fragment="B"))
        assert len(store.get_all()) == 2


# -----------------------------------------------------------------------
# UnsolvedBufferStore (re-export)
# -----------------------------------------------------------------------

class TestUnsolvedBufferStore:
    def test_is_same_class(self):
        assert UnsolvedBufferStore is UnsolvedConceptsBuffer

    def test_basic_usage(self):
        store = UnsolvedBufferStore()
        entry = UnsolvedConceptEntry(
            concept_formulation="What is consciousness?",
            source_engine="E13",
        )
        store.add(entry)
        assert len(store) == 1


# -----------------------------------------------------------------------
# GeneralQuestionStore
# -----------------------------------------------------------------------

class TestGeneralQuestionStore:
    def test_write_and_search(self):
        store = GeneralQuestionStore()
        store.write(GeneralQuestion(
            formulation="What does fairness really mean?",
            domain_hint="philosophical",
        ))
        results = store.search("fairness meaning")
        assert len(results) >= 1

    def test_get_unresolved(self):
        store = GeneralQuestionStore()
        q1 = GeneralQuestion(formulation="Q1", resolved=False)
        q2 = GeneralQuestion(formulation="Q2", resolved=True)
        store.write(q1)
        store.write(q2)
        unresolved = store.get_unresolved()
        assert len(unresolved) == 1
        assert unresolved[0] is q1

    def test_resolve(self):
        store = GeneralQuestionStore()
        q = GeneralQuestion(formulation="What is truth?")
        store.write(q)
        assert not q.resolved

        ok = store.resolve(q.question_id, resolution_note="Found an answer")
        assert ok is True
        assert q.resolved is True
        assert q.resolution_note == "Found an answer"
        assert len(store.get_unresolved()) == 0

    def test_resolve_missing(self):
        store = GeneralQuestionStore()
        assert store.resolve("missing") is False

    def test_tick_stagnation(self):
        store = GeneralQuestionStore()
        q = GeneralQuestion(formulation="Hard question")
        store.write(q)
        assert q.stagnation_count == 0

        ok = store.tick_stagnation(q.question_id)
        assert ok is True
        assert q.stagnation_count == 1

    def test_tick_stagnation_missing(self):
        store = GeneralQuestionStore()
        assert store.tick_stagnation("missing") is False

    def test_get_by_id(self):
        store = GeneralQuestionStore()
        q = GeneralQuestion(formulation="Q")
        store.write(q)
        assert store.get_by_id(q.question_id) is q
        assert store.get_by_id("missing") is None

    def test_get_all(self):
        store = GeneralQuestionStore()
        store.write(GeneralQuestion(formulation="A"))
        store.write(GeneralQuestion(formulation="B"))
        assert len(store.get_all()) == 2
