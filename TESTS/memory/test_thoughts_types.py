"""Tests for thoughts namespace dataclasses."""
import pytest
from zados.memory.long_term.thoughts.types import (
    HeldThinkingBlock, OverviewLogEntry, GeneralQuestion,
)


class TestHeldThinkingBlock:
    def test_defaults(self):
        b = HeldThinkingBlock()
        assert b.block_id
        assert b.reviewed is False

    def test_to_search_text(self):
        b = HeldThinkingBlock(thought_fragment="what if trust is fragile",
                              emotion_tag="anxiety", tags=["cognitive:interrupt"])
        text = b.to_search_text()
        assert "trust" in text
        assert "anxiety" in text


class TestOverviewLogEntry:
    def test_defaults(self):
        o = OverviewLogEntry(session_id="s1", summary="Productive session")
        assert o.log_id
        assert o.mode_sequence == []

    def test_to_search_text(self):
        o = OverviewLogEntry(summary="explored ethics", subject_tags=["domain:philosophical"])
        text = o.to_search_text()
        assert "ethics" in text
        assert "domain:philosophical" in text


class TestGeneralQuestion:
    def test_defaults(self):
        q = GeneralQuestion(formulation="What does fairness mean?")
        assert q.priority == 0.5
        assert q.resolved is False
        assert q.stagnation_count == 0

    def test_to_search_text(self):
        q = GeneralQuestion(formulation="meaning of life",
                            domain_hint="philosophical", tags=["cognitive:unresolved"])
        text = q.to_search_text()
        assert "meaning" in text
        assert "philosophical" in text
