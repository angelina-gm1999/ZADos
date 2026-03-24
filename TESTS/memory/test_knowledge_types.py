"""Tests for knowledge namespace dataclasses."""
import pytest
from zados.memory.long_term.knowledge.types import (
    LessonEntry, KnowledgeNode, KnowledgeLink, KnowledgeMap,
    NotebookEntry, AcademicQuestion, LibraryEntry,
)


class TestLessonEntry:
    def test_defaults(self):
        le = LessonEntry()
        assert le.lesson_id
        assert le.confidence == 0.5
        assert le.validation_status == "pending"
        assert le.reinforcement_count == 0
        assert le.source_refs == []
        assert le.cross_links == []
        assert le.knowledge_map_refs == []

    def test_to_search_text(self):
        le = LessonEntry(content="Newton's laws", subject_category="physics",
                         source_mode="M3", tags=["domain:scientific"])
        text = le.to_search_text()
        assert "Newton" in text
        assert "physics" in text
        assert "M3" in text
        assert "domain:scientific" in text


class TestKnowledgeNode:
    def test_defaults(self):
        n = KnowledgeNode()
        assert n.node_id
        assert n.label == ""
        assert n.confidence == 0.5

    def test_fields(self):
        n = KnowledgeNode(label="gravity", node_type="concept", confidence=0.9)
        assert n.label == "gravity"
        assert n.node_type == "concept"
        assert n.confidence == 0.9


class TestKnowledgeLink:
    def test_defaults(self):
        lk = KnowledgeLink()
        assert lk.link_id
        assert lk.weight == 1.0

    def test_fields(self):
        lk = KnowledgeLink(source_node="n1", target_node="n2",
                           relation="supports", weight=0.8)
        assert lk.source_node == "n1"
        assert lk.target_node == "n2"
        assert lk.relation == "supports"


class TestKnowledgeMap:
    def test_defaults(self):
        km = KnowledgeMap()
        assert km.map_id
        assert km.nodes == []
        assert km.links == []
        assert km.atomspace_ref is None

    def test_to_search_text(self):
        n1 = KnowledgeNode(label="entropy")
        n2 = KnowledgeNode(label="thermodynamics")
        km = KnowledgeMap(title="Thermo Map", description="Heat and energy",
                          subject_category="physics",
                          nodes=[n1, n2], tags=["domain:scientific"])
        text = km.to_search_text()
        assert "Thermo" in text
        assert "entropy" in text
        assert "thermodynamics" in text
        assert "domain:scientific" in text


class TestNotebookEntry:
    def test_defaults(self):
        ne = NotebookEntry()
        assert ne.note_id
        assert ne.nt_snapshot == {}
        assert ne.related_lessons == []
        assert ne.related_questions == []

    def test_to_search_text(self):
        ne = NotebookEntry(content="Studied calculus", subject_category="math",
                           source_mode="homework", tags=["domain:mathematical"])
        text = ne.to_search_text()
        assert "calculus" in text
        assert "math" in text
        assert "domain:mathematical" in text


class TestAcademicQuestion:
    def test_defaults(self):
        aq = AcademicQuestion()
        assert aq.question_id
        assert aq.priority == 0.5
        assert aq.resolved is False
        assert aq.stagnation_count == 0

    def test_mirrors_general_question_fields(self):
        aq = AcademicQuestion(formulation="What is entropy?",
                              source="self_generated",
                              subject_category="physics",
                              domain="thermodynamics")
        assert aq.formulation == "What is entropy?"
        assert aq.subject_category == "physics"
        assert aq.domain == "thermodynamics"

    def test_to_search_text(self):
        aq = AcademicQuestion(formulation="What is entropy?",
                              subject_category="physics",
                              domain="thermodynamics",
                              tags=["domain:scientific"])
        text = aq.to_search_text()
        assert "entropy" in text
        assert "physics" in text
        assert "thermodynamics" in text
        assert "domain:scientific" in text


class TestLibraryEntry:
    def test_defaults(self):
        lib = LibraryEntry()
        assert lib.entry_id
        assert lib.nt_snapshot == {}

    def test_to_search_text(self):
        lib = LibraryEntry(title="Principia Mathematica", content="Laws of motion",
                           domain="physics", tags=["domain:scientific"])
        text = lib.to_search_text()
        assert "Principia" in text
        assert "motion" in text
        assert "physics" in text
        assert "domain:scientific" in text
