"""Tests for Knowledge namespace stores."""
import pytest
from zados.memory.long_term.knowledge.library.store import LibraryStore
from zados.memory.long_term.knowledge.lessons.store import LessonStore
from zados.memory.long_term.knowledge.academic_buffer.store import (
    AcademicBufferStore, AcademicBufferEntry,
)
from zados.memory.long_term.knowledge.academic_questions.store import AcademicQuestionStore
from zados.memory.long_term.knowledge.knowledge_maps.store import KnowledgeMapStore
from zados.memory.long_term.knowledge.cognitools_data.store import CognitoolsDataStore
from zados.memory.long_term.knowledge.notebook.store import NotebookStore
from zados.memory.long_term.knowledge.types import (
    LibraryEntry, LessonEntry, AcademicQuestion,
    KnowledgeMap, KnowledgeNode, NotebookEntry,
)


# -----------------------------------------------------------------------
# LibraryStore
# -----------------------------------------------------------------------

class TestLibraryStore:
    def test_write_and_search(self):
        store = LibraryStore()
        store.write(LibraryEntry(title="Principia Mathematica", content="Laws of motion", domain="physics"))
        results = store.search("motion physics")
        assert len(results) >= 1
        assert "Principia" in results[0][1].title

    def test_get_by_id(self):
        store = LibraryStore()
        e = LibraryEntry(title="Book")
        store.write(e)
        assert store.get_by_id(e.entry_id) is e
        assert store.get_by_id("missing") is None

    def test_get_by_domain(self):
        store = LibraryStore()
        store.write(LibraryEntry(title="A", domain="physics"))
        store.write(LibraryEntry(title="B", domain="math"))
        store.write(LibraryEntry(title="C", domain="physics"))
        assert len(store.get_by_domain("physics")) == 2
        assert len(store.get_by_domain("math")) == 1

    def test_get_all_and_len(self):
        store = LibraryStore()
        store.write(LibraryEntry(title="A"))
        store.write(LibraryEntry(title="B"))
        assert len(store) == 2
        assert len(store.get_all()) == 2


# -----------------------------------------------------------------------
# LessonStore
# -----------------------------------------------------------------------

class TestLessonStore:
    def test_write_and_search(self):
        store = LessonStore()
        store.write(LessonEntry(content="Force equals mass times acceleration", subject_category="physics"))
        results = store.search("force acceleration")
        assert len(results) >= 1

    def test_validate(self):
        store = LessonStore()
        le = LessonEntry(content="Hypothesis A")
        store.write(le)
        assert le.validation_status == "pending"
        assert store.validate(le.lesson_id) is True
        assert le.validation_status == "validated"

    def test_contradict(self):
        store = LessonStore()
        le = LessonEntry(content="Bad hypothesis")
        store.write(le)
        assert store.contradict(le.lesson_id) is True
        assert le.validation_status == "contradicted"

    def test_validate_missing(self):
        store = LessonStore()
        assert store.validate("missing") is False
        assert store.contradict("missing") is False

    def test_reinforce(self):
        store = LessonStore()
        le = LessonEntry(content="Strong hypothesis")
        store.write(le)
        assert le.reinforcement_count == 0
        assert store.reinforce(le.lesson_id) is True
        assert le.reinforcement_count == 1

    def test_reinforce_missing(self):
        store = LessonStore()
        assert store.reinforce("missing") is False

    def test_get_validated(self):
        store = LessonStore()
        le1 = LessonEntry(content="A", validation_status="validated")
        le2 = LessonEntry(content="B", validation_status="pending")
        store.write(le1)
        store.write(le2)
        assert len(store.get_validated()) == 1

    def test_get_by_id_and_all(self):
        store = LessonStore()
        le = LessonEntry(content="A")
        store.write(le)
        assert store.get_by_id(le.lesson_id) is le
        assert len(store.get_all()) == 1


# -----------------------------------------------------------------------
# AcademicBufferStore
# -----------------------------------------------------------------------

class TestAcademicBufferStore:
    def test_add_and_resolve(self):
        store = AcademicBufferStore()
        entry = AcademicBufferEntry(concept_formulation="What is entropy?", subject_category="physics")
        store.add(entry)
        assert len(store) == 1
        store.resolve(entry.entry_id, note="Understood via thermodynamics")
        assert store.get_by_id(entry.entry_id).resolved is True

    def test_tick_all(self):
        store = AcademicBufferStore()
        e1 = AcademicBufferEntry(concept_formulation="Q1")
        e2 = AcademicBufferEntry(concept_formulation="Q2", resolved=True)
        store.add(e1)
        store.add(e2)
        store.tick_all()
        assert e1.stagnation_cycles == 1
        assert e2.stagnation_cycles == 0  # resolved, not ticked

    def test_dream_candidates(self):
        store = AcademicBufferStore()
        entry = AcademicBufferEntry(concept_formulation="Hard problem", stagnation_cycles=5)
        store.add(entry)
        assert len(store.get_dream_candidates(threshold=5)) == 1

    def test_get_all(self):
        store = AcademicBufferStore()
        store.add(AcademicBufferEntry(concept_formulation="A"))
        store.add(AcademicBufferEntry(concept_formulation="B"))
        assert len(store.get_all()) == 2


# -----------------------------------------------------------------------
# AcademicQuestionStore
# -----------------------------------------------------------------------

class TestAcademicQuestionStore:
    def test_write_and_search(self):
        store = AcademicQuestionStore()
        store.write(AcademicQuestion(
            formulation="What is entropy?",
            subject_category="physics",
            domain="thermodynamics",
        ))
        results = store.search("entropy thermodynamics")
        assert len(results) >= 1

    def test_resolve(self):
        store = AcademicQuestionStore()
        q = AcademicQuestion(formulation="Question")
        store.write(q)
        assert store.resolve(q.question_id, resolution_note="Answered") is True
        assert q.resolved is True

    def test_tick_stagnation(self):
        store = AcademicQuestionStore()
        q = AcademicQuestion(formulation="Hard question")
        store.write(q)
        assert store.tick_stagnation(q.question_id) is True
        assert q.stagnation_count == 1

    def test_get_unresolved(self):
        store = AcademicQuestionStore()
        store.write(AcademicQuestion(formulation="A", resolved=False))
        store.write(AcademicQuestion(formulation="B", resolved=True))
        assert len(store.get_unresolved()) == 1

    def test_missing_operations(self):
        store = AcademicQuestionStore()
        assert store.resolve("missing") is False
        assert store.tick_stagnation("missing") is False
        assert store.get_by_id("missing") is None


# -----------------------------------------------------------------------
# KnowledgeMapStore
# -----------------------------------------------------------------------

class TestKnowledgeMapStore:
    def test_write_and_search(self):
        store = KnowledgeMapStore()
        n1 = KnowledgeNode(label="gravity")
        km = KnowledgeMap(title="Physics Map", subject_category="physics", nodes=[n1])
        store.write(km)
        results = store.search("gravity physics")
        assert len(results) >= 1

    def test_get_by_subject(self):
        store = KnowledgeMapStore()
        store.write(KnowledgeMap(title="A", subject_category="physics"))
        store.write(KnowledgeMap(title="B", subject_category="math"))
        assert len(store.get_by_subject("physics")) == 1

    def test_get_by_id(self):
        store = KnowledgeMapStore()
        km = KnowledgeMap(title="Test")
        store.write(km)
        assert store.get_by_id(km.map_id) is km
        assert store.get_by_id("missing") is None

    def test_get_all_and_len(self):
        store = KnowledgeMapStore()
        store.write(KnowledgeMap(title="A"))
        store.write(KnowledgeMap(title="B"))
        assert len(store) == 2
        assert len(store.get_all()) == 2


# -----------------------------------------------------------------------
# CognitoolsDataStore
# -----------------------------------------------------------------------

class TestCognitoolsDataStore:
    def test_write_and_get(self):
        store = CognitoolsDataStore()
        store.write("E9", {"atoms": 42, "links": 100})
        result = store.get_by_id("E9")
        assert result["atoms"] == 42

    def test_overwrite(self):
        store = CognitoolsDataStore()
        store.write("E9", {"v": 1})
        store.write("E9", {"v": 2})
        assert store.get_by_id("E9")["v"] == 2
        assert len(store) == 1

    def test_missing(self):
        store = CognitoolsDataStore()
        assert store.get_by_id("missing") is None

    def test_get_all_engine_ids(self):
        store = CognitoolsDataStore()
        store.write("E9", {})
        store.write("E10", {})
        assert sorted(store.get_all_engine_ids()) == ["E10", "E9"]

    def test_get_all(self):
        store = CognitoolsDataStore()
        store.write("E9", {"a": 1})
        store.write("E10", {"b": 2})
        assert len(store.get_all()) == 2


# -----------------------------------------------------------------------
# NotebookStore
# -----------------------------------------------------------------------

class TestNotebookStore:
    def test_write_and_search(self):
        store = NotebookStore()
        store.write(NotebookEntry(content="Studied derivatives and integrals", subject_category="math"))
        results = store.search("derivatives integrals")
        assert len(results) >= 1

    def test_get_by_subject(self):
        store = NotebookStore()
        store.write(NotebookEntry(content="A", subject_category="math"))
        store.write(NotebookEntry(content="B", subject_category="physics"))
        assert len(store.get_by_subject("math")) == 1

    def test_get_by_id(self):
        store = NotebookStore()
        ne = NotebookEntry(content="Test")
        store.write(ne)
        assert store.get_by_id(ne.note_id) is ne
        assert store.get_by_id("missing") is None

    def test_get_all_and_len(self):
        store = NotebookStore()
        store.write(NotebookEntry(content="A"))
        store.write(NotebookEntry(content="B"))
        assert len(store) == 2
        assert len(store.get_all()) == 2
