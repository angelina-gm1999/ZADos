"""Tests for RetrievalRouter."""
import pytest
from zados.memory import MemoryLayer
from zados.memory.long_term.retrieval_router import RetrievalRouter, RetrievalContext
from zados.memory.long_term.store import LTMMStore
from zados.memory.managers.scope_filter import ScopeFilter
from zados.memory.long_term.knowledge.types import LessonEntry, LibraryEntry
from zados.memory.long_term.identity.types import CoreMemory
from zados.memory.long_term.thoughts.types import OverviewLogEntry, GeneralQuestion


def _make_ml():
    ml = MemoryLayer()
    ml.knowledge.lessons.write(LessonEntry(
        content="Force equals mass times acceleration",
        subject_category="physics",
        tags=["domain:scientific"],
    ))
    ml.knowledge.library.write(LibraryEntry(
        title="Physics Textbook",
        content="Comprehensive guide to Newtonian mechanics",
        domain="physics",
    ))
    ml.identity.core.write(CoreMemory(
        content="I value intellectual curiosity and growth",
        memory_type="value",
        tags=["identity:core"],
    ))
    ml.thoughts.overview_logs.write(OverviewLogEntry(
        session_id="s1",
        summary="Explored physics and force concepts deeply",
    ))
    ml.thoughts.general_questions.write(GeneralQuestion(
        formulation="What does fairness really mean?",
        domain_hint="philosophical",
    ))
    return ml


class TestRetrievalContext:
    def test_defaults(self):
        ctx = RetrievalContext(query_text="test")
        assert ctx.query_type == "general"
        assert ctx.limit == 5
        assert ctx.tags == []
        assert ctx.scope_filter is None


class TestRetrievalRouterKnowledge:
    def test_knowledge_query_finds_lessons(self):
        ml = _make_ml()
        ctx = RetrievalContext(query_text="force acceleration", query_type="knowledge")
        results = ml.router.route(ctx)
        assert len(results) >= 1
        # Should find the lesson about force
        found_force = any("force" in str(getattr(e, "content", "")).lower() for _, e in results)
        assert found_force

    def test_knowledge_query_skips_identity(self):
        ml = _make_ml()
        ctx = RetrievalContext(query_text="curiosity growth", query_type="knowledge")
        results = ml.router.route(ctx)
        # Identity store entries should NOT appear in knowledge queries
        for _, entry in results:
            assert not hasattr(entry, "memory_type")


class TestRetrievalRouterIdentity:
    def test_identity_query_finds_core(self):
        ml = _make_ml()
        ctx = RetrievalContext(query_text="curiosity growth", query_type="identity")
        results = ml.router.route(ctx)
        assert len(results) >= 1
        found_curiosity = any("curiosity" in str(getattr(e, "content", "")).lower() for _, e in results)
        assert found_curiosity


class TestRetrievalRouterThought:
    def test_thought_query_finds_overview(self):
        ml = _make_ml()
        ctx = RetrievalContext(query_text="physics force", query_type="thought")
        results = ml.router.route(ctx)
        assert len(results) >= 1


class TestRetrievalRouterGeneral:
    def test_general_query_searches_both(self):
        ml = _make_ml()
        ctx = RetrievalContext(query_text="force physics", query_type="general")
        results = ml.router.route(ctx)
        assert len(results) >= 1


class TestRetrievalRouterFallback:
    def test_unknown_query_type_falls_back_to_ltmm(self):
        ml = _make_ml()
        ctx = RetrievalContext(query_text="anything", query_type="unknown_type")
        results = ml.router.route(ctx)
        # Flat LTMM is empty, so no results
        assert len(results) == 0

    def test_no_namespaces_falls_back(self):
        ltmm = LTMMStore()
        router = RetrievalRouter(ltmm, namespaces=None)
        ctx = RetrievalContext(query_text="test", query_type="knowledge")
        results = router.route(ctx)
        assert len(results) == 0


class TestRetrievalRouterScopeFilter:
    def test_scope_filter_overrides_query_type(self):
        ml = _make_ml()
        # Query type says "knowledge" but scope_filter says "identity/core"
        sf = ScopeFilter(folders=frozenset({"identity/core"}))
        ctx = RetrievalContext(
            query_text="curiosity",
            query_type="knowledge",
            scope_filter=sf,
        )
        results = ml.router.route(ctx)
        assert len(results) >= 1
        # Should find identity entries despite query_type="knowledge"
        found_curiosity = any("curiosity" in str(getattr(e, "content", "")).lower() for _, e in results)
        assert found_curiosity

    def test_subject_filter(self):
        ml = _make_ml()
        sf = ScopeFilter(
            folders=frozenset({"knowledge/lessons"}),
            subject_filter="physics",
        )
        ctx = RetrievalContext(query_text="force", scope_filter=sf)
        results = ml.router.route(ctx)
        assert len(results) >= 1

    def test_subject_filter_excludes_mismatch(self):
        ml = _make_ml()
        sf = ScopeFilter(
            folders=frozenset({"knowledge/lessons"}),
            subject_filter="biology",  # no biology lessons
        )
        ctx = RetrievalContext(query_text="force", scope_filter=sf)
        results = ml.router.route(ctx)
        assert len(results) == 0


class TestRetrievalRouterTagFiltering:
    def test_tag_filter(self):
        ml = _make_ml()
        ctx = RetrievalContext(
            query_text="force",
            query_type="knowledge",
            tags=["domain:scientific"],
        )
        results = ml.router.route(ctx)
        assert len(results) >= 1

    def test_tag_filter_excludes_unmatched(self):
        ml = _make_ml()
        ctx = RetrievalContext(
            query_text="force",
            query_type="knowledge",
            tags=["nonexistent:tag"],
        )
        results = ml.router.route(ctx)
        assert len(results) == 0


class TestRetrievalRouterLimit:
    def test_respects_limit(self):
        ml = _make_ml()
        ctx = RetrievalContext(query_text="physics", query_type="general", limit=1)
        results = ml.router.route(ctx)
        assert len(results) <= 1
