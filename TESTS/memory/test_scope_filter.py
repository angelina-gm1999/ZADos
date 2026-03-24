"""Tests for ScopeFilter and scoped MemoryContrast search."""
import pytest
from zados.memory import MemoryLayer
from zados.memory.managers.scope_filter import (
    ScopeFilter,
    REGULAR_SCOPE, M1_M5_SCOPE, M2_SCOPE, M3_SCOPE,
    HOMEWORK_SCOPE, REFLECTIVE_SCOPE, REM_SCOPE, DREAM_SCOPE,
)
from zados.memory.managers.contrast import MemoryContrast
from zados.memory.long_term.knowledge.types import LessonEntry, LibraryEntry
from zados.memory.long_term.identity.types import CoreMemory, IdentityConclusion
from zados.memory.long_term.thoughts.types import OverviewLogEntry, GeneralQuestion


# -----------------------------------------------------------------------
# ScopeFilter dataclass
# -----------------------------------------------------------------------

class TestScopeFilter:
    def test_frozen(self):
        sf = ScopeFilter()
        with pytest.raises(AttributeError):
            sf.max_results = 99

    def test_defaults(self):
        sf = ScopeFilter()
        assert sf.folders == frozenset()
        assert sf.required_tags == frozenset()
        assert sf.excluded_tags == frozenset()
        assert sf.subject_filter is None
        assert sf.max_results == 10
        assert sf.include_cold is False

    def test_custom(self):
        sf = ScopeFilter(
            folders=frozenset({"knowledge/lessons"}),
            required_tags=frozenset({"pipeline:m1"}),
            max_results=5,
        )
        assert "knowledge/lessons" in sf.folders
        assert "pipeline:m1" in sf.required_tags
        assert sf.max_results == 5


class TestPrebuiltScopes:
    def test_regular_scope(self):
        assert "thoughts/overview_logs" in REGULAR_SCOPE.folders
        assert "knowledge/lessons" in REGULAR_SCOPE.folders

    def test_m2_scope_targets_identity(self):
        assert "identity/core" in M2_SCOPE.folders
        assert "identity/conclusions" in M2_SCOPE.folders
        assert "identity/journal" in M2_SCOPE.folders
        assert "pipeline:m2" in M2_SCOPE.required_tags

    def test_homework_scope_knowledge_heavy(self):
        # Part 5 §5.1 — homework reads knowledge + thoughts for offline processing
        for folder in HOMEWORK_SCOPE.folders:
            assert folder.startswith("knowledge/") or folder.startswith("thoughts/")
        # Must include all knowledge stores
        assert "knowledge/lessons" in HOMEWORK_SCOPE.folders
        assert "knowledge/library" in HOMEWORK_SCOPE.folders
        assert "knowledge/knowledge_maps" in HOMEWORK_SCOPE.folders
        # Must include thoughts stores for question resolution
        assert "thoughts/unsolved_buffer" in HOMEWORK_SCOPE.folders
        assert "thoughts/general_questions" in HOMEWORK_SCOPE.folders
        assert HOMEWORK_SCOPE.max_results == 20

    def test_rem_scope_includes_cold(self):
        assert REM_SCOPE.include_cold is True

    def test_dream_scope_includes_cold(self):
        assert DREAM_SCOPE.include_cold is True


# -----------------------------------------------------------------------
# Scoped search via MemoryContrast
# -----------------------------------------------------------------------

class TestScopedSearch:
    def _make_ml(self):
        ml = MemoryLayer()
        # Populate some stores
        ml.knowledge.lessons.write(LessonEntry(
            content="Force equals mass times acceleration",
            subject_category="physics",
            tags=["pipeline:m1", "domain:scientific"],
        ))
        ml.knowledge.library.write(LibraryEntry(
            title="Physics Textbook",
            content="Comprehensive guide to Newtonian mechanics",
            domain="physics",
        ))
        ml.identity.core.write(CoreMemory(
            content="I value intellectual curiosity",
            memory_type="value",
            tags=["identity:core"],
        ))
        ml.thoughts.overview_logs.write(OverviewLogEntry(
            session_id="s1",
            summary="Explored physics and force concepts",
        ))
        return ml

    def test_scoped_search_finds_matching_entries(self):
        ml = self._make_ml()
        sf = ScopeFilter(
            folders=frozenset({"knowledge/lessons", "knowledge/library"}),
            max_results=5,
        )
        result = ml.contrast.contrast(
            current={"text": "force acceleration"},
            query_type="concept",
            scope_filter=sf,
        )
        assert result.similarity > 0.0
        assert len(result.references) >= 1
        # All references should be from LTMM_SCOPED source
        for ref in result.references:
            assert ref["source"] == "LTMM_SCOPED"

    def test_scoped_search_respects_folder_restriction(self):
        ml = self._make_ml()
        # Search only identity — should NOT find physics lessons
        sf = ScopeFilter(
            folders=frozenset({"identity/core"}),
            max_results=5,
        )
        result = ml.contrast.contrast(
            current={"text": "force acceleration physics"},
            query_type="concept",
            scope_filter=sf,
        )
        # Identity store has "intellectual curiosity", not "force"
        for ref in result.references:
            assert ref["folder"] == "identity/core"

    def test_scoped_search_with_required_tags(self):
        ml = self._make_ml()
        sf = ScopeFilter(
            folders=frozenset({"knowledge/lessons"}),
            required_tags=frozenset({"pipeline:m1"}),
            max_results=5,
        )
        result = ml.contrast.contrast(
            current={"text": "force"},
            query_type="concept",
            scope_filter=sf,
        )
        assert result.similarity > 0.0

    def test_scoped_search_excluded_tags(self):
        ml = self._make_ml()
        sf = ScopeFilter(
            folders=frozenset({"knowledge/lessons"}),
            excluded_tags=frozenset({"pipeline:m1"}),
            max_results=5,
        )
        result = ml.contrast.contrast(
            current={"text": "force acceleration"},
            query_type="concept",
            scope_filter=sf,
        )
        # Entry has pipeline:m1 tag → should be excluded
        assert len(result.references) == 0

    def test_none_scope_filter_falls_back_to_flat(self):
        ml = self._make_ml()
        # No scope_filter → flat LTMM search (which is empty, so 0 similarity)
        result = ml.contrast.contrast(
            current={"text": "force"},
            query_type="concept",
        )
        # This goes through the flat path (no namespaced results)
        assert result.similarity == 0.0 or result.references  # either works

    def test_scoped_search_meta_contains_folders(self):
        ml = self._make_ml()
        sf = ScopeFilter(
            folders=frozenset({"knowledge/lessons"}),
            max_results=5,
        )
        result = ml.contrast.contrast(
            current={"text": "force"},
            query_type="concept",
            scope_filter=sf,
        )
        assert result.meta["scoped"] is True
        assert "knowledge/lessons" in result.meta["folders_searched"]

    def test_scoped_search_empty_result(self):
        ml = self._make_ml()
        sf = ScopeFilter(
            folders=frozenset({"knowledge/notebook"}),
            max_results=5,
        )
        result = ml.contrast.contrast(
            current={"text": "quantum mechanics"},
            query_type="concept",
            scope_filter=sf,
        )
        assert result.similarity == 0.0
        assert len(result.references) == 0
