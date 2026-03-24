"""Tests for pipeline scope declarations."""
import pytest
from zados.memory.managers.pipeline_scopes import (
    PipelineScope, get_pipeline_scope, get_all_pipeline_names,
    PIPELINE_REGULAR, PIPELINE_M1, PIPELINE_M2, PIPELINE_M3,
    PIPELINE_M4, PIPELINE_M5, PIPELINE_HOMEWORK,
    PIPELINE_SELF_REFLECTIVE, PIPELINE_REM, PIPELINE_DREAM,
)


class TestPipelineScope:
    def test_frozen(self):
        with pytest.raises(AttributeError):
            PIPELINE_REGULAR.pipeline_name = "hacked"

    def test_regular_has_read_and_write(self):
        assert PIPELINE_REGULAR.read_scope.folders
        assert PIPELINE_REGULAR.write_scope.folders

    def test_m2_targets_identity(self):
        """M2 reads from identity/, writes to identity/ + knowledge/lessons + thoughts/held_blocks."""
        for folder in PIPELINE_M2.read_scope.folders:
            assert folder.startswith("identity/")
        # M2 write_scope includes identity + cross-namespace writes per Part 4 §3.3
        identity_writes = [f for f in PIPELINE_M2.write_scope.folders if f.startswith("identity/")]
        assert len(identity_writes) >= 3  # core, conclusions, journal
        assert "knowledge/lessons" in PIPELINE_M2.write_scope.folders
        assert "thoughts/held_blocks" in PIPELINE_M2.write_scope.folders

    def test_homework_targets_knowledge_and_thoughts(self):
        # Part 5 §5.1-5.2 — homework reads/writes knowledge + thoughts + identity/core
        for folder in PIPELINE_HOMEWORK.read_scope.folders:
            assert folder.startswith("knowledge/") or folder.startswith("thoughts/")
        for folder in PIPELINE_HOMEWORK.write_scope.folders:
            assert (folder.startswith("knowledge/") or
                    folder.startswith("thoughts/") or
                    folder == "identity/core")
        # Core memory updates via gate only
        assert "identity/core" in PIPELINE_HOMEWORK.write_scope.folders
        # Homework run summary
        assert "thoughts/overview_logs" in PIPELINE_HOMEWORK.write_scope.folders

    def test_rem_and_dream_include_cold(self):
        assert PIPELINE_REM.read_scope.include_cold is True
        assert PIPELINE_DREAM.read_scope.include_cold is True


class TestLookup:
    def test_get_pipeline_scope(self):
        scope = get_pipeline_scope("regular")
        assert scope is PIPELINE_REGULAR

    def test_get_missing_returns_none(self):
        assert get_pipeline_scope("nonexistent") is None

    def test_all_pipeline_names(self):
        names = get_all_pipeline_names()
        assert "regular" in names
        assert "m1_academic" in names
        assert "m2_peer_review" in names
        assert "homework" in names
        assert "rem_sleep" in names
        assert "dream" in names
        assert len(names) == 10

    def test_all_pipelines_have_unique_names(self):
        names = get_all_pipeline_names()
        assert len(names) == len(set(names))

    def test_all_pipelines_have_non_empty_scopes(self):
        for name in get_all_pipeline_names():
            ps = get_pipeline_scope(name)
            assert ps.read_scope.folders, f"{name} has empty read_scope"
            assert ps.write_scope.folders, f"{name} has empty write_scope"
