"""Tests for all 8 Specialized Log Subsystems."""
import pytest

from zados.memory.long_term.specialized_logs import (
    ContradictionEntry,
    ContradictionLog,
    DreamEntry,
    DreamLog,
    IdentityMemoryEntry,
    IdentityMemoryLog,
    LearningEntry,
    LearningSystemLog,
    ParadoxEntry,
    ParadoxLog,
    SandboxEntry,
    SandboxLog,
    SelfReflectionEntry,
    SelfReflectionLog,
    SpecializedLogs,
    UnsolvedConceptEntry,
    UnsolvedConceptsBuffer,
)


# ---------------------------------------------------------------------------
# 1. LearningSystemLog
# ---------------------------------------------------------------------------

class TestLearningSystemLog:
    def test_record_and_len(self):
        log = LearningSystemLog()
        log.record(LearningEntry(learning_type="contextual", learned_content="users prefer brevity"))
        assert len(log) == 1

    def test_default_status_is_pending(self):
        log = LearningSystemLog()
        eid = log.record(LearningEntry())
        assert log.get_pending()[0].entry_id == eid

    def test_validate_transitions_to_validated(self):
        log = LearningSystemLog()
        eid = log.record(LearningEntry())
        log.validate(eid)
        assert len(log.get_validated()) == 1
        assert len(log.get_pending())   == 0

    def test_invalidate_with_note(self):
        log = LearningSystemLog()
        eid = log.record(LearningEntry())
        log.invalidate(eid, note="turned out wrong")
        entry = [e for e in log._entries if e.entry_id == eid][0]
        assert entry.validation_status == "invalidated"
        assert "wrong" in entry.invalidation_note

    def test_multiple_entries_independent(self):
        log = LearningSystemLog()
        e1 = log.record(LearningEntry(learning_type="reward"))
        e2 = log.record(LearningEntry(learning_type="reflective"))
        log.validate(e1)
        assert len(log.get_pending())   == 1
        assert len(log.get_validated()) == 1


# ---------------------------------------------------------------------------
# 2. SandboxLog
# ---------------------------------------------------------------------------

class TestSandboxLog:
    def test_record_and_promote(self):
        log = SandboxLog()
        eid = log.record(SandboxEntry(trigger="dev_mode_experiment"))
        assert len(log) == 1
        log.promote(eid)
        assert log.get_all()[0].promoted is True

    def test_unpromoted_by_default(self):
        log = SandboxLog()
        log.record(SandboxEntry())
        assert log.get_all()[0].promoted is False


# ---------------------------------------------------------------------------
# 3. ParadoxLog
# ---------------------------------------------------------------------------

class TestParadoxLog:
    def test_record_and_resolve(self):
        log = ParadoxLog()
        eid = log.record(ParadoxEntry(formulation="this statement is false"))
        assert len(log.get_unresolved()) == 1
        log.resolve(eid)
        assert len(log.get_unresolved()) == 0

    def test_classification_preserved(self):
        log = ParadoxLog()
        log.record(ParadoxEntry(classification="productive"))
        assert log._entries[0].classification == "productive"

    def test_multiple_unresolved(self):
        log = ParadoxLog()
        log.record(ParadoxEntry(formulation="P1"))
        log.record(ParadoxEntry(formulation="P2"))
        assert len(log.get_unresolved()) == 2


# ---------------------------------------------------------------------------
# 4. ContradictionLog
# ---------------------------------------------------------------------------

class TestContradictionLog:
    def test_record_and_resolve(self):
        log = ContradictionLog()
        eid = log.record(ContradictionEntry(
            statement_a="X is true",
            statement_b="X is false",
            severity="high",
        ))
        assert len(log.get_unresolved()) == 1
        log.resolve(eid, method="user clarified")
        assert len(log.get_unresolved()) == 0

    def test_get_by_severity(self):
        log = ContradictionLog()
        log.record(ContradictionEntry(severity="high"))
        log.record(ContradictionEntry(severity="low"))
        log.record(ContradictionEntry(severity="high"))
        assert len(log.get_by_severity("high")) == 2

    def test_contradiction_type_preserved(self):
        log = ContradictionLog()
        log.record(ContradictionEntry(contradiction_type="pragmatic"))
        assert log._entries[0].contradiction_type == "pragmatic"


# ---------------------------------------------------------------------------
# 5. UnsolvedConceptsBuffer
# ---------------------------------------------------------------------------

class TestUnsolvedConceptsBuffer:
    def test_add_and_resolve(self):
        buf = UnsolvedConceptsBuffer()
        eid = buf.add(UnsolvedConceptEntry(concept_formulation="What is consciousness?"))
        assert len(buf.get_all_active()) == 1
        buf.resolve(eid, note="defined via IIT theory")
        assert len(buf.get_all_active()) == 0

    def test_stagnation_increments(self):
        buf = UnsolvedConceptsBuffer()
        eid = buf.add(UnsolvedConceptEntry())
        buf.tick_all()
        buf.tick_all()
        entry = buf.get_by_id(eid)
        assert entry.stagnation_cycles == 2

    def test_dream_candidate_threshold(self):
        buf = UnsolvedConceptsBuffer()
        eid = buf.add(UnsolvedConceptEntry())
        for _ in range(5):
            buf.tick_all()
        candidates = buf.get_dream_candidates(threshold=5)
        assert eid in [e.entry_id for e in candidates]

    def test_resolved_not_dream_candidate(self):
        buf = UnsolvedConceptsBuffer()
        eid = buf.add(UnsolvedConceptEntry())
        for _ in range(10):
            buf.tick_all()
        buf.resolve(eid)
        candidates = buf.get_dream_candidates(threshold=5)
        assert candidates == []

    def test_get_by_id_returns_none_for_missing(self):
        buf = UnsolvedConceptsBuffer()
        assert buf.get_by_id("nonexistent") is None

    def test_tick_all_skips_resolved(self):
        buf = UnsolvedConceptsBuffer()
        eid = buf.add(UnsolvedConceptEntry())
        buf.resolve(eid)
        buf.tick_all()   # should not raise or increment
        entry = buf.get_by_id(eid)
        assert entry.stagnation_cycles == 0


# ---------------------------------------------------------------------------
# 6. SelfReflectionLog
# ---------------------------------------------------------------------------

class TestSelfReflectionLog:
    def test_record_and_filter(self):
        log = SelfReflectionLog()
        log.record(SelfReflectionEntry(observation_type="bias_detected", severity="medium"))
        log.record(SelfReflectionEntry(observation_type="identity_drift", severity="high"))
        assert len(log.get_by_type("bias_detected")) == 1
        assert len(log.get_all()) == 2

    def test_severity_preserved(self):
        log = SelfReflectionLog()
        log.record(SelfReflectionEntry(severity="critical"))
        assert log.get_all()[0].severity == "critical"


# ---------------------------------------------------------------------------
# 7. IdentityMemoryLog
# ---------------------------------------------------------------------------

class TestIdentityMemoryLog:
    def test_record_and_filter_by_aspect(self):
        log = IdentityMemoryLog()
        log.record(IdentityMemoryEntry(identity_aspect="values"))
        log.record(IdentityMemoryEntry(identity_aspect="capabilities"))
        assert len(log.get_by_aspect("values"))       == 1
        assert len(log.get_by_aspect("capabilities")) == 1

    def test_all_entries_retrievable(self):
        log = IdentityMemoryLog()
        log.record(IdentityMemoryEntry())
        log.record(IdentityMemoryEntry())
        assert len(log.get_all()) == 2

    def test_stability_assessment_preserved(self):
        log = IdentityMemoryLog()
        log.record(IdentityMemoryEntry(stability_assessment="challenged"))
        assert log.get_all()[0].stability_assessment == "challenged"


# ---------------------------------------------------------------------------
# 8. DreamLog
# ---------------------------------------------------------------------------

class TestDreamLog:
    def test_record_and_validate(self):
        log = DreamLog()
        eid = log.record(DreamEntry(
            dream_trigger_id="ucid_001",
            dream_content="Perhaps consciousness emerges from complexity.",
        ))
        log.validate_candidate(eid, resolved_concept_id="ucid_001")
        assert log.get_all()[0].validated is True
        assert log.get_all()[0].resolution_status == "validated"
        assert log.get_all()[0].cross_link_id == "ucid_001"

    def test_unvalidated_by_default(self):
        log = DreamLog()
        log.record(DreamEntry())
        assert log.get_all()[0].validated is False

    def test_all_episodes_persist(self):
        log = DreamLog()
        for i in range(5):
            log.record(DreamEntry(dream_trigger_id=f"ucid_{i}"))
        assert len(log.get_all()) == 5


# ---------------------------------------------------------------------------
# SpecializedLogs bundle
# ---------------------------------------------------------------------------

class TestSpecializedLogs:
    def test_all_logs_initialized(self):
        logs = SpecializedLogs()
        assert isinstance(logs.learning,      LearningSystemLog)
        assert isinstance(logs.sandbox,       SandboxLog)
        assert isinstance(logs.paradox,       ParadoxLog)
        assert isinstance(logs.contradiction, ContradictionLog)
        assert isinstance(logs.unsolved,      UnsolvedConceptsBuffer)
        assert isinstance(logs.self_reflect,  SelfReflectionLog)
        assert isinstance(logs.identity,      IdentityMemoryLog)
        assert isinstance(logs.dream,         DreamLog)

    def test_each_log_independent(self):
        logs1 = SpecializedLogs()
        logs2 = SpecializedLogs()
        logs1.learning.record(LearningEntry())
        assert len(logs2.learning) == 0
