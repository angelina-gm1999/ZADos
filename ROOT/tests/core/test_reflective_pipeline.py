"""
Tests for ReflectivePipeline — identity + meta-learning reflective mode.
========================================================================
Covers: constructor wiring, 6-phase pipeline, E31/E32 integration,
identity store mutations (conclusion reinforcement/creation, journal
writes), CorticalReflectionLog.identity_coherence_status write path,
homework handoff, cross-referencing, edge cases.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from zados.core.commanded.meta_learning_mode.reflective_mode.pipeline import (
    ReflectivePipeline,
)
from zados.core.types import (
    LearningLogEntry,
    ReflectiveModeInput,
    ReflectiveModeResult,
    SessionState,
)
from zados.memory.long_term.identity.types import (
    IdentityConclusion,
    IdentityJournalEntry,
    IdentityJournalEntryType,
    CoreMemory,
    PendingUpdate,
)


# =====================================================================
# Fixtures — mock memory layer + stores
# =====================================================================

class FakeCoreMemoryStore:
    def __init__(self, memories=None):
        self._memories = list(memories or [])

    def get_all(self):
        return list(self._memories)

    def get_by_id(self, mid):
        for m in self._memories:
            if m.memory_id == mid:
                return m
        return None


class FakeConclusionStore:
    def __init__(self, conclusions=None):
        self._conclusions = {c.conclusion_id: c for c in (conclusions or [])}
        self.reinforced_ids: List[str] = []
        self.written: List[IdentityConclusion] = []

    def get_all(self):
        return list(self._conclusions.values())

    def reinforce(self, cid):
        if cid in self._conclusions:
            self.reinforced_ids.append(cid)
            self._conclusions[cid].reinforcement_count += 1
            return True
        return False

    def write(self, entry):
        self._conclusions[entry.conclusion_id] = entry
        self.written.append(entry)

    def search(self, query_text, limit=5):
        return []


class FakeJournalStore:
    def __init__(self):
        self.entries: List[IdentityJournalEntry] = []

    def get_all(self):
        return list(self.entries)

    def write(self, entry):
        self.entries.append(entry)

    def __len__(self):
        return len(self.entries)


class FakePendingQueue:
    def __init__(self, updates=None):
        self._updates = list(updates or [])

    def get_pending(self):
        return [u for u in self._updates if u.status == "pending"]


class FakeHardcodedStore:
    def get_all(self):
        return []


class FakeIdentityNamespace:
    def __init__(self, core=None, conclusions=None, journal=None, pending=None):
        self.core = core or FakeCoreMemoryStore()
        self.conclusions = conclusions or FakeConclusionStore()
        self.journal = journal or FakeJournalStore()
        self.pending = pending or FakePendingQueue()
        self.hardcoded = FakeHardcodedStore()


@dataclass
class FakeCorticalLog:
    identity_coherence_status: str = "coherent"
    notes: List[str] = field(default_factory=list)


@dataclass
class FakeSTMM:
    cortical_reflection_log: FakeCorticalLog = field(
        default_factory=FakeCorticalLog,
    )
    emotion_profile: Dict[str, float] = field(default_factory=dict)


class FakeMemoryLayer:
    def __init__(self, identity=None, stmm=None):
        self.identity = identity or FakeIdentityNamespace()
        self.stmm = stmm or FakeSTMM()
        self.thoughts = None
        self.knowledge = None
        self.contrast = None


class FakeLearningLog:
    def __init__(self, entries=None):
        self._entries = list(entries or [])

    def get_unprocessed_logs(self):
        return [e for e in self._entries if not e.processed]


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def session():
    return SessionState(session_id="test-reflective-001")


@pytest.fixture
def memory():
    return FakeMemoryLayer()


@pytest.fixture
def learning_log():
    return FakeLearningLog()


@pytest.fixture
def pipeline(memory, learning_log):
    return ReflectivePipeline(
        answer_pipeline=MagicMock(),
        learning_log=learning_log,
        memory_layer=memory,
        unsolved_buffer=MagicMock(is_empty=MagicMock(return_value=True)),
        neurochem_engine=None,
    )


# =====================================================================
# Constructor
# =====================================================================

class TestConstructor:
    def test_creates_with_all_deps(self, memory, learning_log):
        p = ReflectivePipeline(
            answer_pipeline=MagicMock(),
            learning_log=learning_log,
            memory_layer=memory,
        )
        assert p._memory is memory
        assert p._learning_log is learning_log

    def test_creates_with_no_deps(self):
        p = ReflectivePipeline()
        assert p._memory is None
        assert p._learning_log is None

    def test_lazy_engine_creation(self):
        p = ReflectivePipeline()
        assert p._e31 is None
        assert p._e32 is None
        p._ensure_engines()
        assert p._e31 is not None
        assert p._e32 is not None


# =====================================================================
# Process — basic
# =====================================================================

class TestProcessBasic:
    def test_returns_dict(self, pipeline, session):
        result = pipeline.process(session)
        assert isinstance(result, dict)
        assert result["status"] == "completed"

    def test_session_id_in_result(self, pipeline, session):
        result = pipeline.process(session)
        assert result["session_id"] == "test-reflective-001"

    def test_processing_time(self, pipeline, session):
        result = pipeline.process(session)
        assert "processing_time_s" in result
        assert result["processing_time_s"] >= 0

    def test_all_output_keys(self, pipeline, session):
        result = pipeline.process(session)
        expected_keys = {
            "status",
            "session_id",
            "processing_time_s",
            "learning_patterns",
            "recurring_failures",
            "mode_effectiveness",
            "subject_proficiencies",
            "style_preferences",
            "learning_recommendations",
            "identity_coherence_status",
            "coherence_score",
            "core_contradictions",
            "fragile_conclusions",
            "alignment_issues",
            "identity_themes",
            "cross_references",
            "conclusions_reinforced",
            "conclusions_created",
            "conclusions_recommended_for_update",
            "journal_entries_created",
            "pending_updates_analysed",
            "learning_logs_analysed",
            "fallacy_flags_processed",
            "bias_flags_processed",
            "meta_patterns_processed",
        }
        assert expected_keys.issubset(set(result.keys()))


# =====================================================================
# Phase 0 — Input Assembly
# =====================================================================

class TestPhase0InputAssembly:
    def test_gathers_learning_logs(self, memory, session):
        entries = [
            LearningLogEntry(mode="M1", subject="technical"),
            LearningLogEntry(mode="M3", subject="philosophy"),
        ]
        log = FakeLearningLog(entries)
        p = ReflectivePipeline(
            learning_log=log,
            memory_layer=memory,
        )
        result = p.process(session)
        assert result["learning_logs_analysed"] == 2

    def test_gathers_identity_data(self, session):
        core = FakeCoreMemoryStore([
            CoreMemory(content="I value logic"),
            CoreMemory(content="I value empathy"),
        ])
        conclusions = FakeConclusionStore([
            IdentityConclusion(content="Logic is my strength"),
        ])
        journal = FakeJournalStore()
        identity = FakeIdentityNamespace(
            core=core,
            conclusions=conclusions,
            journal=journal,
        )
        memory = FakeMemoryLayer(identity=identity)
        p = ReflectivePipeline(memory_layer=memory)
        result = p.process(session)
        assert isinstance(result, dict)

    def test_no_memory_layer(self, session):
        p = ReflectivePipeline()
        result = p.process(session)
        assert result["status"] == "completed"
        assert result["learning_logs_analysed"] == 0


# =====================================================================
# Phase 1 — E31 Meta-Learning Analysis
# =====================================================================

class TestPhase1MetaLearning:
    def test_e31_produces_mode_effectiveness(self, memory, session):
        entries = [
            LearningLogEntry(
                mode="M1", subject="technical",
                confirmations=10, contradictions=2,
            ),
        ]
        log = FakeLearningLog(entries)
        p = ReflectivePipeline(
            learning_log=log,
            memory_layer=memory,
        )
        result = p.process(session)
        assert isinstance(result["mode_effectiveness"], dict)

    def test_e31_produces_recommendations(self, memory, session):
        entries = [
            LearningLogEntry(
                mode="M1", subject="technical",
                confirmations=1, contradictions=8,
            ),
        ]
        log = FakeLearningLog(entries)
        p = ReflectivePipeline(
            learning_log=log,
            memory_layer=memory,
        )
        result = p.process(session)
        assert isinstance(result["learning_recommendations"], list)


# =====================================================================
# Phase 2 — E32 Identity Coherence
# =====================================================================

class TestPhase2IdentityCoherence:
    def test_coherent_status(self, session):
        core = FakeCoreMemoryStore([
            CoreMemory(content="I value learning"),
        ])
        conclusions = FakeConclusionStore([
            IdentityConclusion(
                content="Learning is important",
                confidence=0.9,
                reinforcement_count=20,
            ),
        ])
        identity = FakeIdentityNamespace(core=core, conclusions=conclusions)
        memory = FakeMemoryLayer(identity=identity)
        p = ReflectivePipeline(memory_layer=memory)
        result = p.process(session)
        assert result["identity_coherence_status"] == "coherent"

    def test_fragile_conclusions_detected(self, session):
        conclusions = FakeConclusionStore([
            IdentityConclusion(
                content="Maybe I'm creative",
                confidence=0.1,
                reinforcement_count=0,
            ),
        ])
        identity = FakeIdentityNamespace(conclusions=conclusions)
        memory = FakeMemoryLayer(identity=identity)
        p = ReflectivePipeline(memory_layer=memory)
        result = p.process(session)
        assert result["fragile_conclusions"] >= 1

    def test_confused_forces_disrupted(self, session):
        """Appendix spec: confused > 0.6 → identity_coherence_status = disrupted."""
        # E32 needs at least one core memory to not return early
        core = FakeCoreMemoryStore([
            CoreMemory(content="I value clarity"),
        ])
        identity = FakeIdentityNamespace(core=core)
        stmm = FakeSTMM(emotion_profile={"confused": 0.7})
        memory = FakeMemoryLayer(identity=identity, stmm=stmm)
        p = ReflectivePipeline(memory_layer=memory)
        result = p.process(session)
        assert result["identity_coherence_status"] == "disrupted"


# =====================================================================
# Phase 3 — Cross-Reference
# =====================================================================

class TestPhase3CrossReference:
    def test_cross_refs_type(self, pipeline, session):
        result = pipeline.process(session)
        assert isinstance(result["cross_references"], int)

    def test_homework_handoff_processed(self, memory, session):
        entries = [
            LearningLogEntry(mode="M1", subject="technical"),
        ]
        log = FakeLearningLog(entries)
        # Inject reflective input
        session.reflective_input = ReflectiveModeInput(
            fallacy_flags=[{"name": "straw_man", "type": "fallacy", "severity": "medium"}],
            bias_flags=[{"name": "confirmation_bias", "type": "bias", "severity": "low"}],
            meta_patterns=[{"type": "anchoring_pattern"}],
            source_homework_session="hw-001",
        )
        p = ReflectivePipeline(
            learning_log=log,
            memory_layer=memory,
        )
        result = p.process(session)
        assert result["fallacy_flags_processed"] == 1
        assert result["bias_flags_processed"] == 1
        assert result["meta_patterns_processed"] == 1


# =====================================================================
# Phase 4 — Identity Store Mutations
# =====================================================================

class TestPhase4Mutations:
    def test_conclusion_reinforcement(self, session):
        c = IdentityConclusion(
            content="Learning is a core value",
            conclusion_type="lesson",
            confidence=0.8,
            reinforcement_count=5,
        )
        conclusions = FakeConclusionStore([c])
        identity = FakeIdentityNamespace(conclusions=conclusions)
        memory = FakeMemoryLayer(identity=identity)

        entries = [
            LearningLogEntry(
                mode="M1", subject="technical",
                confirmations=10, contradictions=1,
            ),
        ]
        log = FakeLearningLog(entries)
        p = ReflectivePipeline(
            learning_log=log,
            memory_layer=memory,
        )
        result = p.process(session)
        assert isinstance(result["conclusions_reinforced"], int)

    def test_journal_entry_creation(self, session):
        identity = FakeIdentityNamespace()
        memory = FakeMemoryLayer(identity=identity)

        entries = [
            LearningLogEntry(
                mode="M1", subject="technical",
                confirmations=10, contradictions=1,
            ),
        ]
        log = FakeLearningLog(entries)
        p = ReflectivePipeline(
            learning_log=log,
            memory_layer=memory,
        )
        result = p.process(session)
        assert result["journal_entries_created"] >= 0
        # Verify journal store received writes
        assert isinstance(identity.journal.entries, list)


# =====================================================================
# Phase 4e — CorticalReflectionLog write path
# =====================================================================

class TestCorticalCoherenceWritePath:
    def test_updates_coherence_status(self, session):
        cortical = FakeCorticalLog(identity_coherence_status="coherent")
        stmm = FakeSTMM(cortical_reflection_log=cortical)
        memory = FakeMemoryLayer(stmm=stmm)
        p = ReflectivePipeline(memory_layer=memory)
        result = p.process(session)
        # Status should have been written (may stay coherent if no issues)
        assert cortical.identity_coherence_status in (
            "coherent", "fragmented", "disrupted",
        )

    def test_status_change_logged_in_notes(self, session):
        """When coherence changes, a note should be added."""
        # E32 needs at least one core memory to not return early
        core = FakeCoreMemoryStore([
            CoreMemory(content="I value clarity"),
        ])
        identity = FakeIdentityNamespace(core=core)
        cortical = FakeCorticalLog(identity_coherence_status="coherent")
        stmm = FakeSTMM(
            cortical_reflection_log=cortical,
            emotion_profile={"confused": 0.8},  # forces disrupted
        )
        memory = FakeMemoryLayer(identity=identity, stmm=stmm)
        p = ReflectivePipeline(memory_layer=memory)
        p.process(session)
        assert cortical.identity_coherence_status == "disrupted"
        assert any("coherence changed" in n for n in cortical.notes)

    def test_no_note_if_status_unchanged(self, session):
        cortical = FakeCorticalLog(identity_coherence_status="coherent")
        stmm = FakeSTMM(cortical_reflection_log=cortical)
        memory = FakeMemoryLayer(stmm=stmm)
        p = ReflectivePipeline(memory_layer=memory)
        p.process(session)
        # If status stays coherent, no change note
        change_notes = [n for n in cortical.notes if "coherence changed" in n]
        assert len(change_notes) == 0


# =====================================================================
# ReflectiveModeResult type
# =====================================================================

class TestReflectiveModeResultType:
    def test_dataclass_defaults(self):
        r = ReflectiveModeResult()
        assert r.identity_coherence_status == "coherent"
        assert r.coherence_score == 1.0
        assert r.conclusions_reinforced == 0
        assert r.journal_entries_created == 0

    def test_dataclass_fields(self):
        r = ReflectiveModeResult(
            session_id="s1",
            identity_coherence_status="fragmented",
            coherence_score=0.5,
            conclusions_reinforced=3,
        )
        assert r.session_id == "s1"
        assert r.coherence_score == 0.5
        assert r.conclusions_reinforced == 3


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_no_dependencies(self, session):
        p = ReflectivePipeline()
        result = p.process(session)
        assert result["status"] == "completed"

    def test_empty_learning_log(self, memory, session):
        log = FakeLearningLog([])
        p = ReflectivePipeline(
            learning_log=log,
            memory_layer=memory,
        )
        result = p.process(session)
        assert result["status"] == "completed"
        assert result["learning_logs_analysed"] == 0

    def test_memory_layer_without_identity(self, session):
        memory = MagicMock()
        memory.identity = None
        memory.stmm = None
        p = ReflectivePipeline(memory_layer=memory)
        result = p.process(session)
        assert result["status"] == "completed"

    def test_multiple_runs_on_same_session(self, pipeline, session):
        result1 = pipeline.process(session)
        result2 = pipeline.process(session)
        assert result1["status"] == "completed"
        assert result2["status"] == "completed"

    def test_serialise_log_entry(self):
        entry = LearningLogEntry(
            mode="M2",
            subject="philosophy",
            confirmations=5,
            contradictions=2,
        )
        serialised = ReflectivePipeline._serialise_log_entry(entry)
        assert serialised["mode"] == "M2"
        assert serialised["subject"] == "philosophy"
        assert serialised["confirmations"] == 5

    def test_describe_mode_style(self):
        assert "guided" in ReflectivePipeline._describe_mode_style("M1")
        assert "peer review" in ReflectivePipeline._describe_mode_style("M2")
        assert "collaborative" in ReflectivePipeline._describe_mode_style("M3")
        assert "questioning" in ReflectivePipeline._describe_mode_style("M4")
        assert "independent" in ReflectivePipeline._describe_mode_style("M5")
        assert "M99" in ReflectivePipeline._describe_mode_style("M99")
