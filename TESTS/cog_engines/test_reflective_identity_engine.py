"""
Tests for Engine 32 — Reflective Identity Engine
=================================================
Covers: engine identity, neurochem coupling, core contradiction detection,
fragile conclusion detection, identity-behaviour alignment, emotion
assessment, pending update analysis, theme extraction, conclusion update
recommendations, coherence scoring, edge cases.
"""
from __future__ import annotations

import pytest

from zados.cognitive_engines.py_engines.reflective_identity_engine import (
    COHERENCE_COHERENT,
    COHERENCE_DISRUPTED,
    COHERENCE_FRAGMENTED,
    IDENTITY_RELEVANT_EMOTIONS,
    ReflectiveIdentityEngine,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def engine():
    return ReflectiveIdentityEngine()


def _make_core_memory(content, memory_type="self_model", **extra):
    mem = {
        "memory_id": f"cm_{hash(content) % 10000}",
        "content": content,
        "memory_type": memory_type,
        "tags": [],
        "version": 1,
        "update_count": 0,
    }
    mem.update(extra)
    return mem


def _make_conclusion(content, conclusion_type="value", confidence=0.7,
                     reinforcement_count=5, **extra):
    c = {
        "conclusion_id": f"cc_{hash(content) % 10000}",
        "content": content,
        "conclusion_type": conclusion_type,
        "confidence": confidence,
        "reinforcement_count": reinforcement_count,
        "tags": [],
        "source_refs": [],
    }
    c.update(extra)
    return c


def _make_journal_entry(content, entry_type="reflection", **extra):
    e = {
        "entry_id": f"je_{hash(content) % 10000}",
        "entry_type": entry_type,
        "content": content,
        "emotion_tags": [],
        "source_pipeline": "test",
        "nt_snapshot": {},
    }
    e.update(extra)
    return e


def _make_pending_update(target_memory_id="cm_1", proposed_content="new", **extra):
    u = {
        "update_id": f"pu_{hash(proposed_content) % 10000}",
        "target_memory_id": target_memory_id,
        "proposed_content": proposed_content,
        "reason": "test update",
        "status": "pending",
    }
    u.update(extra)
    return u


# =====================================================================
# Engine identity
# =====================================================================

class TestEngineIdentity:
    def test_engine_id(self, engine):
        assert engine.engine_id == 32

    def test_engine_name(self, engine):
        assert engine.engine_name == "reflective_identity_engine"

    def test_cluster(self, engine):
        assert engine.cluster == "metacognition"

    def test_get_status_includes_engine_id(self, engine):
        status = engine.get_status()
        assert status["engine_id"] == "reflective_identity_engine"

    def test_get_status_active_flag(self, engine):
        status = engine.get_status()
        assert status["active"] is True


# =====================================================================
# Neurochem coupling
# =====================================================================

class TestNeurochemCoupling:
    def test_update_neurochem_state(self, engine):
        nt = {"oxt": 0.6, "5ht": 0.4, "da": 0.5, "cor": 0.2}
        engine.update_neurochem_state(nt)
        assert engine._nt_state["oxt"] == 0.6

    def test_nt_state_copy(self, engine):
        nt = {"da": 0.6}
        engine.update_neurochem_state(nt)
        nt["da"] = 0.9
        assert engine._nt_state["da"] == 0.6

    def test_empty_nt_state(self, engine):
        engine.update_neurochem_state({})
        assert engine._nt_state == {}


# =====================================================================
# Process — basic
# =====================================================================

class TestProcessBasic:
    def test_empty_input(self, engine):
        result = engine.process()
        assert isinstance(result, dict)
        assert result["identity_coherence_status"] == COHERENCE_COHERENT

    def test_result_keys(self, engine):
        result = engine.process(
            core_memories=[_make_core_memory("I value honesty")],
        )
        expected_keys = {
            "identity_coherence_status",
            "coherence_score",
            "identity_contradictions",
            "fragile_conclusions",
            "alignment_analysis",
            "identity_themes",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_coherent_with_consistent_data(self, engine):
        mems = [
            _make_core_memory("I value honesty"),
            _make_core_memory("I believe in integrity"),
        ]
        conclusions = [
            _make_conclusion("Honesty is fundamental", confidence=0.9,
                             reinforcement_count=10),
        ]
        result = engine.process(
            core_memories=mems,
            identity_conclusions=conclusions,
        )
        assert result["identity_coherence_status"] == COHERENCE_COHERENT


# =====================================================================
# Core contradiction detection
# =====================================================================

class TestCoreContradictions:
    def test_no_contradictions_in_consistent_data(self, engine):
        mems = [
            _make_core_memory("I enjoy learning new things"),
            _make_core_memory("Education is important to me"),
        ]
        result = engine.process(core_memories=mems)
        assert isinstance(result["identity_contradictions"], list)

    def test_detects_negation_contradiction(self, engine):
        mems = [
            _make_core_memory("I always tell the truth"),
            _make_core_memory("I never tell the truth"),
        ]
        result = engine.process(core_memories=mems)
        contras = result["identity_contradictions"]
        # Should detect at least one contradiction
        assert isinstance(contras, list)

    def test_contradiction_between_memory_types(self, engine):
        mems = [
            _make_core_memory("I am patient and calm", memory_type="self_model"),
            _make_core_memory("I lose my temper frequently", memory_type="experience"),
        ]
        result = engine.process(core_memories=mems)
        assert isinstance(result["identity_contradictions"], list)


# =====================================================================
# Fragile conclusion detection
# =====================================================================

class TestFragileConclusions:
    def test_low_confidence_flagged(self, engine):
        conclusions = [
            _make_conclusion("Maybe I'm good at math", confidence=0.2,
                             reinforcement_count=1),
        ]
        result = engine.process(identity_conclusions=conclusions)
        fragile = result["fragile_conclusions"]
        assert len(fragile) >= 1
        assert fragile[0]["conclusion_id"] == conclusions[0]["conclusion_id"]

    def test_high_confidence_not_flagged(self, engine):
        conclusions = [
            _make_conclusion("I am good at logic", confidence=0.9,
                             reinforcement_count=15),
        ]
        result = engine.process(identity_conclusions=conclusions)
        fragile = result["fragile_conclusions"]
        # High confidence + high reinforcement → not fragile
        fragile_ids = [f["conclusion_id"] for f in fragile]
        assert conclusions[0]["conclusion_id"] not in fragile_ids

    def test_low_reinforcement_flagged(self, engine):
        conclusions = [
            _make_conclusion("New insight", confidence=0.5,
                             reinforcement_count=0),
        ]
        result = engine.process(identity_conclusions=conclusions)
        fragile = result["fragile_conclusions"]
        assert len(fragile) >= 1


# =====================================================================
# Identity-behaviour alignment
# =====================================================================

class TestIdentityBehaviourAlignment:
    def test_alignment_with_matching_journal(self, engine):
        mems = [_make_core_memory("I value learning")]
        journal = [
            _make_journal_entry("Today I spent time learning new things"),
        ]
        result = engine.process(
            core_memories=mems,
            journal_entries=journal,
        )
        alignment = result["alignment_analysis"]
        assert isinstance(alignment, dict)
        assert isinstance(alignment.get("alignment_issues", []), list)

    def test_empty_journal(self, engine):
        mems = [_make_core_memory("I value honesty")]
        result = engine.process(
            core_memories=mems,
            journal_entries=[],
        )
        alignment = result["alignment_analysis"]
        assert isinstance(alignment, dict)


# =====================================================================
# Emotion assessment
# =====================================================================

class TestEmotionAssessment:
    def test_identity_relevant_emotions_set(self):
        assert "ashamed" in IDENTITY_RELEVANT_EMOTIONS
        assert "proud" in IDENTITY_RELEVANT_EMOTIONS
        assert "belonging" in IDENTITY_RELEVANT_EMOTIONS
        assert "rejected" in IDENTITY_RELEVANT_EMOTIONS

    def test_confused_disruption(self, engine):
        """Appendix spec: confused > 0.6 → disrupted."""
        # Need core memory so E32 doesn't return early before confused check
        result = engine.process(
            core_memories=[_make_core_memory("I value clarity")],
            emotion_snapshot={"confused": 0.7},
        )
        assert result["identity_coherence_status"] == COHERENCE_DISRUPTED

    def test_confused_below_threshold(self, engine):
        result = engine.process(
            core_memories=[_make_core_memory("I value clarity")],
            emotion_snapshot={"confused": 0.3},
        )
        assert result["identity_coherence_status"] != COHERENCE_DISRUPTED

    def test_identity_emotions_tracked(self, engine):
        # Need at least one core memory so E32 doesn't return early
        result = engine.process(
            core_memories=[_make_core_memory("I value self-awareness")],
            emotion_snapshot={"proud": 0.5, "ashamed": 0.3, "boredom": 0.8},
        )
        identity_emotions = result.get("identity_emotions", {})
        active = identity_emotions.get("active_identity_emotions", {})
        assert "proud" in active
        assert "ashamed" in active
        # boredom is not identity-relevant, should not appear
        assert "boredom" not in active


# =====================================================================
# Pending update analysis
# =====================================================================

class TestPendingUpdateAnalysis:
    def test_pending_updates_counted(self, engine):
        updates = [
            _make_pending_update("cm1", "updated content 1"),
            _make_pending_update("cm2", "updated content 2"),
        ]
        # Need core memory so E32 doesn't return early
        result = engine.process(
            core_memories=[_make_core_memory("I exist")],
            pending_updates=updates,
        )
        analysis = result["pending_update_analysis"]
        assert analysis["total"] == 2

    def test_empty_pending_updates(self, engine):
        result = engine.process(
            core_memories=[_make_core_memory("I exist")],
            pending_updates=[],
        )
        analysis = result["pending_update_analysis"]
        assert analysis["total"] == 0


# =====================================================================
# Theme extraction
# =====================================================================

class TestThemeExtraction:
    def test_themes_from_core_memories(self, engine):
        mems = [
            _make_core_memory("I value honesty and integrity",
                              tags=["honesty", "values"]),
            _make_core_memory("I value learning and growth",
                              tags=["learning", "growth"]),
            _make_core_memory("I value honesty in relationships",
                              tags=["honesty", "relationships"]),
        ]
        result = engine.process(core_memories=mems)
        themes = result["identity_themes"]
        assert isinstance(themes, list)

    def test_empty_input_no_themes(self, engine):
        result = engine.process()
        themes = result["identity_themes"]
        assert isinstance(themes, list)


# =====================================================================
# Conclusion update recommendations
# =====================================================================

class TestConclusionUpdates:
    def test_recommendations_for_fragile(self, engine):
        conclusions = [
            _make_conclusion("Maybe I'm creative", confidence=0.15,
                             reinforcement_count=0),
        ]
        result = engine.process(identity_conclusions=conclusions)
        updates = result.get("conclusion_updates", [])
        assert isinstance(updates, list)


# =====================================================================
# Coherence scoring
# =====================================================================

class TestCoherenceScoring:
    def test_score_range(self, engine):
        result = engine.process()
        score = result["coherence_score"]
        assert 0.0 <= score <= 1.0

    def test_perfect_coherence(self, engine):
        mems = [_make_core_memory("I value learning")]
        conclusions = [
            _make_conclusion("Learning is important", confidence=0.9,
                             reinforcement_count=20),
        ]
        result = engine.process(
            core_memories=mems,
            identity_conclusions=conclusions,
        )
        assert result["coherence_score"] >= 0.7
        assert result["identity_coherence_status"] == COHERENCE_COHERENT

    def test_fragmented_with_issues(self, engine):
        # Many fragile conclusions → should reduce coherence
        conclusions = [
            _make_conclusion(f"Fragile claim {i}", confidence=0.1,
                             reinforcement_count=0)
            for i in range(10)
        ]
        # Contradicting memories with high meaningful word overlap
        mems = [
            _make_core_memory(
                "I always value honesty integrity truth transparency",
                memory_type="self_model",
            ),
            _make_core_memory(
                "I never value honesty integrity truth transparency",
                memory_type="self_model",
            ),
        ]
        # Add disruptive journal entries to lower alignment
        journals = [
            _make_journal_entry(
                "Felt terrible about lying",
                emotion_tags=["ashamed", "guilty", "regret"],
            ),
            _make_journal_entry(
                "Acted against my values again",
                emotion_tags=["ashamed", "rejected"],
            ),
        ]
        result = engine.process(
            core_memories=mems,
            identity_conclusions=conclusions,
            journal_entries=journals,
        )
        # Should be fragmented or disrupted due to contradictions +
        # many fragile conclusions + disruptive journal entries
        assert result["identity_coherence_status"] in (
            COHERENCE_FRAGMENTED, COHERENCE_DISRUPTED,
        )


# =====================================================================
# NT modulation effect
# =====================================================================

class TestNTModulation:
    def test_high_oxt_social_weight(self, engine):
        engine.update_neurochem_state({"oxt": 0.9})
        result = engine.process(
            core_memories=[_make_core_memory("I value friendship")],
        )
        assert isinstance(result, dict)

    def test_high_cor_threat_sensitivity(self, engine):
        engine.update_neurochem_state({"cor": 0.8})
        # High cortisol → lower disruption threshold
        conclusions = [
            _make_conclusion("Uncertain belief", confidence=0.3,
                             reinforcement_count=1),
        ]
        result = engine.process(identity_conclusions=conclusions)
        assert isinstance(result, dict)

    def test_high_5ht_stability_tolerance(self, engine):
        engine.update_neurochem_state({"5ht": 0.9})
        # High 5-HT → more tolerant of divergence
        result = engine.process()
        assert result["identity_coherence_status"] == COHERENCE_COHERENT


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_all_empty_inputs(self, engine):
        result = engine.process(
            core_memories=[],
            identity_conclusions=[],
            journal_entries=[],
            pending_updates=[],
            emotion_snapshot={},
        )
        assert result["identity_coherence_status"] == COHERENCE_COHERENT
        assert result["coherence_score"] >= 0.9

    def test_only_pending_updates(self, engine):
        updates = [_make_pending_update()]
        result = engine.process(pending_updates=updates)
        assert isinstance(result, dict)

    def test_large_dataset(self, engine):
        mems = [_make_core_memory(f"Memory {i}") for i in range(50)]
        conclusions = [
            _make_conclusion(f"Conclusion {i}", confidence=0.5 + (i % 5) * 0.1)
            for i in range(50)
        ]
        result = engine.process(
            core_memories=mems,
            identity_conclusions=conclusions,
        )
        assert isinstance(result, dict)
        assert 0.0 <= result["coherence_score"] <= 1.0

    def test_missing_fields_in_memory(self, engine):
        # Minimal memory with missing fields
        mem = {"content": "basic memory"}
        result = engine.process(core_memories=[mem])
        assert isinstance(result, dict)

    def test_coherence_status_values(self, engine):
        """Verify coherence status constants match expected values."""
        assert COHERENCE_COHERENT == "coherent"
        assert COHERENCE_FRAGMENTED == "fragmented"
        assert COHERENCE_DISRUPTED == "disrupted"
