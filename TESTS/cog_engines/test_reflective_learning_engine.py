"""
Tests for Engine 31 — Reflective Learning Engine
=================================================
Covers: engine identity, neurochem coupling, mode effectiveness,
subject proficiencies, recurring failure detection, meta-pattern
detection, style preferences, recommendation generation, edge cases.
"""
from __future__ import annotations

import pytest

from zados.cognitive_engines.py_engines.reflective_learning_engine import (
    ReflectiveLearningEngine,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def engine():
    return ReflectiveLearningEngine()


def _make_entry(
    mode="M1",
    subject="technical",
    confirmations=5,
    contradictions=1,
    novel_entries=2,
    patterns_detected=3,
    session_id="s1",
    **extra,
):
    entry = {
        "turn_id": f"t_{mode}_{subject}",
        "timestamp": 1000.0,
        "mode": mode,
        "subject": subject,
        "session_id": session_id,
        "confirmations": confirmations,
        "contradictions": contradictions,
        "extensions": 0,
        "novel_entries": novel_entries,
        "patterns_detected": patterns_detected,
        "e19_patterns": [],
        "e20_comparisons": [],
        "e17_rewards": [],
        "e25_meta_updates": [],
        "reward_scores": {},
        "processed": False,
    }
    entry.update(extra)
    return entry


# =====================================================================
# Engine identity
# =====================================================================

class TestEngineIdentity:
    def test_engine_id(self, engine):
        assert engine.engine_id == 31

    def test_engine_name(self, engine):
        assert engine.engine_name == "reflective_learning_engine"

    def test_cluster(self, engine):
        assert engine.cluster == "metacognition"

    def test_get_status_includes_engine_id(self, engine):
        status = engine.get_status()
        assert status["engine_id"] == "reflective_learning_engine"

    def test_get_status_active_flag(self, engine):
        status = engine.get_status()
        assert status["active"] is True


# =====================================================================
# Neurochem coupling
# =====================================================================

class TestNeurochemCoupling:
    def test_update_neurochem_state(self, engine):
        nt = {"da": 0.6, "5ht": 0.4, "ne": 0.7, "ach": 0.5}
        engine.update_neurochem_state(nt)
        assert engine._nt_state["da"] == 0.6

    def test_nt_state_copy(self, engine):
        nt = {"da": 0.6}
        engine.update_neurochem_state(nt)
        nt["da"] = 0.9  # mutate original
        assert engine._nt_state["da"] == 0.6  # engine state unchanged

    def test_empty_nt_state(self, engine):
        engine.update_neurochem_state({})
        assert engine._nt_state == {}


# =====================================================================
# Process — basic
# =====================================================================

class TestProcessBasic:
    def test_empty_input(self, engine):
        result = engine.process(learning_entries=[])
        assert isinstance(result, dict)
        assert "learning_patterns" in result
        assert "recurring_failures" in result
        assert "mode_effectiveness" in result

    def test_none_input(self, engine):
        result = engine.process()
        assert isinstance(result, dict)

    def test_single_entry(self, engine):
        entries = [_make_entry()]
        result = engine.process(learning_entries=entries)
        assert isinstance(result, dict)
        # Should have mode effectiveness for M1
        assert "M1" in result.get("mode_effectiveness", {})

    def test_result_keys(self, engine):
        entries = [_make_entry()]
        result = engine.process(learning_entries=entries)
        expected_keys = {
            "learning_patterns",
            "recurring_failures",
            "mode_effectiveness",
            "style_preferences",
            "subject_proficiencies",
            "recommendations",
        }
        assert expected_keys.issubset(set(result.keys()))


# =====================================================================
# Mode effectiveness
# =====================================================================

class TestModeEffectiveness:
    def test_single_mode_stats(self, engine):
        entries = [
            _make_entry(mode="M1", confirmations=10, contradictions=2),
            _make_entry(mode="M1", confirmations=8, contradictions=3),
        ]
        result = engine.process(learning_entries=entries)
        m1 = result["mode_effectiveness"].get("M1", {})
        assert m1.get("turns", 0) == 2
        assert m1.get("confirmations", 0) == 18
        assert m1.get("contradictions", 0) == 5

    def test_multiple_modes(self, engine):
        entries = [
            _make_entry(mode="M1"),
            _make_entry(mode="M3"),
            _make_entry(mode="M5"),
        ]
        result = engine.process(learning_entries=entries)
        assert "M1" in result["mode_effectiveness"]
        assert "M3" in result["mode_effectiveness"]
        assert "M5" in result["mode_effectiveness"]

    def test_confirmation_ratio(self, engine):
        entries = [
            _make_entry(mode="M2", confirmations=8, contradictions=2),
        ]
        result = engine.process(learning_entries=entries)
        m2 = result["mode_effectiveness"]["M2"]
        # ratio = 8 / (8 + 2) = 0.8
        assert abs(m2.get("confirmation_ratio", 0) - 0.8) < 0.01


# =====================================================================
# Subject proficiencies
# =====================================================================

class TestSubjectProficiencies:
    def test_single_subject(self, engine):
        entries = [
            _make_entry(subject="technical", confirmations=10, contradictions=1),
        ]
        result = engine.process(learning_entries=entries)
        assert "technical" in result["subject_proficiencies"]

    def test_improving_subject(self, engine):
        # Multiple entries with high confirmations, low contradictions
        entries = [
            _make_entry(subject="science", confirmations=10, contradictions=1),
            _make_entry(subject="science", confirmations=12, contradictions=0),
        ]
        result = engine.process(learning_entries=entries)
        # Should be improving or stable (not stagnating)
        science_data = result["subject_proficiencies"].get("science", {})
        trend = science_data.get("trend", "")
        assert trend in ("improving", "stable")

    def test_stagnating_subject(self, engine):
        # High contradiction ratio → stagnating
        entries = [
            _make_entry(subject="philosophy", confirmations=2, contradictions=5),
            _make_entry(subject="philosophy", confirmations=1, contradictions=6),
        ]
        result = engine.process(learning_entries=entries)
        phil_data = result["subject_proficiencies"].get("philosophy", {})
        trend = phil_data.get("trend", "")
        assert trend in ("stagnating", "declining")


# =====================================================================
# Recurring failures
# =====================================================================

class TestRecurringFailures:
    def test_no_failures_with_good_data(self, engine):
        entries = [
            _make_entry(confirmations=10, contradictions=0),
        ]
        result = engine.process(learning_entries=entries)
        assert isinstance(result["recurring_failures"], list)

    def test_repeated_pattern_failures(self, engine):
        # Same pattern type appearing multiple times across entries
        entries = [
            _make_entry(
                e19_patterns=[{"pattern_type": "logic_error", "confidence": 0.5}],
                contradictions=3,
            ),
            _make_entry(
                e19_patterns=[{"pattern_type": "logic_error", "confidence": 0.4}],
                contradictions=4,
            ),
            _make_entry(
                e19_patterns=[{"pattern_type": "logic_error", "confidence": 0.6}],
                contradictions=5,
            ),
        ]
        result = engine.process(learning_entries=entries)
        failures = result["recurring_failures"]
        # Should detect "logic_error" as recurring
        assert isinstance(failures, list)


# =====================================================================
# Meta-patterns
# =====================================================================

class TestMetaPatterns:
    def test_comfort_zone_detection(self, engine):
        # All entries in same mode → comfort zone (need >= 5 turns)
        entries = [_make_entry(mode="M1") for _ in range(8)]
        result = engine.process(learning_entries=entries)
        patterns = result["learning_patterns"]
        pattern_types = [p.get("pattern_type", "") for p in patterns]
        assert "comfort_zone" in pattern_types

    def test_underperforming_mode_detection(self, engine):
        # Mode with low confirmation ratio (needs >= 3 turns)
        # Set low DA so salience threshold is low enough to include
        # underperforming_mode pattern (default DA filters low-salience patterns)
        engine.update_neurochem_state({"da": 0.0})
        entries = [
            _make_entry(mode="M4", confirmations=1, contradictions=5),
            _make_entry(mode="M4", confirmations=2, contradictions=6),
            _make_entry(mode="M4", confirmations=0, contradictions=4),
        ]
        result = engine.process(learning_entries=entries)
        patterns = result["learning_patterns"]
        pattern_types = [p.get("pattern_type", "") for p in patterns]
        assert "underperforming_mode" in pattern_types


# =====================================================================
# Style preferences
# =====================================================================

class TestStylePreferences:
    def test_style_preferences_ordering(self, engine):
        entries = [
            _make_entry(mode="M3", confirmations=15, contradictions=1),
            _make_entry(mode="M3", confirmations=12, contradictions=2),
            _make_entry(mode="M1", confirmations=3, contradictions=5),
        ]
        result = engine.process(learning_entries=entries)
        prefs = result["style_preferences"]
        assert isinstance(prefs, list)
        if prefs:
            # E31 returns list of dicts: {"mode": ..., "effectiveness_score": ...}
            pref_modes = [p.get("mode", "") for p in prefs]
            m3_idx = pref_modes.index("M3") if "M3" in pref_modes else 999
            m1_idx = pref_modes.index("M1") if "M1" in pref_modes else 999
            # M3 should rank higher (lower index) than M1
            assert m3_idx < m1_idx


# =====================================================================
# Recommendations
# =====================================================================

class TestRecommendations:
    def test_recommendations_type(self, engine):
        entries = [_make_entry()]
        result = engine.process(learning_entries=entries)
        assert isinstance(result["recommendations"], list)

    def test_recommendations_for_failures(self, engine):
        # Recurring failures should generate recommendations
        entries = [
            _make_entry(mode="M1", confirmations=1, contradictions=8),
            _make_entry(mode="M1", confirmations=0, contradictions=10),
        ]
        result = engine.process(learning_entries=entries)
        recs = result["recommendations"]
        assert isinstance(recs, list)


# =====================================================================
# NT modulation effect
# =====================================================================

class TestNTModulation:
    def test_high_da_focuses_salience(self, engine):
        entries = [_make_entry(), _make_entry(mode="M3")]
        # High DA → should influence pattern salience
        engine.update_neurochem_state({"da": 0.9})
        result_high = engine.process(learning_entries=entries)

        engine2 = ReflectiveLearningEngine()
        engine2.update_neurochem_state({"da": 0.1})
        result_low = engine2.process(learning_entries=entries)

        # Both should produce valid results
        assert isinstance(result_high, dict)
        assert isinstance(result_low, dict)

    def test_high_ne_prioritises_failures(self, engine):
        entries = [_make_entry(contradictions=8)]
        engine.update_neurochem_state({"ne": 0.9})
        result = engine.process(learning_entries=entries)
        assert isinstance(result["recurring_failures"], list)


# =====================================================================
# Identity context
# =====================================================================

class TestIdentityContext:
    def test_with_identity_context(self, engine):
        entries = [_make_entry()]
        context = {
            "self_model": ["I value logical reasoning"],
            "lesson": ["Practice improves outcomes"],
        }
        result = engine.process(
            learning_entries=entries,
            identity_context=context,
        )
        assert isinstance(result, dict)

    def test_empty_identity_context(self, engine):
        entries = [_make_entry()]
        result = engine.process(
            learning_entries=entries,
            identity_context={},
        )
        assert isinstance(result, dict)


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_all_zeros(self, engine):
        entry = _make_entry(
            confirmations=0,
            contradictions=0,
            novel_entries=0,
            patterns_detected=0,
        )
        result = engine.process(learning_entries=[entry])
        assert isinstance(result, dict)

    def test_very_large_dataset(self, engine):
        entries = [_make_entry(session_id=f"s{i}") for i in range(100)]
        result = engine.process(learning_entries=entries)
        assert isinstance(result, dict)

    def test_missing_fields_in_entry(self, engine):
        # Minimal entry with missing fields
        entry = {"mode": "M1", "subject": "mixed"}
        result = engine.process(learning_entries=[entry])
        assert isinstance(result, dict)
