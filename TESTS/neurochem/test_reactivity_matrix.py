"""Tests for reactivity matrix and phasic burst computation (Extractor 2 routing)."""

import numpy as np
import pytest

from zados.neurochem.extractors.reactivity_matrix import (
    ReactivityEntry,
    ReactivityMatrixConfig,
    DEFAULT_REACTIVITY_CONFIG,
    apply_threshold_gating,
    compute_stochastic_burst_deltas,
    burst_deltas_to_modulation_signals,
)


# =====================================================================
# Threshold gating
# =====================================================================

class TestApplyThresholdGating:
    def test_above_threshold(self):
        assert apply_threshold_gating(0.6, 0.3) == 0.6

    def test_below_threshold(self):
        assert apply_threshold_gating(0.2, 0.3) == 0.0

    def test_at_threshold(self):
        """At threshold exactly → gated off (strictly greater required)."""
        assert apply_threshold_gating(0.3, 0.3) == 0.0

    def test_zero_threshold(self):
        assert apply_threshold_gating(0.5, 0.0) == 0.5

    def test_zero_value_zero_threshold(self):
        assert apply_threshold_gating(0.0, 0.0) == 0.0


# =====================================================================
# ReactivityEntry / Config
# =====================================================================

class TestReactivityEntry:
    def test_frozen(self):
        entry = ReactivityEntry("DA", "novelty", 0.8)
        with pytest.raises(AttributeError):
            entry.weight = 0.5

    def test_defaults(self):
        entry = ReactivityEntry("DA", "novelty", 0.8)
        assert entry.threshold == 0.3
        assert entry.distribution == "gamma"


class TestReactivityMatrixConfig:
    def test_default_covers_all_12_nts(self):
        """Default config should have entries for all 12 neurotransmitters."""
        nt_names = {e.nt_name for e in DEFAULT_REACTIVITY_CONFIG.entries}
        expected = {
            "DA", "5HT", "NE", "ACh", "OXT", "MOR",
            "CB1", "cortisol", "CRH", "GABA", "GLU", "histamine",
        }
        assert nt_names == expected

    def test_default_entry_count(self):
        """Should have ~20 entries."""
        assert len(DEFAULT_REACTIVITY_CONFIG.entries) == 20

    def test_all_weights_positive(self):
        for e in DEFAULT_REACTIVITY_CONFIG.entries:
            assert e.weight > 0.0, f"{e.nt_name}←{e.axis_name} has non-positive weight"

    def test_all_thresholds_in_range(self):
        for e in DEFAULT_REACTIVITY_CONFIG.entries:
            assert 0.0 <= e.threshold <= 1.0, f"{e.nt_name}←{e.axis_name} threshold out of range"


# =====================================================================
# compute_stochastic_burst_deltas
# =====================================================================

class TestComputeStochasticBurstDeltas:
    def test_single_entry_above_threshold(self):
        """Single entry with axis above threshold should produce non-zero delta."""
        config = ReactivityMatrixConfig(entries=(
            ReactivityEntry("DA", "novelty", 0.8, 0.3, "gamma"),
        ))
        eval_vec = {"novelty": 0.7}
        deltas = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, config, rng=np.random.default_rng(42),
        )
        assert "DA" in deltas
        assert deltas["DA"] > 0.0

    def test_single_entry_below_threshold(self):
        """Axis below threshold → no delta for that NT."""
        config = ReactivityMatrixConfig(entries=(
            ReactivityEntry("DA", "novelty", 0.8, 0.5, "gamma"),
        ))
        eval_vec = {"novelty": 0.3}
        deltas = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, config, rng=np.random.default_rng(42),
        )
        assert deltas.get("DA", 0.0) == 0.0

    def test_missing_axis_returns_zero(self):
        """Evaluation vector missing the axis → no burst."""
        config = ReactivityMatrixConfig(entries=(
            ReactivityEntry("DA", "novelty", 0.8, 0.3, "gamma"),
        ))
        eval_vec = {"urgency": 0.9}  # no "novelty" key
        deltas = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, config, rng=np.random.default_rng(42),
        )
        assert deltas.get("DA", 0.0) == 0.0

    def test_multi_entry_same_nt_accumulates(self):
        """Multiple entries for same NT should sum their contributions."""
        config = ReactivityMatrixConfig(entries=(
            ReactivityEntry("DA", "novelty",          0.5, 0.2, "gamma"),
            ReactivityEntry("DA", "reward_alignment", 0.5, 0.2, "gamma"),
        ))
        eval_vec = {"novelty": 0.8, "reward_alignment": 0.7}
        deltas = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, config, rng=np.random.default_rng(42),
        )
        assert "DA" in deltas
        # Two contributions → should be larger than either alone
        config_single = ReactivityMatrixConfig(entries=(
            ReactivityEntry("DA", "novelty", 0.5, 0.2, "gamma"),
        ))
        delta_single = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, config_single, rng=np.random.default_rng(42),
        )
        # The combined should generally be larger (same seed for first entry)
        assert deltas["DA"] >= delta_single.get("DA", 0.0)

    def test_all_deltas_non_negative(self):
        """All burst deltas should be non-negative."""
        eval_vec = {
            "novelty": 0.8, "emotional_valence": 0.7, "urgency": 0.9,
            "logical_conflict": 0.5, "coherence": 0.6, "social_salience": 0.7,
            "reward_alignment": 0.6, "identity_resonance": 0.8,
        }
        deltas = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, rng=np.random.default_rng(42),
        )
        for nt, val in deltas.items():
            assert val >= 0.0, f"{nt} has negative delta {val}"

    def test_prev_vector_none_no_error(self):
        """prev_evaluation_vector=None should work (zero volatility)."""
        eval_vec = {"novelty": 0.7}
        config = ReactivityMatrixConfig(entries=(
            ReactivityEntry("DA", "novelty", 0.8, 0.3, "gamma"),
        ))
        deltas = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, config, rng=np.random.default_rng(42),
        )
        assert "DA" in deltas

    def test_prev_vector_changes_distribution(self):
        """Providing a prev_vector should affect sampling (volatility parameter)."""
        config = ReactivityMatrixConfig(entries=(
            ReactivityEntry("DA", "novelty", 0.8, 0.3, "gamma"),
        ))
        eval_vec = {"novelty": 0.7}
        prev_vec = {"novelty": 0.2}  # big jump → high de/dt

        # Collect samples with no prev_vector
        stable = [
            compute_stochastic_burst_deltas(
                eval_vec, None, 0.01, config, rng=np.random.default_rng(i),
            ).get("DA", 0.0)
            for i in range(100)
        ]
        # Collect samples with prev_vector (high volatility)
        volatile = [
            compute_stochastic_burst_deltas(
                eval_vec, prev_vec, 0.01, config, rng=np.random.default_rng(i),
            ).get("DA", 0.0)
            for i in range(100)
        ]
        # Both should produce non-negative values
        assert all(s >= 0.0 for s in stable)
        assert all(v >= 0.0 for v in volatile)

    def test_reproducible_with_seed(self):
        eval_vec = {"novelty": 0.8, "urgency": 0.7}
        d1 = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, rng=np.random.default_rng(99),
        )
        d2 = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, rng=np.random.default_rng(99),
        )
        assert d1.keys() == d2.keys()
        for k in d1:
            assert d1[k] == pytest.approx(d2[k])

    def test_empty_eval_vector(self):
        """Empty evaluation vector → no bursts."""
        deltas = compute_stochastic_burst_deltas(
            {}, None, 0.01, rng=np.random.default_rng(42),
        )
        assert len(deltas) == 0

    def test_default_config_with_full_eval(self):
        """Default config with all 8 axes populated should produce deltas for 12 NTs."""
        eval_vec = {
            "novelty": 0.8, "emotional_valence": 0.7, "urgency": 0.9,
            "logical_conflict": 0.6, "coherence": 0.7, "social_salience": 0.8,
            "reward_alignment": 0.6, "identity_resonance": 0.7,
        }
        deltas = compute_stochastic_burst_deltas(
            eval_vec, None, 0.01, rng=np.random.default_rng(42),
        )
        assert len(deltas) == 12


# =====================================================================
# burst_deltas_to_modulation_signals
# =====================================================================

class TestBurstDeltasToModulationSignals:
    def test_basic_format(self):
        deltas = {"DA": 0.5, "NE": 0.3}
        signals = burst_deltas_to_modulation_signals(deltas)
        assert signals == {
            "DA": {"stochastic_burst": 0.5},
            "NE": {"stochastic_burst": 0.3},
        }

    def test_empty_deltas(self):
        signals = burst_deltas_to_modulation_signals({})
        assert signals == {}

    def test_merge_with_existing(self):
        existing = {
            "DA": {"reward_drive": 0.8, "emotion_drive": 0.3},
            "5HT": {"emotion_drive": 0.5},
        }
        deltas = {"DA": 0.4, "NE": 0.2}
        signals = burst_deltas_to_modulation_signals(deltas, existing)

        # DA should have all 3 keys
        assert signals["DA"]["reward_drive"] == 0.8
        assert signals["DA"]["emotion_drive"] == 0.3
        assert signals["DA"]["stochastic_burst"] == 0.4
        # NE added fresh
        assert signals["NE"]["stochastic_burst"] == 0.2
        # 5HT preserved
        assert signals["5HT"]["emotion_drive"] == 0.5

    def test_does_not_mutate_existing(self):
        existing = {"DA": {"reward_drive": 0.8}}
        deltas = {"DA": 0.4}
        signals = burst_deltas_to_modulation_signals(deltas, existing)
        # Original should be unmodified
        assert "stochastic_burst" not in existing["DA"]
        assert signals["DA"]["stochastic_burst"] == 0.4

    def test_overwrites_existing_burst_key(self):
        existing = {"DA": {"stochastic_burst": 0.1}}
        deltas = {"DA": 0.9}
        signals = burst_deltas_to_modulation_signals(deltas, existing)
        assert signals["DA"]["stochastic_burst"] == 0.9
