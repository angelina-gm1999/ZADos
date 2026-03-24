"""
Tests for inference matrix (bidirectional NT ↔ cognitive engine arbitration).

Phase 15: Verifies NT→engine priority mapping, engine→NT modulation feedback,
and the full bidirectional arbitration cycle.
"""

import pytest

from zados.neurochem.inference_matrix.nt_to_engine import (
    compute_engine_priority_weights,
    EnginePriorityWeights,
)
from zados.neurochem.inference_matrix.engine_to_nt import (
    compute_nt_modulation_from_evaluation,
)
from zados.neurochem.inference_matrix.arbitration import InferenceArbitrator
from zados.neurochem.neurotransmitters.configs import DEFAULT_NT_CONFIGS


# =====================================================================
# EnginePriorityWeights Tests
# =====================================================================

class TestEnginePriorityWeights:
    """Test EnginePriorityWeights dataclass."""

    def test_default_values(self):
        w = EnginePriorityWeights()
        assert w.exploration == 0.5
        assert w.verification == 0.5

    def test_as_dict(self):
        w = EnginePriorityWeights(exploration=0.8, safety=0.2)
        d = w.as_dict()
        assert d["exploration"] == 0.8
        assert d["safety"] == 0.2

    def test_dominant_engine(self):
        w = EnginePriorityWeights(
            exploration=0.9, verification=0.3,
            attunement=0.1, safety=0.5, integration=0.4,
        )
        assert w.dominant_engine() == "exploration"

    def test_dominant_engine_tie_resolution(self):
        """When tied, any of the max engines is acceptable."""
        w = EnginePriorityWeights(
            exploration=0.7, verification=0.7,
            attunement=0.1, safety=0.1, integration=0.1,
        )
        assert w.dominant_engine() in ("exploration", "verification")

    def test_normalized_sums_to_one(self):
        w = EnginePriorityWeights(
            exploration=0.8, verification=0.4,
            attunement=0.3, safety=0.2, integration=0.1,
        )
        norm = w.normalized()
        assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_normalized_zero_weights(self):
        w = EnginePriorityWeights(
            exploration=0.0, verification=0.0,
            attunement=0.0, safety=0.0, integration=0.0,
        )
        norm = w.normalized()
        # Should be uniform
        assert abs(sum(norm.values()) - 1.0) < 1e-9
        assert all(abs(v - 0.2) < 1e-9 for v in norm.values())

    def test_frozen(self):
        w = EnginePriorityWeights()
        with pytest.raises(AttributeError):
            w.exploration = 0.9


# =====================================================================
# NT → Engine Priority Tests
# =====================================================================

class TestNTtoEngine:
    """Test compute_engine_priority_weights."""

    def test_default_metrics_balanced(self):
        """Default (0.5) metrics should give balanced weights."""
        metrics = {
            "motivation": 0.5, "empathy": 0.5, "cognitive_rigidity": 0.5,
            "fatigue": 0.5, "precision": 0.5, "openness": 0.5,
            "anxiety": 0.5, "social_engagement": 0.5,
        }
        weights = compute_engine_priority_weights(metrics)
        d = weights.as_dict()
        for v in d.values():
            assert 0.0 <= v <= 1.0

    def test_high_motivation_openness_drives_exploration(self):
        metrics = {
            "motivation": 0.9, "openness": 0.9,
            "cognitive_rigidity": 0.1,
        }
        weights = compute_engine_priority_weights(metrics)
        assert weights.exploration > 0.7

    def test_low_motivation_low_exploration(self):
        metrics = {
            "motivation": 0.1, "openness": 0.1,
            "cognitive_rigidity": 0.9,
        }
        weights = compute_engine_priority_weights(metrics)
        assert weights.exploration < 0.3

    def test_high_precision_drives_verification(self):
        metrics = {
            "precision": 0.9, "cognitive_rigidity": 0.8,
            "fatigue": 0.1,
        }
        weights = compute_engine_priority_weights(metrics)
        assert weights.verification > 0.7

    def test_high_empathy_drives_attunement(self):
        metrics = {
            "empathy": 0.9, "social_engagement": 0.8,
        }
        weights = compute_engine_priority_weights(metrics)
        assert weights.attunement > 0.7

    def test_high_anxiety_drives_safety(self):
        metrics = {
            "anxiety": 0.9, "openness": 0.1,
        }
        weights = compute_engine_priority_weights(metrics)
        assert weights.safety > 0.7

    def test_low_rigidity_drives_integration(self):
        metrics = {
            "cognitive_rigidity": 0.1, "fatigue": 0.1,
        }
        weights = compute_engine_priority_weights(metrics)
        assert weights.integration > 0.7

    def test_high_fatigue_reduces_verification(self):
        high_fatigue = compute_engine_priority_weights({
            "precision": 0.8, "fatigue": 0.9,
        })
        low_fatigue = compute_engine_priority_weights({
            "precision": 0.8, "fatigue": 0.1,
        })
        assert high_fatigue.verification < low_fatigue.verification

    def test_all_weights_bounded(self):
        """Extreme metrics should still produce bounded weights."""
        for val in [0.0, 0.5, 1.0]:
            metrics = {k: val for k in [
                "motivation", "empathy", "cognitive_rigidity",
                "fatigue", "precision", "openness",
                "anxiety", "social_engagement",
            ]}
            weights = compute_engine_priority_weights(metrics)
            for v in weights.as_dict().values():
                assert 0.0 <= v <= 1.0

    def test_empty_metrics_uses_defaults(self):
        weights = compute_engine_priority_weights({})
        d = weights.as_dict()
        for v in d.values():
            assert 0.0 <= v <= 1.0


# =====================================================================
# Engine → NT Modulation Tests
# =====================================================================

class TestEngineToNT:
    """Test compute_nt_modulation_from_evaluation."""

    def test_returns_dict_of_dicts(self):
        result = compute_nt_modulation_from_evaluation({})
        assert isinstance(result, dict)
        for nt_name, signals in result.items():
            assert isinstance(signals, dict)

    def test_covers_key_nts(self):
        """Should produce signals for key NTs."""
        result = compute_nt_modulation_from_evaluation({
            "confidence": 0.7,
            "contradictions_found": 2,
            "social_resonance": 0.6,
            "risk_detected": 0.3,
            "novelty_detected": 0.8,
        })
        expected_nts = {"DA", "NE", "OXT", "5HT", "GABA", "cortisol",
                        "CRH", "ACh", "MOR", "CB1", "GLU"}
        assert set(result.keys()) == expected_nts

    def test_all_signals_target_valid_nts(self):
        result = compute_nt_modulation_from_evaluation({
            "confidence": 0.5,
        })
        valid_nts = set(DEFAULT_NT_CONFIGS.keys())
        for nt_name in result:
            assert nt_name in valid_nts, f"Unknown NT: {nt_name}"

    def test_high_novelty_drives_da(self):
        result = compute_nt_modulation_from_evaluation({
            "novelty_detected": 0.9,
        })
        assert result["DA"]["novelty"] > 0.5

    def test_high_confidence_positive_rpe(self):
        result = compute_nt_modulation_from_evaluation({
            "confidence": 0.9,
        })
        assert result["DA"]["rpe"] > 0.0

    def test_low_confidence_negative_rpe(self):
        result = compute_nt_modulation_from_evaluation({
            "confidence": 0.1,
        })
        assert result["DA"]["rpe"] < 0.0

    def test_contradictions_drive_ne(self):
        result = compute_nt_modulation_from_evaluation({
            "contradictions_found": 5,
        })
        assert result["NE"]["contradiction"] > 0.5
        assert result["NE"]["precision"] > 0.3

    def test_social_resonance_drives_oxt(self):
        result = compute_nt_modulation_from_evaluation({
            "social_resonance": 0.9,
        })
        assert result["OXT"]["empathy"] > 0.5
        assert result["OXT"]["social_engagement"] > 0.3

    def test_risk_drives_gaba_and_stress(self):
        result = compute_nt_modulation_from_evaluation({
            "risk_detected": 0.8,
        })
        assert result["GABA"]["inhibition"] > 0.3
        assert result["cortisol"]["stress_level"] > 0.0
        assert result["CRH"]["acute_stress"] > 0.2

    def test_domain_scores_affect_quality(self):
        high_quality = compute_nt_modulation_from_evaluation({
            "domain_scores": {"logic": 0.9, "ethics": 0.8},
        })
        low_quality = compute_nt_modulation_from_evaluation({
            "domain_scores": {"logic": 0.2, "ethics": 0.1},
        })
        # High quality should boost 5HT mood stability
        assert high_quality["5HT"]["mood_stability"] > low_quality["5HT"]["mood_stability"]

    def test_empty_evaluation_safe(self):
        """Empty evaluation should not crash."""
        result = compute_nt_modulation_from_evaluation({})
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_low_confidence_drives_ach(self):
        result = compute_nt_modulation_from_evaluation({
            "confidence": 0.1,
        })
        assert result["ACh"]["attention_demand"] > 0.3

    def test_novelty_drives_cb1_flexibility(self):
        result = compute_nt_modulation_from_evaluation({
            "novelty_detected": 0.8,
        })
        assert result["CB1"]["flexibility"] > 0.3

    def test_contradictions_drive_glu_integration(self):
        result = compute_nt_modulation_from_evaluation({
            "contradictions_found": 4,
        })
        assert result["GLU"]["integration_demand"] > 0.5


# =====================================================================
# Arbitration Tests
# =====================================================================

class TestInferenceArbitrator:
    """Test bidirectional arbitration orchestrator."""

    def test_initial_state(self):
        arb = InferenceArbitrator()
        assert arb.last_priorities is None
        assert arb.last_modulation is None
        assert arb.step_count == 0

    def test_compute_priorities(self):
        arb = InferenceArbitrator()
        metrics = {"motivation": 0.8, "openness": 0.7}
        weights = arb.compute_priorities(metrics)
        assert isinstance(weights, EnginePriorityWeights)
        assert arb.last_priorities is weights

    def test_process_evaluation(self):
        arb = InferenceArbitrator()
        eval_results = {
            "confidence": 0.7,
            "novelty_detected": 0.5,
        }
        modulation = arb.process_evaluation(eval_results)
        assert isinstance(modulation, dict)
        assert arb.last_modulation is modulation
        assert arb.step_count == 1

    def test_full_cycle(self):
        arb = InferenceArbitrator()
        metrics = {
            "motivation": 0.7, "empathy": 0.6,
            "precision": 0.5, "anxiety": 0.3,
        }
        eval_results = {
            "confidence": 0.8,
            "novelty_detected": 0.4,
            "social_resonance": 0.5,
        }
        modulation = arb.full_cycle(metrics, eval_results)
        assert isinstance(modulation, dict)
        assert arb.last_priorities is not None
        assert arb.last_modulation is not None
        assert arb.step_count == 1

    def test_multiple_cycles(self):
        arb = InferenceArbitrator()
        for i in range(5):
            arb.full_cycle(
                {"motivation": 0.5 + i * 0.05},
                {"confidence": 0.5 + i * 0.08},
            )
        assert arb.step_count == 5

    def test_reset(self):
        arb = InferenceArbitrator()
        arb.full_cycle({"motivation": 0.5}, {"confidence": 0.7})
        arb.reset()
        assert arb.last_priorities is None
        assert arb.last_modulation is None
        assert arb.step_count == 0

    def test_cycle_produces_valid_signals(self):
        """Full cycle should produce signals that target valid NTs."""
        arb = InferenceArbitrator()
        modulation = arb.full_cycle(
            {"motivation": 0.8, "empathy": 0.6, "anxiety": 0.4},
            {"confidence": 0.7, "novelty_detected": 0.5,
             "contradictions_found": 1, "social_resonance": 0.4},
        )
        valid_nts = set(DEFAULT_NT_CONFIGS.keys())
        for nt_name in modulation:
            assert nt_name in valid_nts

    def test_high_motivation_exploration_feedback(self):
        """High motivation metrics → exploration priority →
        novelty in evaluation → DA novelty signal."""
        arb = InferenceArbitrator()

        # Step 1: High motivation metrics
        metrics = {
            "motivation": 0.9, "openness": 0.8,
            "cognitive_rigidity": 0.1,
        }
        weights = arb.compute_priorities(metrics)
        assert weights.exploration > 0.7

        # Step 2: Evaluation found novelty (exploration worked)
        eval_results = {"novelty_detected": 0.8, "confidence": 0.8}
        modulation = arb.process_evaluation(eval_results)

        # DA should get high novelty signal (positive feedback)
        assert modulation["DA"]["novelty"] > 0.5

    def test_high_anxiety_safety_feedback(self):
        """High anxiety → safety priority →
        risk in evaluation → GABA/CRH signals."""
        arb = InferenceArbitrator()

        metrics = {"anxiety": 0.9, "openness": 0.1}
        weights = arb.compute_priorities(metrics)
        assert weights.safety > 0.7

        eval_results = {"risk_detected": 0.7, "confidence": 0.6}
        modulation = arb.process_evaluation(eval_results)

        assert modulation["GABA"]["inhibition"] > 0.3
        assert modulation["CRH"]["acute_stress"] > 0.2
