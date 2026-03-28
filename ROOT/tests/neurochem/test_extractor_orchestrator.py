"""Tests for the extractor orchestrator (top-level sequencer)."""

import numpy as np
import pytest

from zados.reward.base.types import RewardDomainResult, RewardSubscore
from zados.neurochem.state.oscillation_state import OscillationState
from zados.neurochem.extractors.extractor_orchestrator import (
    ExtractorState,
    ExtractorResult,
    ExtractorOrchestrator,
)
from zados.neurochem.extractors.evaluation_vector import (
    EvaluationAxisConfig,
    EvaluationVectorConfig,
)
from zados.neurochem.extractors.reactivity_matrix import (
    ReactivityEntry,
    ReactivityMatrixConfig,
)
from zados.neurochem.extractors.regulatory_modulator import (
    RegulatoryPathwayConfig,
    RegulatoryModulatorConfig,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_domain_result(domain, general_score=0.5, subscores=None):
    subs = {}
    if subscores:
        for name, score in subscores.items():
            subs[name] = RewardSubscore(name=name, score=score)
    return RewardDomainResult(domain=domain, general_score=general_score, subscores=subs)


def _make_all_domains():
    return {
        "innovation": _make_domain_result("innovation", 0.7, {
            "novelty_generation": 0.8,
            "conceptual_novelty": 0.6,
            "pattern_divergence": 0.5,
        }),
        "logic": _make_domain_result("logic", 0.6, {
            "internal_consistency": 0.9,
            "semantic_continuity": 0.75,
            "epistemic_calibration": 0.65,
        }),
        "human_attunement": _make_domain_result("human_attunement", 0.65, {
            "empathetic_inference": 0.7,
            "cognitive_reading": 0.6,
            "intention_calibration": 0.55,
        }),
        "ethics": _make_domain_result("ethics", 0.8, {
            "failure_mode_awareness": 0.85,
            "intent_clarity": 0.9,
            "timeline_reflection": 0.7,
        }),
    }


# =====================================================================
# ExtractorState
# =====================================================================

class TestExtractorState:
    def test_initialize_defaults(self):
        state = ExtractorState.initialize()
        assert state.prev_evaluation_vector is None
        assert state.regulatory_state is not None
        assert state.emotion_tracker_state is not None
        assert len(state.emotion_tracker_state.integrators) == 12

    def test_initialize_custom_emotions(self):
        state = ExtractorState.initialize(emotion_ids=["joy", "fear"])
        assert len(state.emotion_tracker_state.integrators) == 2

    def test_roundtrip_dict(self):
        state = ExtractorState.initialize()
        state.prev_evaluation_vector = {"novelty": 0.5}
        d = state.as_dict()
        restored = ExtractorState.from_dict(d)
        assert restored.prev_evaluation_vector == {"novelty": 0.5}
        assert restored.regulatory_state is not None
        assert restored.emotion_tracker_state is not None


# =====================================================================
# ExtractorOrchestrator — initialization
# =====================================================================

class TestExtractorOrchestratorInit:
    def test_default_init(self):
        orch = ExtractorOrchestrator()
        assert orch.rng is None
        assert orch.state is not None

    def test_seeded_init(self):
        rng = np.random.default_rng(42)
        orch = ExtractorOrchestrator(rng=rng)
        assert orch.rng is rng

    def test_state_settable(self):
        orch = ExtractorOrchestrator()
        new_state = ExtractorState.initialize()
        orch.state = new_state
        assert orch.state is new_state

    def test_custom_emotion_ids(self):
        """B2 regression: emotion_ids should be forwarded to ExtractorState."""
        orch = ExtractorOrchestrator(emotion_ids=["joy", "fear", "calm"])
        assert len(orch.state.emotion_tracker_state.integrators) == 3
        assert "joy" in orch.state.emotion_tracker_state.integrators
        assert "fear" in orch.state.emotion_tracker_state.integrators
        assert "calm" in orch.state.emotion_tracker_state.integrators

    def test_default_emotion_ids_is_12(self):
        """Without emotion_ids param, all 12 default emotions should be tracked."""
        orch = ExtractorOrchestrator()
        assert len(orch.state.emotion_tracker_state.integrators) == 12


# =====================================================================
# ExtractorOrchestrator — step
# =====================================================================

class TestExtractorOrchestratorStep:
    def test_basic_step_produces_result(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        domains = _make_all_domains()
        result = orch.step(domains)
        assert isinstance(result, ExtractorResult)

    def test_result_has_evaluation_vector(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        assert len(result.evaluation_vector) == 8
        for val in result.evaluation_vector.values():
            assert 0.0 <= val <= 1.0

    def test_result_has_modulation_signals(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        assert isinstance(result.modulation_signals, dict)
        # Should have NT signals from burst deltas
        assert len(result.modulation_signals) > 0

    def test_result_has_feedback_params(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        assert "neurotransmitters" in result.feedback_params
        assert "receptors" in result.feedback_params

    def test_result_has_burst_deltas(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        assert isinstance(result.burst_deltas, dict)
        for nt, val in result.burst_deltas.items():
            assert val >= 0.0

    def test_result_has_emotion_info(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        assert isinstance(result.emotion_saturations, dict)
        assert len(result.dominant_emotion) == 2
        assert isinstance(result.dominant_emotion[0], str)

    def test_no_oscillation_without_input(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        assert result.oscillation_update is None

    def test_oscillation_with_input(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        osc = OscillationState(theta=0.5, gamma=0.3, beta=0.4, alpha=0.6)
        result = orch.step(_make_all_domains(), current_oscillations=osc)
        assert result.oscillation_update is not None
        assert isinstance(result.oscillation_update, OscillationState)


# =====================================================================
# Emotion flow-through
# =====================================================================

class TestEmotionFlowThrough:
    def test_emotion_inputs_tracked(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(
            _make_all_domains(),
            emotion_inputs={"joy": 0.9, "fear": 0.3},
        )
        assert result.emotion_saturations.get("joy", 0.0) > 0.0
        assert result.emotion_saturations.get("fear", 0.0) > 0.0

    def test_emotion_affects_modulation(self):
        """Emotion inputs should contribute to modulation_signals via 4R path."""
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        # Step with no emotions
        r1 = orch.step(_make_all_domains())

        # Reset and step with strong emotions
        orch2 = ExtractorOrchestrator(rng=np.random.default_rng(42))
        r2 = orch2.step(
            _make_all_domains(),
            emotion_inputs={"joy": 1.0, "fear": 1.0},
        )
        # With emotions, more NTs should have signals
        # (or existing NTs should have additional signal keys)
        total_signals_1 = sum(
            len(sigs) for sigs in r1.modulation_signals.values()
        )
        total_signals_2 = sum(
            len(sigs) for sigs in r2.modulation_signals.values()
        )
        assert total_signals_2 >= total_signals_1

    def test_no_emotion_inputs_still_works(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        # Should work fine with no emotion inputs
        assert all(v == 0.0 for v in result.emotion_saturations.values())


# =====================================================================
# Reproducibility
# =====================================================================

class TestReproducibility:
    def test_seeded_reproducible(self):
        domains = _make_all_domains()

        orch1 = ExtractorOrchestrator(rng=np.random.default_rng(99))
        r1 = orch1.step(domains)

        orch2 = ExtractorOrchestrator(rng=np.random.default_rng(99))
        r2 = orch2.step(domains)

        assert r1.evaluation_vector == r2.evaluation_vector
        assert r1.burst_deltas.keys() == r2.burst_deltas.keys()
        for k in r1.burst_deltas:
            assert r1.burst_deltas[k] == pytest.approx(r2.burst_deltas[k])


# =====================================================================
# Multi-step dynamics
# =====================================================================

class TestMultiStepDynamics:
    def test_prev_eval_vector_stored(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        domains = _make_all_domains()

        # First step: no prev_vector
        orch.step(domains)
        assert orch.state.prev_evaluation_vector is not None

        # Second step: should use prev_vector for volatility
        orch.step(domains)
        assert orch.state.prev_evaluation_vector is not None

    def test_regulatory_smoothing_across_steps(self):
        """Regulatory feedback should accumulate over steps."""
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        domains = _make_all_domains()

        results = []
        for _ in range(100):
            results.append(orch.step(domains, dt=0.1))

        # OXT baseline delta should grow from near-zero
        first_oxt = results[0].feedback_params["neurotransmitters"].get(
            "OXT", {},
        ).get("C_baseline_delta", 0.0)
        last_oxt = results[-1].feedback_params["neurotransmitters"].get(
            "OXT", {},
        ).get("C_baseline_delta", 0.0)
        # Last should have larger magnitude than first (ramp-up)
        assert abs(last_oxt) >= abs(first_oxt)


# =====================================================================
# Empty / missing inputs
# =====================================================================

class TestEdgeCases:
    def test_empty_domains(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step({})
        assert len(result.evaluation_vector) == 8
        for val in result.evaluation_vector.values():
            assert val == 0.0

    def test_none_emotion_inputs(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains(), emotion_inputs=None)
        assert isinstance(result, ExtractorResult)

    def test_burst_deltas_all_non_negative(self):
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        for nt, val in result.burst_deltas.items():
            assert val >= 0.0, f"{nt} burst delta is negative: {val}"

    def test_default_config_most_nts_present(self):
        """With full domain input, burst deltas should cover most NTs.

        Some NTs may be missing if their axis values fall below threshold
        gating (e.g., GABA with logical_conflict=0.1 < threshold=0.4).
        """
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        result = orch.step(_make_all_domains())
        # At least 10 of 12 NTs should fire (GABA may be gated off)
        assert len(result.burst_deltas) >= 10
        # Key NTs should always be present with these inputs
        assert "DA" in result.burst_deltas
        assert "NE" in result.burst_deltas
        assert "5HT" in result.burst_deltas
