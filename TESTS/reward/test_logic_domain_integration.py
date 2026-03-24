"""Integration tests for LogicDomain — verifies all submodules are instantiated."""

from zados.reward.domains.logic.domain import LogicDomain
from zados.reward.base.types import RewardContext


class TestLogicDomainSubmodules:
    def test_semantic_continuity_in_subscores(self):
        """B1 regression: SemanticContinuitySubmodule must be instantiated.

        Without it, evaluation_vector axis 'coherence' (mapped to
        logic/semantic_continuity) silently returns 0.0 every tick.
        """
        domain = LogicDomain()
        state = {"representation": {"meaning": "test"}}
        ctx = RewardContext()
        result = domain.evaluate(state, ctx)
        assert "semantic_continuity" in result.subscores

    def test_all_expected_submodules_present(self):
        """LogicDomain should produce subscores for all instantiated submodules."""
        domain = LogicDomain()
        state = {"representation": {"meaning": "test"}}
        ctx = RewardContext()
        result = domain.evaluate(state, ctx)
        expected = {
            "epistemic_calibration",
            "uncertainty_acknowledgment",
            "abstention_appropriateness",
            "internal_consistency",
            "semantic_continuity",
        }
        assert expected == set(result.subscores.keys())

    def test_domain_name(self):
        domain = LogicDomain()
        assert domain.domain_name == "logic"

    def test_general_score_averages_all_subscores(self):
        domain = LogicDomain()
        state = {"representation": {"meaning": "test"}}
        ctx = RewardContext()
        result = domain.evaluate(state, ctx)
        expected_avg = sum(s.score for s in result.subscores.values()) / len(result.subscores)
        assert abs(result.general_score - expected_avg) < 1e-10
