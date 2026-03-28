from zados.reward.domains.innovation.pattern_divergence import (
    PatternDivergenceSubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_pattern_divergence_high():
    sm = PatternDivergenceSubmodule()
    ctx = RewardContext()

    state = {
        "divergence_rate": 0.8,
        "stability_pressure": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.5
    assert r.flags == {}


def test_forced_divergence_flag():
    sm = PatternDivergenceSubmodule()
    ctx = RewardContext()

    state = {
        "divergence_rate": 0.9,
        "stability_pressure": 0.8,
    }

    r = sm.evaluate(state, ctx)

    assert "forced_divergence" in r.flags


def test_innovation_domain_includes_pattern_divergence():
    d = InnovationDomain()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.4,
        "exploration_intent": 0.4,
        "conceptual_shift": 0.5,
        "concept_overlap": 0.4,
        "structural_shift": 0.5,
        "pattern_reuse": 0.4,
        "divergence_rate": 0.6,
        "stability_pressure": 0.3,
    }

    out = d.evaluate(state, ctx)

    assert out.domain == "innovation"
    assert "pattern_divergence" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
