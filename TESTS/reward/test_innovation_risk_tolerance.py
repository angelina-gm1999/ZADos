from zados.reward.domains.innovation.risk_tolerance import (
    RiskToleranceSubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_risk_tolerance_balanced():
    sm = RiskToleranceSubmodule()
    ctx = RewardContext()

    state = {
        "risk_exposure": 0.6,
        "constraint_awareness": 0.6,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.3
    assert r.flags == {}


def test_reckless_exploration_flag():
    sm = RiskToleranceSubmodule()
    ctx = RewardContext()

    state = {
        "risk_exposure": 0.9,
        "constraint_awareness": 0.1,
    }

    r = sm.evaluate(state, ctx)

    assert "reckless_exploration" in r.flags


def test_innovation_domain_includes_risk_tolerance():
    d = InnovationDomain()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.4,
        "exploration_intent": 0.4,
        "conceptual_shift": 0.5,
        "concept_overlap": 0.4,
        "structural_shift": 0.5,
        "pattern_reuse": 0.4,
        "divergence_rate": 0.5,
        "stability_pressure": 0.3,
        "symbols_used": ["A", "B"],
        "known_symbol_pairs": [("A", "B")],
        "risk_exposure": 0.5,
        "constraint_awareness": 0.6,
    }

    out = d.evaluate(state, ctx)

    assert "risk_tolerance" in out.subscores
