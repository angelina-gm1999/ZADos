from zados.reward.domains.innovation.structural_novelty import (
    StructuralNoveltySubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_structural_novelty_high():
    sm = StructuralNoveltySubmodule()
    ctx = RewardContext()

    state = {
        "structural_shift": 0.8,
        "pattern_reuse": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.5
    assert r.flags == {}


def test_cosmetic_variation_flag():
    sm = StructuralNoveltySubmodule()
    ctx = RewardContext()

    state = {
        "structural_shift": 0.9,
        "pattern_reuse": 0.9,
    }

    r = sm.evaluate(state, ctx)

    assert "cosmetic_variation" in r.flags


def test_innovation_domain_includes_structural():
    d = InnovationDomain()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.4,
        "exploration_intent": 0.4,
        "conceptual_shift": 0.5,
        "concept_overlap": 0.4,
        "structural_shift": 0.6,
        "pattern_reuse": 0.3,
    }

    out = d.evaluate(state, ctx)

    assert out.domain == "innovation"
    assert "structural_novelty" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
