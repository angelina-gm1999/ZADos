from zados.reward.domains.innovation.conceptual_novelty import (
    ConceptualNoveltySubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_conceptual_novelty_high():
    sm = ConceptualNoveltySubmodule()
    ctx = RewardContext()

    state = {
        "conceptual_shift": 0.8,
        "concept_overlap": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.5
    assert r.flags == {}


def test_shallow_relabeling_flag():
    sm = ConceptualNoveltySubmodule()
    ctx = RewardContext()

    state = {
        "conceptual_shift": 0.9,
        "concept_overlap": 0.9,
    }

    r = sm.evaluate(state, ctx)

    assert "shallow_relabeling" in r.flags


def test_innovation_domain_includes_conceptual():
    d = InnovationDomain()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.4,
        "exploration_intent": 0.4,
        "conceptual_shift": 0.6,
        "concept_overlap": 0.3,
    }

    out = d.evaluate(state, ctx)

    assert out.domain == "innovation"
    assert "conceptual_novelty" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
