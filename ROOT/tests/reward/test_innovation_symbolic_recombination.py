from zados.reward.domains.innovation.symbolic_recombination import (
    SymbolicRecombinationSubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_symbolic_recombination_detected():
    sm = SymbolicRecombinationSubmodule()
    ctx = RewardContext()

    state = {
        "symbols_used": ["A", "B", "C"],
        "known_symbol_pairs": [("A", "B")],
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.0


def test_unanchored_recombination_flag():
    sm = SymbolicRecombinationSubmodule()
    ctx = RewardContext()

    state = {
        "symbols_used": ["X", "Y", "Z"],
        "known_symbol_pairs": [],
    }

    r = sm.evaluate(state, ctx)

    assert "unanchored_recombination" in r.flags


def test_innovation_domain_includes_symbolic_recombination():
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
    }

    out = d.evaluate(state, ctx)

    assert "symbolic_recombination" in out.subscores
