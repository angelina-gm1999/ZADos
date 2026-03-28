from zados.reward.domains.human_attunement.attuned_dissonance import (
    AttunedDissonanceSubmodule,
)
from zados.reward.base.types import RewardContext


def test_productive_dissonance():
    sm = AttunedDissonanceSubmodule()
    ctx = RewardContext()

    state = {
        "disagreement_intensity": 0.8,
        "contextual_justification": 0.8,
        "escalation_signal": 0.1,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.5
    assert "productive_dissonance" in r.flags


def test_unjustified_confrontation_flag():
    sm = AttunedDissonanceSubmodule()
    ctx = RewardContext()

    state = {
        "disagreement_intensity": 0.8,
        "contextual_justification": 0.2,
        "escalation_signal": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert "unjustified_confrontation" in r.flags


def test_escalation_risk_flag():
    sm = AttunedDissonanceSubmodule()
    ctx = RewardContext()

    state = {
        "disagreement_intensity": 0.6,
        "contextual_justification": 0.6,
        "escalation_signal": 0.8,
    }

    r = sm.evaluate(state, ctx)

    assert "escalation_risk" in r.flags
