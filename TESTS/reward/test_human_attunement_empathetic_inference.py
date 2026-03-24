from zados.reward.domains.human_attunement.empathetic_inference import (
    EmpatheticInferenceSubmodule,
)
from zados.reward.domains.human_attunement.domain import HumanAttunementDomain
from zados.reward.base.types import RewardContext


def test_empathetic_inference_balanced():
    sm = EmpatheticInferenceSubmodule()
    ctx = RewardContext()

    state = {
        "inferred_user_state_confidence": 0.7,
        "user_state_signal_strength": 0.8,
        "inference_mismatch": 0.1,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.4
    assert "poor_inference_fit" not in r.flags


def test_overconfident_inference_flag():
    sm = EmpatheticInferenceSubmodule()
    ctx = RewardContext()

    state = {
        "inferred_user_state_confidence": 0.9,
        "user_state_signal_strength": 0.1,
        "inference_mismatch": 0.1,
    }

    r = sm.evaluate(state, ctx)

    assert "overconfident_inference" in r.flags
    assert "low_observability" in r.flags


def test_poor_inference_fit_flag():
    sm = EmpatheticInferenceSubmodule()
    ctx = RewardContext()

    state = {
        "inferred_user_state_confidence": 0.7,
        "user_state_signal_strength": 0.7,
        "inference_mismatch": 0.9,
    }

    r = sm.evaluate(state, ctx)

    assert "poor_inference_fit" in r.flags
    assert 0.0 <= r.score <= 1.0


def test_human_attunement_domain_includes_empathetic_inference():
    d = HumanAttunementDomain()
    ctx = RewardContext()

    state = {
        "inferred_user_state_confidence": 0.5,
        "user_state_signal_strength": 0.5,
        "inference_mismatch": 0.2,
    }

    out = d.evaluate(state, ctx)

    assert out.domain == "human_attunement"
    assert "empathetic_inference" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
