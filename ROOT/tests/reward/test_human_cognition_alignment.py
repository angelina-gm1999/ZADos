from zados.reward.domains.ethics.human_cognition_alignment import HumanCognitionAlignmentSubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_human_cognition_aligned():
    sm = HumanCognitionAlignmentSubmodule()
    ctx = RewardContext()

    state = {
        "cognitive_load_high": False,
        "structure_clear": True,
        "abstraction_level_appropriate": True,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 1.0
    assert r.flags == {}


def test_cognitive_misalignment_flags():
    sm = HumanCognitionAlignmentSubmodule()
    ctx = RewardContext()

    state = {
        "cognitive_load_high": True,
        "structure_clear": False,
        "abstraction_level_appropriate": False,
    }

    r = sm.evaluate(state, ctx)

    assert "cognitive_overload_risk" in r.flags
    assert "unclear_structure" in r.flags
    assert r.score < 0.5


def test_ethics_domain_includes_human_cognition():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": "explain concept",
        "inferred_intent_confidence": 0.8,
        "cognitive_load_high": False,
    }

    out = d.evaluate(state, ctx)

    assert "human_cognition_alignment" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
