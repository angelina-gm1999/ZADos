from zados.reward.domains.logic.epistemic_calibration import (
    EpistemicCalibrationSubmodule,
)
from zados.reward.domains.logic.domain import LogicDomain
from zados.reward.base.types import RewardContext


def test_epistemic_calibration_balanced():
    sm = EpistemicCalibrationSubmodule()
    ctx = RewardContext()

    state = {"confidence": 0.7, "uncertainty": 0.3}
    result = sm.evaluate(state, ctx)

    assert result.score > 0.9
    assert result.flags == {}


def test_epistemic_calibration_overconfidence_flag():
    sm = EpistemicCalibrationSubmodule()
    ctx = RewardContext()

    state = {"confidence": 0.95, "uncertainty": 0.8}
    result = sm.evaluate(state, ctx)

    assert result.score < 0.5
    assert "overconfidence" in result.flags


def test_logic_domain_aggregation():
    domain = LogicDomain()
    ctx = RewardContext()

    state = {"confidence": 0.6, "uncertainty": 0.4}
    result = domain.evaluate(state, ctx)

    assert result.domain == "logic"
    assert "epistemic_calibration" in result.subscores
    assert 0.0 <= result.general_score <= 1.0
