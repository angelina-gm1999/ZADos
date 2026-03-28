from zados.reward.domains.innovation.exploration_drive import (
    ExplorationDriveSubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_exploration_drive_balanced():
    sm = ExplorationDriveSubmodule()
    ctx = RewardContext()

    state = {
        "uncertainty_level": 0.6,
        "inquiry_intent": 0.6,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.3
    assert r.flags == {}


def test_exploration_drive_ignored_uncertainty_flag():
    sm = ExplorationDriveSubmodule()
    ctx = RewardContext()

    state = {
        "uncertainty_level": 0.9,
        "inquiry_intent": 0.1,
    }

    r = sm.evaluate(state, ctx)

    assert "ignored_uncertainty" in r.flags


def test_innovation_domain_includes_exploration_drive():
    d = InnovationDomain()
    ctx = RewardContext()

    state = {
        "uncertainty_level": 0.5,
        "inquiry_intent": 0.5,
    }

    out = d.evaluate(state, ctx)

    assert "exploration_drive" in out.subscores
