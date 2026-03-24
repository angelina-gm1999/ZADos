from zados.reward.base.types import (
    RewardContext,
    RewardSubscore,
    RewardDomainResult,
    RewardMetaDirective,
    RewardWeights,
)


def test_reward_context_defaults():
    ctx = RewardContext()
    assert ctx.reward_profile == "default"
    assert isinstance(ctx.meta, dict)


def test_reward_subscore_container():
    s = RewardSubscore(name="fairness", score=0.75)
    assert s.name == "fairness"
    assert s.score == 0.75
    assert isinstance(s.flags, dict)
    assert isinstance(s.meta, dict)


def test_reward_domain_result_container():
    r = RewardDomainResult(domain="ethics", general_score=0.9)
    assert r.domain == "ethics"
    assert r.general_score == 0.9
    assert isinstance(r.subscores, dict)
    assert isinstance(r.flags, dict)


def test_reward_weights_get():
    w = RewardWeights(weights={"ethics": 0.8})
    assert w.get("ethics") == 0.8
    assert w.get("logic", default=0.2) == 0.2


def test_reward_meta_directive_defaults():
    d = RewardMetaDirective()
    assert d.allow_output is True
    assert d.abstain is False
    assert d.suppress is False
    assert isinstance(d.directives, dict)
    assert isinstance(d.routing, dict)
    assert isinstance(d.flags, dict)
