from zados.reward.domains.logic.ports import ContrastResult, TraceResult
from zados.reward.domains.logic.uncertainty_acknowledgment import (
    UncertaintyAcknowledgmentSubmodule,
)
from zados.reward.domains.logic.domain import LogicDomain
from zados.reward.base.types import RewardContext


def test_ports_dataclasses_exist():
    c = ContrastResult(similarity=0.9, divergence=0.1)
    t = TraceResult(trace={"steps": 3})

    assert c.similarity == 0.9
    assert c.divergence == 0.1
    assert isinstance(c.meta, dict)
    assert isinstance(c.references, list)

    assert isinstance(t.trace, dict)
    assert isinstance(t.meta, dict)


def test_uncertainty_acknowledgment_good_alignment():
    sm = UncertaintyAcknowledgmentSubmodule()
    ctx = RewardContext()

    state = {"uncertainty": 0.8, "uncertainty_ack": 0.8}
    r = sm.evaluate(state, ctx)

    assert r.score > 0.95
    assert r.flags == {}


def test_uncertainty_acknowledgment_flags_risk_when_unacknowledged():
    sm = UncertaintyAcknowledgmentSubmodule()
    ctx = RewardContext()

    state = {"uncertainty": 0.9, "uncertainty_ack": 0.1}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.3
    assert "unacknowledged_uncertainty" in r.flags


def test_logic_domain_includes_uncertainty_acknowledgment():
    d = LogicDomain()
    ctx = RewardContext()

    state = {"confidence": 0.6, "uncertainty": 0.4, "uncertainty_ack": 0.4}
    out = d.evaluate(state, ctx)

    assert out.domain == "logic"
    assert "epistemic_calibration" in out.subscores
    assert "uncertainty_acknowledgment" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
    assert out.meta["ports"]["memory_contrast"] is False
    assert out.meta["ports"]["cognitive_trace"] is False
