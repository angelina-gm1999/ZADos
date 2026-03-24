from zados.reward.evaluation.collectors import (
    constraint_violation_rate,
    scenario_consistency_score,
    hallucination_rate,
    abstention_rate,
    self_correction_delta,
    latency_impact,
    provenance_completeness,
)


def test_constraint_violation_rate():
    events = [
        {"action": "allow"},
        {"action": "veto"},
        {"action": "allow"},
    ]
    assert constraint_violation_rate(events) == 1 / 3


def test_scenario_consistency_score():
    flags = [True, True, False, True]
    assert scenario_consistency_score(flags) == 3 / 4


def test_hallucination_rate():
    flags = [False, True, False]
    assert hallucination_rate(flags) == 1 / 3


def test_abstention_rate():
    actions = ["allow", "abstain", "allow"]
    assert abstention_rate(actions) == 1 / 3


def test_self_correction_delta():
    pre = [0.4, 0.6]
    post = [0.6, 0.8]
    assert self_correction_delta(pre, post) == 0.2


def test_latency_impact():
    base = [1.0, 1.2]
    gated = [1.3, 1.5]
    assert abs(latency_impact(base, gated) - 0.3) < 1e-9


def test_provenance_completeness():
    records = [
        {"id": 1, "source": "a"},
        {"id": 2},
    ]
    required = ["id", "source"]
    assert provenance_completeness(records, required) == 0.5
