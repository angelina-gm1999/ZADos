from zados.reward.base.structure import (
    ThresholdSpec,
    RewardFlag,
    RewardFlagSet,
    ProvenanceRecord,
)


def test_threshold_spec_basic():
    t = ThresholdSpec(lower=0.2, upper=0.8, label="safe")
    assert t.in_range(0.5) is True
    assert t.in_range(0.1) is False


def test_reward_flag_and_flagset():
    f1 = RewardFlag(name="high_creativity_low_logic", severity="risk")
    f2 = RewardFlag(name="ethics_violation", severity="critical")

    fs = RewardFlagSet(flags=(f1, f2))

    assert fs.has_severity("risk") is True
    assert fs.has_severity("warning") is False
    assert "ethics_violation" in fs.names()


def test_provenance_record_defaults():
    p = ProvenanceRecord(source="reward_synthesis")

    assert isinstance(p.provenance_id, str)
    assert isinstance(p.timestamp, float)
    assert p.source == "reward_synthesis"
    assert isinstance(p.notes, dict)
