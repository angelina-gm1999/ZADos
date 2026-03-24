from zados.reward.domains.logic.external_consistency import (
    ExternalConsistencySubmodule,
)
from zados.reward.domains.logic.ports import ContrastResult
from zados.reward.base.types import RewardContext


class DummyMemoryContrast:
    def contrast(self, *, current, query_type, ctx_id=None, limit=5, meta=None):
        assert query_type == "external"
        return ContrastResult(
            similarity=0.3,
            divergence=0.7,
            references=[{"prior": "Previously asserted B"}],
        )


def test_external_consistency_flags_contradiction():
    sm = ExternalConsistencySubmodule(memory_contrast=DummyMemoryContrast())
    ctx = RewardContext()

    state = {"representation": {"claims": ["not B"]}}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.5
    assert "external_contradiction" in r.flags


def test_external_consistency_skipped_without_port():
    sm = ExternalConsistencySubmodule(memory_contrast=None)
    ctx = RewardContext()

    state = {"representation": {"claims": ["B"]}}
    r = sm.evaluate(state, ctx)

    assert r.meta.get("skipped") is True
    assert "missing_memory_contrast" in r.flags
