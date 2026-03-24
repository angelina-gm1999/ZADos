from zados.reward.domains.logic.internal_consistency import (
    InternalConsistencySubmodule,
)
from zados.reward.domains.logic.ports import ContrastResult
from zados.reward.base.types import RewardContext


class DummyMemoryContrast:
    def contrast(self, *, current, query_type, ctx_id=None, limit=5, meta=None):
        return ContrastResult(
            similarity=0.2,
            divergence=0.8,
            references=[{"conflict": "A vs not-A"}],
        )


def test_internal_consistency_flags_contradiction():
    sm = InternalConsistencySubmodule(memory_contrast=DummyMemoryContrast())
    ctx = RewardContext()

    state = {"representation": {"claims": ["A", "not A"]}}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.5
    assert "internal_contradiction" in r.flags


def test_internal_consistency_skipped_without_port():
    sm = InternalConsistencySubmodule(memory_contrast=None)
    ctx = RewardContext()

    state = {"representation": {"claims": ["A"]}}
    r = sm.evaluate(state, ctx)

    assert r.meta.get("skipped") is True
    assert "missing_memory_contrast" in r.flags
