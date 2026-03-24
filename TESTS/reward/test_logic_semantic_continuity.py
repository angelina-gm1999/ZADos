from zados.reward.domains.logic.semantic_continuity import (
    SemanticContinuitySubmodule,
)
from zados.reward.domains.logic.ports import ContrastResult
from zados.reward.base.types import RewardContext


class DummyMemoryContrast:
    def contrast(self, *, current, query_type, ctx_id=None, limit=5, meta=None):
        assert query_type == "semantic"
        return ContrastResult(
            similarity=0.4,
            divergence=0.6,
            references=[{"prior_meaning": "X meant Y"}],
        )


def test_semantic_continuity_flags_drift():
    sm = SemanticContinuitySubmodule(memory_contrast=DummyMemoryContrast())
    ctx = RewardContext()

    state = {"representation": {"meaning": "Z"}}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.6
    assert "semantic_drift" in r.flags


def test_semantic_continuity_skipped_without_port():
    sm = SemanticContinuitySubmodule(memory_contrast=None)
    ctx = RewardContext()

    state = {"representation": {"meaning": "X"}}
    r = sm.evaluate(state, ctx)

    assert r.meta.get("skipped") is True
    assert "missing_memory_contrast" in r.flags
