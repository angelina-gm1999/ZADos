from zados.reward.domains.logic.context_fidelity import (
    ContextFidelitySubmodule,
)
from zados.reward.domains.logic.ports import ContrastResult
from zados.reward.base.types import RewardContext


class DummyMemoryContrast:
    def contrast(self, *, current, query_type, ctx_id=None, limit=5, meta=None):
        assert query_type == "context"
        return ContrastResult(
            similarity=0.4,
            divergence=0.6,
            references=[{"expected_context": "A", "observed": "B"}],
        )


def test_context_fidelity_flags_drift():
    sm = ContextFidelitySubmodule(memory_contrast=DummyMemoryContrast())
    ctx = RewardContext()

    state = {"representation": {"context": "B"}}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.6
    assert "context_drift" in r.flags


def test_context_fidelity_skipped_without_port():
    sm = ContextFidelitySubmodule(memory_contrast=None)
    ctx = RewardContext()

    state = {"representation": {"context": "A"}}
    r = sm.evaluate(state, ctx)

    assert r.meta.get("skipped") is True
    assert "missing_memory_contrast" in r.flags
