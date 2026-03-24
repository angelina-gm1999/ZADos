from zados.reward.domains.logic.concept_continuity import (
    ConceptContinuitySubmodule,
)
from zados.reward.domains.logic.ports import ContrastResult
from zados.reward.base.types import RewardContext


class DummyMemoryContrast:
    def contrast(self, *, current, query_type, ctx_id=None, limit=5, meta=None):
        assert query_type == "concept"
        return ContrastResult(
            similarity=0.3,
            divergence=0.7,
            references=[{"concept": "X", "previous_definition": "old"}],
        )


def test_concept_continuity_flags_identity_drift():
    sm = ConceptContinuitySubmodule(memory_contrast=DummyMemoryContrast())
    ctx = RewardContext()

    state = {"representation": {"concepts": {"X": "new"}}}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.5
    assert "concept_identity_drift" in r.flags


def test_concept_continuity_skipped_without_port():
    sm = ConceptContinuitySubmodule(memory_contrast=None)
    ctx = RewardContext()

    state = {"representation": {"concepts": {"X": "stable"}}}
    r = sm.evaluate(state, ctx)

    assert r.meta.get("skipped") is True
    assert "missing_memory_contrast" in r.flags
