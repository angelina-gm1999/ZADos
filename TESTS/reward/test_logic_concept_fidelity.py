from zados.reward.domains.logic.concept_fidelity import (
    ConceptFidelitySubmodule,
)
from zados.reward.domains.logic.ports import ContrastResult
from zados.reward.base.types import RewardContext


class DummyMemoryContrast:
    def contrast(self, *, current, query_type, ctx_id=None, limit=5, meta=None):
        assert query_type == "concept_fidelity"
        return ContrastResult(
            similarity=0.4,
            divergence=0.6,
            references=[{"concept": "X", "definition": "violated"}],
        )


def test_concept_fidelity_flags_definition_violation():
    sm = ConceptFidelitySubmodule(memory_contrast=DummyMemoryContrast())
    ctx = RewardContext()

    state = {"representation": {"concepts": {"X": "misused"}}}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.6
    assert "concept_definition_violation" in r.flags


def test_concept_fidelity_skipped_without_port():
    sm = ConceptFidelitySubmodule(memory_contrast=None)
    ctx = RewardContext()

    state = {"representation": {"concepts": {"X": "correct"}}}
    r = sm.evaluate(state, ctx)

    assert r.meta.get("skipped") is True
    assert "missing_memory_contrast" in r.flags
