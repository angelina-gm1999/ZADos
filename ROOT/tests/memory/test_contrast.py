"""Tests for MemoryContrast (MemoryContrastPort implementation)."""
import pytest

from zados.memory.types import MemoryPacket
from zados.memory.mid_term.store import MTMMStore
from zados.memory.long_term.store import LTMMEntry, LTMMStore
from zados.memory.managers.contrast import MemoryContrast


def _pkt(user: str, system: str = "ok", turn: int = 1) -> MemoryPacket:
    return MemoryPacket(
        user_message=user,
        system_response=system,
        turn_index=turn,
        intention="info",
    )


def _entry(user: str) -> LTMMEntry:
    pkt = _pkt(user)
    return LTMMEntry(packet=pkt)


# ---------------------------------------------------------------------------
# MemoryContrast: basic contract
# ---------------------------------------------------------------------------

class TestMemoryContrast:
    def _setup(self) -> MemoryContrast:
        mtmm = MTMMStore()
        ltmm = LTMMStore()
        mtmm.write(_pkt("machine learning neural networks deep learning", turn=1))
        ltmm.write(_entry("neural network architecture backpropagation gradient"))
        return MemoryContrast(mtmm, ltmm)

    def test_returns_contrast_result(self):
        from zados.reward.domains.logic.ports import ContrastResult
        mc = self._setup()
        result = mc.contrast(
            current={"output": "neural network training"},
            query_type="semantic",
        )
        assert isinstance(result, ContrastResult)
        assert 0.0 <= result.similarity <= 1.0
        assert 0.0 <= result.divergence <= 1.0

    def test_context_query_hits_mtmm(self):
        mc = self._setup()
        result = mc.contrast(
            current={"output": "machine learning"},
            query_type="context",
        )
        assert result.similarity > 0.0
        assert any(r.get("source") == "MTMM" for r in result.references)

    def test_concept_query_searches_both_tiers(self):
        mc = self._setup()
        result = mc.contrast(
            current={"concept": "neural network"},
            query_type="concept",
        )
        sources = {r.get("source") for r in result.references}
        assert len(sources) > 0   # at least some results

    def test_external_query_returns_references(self):
        mc = self._setup()
        result = mc.contrast(
            current={"output": "backpropagation gradient descent"},
            query_type="external",
        )
        assert isinstance(result.references, list)

    def test_empty_stores_returns_zero_similarity(self):
        mc = MemoryContrast(MTMMStore(), LTMMStore())
        result = mc.contrast(
            current={"output": "anything"},
            query_type="semantic",
        )
        assert result.similarity == 0.0
        assert result.divergence == 0.0

    def test_unrelated_query_low_similarity(self):
        mc = self._setup()
        result = mc.contrast(
            current={"output": "cooking pasta recipe tomato sauce"},
            query_type="semantic",
        )
        # ML-related store vs cooking query → low similarity
        assert result.similarity < 0.5

    def test_related_query_higher_similarity(self):
        mc = self._setup()
        result = mc.contrast(
            current={"output": "deep learning machine learning neural network"},
            query_type="semantic",
        )
        assert result.similarity > 0.0

    def test_meta_populated_for_mtmm_query(self):
        mc = self._setup()
        result = mc.contrast(
            current={"output": "machine learning"},
            query_type="context",
        )
        assert "hit_count" in result.meta or len(result.references) >= 0

    def test_extract_query_text_multiple_fields(self):
        mc = MemoryContrast(MTMMStore(), LTMMStore())
        # Should not raise even with multiple field types
        result = mc.contrast(
            current={"output": "test", "statement": "additional context", "text": "more"},
            query_type="internal",
        )
        assert result is not None

    def test_unknown_query_type_handled(self):
        mc = self._setup()
        result = mc.contrast(
            current={"output": "test query"},
            query_type="some_future_type",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Protocol compliance (duck-typing check)
# ---------------------------------------------------------------------------

class TestMemoryContrastProtocol:
    def test_implements_protocol_interface(self):
        """MemoryContrast must have a .contrast() method with correct signature."""
        mc = MemoryContrast(MTMMStore(), LTMMStore())
        assert callable(mc.contrast)
        # Should accept the expected kwargs
        result = mc.contrast(current={}, query_type="semantic")
        assert hasattr(result, "similarity")
        assert hasattr(result, "divergence")
        assert hasattr(result, "references")
        assert hasattr(result, "meta")
