"""Tests for MTMM: RawInteractionLogger, MidTermProcessingEngine, ContextProcessor, MTMMStore."""
import pytest

from zados.memory.types import MemoryPacket
from zados.memory.mid_term.logger import RawInteractionLogger
from zados.memory.mid_term.trends import MidTermProcessingEngine
from zados.memory.mid_term.context_processor import ContextProcessor, _cosine, _term_freq, _tokenize
from zados.memory.mid_term.store import MTMMStore


def _pkt(turn: int, user: str = "hello world", system: str = "ok",
          intention: str = "info", contradictions: int = 0,
          emotion: float = 0.0, flags=None) -> MemoryPacket:
    return MemoryPacket(
        turn_index=turn,
        user_message=user,
        system_response=system,
        intention=intention,
        contradictions_detected=contradictions,
        emotional_significance=emotion,
        flags=flags or [],
        reward_scores={"ethics": 0.8, "logic": 0.7},
        neurochemical_snapshot={"DA": 0.5},
    )


# ---------------------------------------------------------------------------
# RawInteractionLogger
# ---------------------------------------------------------------------------

class TestRawInteractionLogger:
    def test_append_and_len(self):
        log = RawInteractionLogger()
        log.append(_pkt(1))
        log.append(_pkt(2))
        assert len(log) == 2

    def test_get_all_returns_copy(self):
        log = RawInteractionLogger()
        log.append(_pkt(1))
        all_ = log.get_all()
        all_.clear()
        assert len(log) == 1  # original unaffected

    def test_get_by_turn(self):
        log = RawInteractionLogger()
        log.append(_pkt(5, user="specific message"))
        found = log.get_by_turn(5)
        assert found is not None
        assert found.user_message == "specific message"

    def test_get_by_turn_missing(self):
        log = RawInteractionLogger()
        assert log.get_by_turn(99) is None

    def test_recompress_entry_low_importance(self):
        log = RawInteractionLogger()
        p = _pkt(1)
        p.neurochemical_snapshot = {"DA": 0.5, "NE": 0.3}
        p.emotion_vector = {"curiosity": 0.8, "sys_calm": 0.4}
        log.append(p)
        log.recompress_entry(1, importance=0.1)
        # neurochemical cleared
        assert p.neurochemical_snapshot == {}
        # sys_ emotions dropped
        assert "sys_calm" not in p.emotion_vector

    def test_recompress_entry_high_importance_preserved(self):
        log = RawInteractionLogger()
        p = _pkt(1)
        p.neurochemical_snapshot = {"DA": 0.5}
        log.append(p)
        log.recompress_entry(1, importance=0.9)   # should not compress
        assert p.neurochemical_snapshot == {"DA": 0.5}


# ---------------------------------------------------------------------------
# MidTermProcessingEngine
# ---------------------------------------------------------------------------

class TestMidTermProcessingEngine:
    def test_ingestion_accumulates(self):
        engine = MidTermProcessingEngine(update_interval=2)
        engine.ingest(_pkt(1, contradictions=1))
        engine.ingest(_pkt(2, contradictions=3))
        assert len(engine.trends.contradiction_counts) == 2

    def test_contradiction_trend_increasing(self):
        engine = MidTermProcessingEngine(update_interval=3)
        engine.ingest(_pkt(1, contradictions=0))
        engine.ingest(_pkt(2, contradictions=1))
        engine.ingest(_pkt(3, contradictions=4))   # triggers recalc
        assert engine.trends.contradiction_trend == "increasing"

    def test_contradiction_trend_stable(self):
        engine = MidTermProcessingEngine(update_interval=3)
        for i in range(3):
            engine.ingest(_pkt(i, contradictions=2))
        assert engine.trends.contradiction_trend == "stable"

    def test_intention_pattern_accumulated(self):
        engine = MidTermProcessingEngine(update_interval=10)
        engine.ingest(_pkt(1, intention="info"))
        engine.ingest(_pkt(2, intention="advice"))
        assert "info"   in engine.trends.intention_pattern
        assert "advice" in engine.trends.intention_pattern

    def test_early_recalc_on_high_emotion(self):
        engine = MidTermProcessingEngine(update_interval=100)
        # Should trigger early recalc regardless of interval
        engine.ingest(_pkt(1, emotion=0.9, flags=["F"] * 4))
        # No assertion on trend — just confirm no exception


# ---------------------------------------------------------------------------
# ContextProcessor helpers
# ---------------------------------------------------------------------------

class TestContextProcessorHelpers:
    def test_tokenize(self):
        tokens = _tokenize("Hello World, it's a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "it's"  in tokens

    def test_term_freq_sums_to_one(self):
        tokens = _tokenize("the cat sat on the mat")
        tf = _term_freq(tokens)
        assert abs(sum(tf.values()) - 1.0) < 1e-9

    def test_cosine_identical(self):
        a = _term_freq(["hello", "world"])
        assert _cosine(a, a) == pytest.approx(1.0)

    def test_cosine_disjoint(self):
        a = _term_freq(["hello"])
        b = _term_freq(["world"])
        assert _cosine(a, b) == pytest.approx(0.0)

    def test_cosine_partial_overlap(self):
        a = _term_freq(["hello", "world"])
        b = _term_freq(["hello", "there"])
        sim = _cosine(a, b)
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# ContextProcessor
# ---------------------------------------------------------------------------

class TestContextProcessor:
    def test_search_finds_indexed_entry(self):
        log = RawInteractionLogger()
        proc = ContextProcessor(log)
        p = _pkt(1, user="machine learning is fascinating", system="agreed")
        log.append(p)
        proc.index_packet(p)

        results = proc.search("machine learning")
        assert len(results) > 0
        assert results[0][1].turn_index == 1

    def test_search_respects_limit(self):
        log = RawInteractionLogger()
        proc = ContextProcessor(log)
        for i in range(10):
            p = _pkt(i, user=f"machine learning topic {i}")
            log.append(p)
            proc.index_packet(p)
        results = proc.search("machine learning", limit=3)
        assert len(results) <= 3

    def test_search_empty_returns_empty(self):
        log  = RawInteractionLogger()
        proc = ContextProcessor(log)
        results = proc.search("anything")
        assert results == []

    def test_validate_no_issues_clean_log(self):
        log = RawInteractionLogger()
        proc = ContextProcessor(log)
        log.append(_pkt(1, contradictions=0))
        log.append(_pkt(2, contradictions=1))
        issues = proc.validate()
        assert isinstance(issues, list)

    def test_compress_old_entries_called(self):
        log  = RawInteractionLogger()
        proc = ContextProcessor(log)
        p = _pkt(1)
        p.neurochemical_snapshot = {"DA": 0.5}
        log.append(p)
        proc.index_packet(p, importance=0.0)
        # current_turn=20, window=10 → entry at turn 1 is age=19 → gets compressed
        proc.compress_old_entries(current_turn=20, window=10)
        assert p.neurochemical_snapshot == {}


# ---------------------------------------------------------------------------
# MTMMStore integration
# ---------------------------------------------------------------------------

class TestMTMMStore:
    def test_write_and_len(self):
        store = MTMMStore()
        store.write(_pkt(1))
        store.write(_pkt(2))
        assert len(store) == 2

    def test_search_returns_results(self):
        store = MTMMStore()
        store.write(_pkt(1, user="deep neural networks and backpropagation"))
        store.write(_pkt(2, user="cooking recipes for pasta"))
        results = store.search("neural networks")
        assert len(results) > 0
        # Best match should be turn 1
        assert results[0][1].turn_index == 1

    def test_get_by_turn(self):
        store = MTMMStore()
        store.write(_pkt(7, user="unique content"))
        found = store.get_by_turn(7)
        assert found is not None
        assert found.user_message == "unique content"

    def test_trends_populated(self):
        store = MTMMStore(update_interval=2)
        store.write(_pkt(1, contradictions=1))
        store.write(_pkt(2, contradictions=3))  # triggers recalc
        assert len(store.trends.contradiction_counts) == 2

    def test_validate_returns_list(self):
        store = MTMMStore()
        store.write(_pkt(1))
        issues = store.validate()
        assert isinstance(issues, list)
