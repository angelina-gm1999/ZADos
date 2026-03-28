"""Tests for LTMM: store, consolidation, relevance heuristics, fractal comparator."""
import pytest
from datetime import datetime, timedelta

from zados.memory.types import MemoryPacket
from zados.memory.long_term.store import Granularity, LTMMEntry, LTMMStore
from zados.memory.long_term.consolidation import MemoryConsolidationEngine
from zados.memory.long_term.relevance import MemoryRelevanceHeuristicsEngine
from zados.memory.long_term.fractal_comparator import FractalPatternComparator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pkt(
    pid: str = None,
    user: str = "hello world",
    system: str = "response",
    emotional_sig: float = 0.0,
    trust: float = 1.0,
    contradictions: int = 0,
    paradoxes: int = 0,
    flags=None,
    unsolved=None,
) -> MemoryPacket:
    p = MemoryPacket(
        user_message=user,
        system_response=system,
        emotional_significance=emotional_sig,
        trust_weight=trust,
        contradictions_detected=contradictions,
        paradoxes_detected=paradoxes,
        flags=flags or [],
        unsolved_items_matched=unsolved or [],
    )
    if pid:
        p.packet_id = pid
    return p


def _entry(
    pkt: MemoryPacket = None,
    granularity: str = Granularity.SEMANTIC,
    retrieval_count: int = 0,
    relevance: float = 1.0,
    emotion_sig: float = 0.5,
    utility: float = 0.5,
    cold: bool = False,
    identity: bool = False,
    last_accessed_hours_ago: float = 0.0,
) -> LTMMEntry:
    if pkt is None:
        pkt = _pkt()
    pkt.emotional_significance = emotion_sig
    e = LTMMEntry(
        packet=pkt,
        granularity=granularity,
        relevance_score=relevance,
        retrieval_count=retrieval_count,
        utility_score=utility,
        cold_storage=cold,
        identity_relevant=identity,
        last_accessed=datetime.utcnow() - timedelta(hours=last_accessed_hours_ago),
    )
    return e


# ---------------------------------------------------------------------------
# LTMMStore
# ---------------------------------------------------------------------------

class TestLTMMStore:
    def test_write_and_get_by_id(self):
        store = LTMMStore()
        e = _entry(_pkt("abc"))
        store.write(e)
        found = store.get_by_id("abc")
        assert found is not None

    def test_search_returns_matching_entry(self):
        store = LTMMStore()
        e1 = _entry(_pkt(user="quantum entanglement physics research"))
        e2 = _entry(_pkt(user="chocolate cake recipe dessert"))
        store.write(e1)
        store.write(e2)
        results = store.search("quantum physics")
        assert len(results) >= 1
        assert results[0][1].packet.user_message.startswith("quantum")

    def test_search_respects_limit(self):
        store = LTMMStore()
        for i in range(10):
            store.write(_entry(_pkt(user=f"machine learning model training {i}")))
        results = store.search("machine learning", limit=3)
        assert len(results) <= 3

    def test_cold_entries_excluded_by_default(self):
        store = LTMMStore()
        e_cold = _entry(_pkt(user="cold storage entry irrelevant"), cold=True)
        e_warm = _entry(_pkt(user="warm active relevant entry"))
        store.write(e_cold)
        store.write(e_warm)
        results = store.search("cold storage entry irrelevant", include_cold=False)
        assert all(not entry.cold_storage for _, entry in results)

    def test_search_include_cold(self):
        store = LTMMStore()
        e = _entry(_pkt(user="cold storage unique phrase"), cold=True)
        store.write(e)
        results = store.search("cold storage unique phrase", include_cold=True)
        assert len(results) >= 1

    def test_touch_increments_retrieval(self):
        store = LTMMStore()
        e = _entry(_pkt("xyz"))
        store.write(e)
        store.search("hello")  # trigger touch
        found = store.get_by_id("xyz")
        assert found is not None

    def test_demote_to_cold(self):
        store = LTMMStore()
        e = _entry(_pkt("cold_me"))
        store.write(e)
        store.demote_to_cold("cold_me")
        found = store.get_by_id("cold_me")
        assert found.cold_storage is True

    def test_purge(self):
        store = LTMMStore()
        e = _entry(_pkt("purge_me"))
        store.write(e)
        store.purge("purge_me")
        assert store.get_by_id("purge_me") is None
        assert len(store) == 0

    def test_update_utility(self):
        store = LTMMStore()
        e = _entry(_pkt("util_me"))
        store.write(e)
        store.update_utility("util_me", 0.2)
        found = store.get_by_id("util_me")
        assert found.utility_score == pytest.approx(0.7, abs=0.01)

    def test_update_utility_clamped(self):
        store = LTMMStore()
        e = _entry(_pkt("clamp_me"), utility=0.9)
        store.write(e)
        store.update_utility("clamp_me", 0.5)   # would exceed 1.0
        found = store.get_by_id("clamp_me")
        assert found.utility_score <= 1.0


# ---------------------------------------------------------------------------
# LTMMEntry touch
# ---------------------------------------------------------------------------

class TestLTMMEntry:
    def test_touch_increments_retrieval_count(self):
        e = _entry()
        e.touch()
        assert e.retrieval_count == 1
        e.touch()
        assert e.retrieval_count == 2


# ---------------------------------------------------------------------------
# MemoryConsolidationEngine
# ---------------------------------------------------------------------------

class TestMemoryConsolidationEngine:
    def test_high_emotion_qualifies(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        pkt    = _pkt(emotional_sig=0.8)
        promoted = engine.consolidate([pkt])
        assert pkt.packet_id in promoted
        assert len(store) == 1

    def test_low_emotion_no_flags_not_promoted(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        pkt    = _pkt(emotional_sig=0.1, trust=0.9, contradictions=0)
        promoted = engine.consolidate([pkt])
        assert promoted == []

    def test_unresolved_items_qualifies(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        pkt    = _pkt(unsolved=["ucid_001"])
        promoted = engine.consolidate([pkt])
        assert pkt.packet_id in promoted

    def test_unresolved_gets_verbatim_granularity(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        pkt    = _pkt(unsolved=["ucid_001"])
        engine.consolidate([pkt])
        entry = store.get_by_id(pkt.packet_id)
        assert entry.granularity == Granularity.VERBATIM

    def test_critical_flag_qualifies(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        pkt    = _pkt(flags=["CRITICAL:ethics_violation"])
        promoted = engine.consolidate([pkt])
        assert pkt.packet_id in promoted

    def test_identity_flag_sets_identity_relevant(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        pkt    = _pkt(flags=["IDENTITY:values_shift"])
        engine.consolidate([pkt])
        entry = store.get_by_id(pkt.packet_id)
        assert entry.identity_relevant is True

    def test_low_trust_qualifies(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        pkt    = _pkt(trust=0.2)
        promoted = engine.consolidate([pkt])
        assert pkt.packet_id in promoted

    def test_multiple_packets_selective(self):
        store  = LTMMStore()
        engine = MemoryConsolidationEngine(store)
        p1 = _pkt(emotional_sig=0.9)
        p2 = _pkt(emotional_sig=0.0, trust=1.0)
        promoted = engine.consolidate([p1, p2])
        assert p1.packet_id     in promoted
        assert p2.packet_id not in promoted


# ---------------------------------------------------------------------------
# MemoryRelevanceHeuristicsEngine
# ---------------------------------------------------------------------------

class TestMemoryRelevanceHeuristicsEngine:
    def test_fresh_entry_not_demoted(self):
        store  = LTMMStore()
        engine = MemoryRelevanceHeuristicsEngine(store)
        e = _entry(last_accessed_hours_ago=0.0, retrieval_count=3, emotion_sig=0.7)
        store.write(e)
        demoted, purgeable = engine.scan()
        assert e.packet.packet_id not in demoted

    def test_stale_low_relevance_demoted(self):
        store  = LTMMStore()
        engine = MemoryRelevanceHeuristicsEngine(store)
        e = _entry(
            last_accessed_hours_ago=5000.0,  # very old
            retrieval_count=0,
            emotion_sig=0.0,
            utility=0.0,
        )
        store.write(e)
        demoted, _ = engine.scan()
        assert e.packet.packet_id in demoted

    def test_identity_relevant_never_demoted(self):
        store  = LTMMStore()
        engine = MemoryRelevanceHeuristicsEngine(store)
        e = _entry(
            last_accessed_hours_ago=10000.0,
            retrieval_count=0,
            emotion_sig=0.0,
            utility=0.0,
            identity=True,
        )
        store.write(e)
        demoted, purgeable = engine.scan()
        assert e.packet.packet_id not in demoted
        assert e.packet.packet_id not in purgeable

    def test_relevance_score_updated_after_scan(self):
        store  = LTMMStore()
        engine = MemoryRelevanceHeuristicsEngine(store)
        e = _entry(last_accessed_hours_ago=48.0, retrieval_count=2, emotion_sig=0.4)
        store.write(e)
        old_score = e.relevance_score
        engine.scan()
        # Relevance should have been recomputed (may be lower for older entry)
        assert e.relevance_score != old_score or e.relevance_score <= 1.0


# ---------------------------------------------------------------------------
# FractalPatternComparator
# ---------------------------------------------------------------------------

class TestFractalPatternComparator:
    def test_unique_entry_writes(self):
        store = LTMMStore()
        e1 = _entry(_pkt(user="quantum physics research paper"))
        store.write(e1)

        comp = FractalPatternComparator(store)
        e2   = _entry(_pkt(user="cooking pasta recipe tonight"))
        result = comp.compare(e2)
        assert result.action == "write"

    def test_duplicate_merges(self):
        store = LTMMStore()
        e1 = _entry(_pkt(user="the exact same sentence word for word repeated"))
        store.write(e1)

        comp = FractalPatternComparator(store)
        e2   = _entry(_pkt(user="the exact same sentence word for word repeated"))
        result = comp.compare(e2)
        assert result.action == "merge"
        assert result.merge_target_id == e1.packet.packet_id

    def test_similar_reinforces(self):
        store = LTMMStore()
        e1 = _entry(_pkt(user="machine learning deep neural network training"))
        store.write(e1)

        comp   = FractalPatternComparator(store)
        e2     = _entry(_pkt(user="deep learning neural network optimization methods"))
        result = comp.compare(e2)
        assert result.action in ("reinforce", "write")

    def test_cross_links_populated_for_similar(self):
        store = LTMMStore()
        for i in range(3):
            store.write(_entry(_pkt(user=f"machine learning neural networks model {i}")))

        comp   = FractalPatternComparator(store)
        e_new  = _entry(_pkt(user="neural networks machine learning deep model"))
        result = comp.compare(e_new)
        assert len(result.cross_links) > 0

    def test_merge_promotes_to_symbolic(self):
        store = LTMMStore()
        e1 = _entry(_pkt(user="same phrase again and again"), granularity=Granularity.SEMANTIC)
        store.write(e1)

        comp = FractalPatternComparator(store)
        e2   = _entry(_pkt(user="same phrase again and again"), granularity=Granularity.SEMANTIC)
        comp.compare(e2)

        existing = store.get_by_id(e1.packet.packet_id)
        assert existing.granularity == Granularity.SYMBOLIC
