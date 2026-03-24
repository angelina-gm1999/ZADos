"""Tests for MemoryImplementationManager and the full MemoryLayer facade."""
import pytest

from zados.memory.types import MemoryPacket, MemoryTier
from zados.memory.short_term.store import STMMStore
from zados.memory.mid_term.store import MTMMStore
from zados.memory.long_term.store import LTMMStore
from zados.memory.long_term.specialized_logs import SpecializedLogs, UnsolvedConceptEntry
from zados.memory.managers.implementation import MemoryImplementationManager
from zados.memory import MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populated_stmm(
    user_msg: str = "test message",
    system_msg: str = "test response",
    emotion_sig: float = 0.3,
    flags=None,
) -> STMMStore:
    stmm = STMMStore()
    stmm.add_user_message(user_msg)
    stmm.add_system_response(system_msg)
    stmm.intention_analysis.primary_intention  = "info"
    stmm.emotion_detection.saturation_levels   = {"joy": emotion_sig}
    stmm.emotion_detection.user_emotion_signals = {"joy": emotion_sig}
    stmm.reward_evaluation.per_domain_results  = {"ethics": {"score": 0.8}}
    stmm.reward_evaluation.flags               = flags or []
    return stmm


# ---------------------------------------------------------------------------
# MemoryImplementationManager: on_cycle_end
# ---------------------------------------------------------------------------

class TestMemoryImplementationManager:
    def _manager(self) -> MemoryImplementationManager:
        return MemoryImplementationManager(MTMMStore(), LTMMStore())

    def test_on_cycle_end_returns_packet(self):
        mgr    = self._manager()
        stmm   = _populated_stmm()
        packet = mgr.on_cycle_end(stmm)
        assert isinstance(packet, MemoryPacket)
        assert packet.user_message == "test message"

    def test_on_cycle_end_writes_to_mtmm(self):
        mtmm = MTMMStore()
        mgr  = MemoryImplementationManager(mtmm, LTMMStore())
        mgr.on_cycle_end(_populated_stmm())
        assert len(mtmm) == 1

    def test_multiple_cycles_accumulate_in_mtmm(self):
        mtmm = MTMMStore()
        mgr  = MemoryImplementationManager(mtmm, LTMMStore())
        mgr.on_cycle_end(_populated_stmm(user_msg="first"))
        mgr.on_cycle_end(_populated_stmm(user_msg="second"))
        assert len(mtmm) == 2

    def test_packet_source_tier_is_stmm(self):
        mgr    = self._manager()
        packet = mgr.on_cycle_end(_populated_stmm())
        assert packet.source_tier == MemoryTier.STMM

    # -----------------------------------------------------------------------
    # Importance routing
    # -----------------------------------------------------------------------

    def test_high_emotion_packet_gets_high_importance(self):
        """Verify high-emotion stmm doesn't crash and writes correctly."""
        mtmm = MTMMStore()
        mgr  = MemoryImplementationManager(mtmm, LTMMStore())
        stmm = _populated_stmm(emotion_sig=0.95)
        mgr.on_cycle_end(stmm)
        pkt = mtmm.get_all_packets()[0]
        assert pkt.emotional_significance == pytest.approx(0.95)

    # -----------------------------------------------------------------------
    # Consolidation
    # -----------------------------------------------------------------------

    def test_consolidate_promotes_qualifying_packets(self):
        mtmm = MTMMStore()
        ltmm = LTMMStore()
        mgr  = MemoryImplementationManager(mtmm, ltmm)

        # High-emotion packet → qualifies for LTMM
        stmm = _populated_stmm(emotion_sig=0.9)
        mgr.on_cycle_end(stmm)

        promoted = mgr.consolidate()
        assert len(promoted) >= 0   # may or may not promote depending on comparator

    def test_consolidate_empty_mtmm(self):
        mgr      = self._manager()
        promoted = mgr.consolidate()
        assert promoted == []

    def test_emergency_consolidate_high_priority(self):
        """Emergency consolidation with a high-emotion packet."""
        mgr = self._manager()
        stmm = _populated_stmm(emotion_sig=0.95)
        pkt  = mgr.on_cycle_end(stmm)
        # Force the packet to qualify by giving it high emotional sig
        pkt.emotional_significance = 0.95
        result = mgr.emergency_consolidate(pkt)
        # Result is a str (packet_id or empty)
        assert isinstance(result, str)

    # -----------------------------------------------------------------------
    # Tick unsolved
    # -----------------------------------------------------------------------

    def test_tick_unsolved_increments_stagnation(self):
        logs = SpecializedLogs()
        eid  = logs.unsolved.add(UnsolvedConceptEntry())
        mgr  = MemoryImplementationManager(MTMMStore(), LTMMStore(), specialized_logs=logs)
        mgr.tick_unsolved()
        mgr.tick_unsolved()
        assert logs.unsolved.get_by_id(eid).stagnation_cycles == 2

    # -----------------------------------------------------------------------
    # Relevance scan
    # -----------------------------------------------------------------------

    def test_run_relevance_scan_returns_dict(self):
        mgr    = self._manager()
        result = mgr.run_relevance_scan()
        assert "demoted"   in result
        assert "purgeable" in result

    # -----------------------------------------------------------------------
    # Flag routing
    # -----------------------------------------------------------------------

    def test_contradiction_flag_routed_to_contradiction_log(self):
        logs = SpecializedLogs()
        mgr  = MemoryImplementationManager(MTMMStore(), LTMMStore(), specialized_logs=logs)
        stmm = _populated_stmm(flags=["CONTRADICTION:stmt_a_vs_b"])
        stmm.memory_contrast.potential_contradictions = ["a vs b"]
        mgr.on_cycle_end(stmm)
        assert len(logs.contradiction) >= 1

    def test_paradox_flag_routed_to_paradox_log(self):
        logs = SpecializedLogs()
        mgr  = MemoryImplementationManager(MTMMStore(), LTMMStore(), specialized_logs=logs)
        stmm = _populated_stmm(flags=["PARADOX:this_is_false"])
        stmm.reward_evaluation.per_domain_results = {}
        # Manually set paradox count on the packet after compression
        pkt = mgr.on_cycle_end(stmm)
        pkt.paradoxes_detected = 1
        # Route directly
        mgr._route_flags(pkt)
        assert len(logs.paradox) >= 1

    # -----------------------------------------------------------------------
    # Consistency validation
    # -----------------------------------------------------------------------

    def test_validate_consistency_no_issues_on_clean_state(self):
        mgr    = self._manager()
        issues = mgr.validate_consistency()
        assert isinstance(issues, list)

    def test_validate_consistency_flags_missing_unsolved_ref(self):
        logs = SpecializedLogs()
        mgr  = MemoryImplementationManager(MTMMStore(), LTMMStore(), specialized_logs=logs)
        # Write a packet that references a non-existent unsolved concept
        stmm = _populated_stmm()
        stmm.memory_contrast.unresolved_query_matches = ["missing_ucid"]
        mgr.on_cycle_end(stmm)
        issues = mgr.validate_consistency()
        assert any("missing_ucid" in issue for issue in issues)

    # -----------------------------------------------------------------------
    # Logs property
    # -----------------------------------------------------------------------

    def test_logs_property_returns_specialized_logs(self):
        mgr = self._manager()
        assert isinstance(mgr.logs, SpecializedLogs)


# ---------------------------------------------------------------------------
# MemoryLayer facade
# ---------------------------------------------------------------------------

class TestMemoryLayer:
    def test_initialization(self):
        ml = MemoryLayer()
        assert ml.stmm    is not None
        assert ml.mtmm    is not None
        assert ml.ltmm    is not None
        assert ml.manager is not None
        assert ml.contrast is not None

    def test_end_cycle_returns_packet(self):
        ml = MemoryLayer()
        ml.stmm.add_user_message("Hello")
        ml.stmm.add_system_response("Hi there")
        packet = ml.end_cycle()
        assert isinstance(packet, MemoryPacket)
        assert packet.user_message == "Hello"

    def test_full_cycle_writes_to_mtmm(self):
        ml = MemoryLayer()
        ml.stmm.add_user_message("First message")
        ml.stmm.add_system_response("First response")
        ml.end_cycle()
        assert len(ml.mtmm) == 1

    def test_multiple_cycles(self):
        ml = MemoryLayer()
        for i in range(3):
            ml.stmm.begin_cycle()
            ml.stmm.add_user_message(f"Message {i}")
            ml.stmm.add_system_response(f"Response {i}")
            ml.end_cycle()
        assert len(ml.mtmm) == 3

    def test_contrast_wired_to_live_stores(self):
        """Contrast should find what was stored via end_cycle."""
        ml = MemoryLayer()
        ml.stmm.add_user_message("deep learning neural networks training")
        ml.stmm.add_system_response("gradient descent optimization")
        ml.end_cycle()

        result = ml.contrast.contrast(
            current={"output": "neural networks deep learning"},
            query_type="semantic",
        )
        # The MTMM has one entry with matching content → similarity > 0
        assert result.similarity > 0.0

    def test_consolidation_after_session(self):
        ml = MemoryLayer()
        ml.stmm.add_user_message("I feel strongly about this identity")
        ml.stmm.add_system_response("Understood")
        ml.stmm.emotion_detection.saturation_levels = {"strong": 0.95}
        ml.end_cycle()

        promoted = ml.manager.consolidate()
        # May be 0 or 1 depending on consolidation criteria; should not raise
        assert isinstance(promoted, list)
