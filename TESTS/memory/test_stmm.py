"""Tests for STMM components, STMMStore, and MemoryExitCompressor."""
import pytest

from zados.memory.types import SpeakerID
from zados.memory.short_term.components import (
    ActiveMessageBuffer,
    BrainProcessTracker,
    EngineExecution,
    FractalDecompositionResults,
    IntentionAnalysisResults,
    MemoryContrastResults,
    MemoryMatch,
)
from zados.memory.short_term.store import STMMStore
from zados.memory.short_term.compressor import MemoryExitCompressor


# ---------------------------------------------------------------------------
# ActiveMessageBuffer
# ---------------------------------------------------------------------------

class TestActiveMessageBuffer:
    def test_add_user_message_fifo(self):
        stmm = STMMStore()
        stmm.add_user_message("msg1")
        stmm.add_user_message("msg2")
        stmm.add_user_message("msg3")   # should evict msg1

        user_msgs = stmm.active_message_buffer.get_by_speaker(SpeakerID.USER)
        assert len(user_msgs) == 2
        assert user_msgs[0].text == "msg2"
        assert user_msgs[1].text == "msg3"

    def test_add_system_message_fifo(self):
        stmm = STMMStore()
        stmm.add_system_response("resp1")
        stmm.add_system_response("resp2")
        stmm.add_system_response("resp3")

        sys_msgs = stmm.active_message_buffer.get_by_speaker(SpeakerID.SYSTEM)
        assert len(sys_msgs) == 2
        assert sys_msgs[0].text == "resp2"

    def test_latest_user(self):
        stmm = STMMStore()
        stmm.add_user_message("first")
        stmm.add_user_message("second")
        assert stmm.active_message_buffer.latest_user().text == "second"

    def test_latest_system_none_when_empty(self):
        stmm = STMMStore()
        assert stmm.active_message_buffer.latest_system() is None

    def test_user_and_system_independent_fifos(self):
        stmm = STMMStore()
        stmm.add_user_message("u1")
        stmm.add_system_response("s1")
        stmm.add_user_message("u2")
        stmm.add_user_message("u3")   # evicts u1 only

        assert len(stmm.active_message_buffer.get_by_speaker(SpeakerID.USER))   == 2
        assert len(stmm.active_message_buffer.get_by_speaker(SpeakerID.SYSTEM)) == 1


# ---------------------------------------------------------------------------
# BrainProcessTracker
# ---------------------------------------------------------------------------

class TestBrainProcessTracker:
    def test_record_and_retrieve(self):
        tracker = BrainProcessTracker()
        exec_ = EngineExecution(engine_id="E01", timing_ms=12.5, output_summary="ok")
        tracker.record(exec_)
        assert len(tracker.executions) == 1
        assert tracker.executions[0].engine_id == "E01"

    def test_engine_ids_run_skips_skipped(self):
        tracker = BrainProcessTracker()
        tracker.record(EngineExecution("E01", 10.0, "ok"))
        tracker.record(EngineExecution("E02", 0.0, "", skipped=True, skip_reason="no port"))
        assert tracker.engine_ids_run() == ["E01"]

    def test_mark_stage(self):
        tracker = BrainProcessTracker()
        tracker.mark_stage("step_a", True)
        assert tracker.pipeline_stage_flags["step_a"] is True


# ---------------------------------------------------------------------------
# STMMStore lifecycle
# ---------------------------------------------------------------------------

class TestSTMMStore:
    def test_turn_index_increments_on_user_message(self):
        stmm = STMMStore()
        assert stmm.turn_index == 0
        stmm.add_user_message("hello")
        assert stmm.turn_index == 1
        stmm.add_user_message("world")
        assert stmm.turn_index == 2

    def test_begin_cycle_clears_analysis(self):
        stmm = STMMStore()
        stmm.add_user_message("hi")
        stmm.intention_analysis.primary_intention = "advice-seeking"
        stmm.reward_evaluation.composite_score    = 0.9
        stmm.cortical_reflection.active_mode      = "Dev"

        stmm.begin_cycle()

        assert stmm.intention_analysis.primary_intention == ""
        assert stmm.reward_evaluation.composite_score    == 0.0
        assert stmm.cortical_reflection.active_mode      == "Normal"

    def test_begin_cycle_preserves_message_buffer(self):
        stmm = STMMStore()
        stmm.add_user_message("hello")
        stmm.begin_cycle()
        # Buffer should still hold the message
        assert stmm.active_message_buffer.latest_user().text == "hello"

    def test_snapshot_keys(self):
        stmm = STMMStore()
        snap = stmm.snapshot()
        required_keys = {
            "turn_index", "user_messages", "system_messages",
            "intention", "stability_passed", "composite_reward",
            "engines_run", "active_mode", "contradictions",
        }
        assert required_keys <= snap.keys()


# ---------------------------------------------------------------------------
# MemoryExitCompressor
# ---------------------------------------------------------------------------

class TestMemoryExitCompressor:
    def _make_populated_stmm(self) -> STMMStore:
        stmm = STMMStore()
        stmm.add_user_message("What is the meaning of life?")
        stmm.add_system_response("42.")
        stmm.intention_analysis.primary_intention = "information-gathering"
        stmm.emotion_detection.user_emotion_signals = {"curiosity": 0.8}
        stmm.emotion_detection.saturation_levels    = {"curiosity": 0.6}
        stmm.cephalic_liquid_logger.nt_concentrations = {"DA": 0.7, "5HT": 0.5}
        stmm.reward_evaluation.per_domain_results   = {
            "ethics": {"score": 0.9},
            "logic":  {"score": 0.8},
        }
        stmm.reward_evaluation.flags = ["FLAG_ETHICS_HIGH"]
        stmm.memory_contrast.potential_contradictions = ["stmt_a vs stmt_b"]
        return stmm

    def test_produces_packet(self):
        stmm   = self._make_populated_stmm()
        comp   = MemoryExitCompressor()
        packet = comp.compress(stmm)
        assert packet is not None
        assert packet.user_message   == "What is the meaning of life?"
        assert packet.system_response == "42."

    def test_lossless_message_text(self):
        stmm   = self._make_populated_stmm()
        packet = MemoryExitCompressor().compress(stmm)
        assert "meaning of life" in packet.user_message

    def test_intention_compressed(self):
        stmm   = self._make_populated_stmm()
        packet = MemoryExitCompressor().compress(stmm)
        assert packet.intention == "information-gathering"

    def test_nt_snapshot(self):
        stmm   = self._make_populated_stmm()
        packet = MemoryExitCompressor().compress(stmm)
        assert packet.neurochemical_snapshot["DA"]  == 0.7
        assert packet.neurochemical_snapshot["5HT"] == 0.5

    def test_reward_scores_domain_level_only(self):
        stmm   = self._make_populated_stmm()
        packet = MemoryExitCompressor().compress(stmm)
        assert "ethics" in packet.reward_scores
        assert "logic"  in packet.reward_scores

    def test_contradiction_count(self):
        stmm   = self._make_populated_stmm()
        packet = MemoryExitCompressor().compress(stmm)
        assert packet.contradictions_detected == 1

    def test_emotional_significance_from_saturation(self):
        stmm   = self._make_populated_stmm()
        packet = MemoryExitCompressor().compress(stmm)
        assert packet.emotional_significance == pytest.approx(0.6)

    def test_empty_stmm_produces_valid_packet(self):
        stmm   = STMMStore()
        packet = MemoryExitCompressor().compress(stmm)
        assert packet.user_message     == ""
        assert packet.system_response  == ""
        assert packet.contradictions_detected == 0
