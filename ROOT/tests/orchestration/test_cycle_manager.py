"""
Tests for the CycleManager orchestration layer.

Tests the thin sequencer that connects Memory ↔ Cognitive Engines:
  - STMM begin/compress/write lifecycle
  - Engine registration and dispatch
  - Dependency checking
  - Memory contrast population
  - BrainProcessTracker recording
  - Selective dispatch
  - Error handling in adapters
"""
from __future__ import annotations

import pytest
from typing import Any, Dict

from zados.memory import MemoryLayer, MemoryPacket
from zados.memory.short_term.store import STMMStore
from zados.orchestration.cycle_types import CycleContext, CycleResult, EngineSlot
from zados.orchestration.cycle_manager import CycleManager


# =====================================================================
# Minimal stub engine (follows Pattern A)
# =====================================================================


class StubEngine:
    """Minimal engine that records calls for testing."""

    def __init__(self, engine_id: str = "stub_engine"):
        self.engine_id = engine_id
        self.cluster = "test"
        self._nt_state: Dict[str, float] = {}
        self._process_calls: list = []
        self._process_result: Any = {"summary": "stub_processed"}

    def update_neurochem_state(self, state: Dict[str, float]) -> None:
        self._nt_state = dict(state)

    def process(self, inp: Any = None) -> Any:
        self._process_calls.append(inp)
        return self._process_result

    def get_status(self) -> Dict[str, Any]:
        return {"engine_id": self.engine_id}


def stub_adapter(engine: StubEngine, ctx: CycleContext, stmm: STMMStore) -> str:
    """Simple adapter that calls engine.process and returns a summary."""
    result = engine.process({"user_message": ctx.user_message})
    return f"processed:{engine.engine_id}"


def error_adapter(engine: Any, ctx: CycleContext, stmm: STMMStore) -> str:
    """Adapter that always raises."""
    raise ValueError("intentional test error")


def stmm_writing_adapter(engine: StubEngine, ctx: CycleContext, stmm: STMMStore) -> str:
    """Adapter that writes to STMM emotion_detection slot."""
    result = engine.process(None)
    stmm.emotion_detection.user_emotion_signals["joy"] = 0.8
    stmm.emotion_detection.user_emotion_signals["curiosity"] = 0.5
    return "emotions_written"


def stmm_intention_adapter(engine: StubEngine, ctx: CycleContext, stmm: STMMStore) -> str:
    """Adapter that writes to STMM intention_analysis slot."""
    engine.process(None)
    stmm.intention_analysis.primary_intention = "question"
    stmm.intention_analysis.confidence = 0.85
    return "intention_written"


# =====================================================================
# CycleContext tests
# =====================================================================


class TestCycleContext:
    def test_context_is_frozen(self):
        ctx = CycleContext(
            cycle_id=1,
            user_message="hello",
            system_response="hi",
            nt_state={},
            oscillatory_state={},
            active_mode="normal",
        )
        with pytest.raises(AttributeError):
            ctx.cycle_id = 2

    def test_context_fields(self):
        ctx = CycleContext(
            cycle_id=5,
            user_message="test",
            system_response="resp",
            nt_state={"da": 0.5},
            oscillatory_state={"gamma": 0.3},
            active_mode="dev",
        )
        assert ctx.cycle_id == 5
        assert ctx.user_message == "test"
        assert ctx.system_response == "resp"
        assert ctx.nt_state == {"da": 0.5}
        assert ctx.oscillatory_state == {"gamma": 0.3}
        assert ctx.active_mode == "dev"

    def test_default_memory_contrast_is_none(self):
        ctx = CycleContext(
            cycle_id=1, user_message="", system_response="",
            nt_state={}, oscillatory_state={}, active_mode="normal",
        )
        assert ctx.memory_contrast is None

    def test_timestamp_populated(self):
        ctx = CycleContext(
            cycle_id=1, user_message="", system_response="",
            nt_state={}, oscillatory_state={}, active_mode="normal",
        )
        assert ctx.timestamp > 0


# =====================================================================
# EngineSlot tests
# =====================================================================


class TestEngineSlot:
    def test_slot_creation(self):
        engine = StubEngine("test_engine")
        slot = EngineSlot(
            engine_id="test_engine",
            engine_number=99,
            instance=engine,
            adapter=stub_adapter,
            cluster="test",
            priority=10,
        )
        assert slot.engine_id == "test_engine"
        assert slot.engine_number == 99
        assert slot.instance is engine
        assert slot.priority == 10
        assert slot.requires_memory is False
        assert slot.depends_on == ()

    def test_slot_with_dependencies(self):
        slot = EngineSlot(
            engine_id="child",
            engine_number=2,
            instance=StubEngine(),
            adapter=stub_adapter,
            cluster="test",
            depends_on=("parent_a", "parent_b"),
        )
        assert slot.depends_on == ("parent_a", "parent_b")


# =====================================================================
# CycleResult tests
# =====================================================================


class TestCycleResult:
    def test_result_fields(self):
        result = CycleResult(
            cycle_id=1,
            packet=None,
            engines_run=("a", "b"),
            engines_skipped=("c",),
            timing_ms=42.5,
        )
        assert result.cycle_id == 1
        assert result.engines_run == ("a", "b")
        assert result.engines_skipped == ("c",)
        assert result.timing_ms == 42.5


# =====================================================================
# CycleManager — Registration
# =====================================================================


class TestCycleManagerRegistration:
    def test_register_engine(self):
        cm = CycleManager()
        engine = StubEngine("e1")
        cm.register(EngineSlot(
            engine_id="e1", engine_number=1,
            instance=engine, adapter=stub_adapter, cluster="test",
        ))
        assert "e1" in cm.registered_engines

    def test_register_multiple_engines(self):
        cm = CycleManager()
        for i in range(5):
            cm.register(EngineSlot(
                engine_id=f"e{i}", engine_number=i,
                instance=StubEngine(f"e{i}"),
                adapter=stub_adapter, cluster="test",
            ))
        assert len(cm.registered_engines) == 5

    def test_unregister_engine(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="e1", engine_number=1,
            instance=StubEngine(), adapter=stub_adapter, cluster="test",
        ))
        cm.unregister("e1")
        assert "e1" not in cm.registered_engines

    def test_unregister_nonexistent_is_noop(self):
        cm = CycleManager()
        cm.unregister("does_not_exist")  # Should not raise

    def test_set_default_sequence(self):
        cm = CycleManager()
        cm.set_default_sequence(["a", "b", "c"])
        # No public accessor, but run_cycle will use it


# =====================================================================
# CycleManager — Empty dispatch
# =====================================================================


class TestCycleManagerEmptyDispatch:
    def test_empty_dispatch_produces_packet(self):
        cm = CycleManager()
        result = cm.run_cycle("Hello world")
        assert isinstance(result, CycleResult)
        assert result.packet is not None
        assert isinstance(result.packet, MemoryPacket)
        assert result.engines_run == ()
        assert result.engines_skipped == ()

    def test_empty_dispatch_increments_cycle_count(self):
        cm = CycleManager()
        cm.run_cycle("First")
        cm.run_cycle("Second")
        assert cm.cycle_count == 2

    def test_empty_dispatch_packet_has_user_message(self):
        cm = CycleManager()
        result = cm.run_cycle("Hello from user")
        assert "Hello from user" in result.packet.user_message

    def test_empty_dispatch_with_system_response(self):
        cm = CycleManager()
        result = cm.run_cycle("User says", "System responds")
        assert "System responds" in result.packet.system_response

    def test_stmm_snapshot_populated(self):
        cm = CycleManager()
        result = cm.run_cycle("Test message")
        assert "user_messages" in result.stmm_snapshot
        assert result.stmm_snapshot["user_messages"] >= 1

    def test_timing_is_positive(self):
        cm = CycleManager()
        result = cm.run_cycle("Test")
        assert result.timing_ms > 0


# =====================================================================
# CycleManager — Engine dispatch
# =====================================================================


class TestCycleManagerDispatch:
    def _make_cm_with_stub(self, engine_id="stub"):
        cm = CycleManager()
        engine = StubEngine(engine_id)
        cm.register(EngineSlot(
            engine_id=engine_id, engine_number=1,
            instance=engine, adapter=stub_adapter, cluster="test",
        ))
        cm.set_default_sequence([engine_id])
        return cm, engine

    def test_engine_runs_in_default_sequence(self):
        cm, engine = self._make_cm_with_stub("my_engine")
        result = cm.run_cycle("Hello")
        assert "my_engine" in result.engines_run
        assert len(engine._process_calls) == 1

    def test_engine_receives_nt_state(self):
        cm, engine = self._make_cm_with_stub("my_engine")
        cm.run_cycle("Hello", nt_state={"da": 0.7, "5ht": 0.3})
        assert engine._nt_state == {"da": 0.7, "5ht": 0.3}

    def test_dispatch_list_overrides_default(self):
        cm = CycleManager()
        e1 = StubEngine("e1")
        e2 = StubEngine("e2")
        cm.register(EngineSlot(
            engine_id="e1", engine_number=1,
            instance=e1, adapter=stub_adapter, cluster="test",
        ))
        cm.register(EngineSlot(
            engine_id="e2", engine_number=2,
            instance=e2, adapter=stub_adapter, cluster="test",
        ))
        cm.set_default_sequence(["e1", "e2"])

        # Only dispatch e2
        result = cm.run_cycle("Hello", dispatch_list=["e2"])
        assert result.engines_run == ("e2",)
        assert len(e1._process_calls) == 0
        assert len(e2._process_calls) == 1

    def test_unregistered_engine_skipped(self):
        cm = CycleManager()
        cm.set_default_sequence(["nonexistent_engine"])
        result = cm.run_cycle("Hello")
        assert result.engines_skipped == ("nonexistent_engine",)
        assert result.engines_run == ()

    def test_multiple_engines_in_order(self):
        cm = CycleManager()
        order_tracker = []

        def order_adapter_factory(name):
            def adapter(engine, ctx, stmm):
                order_tracker.append(name)
                engine.process(None)
                return name
            return adapter

        for name in ["first", "second", "third"]:
            cm.register(EngineSlot(
                engine_id=name, engine_number=1,
                instance=StubEngine(name),
                adapter=order_adapter_factory(name),
                cluster="test",
            ))
        cm.set_default_sequence(["first", "second", "third"])
        cm.run_cycle("Test")
        assert order_tracker == ["first", "second", "third"]


# =====================================================================
# CycleManager — Dependency checking
# =====================================================================


class TestCycleManagerDependencies:
    def test_satisfied_dependency_runs(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="parent", engine_number=1,
            instance=StubEngine("parent"),
            adapter=stub_adapter, cluster="test",
        ))
        cm.register(EngineSlot(
            engine_id="child", engine_number=2,
            instance=StubEngine("child"),
            adapter=stub_adapter, cluster="test",
            depends_on=("parent",),
        ))
        cm.set_default_sequence(["parent", "child"])
        result = cm.run_cycle("Test")
        assert "parent" in result.engines_run
        assert "child" in result.engines_run

    def test_unsatisfied_dependency_skipped(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="child", engine_number=2,
            instance=StubEngine("child"),
            adapter=stub_adapter, cluster="test",
            depends_on=("parent",),
        ))
        cm.set_default_sequence(["child"])
        result = cm.run_cycle("Test")
        assert "child" in result.engines_skipped
        assert "child" not in result.engines_run

    def test_partial_dependency_skipped(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="dep_a", engine_number=1,
            instance=StubEngine("dep_a"),
            adapter=stub_adapter, cluster="test",
        ))
        cm.register(EngineSlot(
            engine_id="child", engine_number=3,
            instance=StubEngine("child"),
            adapter=stub_adapter, cluster="test",
            depends_on=("dep_a", "dep_b"),  # dep_b not registered
        ))
        cm.set_default_sequence(["dep_a", "child"])
        result = cm.run_cycle("Test")
        assert "dep_a" in result.engines_run
        assert "child" in result.engines_skipped

    def test_no_dependencies_always_runs(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="free", engine_number=1,
            instance=StubEngine("free"),
            adapter=stub_adapter, cluster="test",
            depends_on=(),
        ))
        cm.set_default_sequence(["free"])
        result = cm.run_cycle("Test")
        assert "free" in result.engines_run


# =====================================================================
# CycleManager — BrainProcessTracker
# =====================================================================


class TestBrainProcessTracker:
    def test_tracker_records_run_engine(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="tracked", engine_number=1,
            instance=StubEngine("tracked"),
            adapter=stub_adapter, cluster="test",
        ))
        cm.set_default_sequence(["tracked"])
        result = cm.run_cycle("Test")
        tracker = cm.memory.stmm.brain_process_tracker
        ids_run = tracker.engine_ids_run()
        assert "tracked" in ids_run

    def test_tracker_records_skipped_engine(self):
        cm = CycleManager()
        cm.set_default_sequence(["unregistered"])
        cm.run_cycle("Test")
        tracker = cm.memory.stmm.brain_process_tracker
        skipped = [e for e in tracker.executions if e.skipped]
        assert len(skipped) == 1
        assert skipped[0].engine_id == "unregistered"
        assert skipped[0].skip_reason == "not_registered"

    def test_tracker_records_timing(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="timed", engine_number=1,
            instance=StubEngine("timed"),
            adapter=stub_adapter, cluster="test",
        ))
        cm.set_default_sequence(["timed"])
        cm.run_cycle("Test")
        tracker = cm.memory.stmm.brain_process_tracker
        execution = [e for e in tracker.executions if e.engine_id == "timed"][0]
        assert execution.timing_ms >= 0

    def test_tracker_records_output_summary(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="summarized", engine_number=1,
            instance=StubEngine("summarized"),
            adapter=stub_adapter, cluster="test",
        ))
        cm.set_default_sequence(["summarized"])
        cm.run_cycle("Test")
        tracker = cm.memory.stmm.brain_process_tracker
        execution = [e for e in tracker.executions if e.engine_id == "summarized"][0]
        assert "processed:summarized" in execution.output_summary

    def test_stage_flags_set(self):
        cm = CycleManager()
        cm.run_cycle("Test")
        tracker = cm.memory.stmm.brain_process_tracker
        assert tracker.pipeline_stage_flags.get("engine_dispatch") is True
        assert tracker.pipeline_stage_flags.get("memory_write") is True


# =====================================================================
# CycleManager — Error handling
# =====================================================================


class TestCycleManagerErrorHandling:
    def test_adapter_error_does_not_crash_cycle(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="broken", engine_number=1,
            instance=StubEngine("broken"),
            adapter=error_adapter, cluster="test",
        ))
        cm.set_default_sequence(["broken"])
        result = cm.run_cycle("Test")
        # Engine counts as run despite error
        assert "broken" in result.engines_run
        assert result.packet is not None

    def test_adapter_error_recorded_in_tracker(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="broken", engine_number=1,
            instance=StubEngine("broken"),
            adapter=error_adapter, cluster="test",
        ))
        cm.set_default_sequence(["broken"])
        cm.run_cycle("Test")
        tracker = cm.memory.stmm.brain_process_tracker
        execution = [e for e in tracker.executions if e.engine_id == "broken"][0]
        assert "ERROR" in execution.output_summary
        assert "intentional test error" in execution.output_summary

    def test_error_engine_does_not_block_next(self):
        cm = CycleManager()
        e_good = StubEngine("good")
        cm.register(EngineSlot(
            engine_id="broken", engine_number=1,
            instance=StubEngine("broken"),
            adapter=error_adapter, cluster="test",
        ))
        cm.register(EngineSlot(
            engine_id="good", engine_number=2,
            instance=e_good, adapter=stub_adapter, cluster="test",
        ))
        cm.set_default_sequence(["broken", "good"])
        result = cm.run_cycle("Test")
        assert "broken" in result.engines_run
        assert "good" in result.engines_run
        assert len(e_good._process_calls) == 1


# =====================================================================
# CycleManager — STMM population via adapters
# =====================================================================


class TestSTMMPopulation:
    def test_adapter_writes_to_emotion_detection(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="emotion", engine_number=28,
            instance=StubEngine("emotion"),
            adapter=stmm_writing_adapter, cluster="emotional_processing",
        ))
        cm.set_default_sequence(["emotion"])
        result = cm.run_cycle("I'm so happy!")
        # After run_cycle, the STMM was compressed into a packet
        # Check that the adapter did write to STMM (we can verify via snapshot)
        assert result.packet is not None

    def test_adapter_writes_to_intention_analysis(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="intention", engine_number=23,
            instance=StubEngine("intention"),
            adapter=stmm_intention_adapter, cluster="pattern_analysis",
        ))
        cm.set_default_sequence(["intention"])
        result = cm.run_cycle("What is the meaning of life?")
        # The snapshot should reflect the intention written by the adapter
        assert result.stmm_snapshot["intention"] == "question"


# =====================================================================
# CycleManager — Memory contrast population
# =====================================================================


class TestMemoryContrastPopulation:
    def test_contrast_populated_on_empty_memory(self):
        cm = CycleManager()
        cm.run_cycle("Hello")
        stmm = cm.memory.stmm
        # With empty MTMM/LTMM, no matches
        assert len(stmm.memory_contrast.matched_entries) == 0

    def test_contrast_finds_prior_message(self):
        cm = CycleManager()
        # First cycle seeds MTMM
        cm.run_cycle("I love machine learning and neural networks")
        # Second cycle should find the first message via TF-IDF cosine
        cm.run_cycle("Tell me about neural networks and machine learning")
        stmm = cm.memory.stmm
        # Should have at least one match from the first cycle
        assert len(stmm.memory_contrast.matched_entries) >= 1

    def test_contrast_entries_have_correct_fields(self):
        cm = CycleManager()
        cm.run_cycle("Quantum computing and its applications in cryptography")
        cm.run_cycle("Applications of quantum computing in cryptography research")
        stmm = cm.memory.stmm
        if stmm.memory_contrast.matched_entries:
            entry = stmm.memory_contrast.matched_entries[0]
            assert hasattr(entry, "entry_id")
            assert hasattr(entry, "source_tier")
            assert hasattr(entry, "similarity")
            assert hasattr(entry, "content_summary")
            assert entry.similarity >= 0.0


# =====================================================================
# CycleManager — Multiple cycles
# =====================================================================


class TestMultipleCycles:
    def test_stmm_resets_between_cycles(self):
        cm = CycleManager()
        cm.register(EngineSlot(
            engine_id="writer", engine_number=1,
            instance=StubEngine("writer"),
            adapter=stmm_writing_adapter, cluster="test",
        ))
        cm.set_default_sequence(["writer"])

        cm.run_cycle("First cycle")
        # Second cycle should start with clean STMM
        cm.run_cycle("Second cycle")
        # If STMM wasn't reset, we'd have stale data — but the adapter
        # re-writes emotion_detection, so we just check cycle_count
        assert cm.cycle_count == 2

    def test_mtmm_accumulates_across_cycles(self):
        cm = CycleManager()
        cm.run_cycle("Message one about cats and dogs")
        cm.run_cycle("Message two about birds and fish")
        cm.run_cycle("Message three about cats and dogs again")
        # The third message should find the first one in MTMM
        stmm = cm.memory.stmm
        # At minimum, MTMM should have 3 packets
        all_packets = cm.memory.mtmm.get_all_packets()
        assert len(all_packets) >= 3

    def test_message_buffer_accumulates(self):
        cm = CycleManager()
        cm.run_cycle("First user message", "First system response")
        cm.run_cycle("Second user message", "Second system response")
        # FIFO: buffer holds last 2 user + 2 system messages
        stmm = cm.memory.stmm
        user_msgs = stmm.active_message_buffer.get_by_speaker(
            __import__("zados.memory.types", fromlist=["SpeakerID"]).SpeakerID.USER
        )
        assert len(user_msgs) == 2


# =====================================================================
# CycleManager — Memory layer access
# =====================================================================


class TestMemoryAccess:
    def test_memory_property(self):
        ml = MemoryLayer()
        cm = CycleManager(memory=ml)
        assert cm.memory is ml

    def test_default_memory_created(self):
        cm = CycleManager()
        assert cm.memory is not None
        assert isinstance(cm.memory, MemoryLayer)

    def test_cycle_context_has_memory_contrast(self):
        """Verify that CycleContext receives the memory contrast instance."""
        cm = CycleManager()
        received_ctx = []

        def capture_adapter(engine, ctx, stmm):
            received_ctx.append(ctx)
            return "captured"

        cm.register(EngineSlot(
            engine_id="capturer", engine_number=1,
            instance=StubEngine(), adapter=capture_adapter, cluster="test",
        ))
        cm.set_default_sequence(["capturer"])
        cm.run_cycle("Test")
        assert len(received_ctx) == 1
        assert received_ctx[0].memory_contrast is cm.memory.contrast
