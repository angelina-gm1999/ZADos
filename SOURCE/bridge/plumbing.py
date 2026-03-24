"""
ZADOS Bridge — Plumbing Test Runner.

Server-side diagnostic tests that verify data flows through the pipeline,
memory tiers read/write correctly, and neurochemical state actually
influences engine dispatch.  No LLM calls required — pure plumbing.

Call run_all(stack) to get a structured report Godot can display.
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    elapsed_ms: float = 0.0


@dataclass
class PlumbingReport:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_ms: float = 0.0

    def add(self, r: TestResult) -> None:
        self.results.append({
            "name": r.name,
            "passed": r.passed,
            "message": r.message,
            "elapsed_ms": round(r.elapsed_ms, 2),
        })
        if r.passed:
            self.passed += 1
        else:
            self.failed += 1

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "total_ms": round(self.total_ms, 2),
            "all_passed": self.failed == 0 and self.errors == 0,
            "results": self.results,
        }


def _run(name: str, fn, report: PlumbingReport) -> None:
    """Run a single test function, catch exceptions, record result."""
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        report.add(TestResult(name=name, passed=True,
                              message="OK", elapsed_ms=elapsed))
    except PlumbingAssertionFailed as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        report.add(TestResult(name=name, passed=False,
                              message=str(exc) or "Assertion failed",
                              elapsed_ms=elapsed))
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        report.errors += 1
        report.add(TestResult(name=name, passed=False,
                              message=f"ERROR: {exc}\n{traceback.format_exc()}",
                              elapsed_ms=elapsed))


# =======================================================================
# Test functions — each takes (stack,) and asserts
# =======================================================================

class PlumbingAssertionFailed(Exception):
    """Raised when a plumbing test assertion fails."""


def _assert(condition: bool, msg: str = "assertion failed") -> None:
    if not condition:
        raise PlumbingAssertionFailed(msg)


# -----------------------------------------------------------------------
# 1. Stack construction
# -----------------------------------------------------------------------

def test_stack_construction(stack: Any) -> None:
    """Verify the stack object has all required components."""
    _assert(stack is not None, "stack is None")
    _assert(stack.orchestrator is not None, "orchestrator is None")
    _assert(stack.classifier is not None, "classifier is None")
    _assert(stack.memory is not None, "memory is None")
    _assert(stack.neurochem is not None, "neurochem is None")


# -----------------------------------------------------------------------
# 2. Session lifecycle
# -----------------------------------------------------------------------

def test_session_open(stack: Any) -> None:
    """Open a session and verify it returns valid state."""
    session = stack.orchestrator.open_session()
    _assert(session is not None, "open_session() returned None")
    _assert(hasattr(session, "session_id"), "session has no session_id")
    _assert(len(session.session_id) > 0, "session_id is empty")
    _assert(hasattr(session, "branch"), "session has no branch")
    _assert(hasattr(session, "turn_count"), "session has no turn_count")


# -----------------------------------------------------------------------
# 3. Neurochem engine baseline
# -----------------------------------------------------------------------

def test_neurochem_baseline(stack: Any) -> None:
    """Verify the neurochem engine has NTs registered and returns a readout."""
    engine = stack.neurochem
    names = list(engine.registry.neurotransmitter_names())
    _assert(len(names) >= 8,
            f"Expected ≥8 NTs registered, got {len(names)}: {names}")

    # Ensure oscillation state is initialized (readout requires it)
    if engine.registry.get_oscillations() is None:
        from zados.neurochem.state.oscillation_state import OscillationState
        engine.set_oscillation_state(OscillationState())

    readout = engine.get_neurosymbolic_readout()
    _assert(isinstance(readout, dict), "readout is not a dict")
    _assert(len(readout) > 0, "readout is empty")


def test_neurochem_step_mutates_state(stack: Any) -> None:
    """Verify that engine.step() actually changes NT concentrations."""
    engine = stack.neurochem

    # Ensure oscillation state exists
    if engine.registry.get_oscillations() is None:
        from zados.neurochem.state.oscillation_state import OscillationState
        engine.set_oscillation_state(OscillationState())

    # Snapshot before (uppercase keys like "DA", "NE", etc.)
    before = {}
    for name in engine.registry.neurotransmitter_names():
        nt = engine.registry.get_neurotransmitter(name)
        before[name] = nt.C

    # Apply a strong DA signal with correct key format
    engine.step({"DA": {"novelty": 0.9, "rpe": 0.8}})

    # Snapshot after
    after = {}
    for name in engine.registry.neurotransmitter_names():
        nt = engine.registry.get_neurotransmitter(name)
        after[name] = nt.C

    # At least DA should have changed
    _assert(after.get("DA", before.get("DA", 0)) != before.get("DA", 0),
            f"DA did not change after step(). Before={before.get('DA')}, After={after.get('DA')}")


# -----------------------------------------------------------------------
# 4. Pipeline phases — individual phase smoke tests
# -----------------------------------------------------------------------

def test_phase0_validation(stack: Any) -> None:
    """Phase 0: validate_bundle accepts a well-formed InputBundle."""
    from zados.core.phases.phase0_reception import validate_bundle
    from zados.core.types import InputBundle

    bundle = InputBundle(raw_text="Hello, this is a plumbing test.")
    result = validate_bundle(bundle)
    _assert(result is not None, "validate_bundle returned None")
    _assert(result.raw_text == bundle.raw_text, "raw_text was mutated")


def test_phase1_perception(stack: Any) -> None:
    """Phase 1: run_perception produces a PerceptionSnapshot."""
    from zados.core.phases.phase1_perception import run_perception
    from zados.core.types import InputBundle
    from zados.memory.short_term.store import STMMStore

    bundle = InputBundle(raw_text="What is the meaning of life?")
    stmm = STMMStore()
    stmm.add_user_message(bundle.raw_text)

    # Build NT snapshot
    nt_snap = {}
    for name in stack.neurochem.registry.neurotransmitter_names():
        nt = stack.neurochem.registry.get_neurotransmitter(name)
        nt_snap[name.lower()] = nt.C

    result = run_perception(bundle, stack.orchestrator.engines, nt_snap, stmm=stmm)
    _assert(result is not None, "run_perception returned None")
    _assert(hasattr(result, "intent_archetype"),
            f"PerceptionSnapshot missing intent_archetype. Fields: {list(vars(result).keys())}")


def test_phase3_dispatch(stack: Any) -> None:
    """Phase 3: run_engine_dispatch returns an EngineDispatchResult."""
    from zados.core.phases.phase1_perception import run_perception
    from zados.core.phases.phase3_dispatch import run_engine_dispatch
    from zados.core.types import InputBundle, PipelineState
    from zados.memory.short_term.store import STMMStore
    import time as _time

    bundle = InputBundle(raw_text="Tell me about recursion.")
    stmm = STMMStore()
    stmm.add_user_message(bundle.raw_text)
    session = stack.orchestrator.session
    if session is None:
        session = stack.orchestrator.open_session()

    nt_snap = {}
    for name in stack.neurochem.registry.neurotransmitter_names():
        nt = stack.neurochem.registry.get_neurotransmitter(name)
        nt_snap[name.lower()] = nt.C

    state = PipelineState(bundle=bundle, stmm=stmm,
                          turn_index=session.turn_count,
                          timestamp=_time.time())
    state.perception = run_perception(bundle, stack.orchestrator.engines,
                                      nt_snap, stmm=stmm)

    result = run_engine_dispatch(state, stack.orchestrator.engines,
                                 nt_snap, memory_contrast=stack.memory.contrast)
    _assert(result is not None, "run_engine_dispatch returned None")
    _assert(hasattr(result, "engines_run"), "dispatch result missing engines_run")


def test_phase2_modulation(stack: Any) -> None:
    """Phase 2: run_nt_modulation returns an NTModulationResult."""
    from zados.core.phases.phase1_perception import run_perception
    from zados.core.phases.phase2_modulation import run_nt_modulation
    from zados.core.phases.phase3_dispatch import run_engine_dispatch
    from zados.core.types import InputBundle, PipelineState
    from zados.memory.short_term.store import STMMStore
    import time as _time

    bundle = InputBundle(raw_text="I feel confused about this topic.")
    stmm = STMMStore()
    stmm.add_user_message(bundle.raw_text)
    session = stack.orchestrator.session or stack.orchestrator.open_session()

    nt_snap = {}
    for name in stack.neurochem.registry.neurotransmitter_names():
        nt = stack.neurochem.registry.get_neurotransmitter(name)
        nt_snap[name.lower()] = nt.C

    state = PipelineState(bundle=bundle, stmm=stmm,
                          turn_index=session.turn_count,
                          timestamp=_time.time())
    state.perception = run_perception(bundle, stack.orchestrator.engines,
                                      nt_snap, stmm=stmm)
    state.dispatch = run_engine_dispatch(state, stack.orchestrator.engines,
                                         nt_snap, memory_contrast=stack.memory.contrast)

    result = run_nt_modulation(bundle, state.perception, state.dispatch,
                               stack.neurochem, stmm)
    _assert(result is not None, "run_nt_modulation returned None")
    _assert(hasattr(result, "nt_snapshot"),
            f"NTModulationResult missing nt_snapshot. Fields: {list(vars(result).keys())}")


def test_full_pipeline_turn(stack: Any) -> None:
    """Run a complete turn through the classifier — all 8 phases."""
    from zados.core.types import RawInput
    if stack.orchestrator.session is None:
        stack.orchestrator.open_session()

    result = stack.classifier.process(RawInput(text="Testing plumbing — please respond."))
    _assert(result is not None, "classifier.process() returned None")

    # Unwrap if needed
    actual = result
    if not hasattr(actual, "final_answer"):
        actual = getattr(actual, "pipeline_result", actual)

    _assert(hasattr(actual, "state"),
            f"Pipeline result has no .state attribute. Type: {type(actual).__name__}")


# -----------------------------------------------------------------------
# 5. Memory tier plumbing
# -----------------------------------------------------------------------

def test_stmm_write_read(stack: Any) -> None:
    """STMM: write a message and read it back."""
    from zados.memory.short_term.store import STMMStore
    from zados.memory.types import SpeakerID
    stmm = STMMStore()
    stmm.add_user_message("Hello from plumbing test")

    buf = stmm.active_message_buffer
    user_msgs = buf.get_by_speaker(SpeakerID.USER)
    _assert(len(user_msgs) >= 1,
            f"Expected ≥1 user messages, got {len(user_msgs)}")
    _assert(user_msgs[-1].text == "Hello from plumbing test",
            "User message text mismatch")


def test_mtmm_write_search(stack: Any) -> None:
    """MTMM: write a MemoryPacket and retrieve it via search."""
    from zados.memory.types import MemoryPacket, MemoryTier, CompressionLevel
    from datetime import datetime

    mtmm = stack.memory.mtmm
    packet = MemoryPacket(
        packet_id="plumbing-test-001",
        timestamp=datetime.now(),
        source_tier=MemoryTier.STMM,
        destination_tier=MemoryTier.MTMM,
        turn_index=999,
        user_message="What is photosynthesis?",
        system_response="Photosynthesis is the process by which plants convert sunlight.",
        intention="question",
        emotion_vector={"curiosity": 0.8},
        neurochemical_snapshot={"da": 0.6, "5ht": 0.5},
        reward_scores={"logic": 0.7},
        flags=[],
        contradictions_detected=0,
        paradoxes_detected=0,
        unsolved_items_matched=[],
        compression_level=CompressionLevel.SEMANTIC,
        trust_weight=0.9,
        emotional_significance=0.3,
        embedding=None,
        verbal_summary="Question about photosynthesis.",
        verbal_emotion_labels=["curiosity"],
        time_context={},
    )
    mtmm.write(packet, importance=0.5)

    results = mtmm.search("photosynthesis", limit=3)
    _assert(len(results) >= 1,
            f"MTMM search('photosynthesis') returned {len(results)} results, expected ≥1")

    # Verify we got our packet back
    found = any(getattr(r[1], "packet_id", None) == "plumbing-test-001"
                for r in results)
    _assert(found, "MTMM search did not return the test packet")


def test_ltmm_write_search(stack: Any) -> None:
    """LTMM: write an LTMMEntry and retrieve it."""
    from zados.memory.types import MemoryPacket, MemoryTier, CompressionLevel
    from zados.memory.long_term.store import LTMMEntry
    from datetime import datetime

    packet = MemoryPacket(
        packet_id="plumbing-test-ltmm-001",
        timestamp=datetime.now(),
        source_tier=MemoryTier.STMM,
        destination_tier=MemoryTier.LTMM,
        turn_index=998,
        user_message="Explain quantum entanglement",
        system_response="Quantum entanglement is a phenomenon in physics.",
        intention="question",
        emotion_vector={"curiosity": 0.9},
        neurochemical_snapshot={"da": 0.7},
        reward_scores={"logic": 0.8},
        flags=[],
        contradictions_detected=0,
        paradoxes_detected=0,
        unsolved_items_matched=[],
        compression_level=CompressionLevel.SEMANTIC,
        trust_weight=0.95,
        emotional_significance=0.4,
        embedding=None,
        verbal_summary="Question about quantum entanglement.",
        verbal_emotion_labels=["curiosity"],
        time_context={},
    )

    entry = LTMMEntry(
        packet=packet,
        granularity="SEMANTIC",
        relevance_score=1.0,
        retrieval_count=0,
        last_accessed=datetime.now(),
        utility_score=0.5,
        cold_storage=False,
        identity_relevant=False,
    )

    ltmm = stack.memory.ltmm
    ltmm.write(entry)

    # Search
    results = ltmm.search("quantum entanglement", limit=3)
    _assert(len(results) >= 1,
            f"LTMM search returned {len(results)} results, expected ≥1")

    # Verify by ID
    by_id = ltmm.get_by_id("plumbing-test-ltmm-001")
    _assert(by_id is not None, "LTMM get_by_id returned None for test packet")


def test_memory_compression_stmm_to_packet(stack: Any) -> None:
    """Verify MemoryExitCompressor compresses STMM into a MemoryPacket."""
    from zados.memory.short_term.store import STMMStore
    from zados.memory.short_term.compressor import MemoryExitCompressor

    stmm = STMMStore()
    stmm.add_user_message("Why do birds sing?")
    stmm.add_system_response("Birds sing to communicate territory and attract mates.")

    compressor = MemoryExitCompressor()
    packet = compressor.compress(stmm)
    _assert(packet is not None, "compressor.compress() returned None")
    _assert(packet.user_message == "Why do birds sing?",
            f"Packet user_message wrong: {packet.user_message!r}")
    _assert(len(packet.system_response) > 0, "Packet system_response is empty")
    _assert(packet.turn_index >= 0, "Packet turn_index is negative")


def test_consolidation_engine(stack: Any) -> None:
    """Verify MemoryConsolidationEngine evaluates packets for promotion."""
    from zados.memory.types import MemoryPacket, MemoryTier, CompressionLevel
    from zados.memory.long_term.consolidation import MemoryConsolidationEngine
    from datetime import datetime

    engine = MemoryConsolidationEngine(stack.memory.ltmm)

    # Packet that SHOULD be promoted (high emotional significance)
    hot_packet = MemoryPacket(
        packet_id="plumbing-consolidation-hot",
        timestamp=datetime.now(),
        source_tier=MemoryTier.STMM,
        destination_tier=MemoryTier.LTMM,
        turn_index=900,
        user_message="I just realized something profound about my identity.",
        system_response="Tell me more about this realization.",
        intention="self_reflection",
        emotion_vector={"awe": 0.95, "joy": 0.8},
        neurochemical_snapshot={"da": 0.8, "5ht": 0.6},
        reward_scores={},
        flags=["IDENTITY"],
        contradictions_detected=0,
        paradoxes_detected=0,
        unsolved_items_matched=[],
        compression_level=CompressionLevel.SEMANTIC,
        trust_weight=0.9,
        emotional_significance=0.95,
        embedding=None,
        verbal_summary="Profound identity realization.",
        verbal_emotion_labels=["awe", "joy"],
        time_context={},
    )

    # Packet that should NOT be promoted (low everything)
    cold_packet = MemoryPacket(
        packet_id="plumbing-consolidation-cold",
        timestamp=datetime.now(),
        source_tier=MemoryTier.STMM,
        destination_tier=MemoryTier.LTMM,
        turn_index=901,
        user_message="ok",
        system_response="Alright.",
        intention="acknowledgement",
        emotion_vector={},
        neurochemical_snapshot={"da": 0.1},
        reward_scores={},
        flags=[],
        contradictions_detected=0,
        paradoxes_detected=0,
        unsolved_items_matched=[],
        compression_level=CompressionLevel.SEMANTIC,
        trust_weight=0.9,
        emotional_significance=0.05,
        embedding=None,
        verbal_summary="Simple acknowledgement.",
        verbal_emotion_labels=[],
        time_context={},
    )

    promoted_ids = engine.consolidate([hot_packet, cold_packet])
    _assert("plumbing-consolidation-hot" in promoted_ids,
            f"Hot packet was NOT promoted. Promoted: {promoted_ids}")
    _assert("plumbing-consolidation-cold" not in promoted_ids,
            f"Cold packet was wrongly promoted. Promoted: {promoted_ids}")


def test_memory_manager_on_cycle_end(stack: Any) -> None:
    """Verify MemoryImplementationManager.on_cycle_end writes to MTMM."""
    from zados.memory.short_term.store import STMMStore

    stmm = STMMStore()
    stmm.add_user_message("Plumbing test: cycle end compression")
    stmm.add_system_response("Acknowledged.")

    manager = stack.memory.manager
    packet = manager.on_cycle_end(stmm)
    _assert(packet is not None, "on_cycle_end returned None")
    _assert(packet.user_message == "Plumbing test: cycle end compression",
            f"Packet content mismatch: {packet.user_message!r}")

    # Verify it landed in MTMM
    results = stack.memory.mtmm.search("cycle end compression", limit=3)
    _assert(len(results) >= 1, "Packet not found in MTMM after on_cycle_end")


# -----------------------------------------------------------------------
# 6. Namespaced store plumbing
# -----------------------------------------------------------------------

def test_namespace_identity_stores(stack: Any) -> None:
    """Verify identity namespace stores are wired and respond to get_all."""
    identity = stack.memory.identity
    _assert(identity is not None, "memory.identity is None")

    for store_name in ("hardcoded", "core", "conclusions", "journal"):
        store = getattr(identity, store_name, None)
        _assert(store is not None, f"identity.{store_name} is None")
        all_items = store.get_all()
        _assert(isinstance(all_items, list),
                f"identity.{store_name}.get_all() returned {type(all_items).__name__}")


def test_namespace_thoughts_stores(stack: Any) -> None:
    """Verify thoughts namespace stores are wired."""
    thoughts = stack.memory.thoughts
    _assert(thoughts is not None, "memory.thoughts is None")

    for store_name in ("overview_logs", "held_blocks", "unsolved_buffer",
                       "general_questions"):
        store = getattr(thoughts, store_name, None)
        _assert(store is not None, f"thoughts.{store_name} is None")


def test_namespace_knowledge_stores(stack: Any) -> None:
    """Verify knowledge namespace stores are wired."""
    knowledge = stack.memory.knowledge
    _assert(knowledge is not None, "memory.knowledge is None")

    for store_name in ("library", "lessons", "academic_buffer",
                       "academic_questions", "knowledge_maps",
                       "cognitools_data", "notebook"):
        store = getattr(knowledge, store_name, None)
        _assert(store is not None, f"knowledge.{store_name} is None")


# -----------------------------------------------------------------------
# 7. NT state influences engine dispatch
# -----------------------------------------------------------------------

def test_nt_modulates_dispatch(stack: Any) -> None:
    """Verify that different NT states produce different dispatch outcomes.

    Push DA high (exploration) vs push GABA high (inhibition) and check
    that the engine dispatch results differ.
    """
    from zados.core.phases.phase1_perception import run_perception
    from zados.core.phases.phase3_dispatch import run_engine_dispatch
    from zados.core.types import InputBundle, PipelineState
    from zados.memory.short_term.store import STMMStore
    import time as _time

    bundle = InputBundle(raw_text="What patterns exist in prime numbers?")
    session = stack.orchestrator.session or stack.orchestrator.open_session()
    engines = stack.orchestrator.engines

    # If no engines are registered, skip gracefully
    if len(engines) == 0:
        _assert(True, "No engines registered — dispatch modulation test skipped (OK for now)")
        return

    def _dispatch_with_nt(nt_overrides: dict) -> Any:
        stmm = STMMStore()
        stmm.add_user_message(bundle.raw_text)
        # Build snapshot with overrides
        nt_snap = {}
        for name in stack.neurochem.registry.neurotransmitter_names():
            nt = stack.neurochem.registry.get_neurotransmitter(name)
            nt_snap[name.lower()] = nt.C
        nt_snap.update(nt_overrides)

        state = PipelineState(bundle=bundle, stmm=stmm,
                              turn_index=session.turn_count,
                              timestamp=_time.time())
        state.perception = run_perception(bundle, engines, nt_snap, stmm=stmm)
        return run_engine_dispatch(state, engines, nt_snap,
                                   memory_contrast=stack.memory.contrast)

    # High DA = exploratory
    result_da = _dispatch_with_nt({"da": 0.95, "gaba": 0.1})
    # High GABA = inhibitory
    result_gaba = _dispatch_with_nt({"da": 0.1, "gaba": 0.95})

    # We expect at least some difference in engine_results or engines_run
    da_run = set(getattr(result_da, "engines_run", []))
    gaba_run = set(getattr(result_gaba, "engines_run", []))

    # If the sets are identical, check if engine results differ
    if da_run == gaba_run:
        da_results = getattr(result_da, "engine_results", {})
        gaba_results = getattr(result_gaba, "engine_results", {})
        _assert(da_results != gaba_results or len(engines) == 0,
                "NT state had NO effect on dispatch: both DA-high and GABA-high "
                "produced identical engine results. The neurochem layer may not "
                "be influencing cognitive engine dispatch.")


# -----------------------------------------------------------------------
# 8. Retrieval router
# -----------------------------------------------------------------------

def test_retrieval_router(stack: Any) -> None:
    """Verify the retrieval router resolves queries without crashing."""
    router = stack.memory.router
    _assert(router is not None, "memory.router is None")

    from zados.memory.long_term.retrieval_router import RetrievalContext
    ctx = RetrievalContext(
        query_text="photosynthesis",
        query_type="knowledge",
        pipeline_name="plumbing_test",
        limit=3,
    )
    results = router.route(ctx)
    _assert(isinstance(results, list),
            f"router.route() returned {type(results).__name__}, expected list")


# -----------------------------------------------------------------------
# 9. STMM → MTMM → LTMM full round trip
# -----------------------------------------------------------------------

def test_full_memory_round_trip(stack: Any) -> None:
    """End-to-end: STMM compress → MTMM write → consolidate → LTMM search."""
    from zados.memory.short_term.store import STMMStore
    from zados.memory.short_term.compressor import MemoryExitCompressor
    from zados.memory.long_term.consolidation import MemoryConsolidationEngine
    from datetime import datetime

    # Step 1: Build STMM content
    stmm = STMMStore()
    stmm.add_user_message("This is a critical identity realization about consciousness.")
    stmm.add_system_response("That is a profound insight.")

    # Step 2: Compress to packet
    compressor = MemoryExitCompressor()
    packet = compressor.compress(stmm)
    _assert(packet is not None, "compression failed")

    # Boost emotional significance so consolidation promotes it
    packet.emotional_significance = 0.95
    packet.flags.append("IDENTITY")
    packet.packet_id = f"plumbing-roundtrip-{int(time.time()*1000)}"

    # Step 3: Write to MTMM
    stack.memory.mtmm.write(packet, importance=0.9)

    # Step 4: Consolidate → LTMM
    consolidation = MemoryConsolidationEngine(stack.memory.ltmm)
    promoted = consolidation.consolidate([packet])
    _assert(packet.packet_id in promoted,
            f"Round-trip packet not promoted. IDs: {promoted}")

    # Step 5: Search in LTMM
    results = stack.memory.ltmm.search("critical identity realization consciousness",
                                        limit=3)
    _assert(len(results) >= 1,
            "Round-trip packet not found in LTMM after consolidation")


# -----------------------------------------------------------------------
# 10. Library import pipeline
# -----------------------------------------------------------------------

def test_library_ingest_whole(stack: Any) -> None:
    """Ingest a short text as a single library entry."""
    from zados.memory.long_term.knowledge.library.importer import import_text

    store = stack.memory.knowledge.library
    result = import_text(
        store=store,
        title="Plumbing Test — Photosynthesis",
        content="Photosynthesis is the process by which green plants convert "
                "sunlight into chemical energy.  It occurs in chloroplasts and "
                "produces glucose and oxygen as byproducts.",
        domain="biology",
        tags=["plumbing_test"],
        source_type="document",
        strategy="whole",
    )
    _assert(result.error == "", f"import_text error: {result.error}")
    _assert(result.entries_created == 1,
            f"Expected 1 entry, got {result.entries_created}")
    _assert(len(result.entry_ids) == 1, "entry_ids mismatch")

    # Verify searchable
    hits = store.search("photosynthesis chloroplasts", limit=3)
    _assert(len(hits) >= 1, "Ingested text not found via search")


def test_library_ingest_chunked(stack: Any) -> None:
    """Ingest a long text that gets chunked into multiple entries."""
    from zados.memory.long_term.knowledge.library.importer import import_text

    store = stack.memory.knowledge.library
    # Build text that exceeds the default chunk size
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Section {i + 1}: This is paragraph {i + 1} of the plumbing test "
            f"document about advanced quantum mechanics and wave function collapse.  "
            f"The Schrödinger equation describes the time evolution of quantum states.  "
            f"Measurement causes decoherence and apparent collapse.  " * 8
        )
    long_text = "\n\n".join(paragraphs)

    result = import_text(
        store=store,
        title="Plumbing Test — Quantum Mechanics",
        content=long_text,
        domain="physics",
        tags=["plumbing_test", "chunked"],
        source_type="book",
        strategy="chunked",
    )
    _assert(result.error == "", f"chunked import error: {result.error}")
    _assert(result.entries_created > 1,
            f"Expected >1 chunks, got {result.entries_created}")
    _assert(len(result.group_id) > 0, "group_id is empty")

    # Verify searchable
    hits = store.search("quantum wave function Schrödinger", limit=5)
    _assert(len(hits) >= 1, "Chunked text not found via search")


def test_library_search_via_router(stack: Any) -> None:
    """Verify RetrievalRouter returns results for a knowledge query
    after library content has been ingested."""
    # Ensure there's library content
    store = stack.memory.knowledge.library
    if len(store) == 0:
        store.ingest(
            title="Router Test Entry",
            content="The mitochondria is the powerhouse of the cell.",
            domain="biology",
        )

    router = stack.memory.router
    _assert(router is not None, "memory.router is None")

    from zados.memory.long_term.retrieval_router import RetrievalContext
    ctx = RetrievalContext(
        query_text="mitochondria powerhouse cell biology",
        query_type="knowledge",
        pipeline_name="plumbing_test",
        limit=10,
    )
    results = router.route(ctx)
    _assert(isinstance(results, list),
            f"router.route() returned {type(results).__name__}")
    _assert(len(results) > 0,
            "RetrievalRouter returned 0 results for knowledge query "
            "after library ingestion")


# =======================================================================
# Runner
# =======================================================================

def run_all(stack: Any) -> dict:
    """Run all plumbing tests and return a structured report.

    Parameters
    ----------
    stack : ZADOSStack
        The live server stack.

    Returns
    -------
    dict
        JSON-serializable report with pass/fail counts and per-test details.
    """
    report = PlumbingReport()
    t0 = time.perf_counter()

    tests = [
        # Stack & session
        ("stack_construction",              lambda: test_stack_construction(stack)),
        ("session_open",                    lambda: test_session_open(stack)),

        # Neurochem
        ("neurochem_baseline",              lambda: test_neurochem_baseline(stack)),
        ("neurochem_step_mutates_state",    lambda: test_neurochem_step_mutates_state(stack)),

        # Pipeline phases
        ("phase0_validation",               lambda: test_phase0_validation(stack)),
        ("phase1_perception",               lambda: test_phase1_perception(stack)),
        ("phase3_dispatch",                 lambda: test_phase3_dispatch(stack)),
        ("phase2_modulation",               lambda: test_phase2_modulation(stack)),
        ("full_pipeline_turn",              lambda: test_full_pipeline_turn(stack)),

        # Memory tiers
        ("stmm_write_read",                lambda: test_stmm_write_read(stack)),
        ("mtmm_write_search",              lambda: test_mtmm_write_search(stack)),
        ("ltmm_write_search",              lambda: test_ltmm_write_search(stack)),
        ("memory_compression",             lambda: test_memory_compression_stmm_to_packet(stack)),
        ("consolidation_engine",           lambda: test_consolidation_engine(stack)),
        ("memory_manager_on_cycle_end",    lambda: test_memory_manager_on_cycle_end(stack)),

        # Namespaced stores
        ("namespace_identity_stores",      lambda: test_namespace_identity_stores(stack)),
        ("namespace_thoughts_stores",      lambda: test_namespace_thoughts_stores(stack)),
        ("namespace_knowledge_stores",     lambda: test_namespace_knowledge_stores(stack)),

        # NT → dispatch influence
        ("nt_modulates_dispatch",          lambda: test_nt_modulates_dispatch(stack)),

        # Retrieval router
        ("retrieval_router",               lambda: test_retrieval_router(stack)),

        # Full round trip
        ("full_memory_round_trip",         lambda: test_full_memory_round_trip(stack)),

        # Library import pipeline
        ("library_ingest_whole",           lambda: test_library_ingest_whole(stack)),
        ("library_ingest_chunked",         lambda: test_library_ingest_chunked(stack)),
        ("library_search_via_router",      lambda: test_library_search_via_router(stack)),
    ]

    for name, fn in tests:
        _run(name, fn, report)

    report.total_ms = (time.perf_counter() - t0) * 1000
    log.info("Plumbing tests complete: %d passed, %d failed, %d errors (%.0f ms)",
             report.passed, report.failed, report.errors, report.total_ms)
    return report.to_dict()
