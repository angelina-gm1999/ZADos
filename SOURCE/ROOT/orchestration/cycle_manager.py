"""
CycleManager — thin sequencer that orchestrates one processing cycle.

Responsibilities:
  1. Reset STMM (begin_cycle)
  2. Ingest user message + system response into STMM buffer
  3. Query MemoryContrast and populate STMM.memory_contrast
  4. Push NT state to each engine and run adapters in topological order
  5. Record each engine execution in BrainProcessTracker
  6. Compress STMM → MemoryPacket → MTMM write (end_cycle)

Not a god object. Delegates to:
  - Per-engine adapter functions for data marshalling
  - MemoryLayer for all memory operations
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from zados.memory import MemoryLayer
from zados.memory.short_term.components import EngineExecution, MemoryMatch
from zados.orchestration.cycle_types import CycleContext, CycleResult, EngineSlot

logger = logging.getLogger(__name__)


class CycleManager:
    """
    Orchestrates one processing cycle: STMM begin → engine dispatch →
    STMM compress → MTMM write.

    Usage::

        cm = CycleManager(memory=MemoryLayer())
        cm.register(slot)
        cm.set_default_sequence(["emotional_detection_engine", ...])
        result = cm.run_cycle("Hello", "Hi there!")
    """

    def __init__(
        self,
        memory: Optional[MemoryLayer] = None,
    ) -> None:
        self._memory = memory or MemoryLayer()
        self._slots: Dict[str, EngineSlot] = {}
        self._default_sequence: List[str] = []
        self._cycle_count = 0

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def register(self, slot: EngineSlot) -> None:
        """Register an engine with its adapter."""
        self._slots[slot.engine_id] = slot

    def unregister(self, engine_id: str) -> None:
        """Remove an engine from the dispatch table."""
        self._slots.pop(engine_id, None)

    def set_default_sequence(self, engine_ids: List[str]) -> None:
        """Set the default dispatch order (list of engine_id strings)."""
        self._default_sequence = list(engine_ids)

    @property
    def registered_engines(self) -> List[str]:
        return list(self._slots.keys())

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def memory(self) -> MemoryLayer:
        return self._memory

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def run_cycle(
        self,
        user_message: str,
        system_response: str = "",
        *,
        dispatch_list: Optional[List[str]] = None,
        nt_state: Optional[Dict[str, float]] = None,
        oscillatory_state: Optional[Dict[str, float]] = None,
        active_mode: str = "normal",
    ) -> CycleResult:
        """Execute one full processing cycle.

        Args:
            user_message:      Current user input text.
            system_response:   System's response (may be empty if not yet generated).
            dispatch_list:     Override which engines to run (default: default_sequence).
            nt_state:          Current NT concentrations (lowercase keys).
            oscillatory_state: Current oscillatory band powers.
            active_mode:       Operating mode string ("normal", "dev", etc.).

        Returns:
            CycleResult with the produced MemoryPacket, timing, and engine lists.
        """
        self._cycle_count += 1
        t0 = time.perf_counter()

        stmm = self._memory.stmm

        # ---- Step 1: Begin cycle (reset analysis slots) ----
        stmm.begin_cycle()

        # ---- Step 2: Ingest messages ----
        stmm.add_user_message(user_message)
        if system_response:
            stmm.add_system_response(system_response)

        # ---- Step 3: Memory contrast query ----
        self._populate_memory_contrast(user_message)

        # ---- Step 4: Build cycle context ----
        ctx = CycleContext(
            cycle_id=self._cycle_count,
            user_message=user_message,
            system_response=system_response,
            nt_state=nt_state or {},
            oscillatory_state=oscillatory_state or {},
            active_mode=active_mode,
            memory_contrast=self._memory.contrast,
        )

        # ---- Step 5: Dispatch engines ----
        sequence = dispatch_list if dispatch_list is not None else self._default_sequence
        engines_run: List[str] = []
        engines_skipped: List[str] = []

        for engine_id in sequence:
            slot = self._slots.get(engine_id)
            if slot is None:
                engines_skipped.append(engine_id)
                stmm.brain_process_tracker.record(EngineExecution(
                    engine_id=engine_id,
                    timing_ms=0.0,
                    output_summary="",
                    skipped=True,
                    skip_reason="not_registered",
                ))
                continue

            # Check dependency satisfaction
            if not self._deps_satisfied(slot, engines_run):
                engines_skipped.append(engine_id)
                stmm.brain_process_tracker.record(EngineExecution(
                    engine_id=engine_id,
                    timing_ms=0.0,
                    output_summary="",
                    skipped=True,
                    skip_reason="unsatisfied_dependency",
                ))
                continue

            # Push NT state
            if nt_state:
                slot.instance.update_neurochem_state(nt_state)

            # Run adapter
            et0 = time.perf_counter()
            output_summary = ""
            try:
                output_summary = slot.adapter(slot.instance, ctx, stmm)
            except Exception as exc:
                output_summary = f"ERROR: {type(exc).__name__}: {exc}"
                logger.warning(
                    "Engine %s adapter raised %s: %s",
                    engine_id, type(exc).__name__, exc,
                )
            elapsed_ms = (time.perf_counter() - et0) * 1000.0

            stmm.brain_process_tracker.record(EngineExecution(
                engine_id=engine_id,
                timing_ms=round(elapsed_ms, 3),
                output_summary=str(output_summary)[:200],
            ))
            engines_run.append(engine_id)

        stmm.brain_process_tracker.mark_stage("engine_dispatch", True)

        # ---- Step 6: Compress STMM → MemoryPacket → MTMM ----
        packet = self._memory.end_cycle()

        # ---- Step 7: Tick unsolved concepts ----
        self._memory.manager.tick_unsolved()

        stmm.brain_process_tracker.mark_stage("memory_write", True)

        total_ms = (time.perf_counter() - t0) * 1000.0

        return CycleResult(
            cycle_id=self._cycle_count,
            packet=packet,
            engines_run=tuple(engines_run),
            engines_skipped=tuple(engines_skipped),
            timing_ms=round(total_ms, 3),
            stmm_snapshot=stmm.snapshot(),
        )

    # -----------------------------------------------------------------
    # Memory contrast population
    # -----------------------------------------------------------------

    def _populate_memory_contrast(self, user_message: str) -> None:
        """Query MemoryContrast and populate STMM.memory_contrast slot."""
        contrast = self._memory.contrast
        stmm = self._memory.stmm

        try:
            result = contrast.contrast(
                current={"text": user_message},
                query_type="context",
                limit=5,
            )
        except Exception:
            logger.debug("Memory contrast query failed — STMM slot stays empty")
            return

        for ref in result.references:
            stmm.memory_contrast.matched_entries.append(
                MemoryMatch(
                    entry_id=ref.get("packet_id", ""),
                    source_tier=ref.get("source", "MTMM"),
                    similarity=ref.get("similarity", 0.0),
                    content_summary=ref.get("summary", ""),
                    metadata=ref,
                )
            )

    # -----------------------------------------------------------------
    # Dependency checking
    # -----------------------------------------------------------------

    @staticmethod
    def _deps_satisfied(slot: EngineSlot, already_run: List[str]) -> bool:
        """Check that all dependencies have already run."""
        if not slot.depends_on:
            return True
        return all(dep in already_run for dep in slot.depends_on)
