"""
Memory Implementation Manager (Engine 22 in the spec).

Enforces the complete memory lifecycle:
  - Receives MemoryPackets from MemoryExitCompressor and writes to MTMM
  - Runs LTMM consolidation (session end / emergency)
  - Coordinates FractalPatternComparator before LTMM writes
  - Enforces cross-tier consistency (no orphaned refs, no contradictory states)
  - Runs MemoryRelevanceHeuristicsEngine scans
  - Routes unsolved/paradox/contradiction flags to specialized logs

This is the single authoritative write path to LTMM.
No engine writes directly to LTMM — they go through this manager.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from zados.memory.long_term.consolidation import MemoryConsolidationEngine
from zados.memory.long_term.fractal_comparator import FractalPatternComparator
from zados.memory.long_term.relevance import MemoryRelevanceHeuristicsEngine
from zados.memory.long_term.specialized_logs import (
    ContradictionEntry,
    ContradictionLog,
    DreamLog,
    IdentityMemoryLog,
    LearningSystemLog,
    ParadoxEntry,
    ParadoxLog,
    SandboxLog,
    SelfReflectionLog,
    SpecializedLogs,
    UnsolvedConceptEntry,
    UnsolvedConceptsBuffer,
)
from zados.memory.long_term.store import LTMMEntry, LTMMStore
from zados.memory.mid_term.store import MTMMStore
from zados.memory.short_term.compressor import MemoryExitCompressor
from zados.memory.short_term.store import STMMStore
from zados.memory.types import MemoryPacket

logger = logging.getLogger(__name__)


class MemoryImplementationManager:
    """
    Orchestrates the full memory lifecycle.

    Typical call sequence per processing cycle:
        1. manager.on_cycle_end(stmm)           → compress + write to MTMM
        2. manager.tick_unsolved()              → increment stagnation counters
        3. [at session end] manager.consolidate()  → MTMM → LTMM

    Additional calls:
        manager.emergency_consolidate(packet)   → critical-severity event
        manager.run_relevance_scan()            → periodic LTMM housekeeping
        manager.write_session_overview(session)  → OverviewLogEntry at session end
    """

    def __init__(
        self,
        mtmm: MTMMStore,
        ltmm: LTMMStore,
        specialized_logs: Optional[SpecializedLogs] = None,
        overview_log_store: Any = None,
    ) -> None:
        self._mtmm    = mtmm
        self._ltmm    = ltmm
        self._logs    = specialized_logs or SpecializedLogs()
        self._overview_log_store = overview_log_store

        self._compressor   = MemoryExitCompressor()
        self._consolidator = MemoryConsolidationEngine(ltmm)
        self._comparator   = FractalPatternComparator(ltmm)
        self._relevance    = MemoryRelevanceHeuristicsEngine(ltmm)

    # -----------------------------------------------------------------------
    # Per-cycle entry point
    # -----------------------------------------------------------------------

    def on_cycle_end(self, stmm: STMMStore) -> MemoryPacket:
        """
        Called at end of each processing cycle.
        Compresses STMM → MemoryPacket and writes to MTMM.
        Returns the produced MemoryPacket.
        """
        packet = self._compressor.compress(stmm)

        # Importance heuristic for MTMM context processor
        importance = self._compute_importance(packet)
        self._mtmm.write(packet, importance)

        # Route unsolved matches to the UnsolvedConceptsBuffer stagnation tracker
        for ucid in packet.unsolved_items_matched:
            entry = self._logs.unsolved.get_by_id(ucid)
            if entry:
                entry.evidence_accumulated.append(
                    f"Matched at turn {packet.turn_index}"
                )

        # Route high-severity flags to specialized logs
        self._route_flags(packet)

        logger.debug(
            "Cycle %d compressed → MTMM (importance=%.2f, flags=%s)",
            packet.turn_index, importance, packet.flags,
        )
        return packet

    def tick_unsolved(self) -> None:
        """Increment stagnation counters on all active unsolved concepts."""
        self._logs.unsolved.tick_all()

    # -----------------------------------------------------------------------
    # Consolidation (MTMM → LTMM)
    # -----------------------------------------------------------------------

    def consolidate(self) -> List[str]:
        """
        Promote qualifying MTMM packets to LTMM.
        Called at session end or during scheduled REM processing.
        Returns list of promoted packet_ids.
        """
        packets = self._mtmm.get_all_packets()
        return self._consolidate_packets(packets)

    def emergency_consolidate(self, packet: MemoryPacket) -> str:
        """
        Immediately promote a critical-severity packet to LTMM.
        Called mid-session when a critical event occurs.
        """
        promoted = self._consolidate_packets([packet])
        return promoted[0] if promoted else ""

    def _consolidate_packets(self, packets: List[MemoryPacket]) -> List[str]:
        promoted_ids: List[str] = []

        # MemoryConsolidationEngine evaluates and creates LTMMEntry candidates
        candidates = self._consolidator.consolidate(packets)

        for packet_id in candidates:
            # Retrieve the newly-written LTMM entry
            entry = self._ltmm.get_by_id(packet_id)
            if entry is None:
                continue

            # FractalPatternComparator: check against existing LTMM
            comparison = self._comparator.compare(entry)

            if comparison.action == "merge":
                # Don't keep new duplicate — the existing entry was reinforced
                self._ltmm.purge(packet_id)
                logger.debug("Merged %s into %s", packet_id, comparison.merge_target_id)
            else:
                promoted_ids.append(packet_id)
                logger.debug("Promoted to LTMM: %s (action=%s)", packet_id, comparison.action)

        return promoted_ids

    # -----------------------------------------------------------------------
    # Session overview (OverviewLogStore)
    # -----------------------------------------------------------------------

    def write_session_overview(self, session: Any) -> Optional[str]:
        """Write a brief cognitive overview entry at session end.

        Reads session metadata (mode_sequence, dominant emotions, open threads)
        and writes an OverviewLogEntry to the OverviewLogStore in LTMM/thoughts.

        Parameters
        ----------
        session : SessionState
            The session state with accumulated metadata.

        Returns
        -------
        str or None
            The log_id of the written entry, or None if store unavailable.
        """
        if self._overview_log_store is None:
            return None

        try:
            from zados.memory.long_term.thoughts.types import OverviewLogEntry

            # Extract session metadata
            session_id = getattr(session, "session_id", "")
            mode_seq = getattr(session, "mode_sequence", [])
            if not mode_seq:
                # Build from active_learning_mode if mode_sequence not tracked
                active = getattr(session, "active_learning_mode", None)
                mode_seq = [active] if active else ["regular"]

            dominant_emotions = getattr(session, "dominant_emotions", [])
            subject_tags = getattr(session, "subject_tags", [])
            open_threads = getattr(session, "open_threads", [])
            nt_arc = getattr(session, "nt_arc", {})

            # Build summary from available info
            summary_parts = [f"Session {session_id[:8]}"]
            if mode_seq:
                summary_parts.append(f"modes: {', '.join(str(m) for m in mode_seq)}")
            if dominant_emotions:
                summary_parts.append(
                    f"dominant emotions: {', '.join(dominant_emotions[:5])}"
                )
            turn_count = getattr(session, "turn_index", 0)
            summary_parts.append(f"turns: {turn_count}")

            entry = OverviewLogEntry(
                session_id=session_id,
                summary="; ".join(summary_parts),
                mode_sequence=list(mode_seq),
                subject_tags=list(subject_tags),
                dominant_emotions=list(dominant_emotions),
                nt_arc=dict(nt_arc) if nt_arc else {},
                open_threads=list(open_threads),
            )
            self._overview_log_store.write(entry)
            logger.info(
                "Session overview written: %s (%d turns, modes=%s)",
                entry.log_id, turn_count, mode_seq,
            )
            return entry.log_id

        except Exception:
            logger.debug("OverviewLogStore write failed.", exc_info=True)
            return None

    # -----------------------------------------------------------------------
    # LTMM housekeeping
    # -----------------------------------------------------------------------

    def run_relevance_scan(self) -> dict:
        """
        Run MemoryRelevanceHeuristicsEngine.
        Returns summary of demoted and purgeable entries.
        """
        demoted, purgeable = self._relevance.scan()
        logger.info(
            "Relevance scan: %d demoted to cold, %d purge candidates",
            len(demoted), len(purgeable),
        )
        return {"demoted": demoted, "purgeable": purgeable}

    # -----------------------------------------------------------------------
    # Specialized log access (read-only externally via properties)
    # -----------------------------------------------------------------------

    @property
    def logs(self) -> SpecializedLogs:
        return self._logs

    # -----------------------------------------------------------------------
    # Cross-tier consistency enforcement
    # -----------------------------------------------------------------------

    def validate_consistency(self) -> List[str]:
        """
        Check for cross-tier inconsistencies.
        Returns list of issue descriptions.
        """
        issues = self._mtmm.validate()
        # Additional: check that unsolved buffer IDs referenced by MTMM exist
        for pkt in self._mtmm.get_all_packets():
            for ucid in pkt.unsolved_items_matched:
                if self._logs.unsolved.get_by_id(ucid) is None:
                    issues.append(
                        f"MTMM packet {pkt.packet_id} references unknown "
                        f"unsolved concept {ucid}"
                    )
        return issues

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _compute_importance(self, packet: MemoryPacket) -> float:
        """
        Heuristic importance score for MTMM context processor.
        High-importance entries resist compression.
        """
        score = 0.0
        score += 0.4 * packet.emotional_significance
        score += 0.2 * (1.0 - packet.trust_weight)           # low trust = anomaly
        score += 0.2 * min(1.0, packet.contradictions_detected * 0.3)
        score += 0.2 * min(1.0, len(packet.flags) * 0.1)
        return round(min(1.0, score), 3)

    def _route_flags(self, packet: MemoryPacket) -> None:
        """Route flag-tagged events to appropriate specialized logs."""
        for flag in packet.flags:
            flag_upper = flag.upper()
            if "CONTRADICTION" in flag_upper and packet.contradictions_detected > 0:
                self._logs.contradiction.record(
                    ContradictionEntry(
                        statement_a=packet.user_message[:200],
                        statement_b=packet.system_response[:200],
                        source_a_ref=packet.packet_id,
                        severity="high" if "CRITICAL" in flag_upper else "medium",
                    )
                )
            if "PARADOX" in flag_upper and packet.paradoxes_detected > 0:
                self._logs.paradox.record(
                    ParadoxEntry(
                        formulation=packet.user_message[:200],
                        source_ref=packet.packet_id,
                    )
                )
