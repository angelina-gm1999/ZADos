"""
MTMMStore — session-scoped memory.

Wraps RawInteractionLogger, MidTermProcessingEngine, and ContextProcessor.
The Memory Implementation Manager calls write() to promote MemoryPackets
from STMM.  Memory Contrast Engine calls search() for session retrieval.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from zados.memory.mid_term.context_processor import ContextProcessor
from zados.memory.mid_term.logger import RawInteractionLogger
from zados.memory.mid_term.trends import MidTermProcessingEngine, SessionTrends
from zados.memory.types import MemoryPacket


class MTMMStore:
    """
    Mid-Term Memory Module.

    Lifecycle: created at session start, consulted throughout, evaluated for
    LTMM promotion at session end (by Memory Consolidation Engine).
    """

    def __init__(self, update_interval: int = 3) -> None:
        self._logger    = RawInteractionLogger()
        self._engine    = MidTermProcessingEngine(update_interval)
        self._processor = ContextProcessor(self._logger)

    # -----------------------------------------------------------------------
    # Write (called by Memory Implementation Manager)
    # -----------------------------------------------------------------------

    def write(self, packet: MemoryPacket, importance: float = 0.5) -> None:
        """
        Ingest a MemoryPacket:
          1. Append to raw log
          2. Update trend engine
          3. Index for semantic search
          4. Run periodic compression
        """
        self._logger.append(packet)
        self._engine.ingest(packet)
        self._processor.index_packet(packet, importance)
        self._processor.compress_old_entries(packet.turn_index)

    # -----------------------------------------------------------------------
    # Read (called by Memory Contrast Engine)
    # -----------------------------------------------------------------------

    def search(self, query_text: str, limit: int = 5) -> List[Tuple[float, MemoryPacket]]:
        """Semantic similarity search over session history."""
        return self._processor.search(query_text, limit)

    def get_all_packets(self) -> List[MemoryPacket]:
        return self._logger.get_all()

    def get_by_turn(self, turn_index: int) -> Optional[MemoryPacket]:
        return self._logger.get_by_turn(turn_index)

    # -----------------------------------------------------------------------
    # Diagnostics / validation
    # -----------------------------------------------------------------------

    @property
    def trends(self) -> SessionTrends:
        return self._engine.trends

    def validate(self) -> List[str]:
        """Cross-check internal consistency; returns list of issues."""
        return self._processor.validate()

    def __len__(self) -> int:
        return len(self._logger)
