"""
MTMM §2.1 — Raw Interaction Logger.

Sequential record of all exchanges in the current session.
Each entry is a MemoryPacket promoted from STMM via MemoryExitCompressor.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from zados.memory.types import MemoryPacket


class RawInteractionLogger:
    """
    Ordered log of all MemoryPackets for the current session.

    Supports:
    - append()   — add a new packet (called by MemoryImplementationManager)
    - get_all()  — return full log (for trend analysis, serialisation)
    - get_by_turn() — retrieve a specific turn
    - semantic_scan() — naive keyword/field scan used by ContextProcessor index
    """

    def __init__(self) -> None:
        self._entries: List[MemoryPacket] = []

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def append(self, packet: MemoryPacket) -> None:
        self._entries.append(packet)

    # -----------------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------------

    def get_all(self) -> List[MemoryPacket]:
        return list(self._entries)

    def get_by_turn(self, turn_index: int) -> Optional[MemoryPacket]:
        for p in self._entries:
            if p.turn_index == turn_index:
                return p
        return None

    def __len__(self) -> int:
        return len(self._entries)

    # -----------------------------------------------------------------------
    # Progressive re-compression
    # -----------------------------------------------------------------------

    def recompress_entry(self, turn_index: int, importance: float) -> None:
        """
        Aggressively compress older low-importance entries.
        High-importance entries (importance > 0.6) are kept at SEMANTIC level.
        Low-importance entries are stripped to minimal fields.
        This simulates the spec's progressive compression.

        Uses dict *replacement* (not in-place mutation) so that existing
        references held by the ContextProcessor index or trend analyser
        are not silently corrupted.
        """
        from zados.memory.types import CompressionLevel
        pkt = self.get_by_turn(turn_index)
        if pkt is None or importance > 0.6:
            return
        # Downgrade: replace (not mutate) dicts to avoid aliasing bugs
        pkt.neurochemical_snapshot = {}
        pkt.emotion_vector = {
            k: v for k, v in pkt.emotion_vector.items()
            if not k.startswith("sys_")      # drop system state, keep user emotions
        }
        pkt.compression_level = CompressionLevel.SYMBOLIC
