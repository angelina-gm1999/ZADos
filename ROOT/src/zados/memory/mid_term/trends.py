"""
MTMM §2.2 — Mid-Term Processing Engine.

Accumulates processing TRENDS across the session from MemoryPackets.
Does NOT store individual results — it maintains running statistics.

Updated every N turns (default 3) or on significant state change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from zados.memory.types import MemoryPacket


@dataclass
class SessionTrends:
    """Accumulated trend data for the current session."""
    contradiction_counts:  List[int]           = field(default_factory=list)
    emotional_trajectory:  List[Dict[str, float]] = field(default_factory=list)
    reward_trajectories:   Dict[str, List[float]] = field(default_factory=dict)
    intention_pattern:     List[str]           = field(default_factory=list)
    nt_trajectories:       Dict[str, List[float]] = field(default_factory=dict)

    # Derived trend labels (updated by MidTermProcessingEngine)
    contradiction_trend:   str  = "stable"      # increasing | stable | decreasing
    emotional_trend:       str  = "neutral"
    reward_trend:          Dict[str, str] = field(default_factory=dict)
    intention_shift:       bool = False


class MidTermProcessingEngine:
    """
    Aggregates MemoryPackets into SessionTrends.

    Call update() after every N packets or on significant change.
    """

    def __init__(self, update_interval: int = 3) -> None:
        self._interval = update_interval
        self._packet_count = 0
        self.trends = SessionTrends()

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, packet: MemoryPacket) -> None:
        """Ingest a single packet; recalculate trends every _interval packets."""
        self._packet_count += 1
        t = self.trends

        t.contradiction_counts.append(packet.contradictions_detected)
        t.emotional_trajectory.append(dict(packet.emotion_vector))
        t.intention_pattern.append(packet.intention)

        for domain, score in packet.reward_scores.items():
            t.reward_trajectories.setdefault(domain, []).append(score)

        for nt, conc in packet.neurochemical_snapshot.items():
            t.nt_trajectories.setdefault(nt, []).append(conc)

        if self._packet_count % self._interval == 0 or self._is_significant_change(packet):
            self._recalculate()

    # -----------------------------------------------------------------------
    # Trend computation
    # -----------------------------------------------------------------------

    def _recalculate(self) -> None:
        t = self.trends

        # Contradiction trend
        counts = t.contradiction_counts
        if len(counts) >= 3:
            recent = counts[-3:]
            slope = recent[-1] - recent[0]
            if slope > 0:
                t.contradiction_trend = "increasing"
            elif slope < 0:
                t.contradiction_trend = "decreasing"
            else:
                t.contradiction_trend = "stable"

        # Reward trends per domain
        for domain, scores in t.reward_trajectories.items():
            if len(scores) >= 2:
                slope = scores[-1] - scores[-2]
                t.reward_trend[domain] = (
                    "improving" if slope > 0.05 else
                    "declining" if slope < -0.05 else
                    "stable"
                )

        # Intention shift: last intention differs from majority
        if len(t.intention_pattern) >= 4:
            majority = max(set(t.intention_pattern[:-1]), key=t.intention_pattern[:-1].count)
            t.intention_shift = t.intention_pattern[-1] != majority

    def _is_significant_change(self, packet: MemoryPacket) -> bool:
        """Trigger early recalculation on high emotional significance."""
        return packet.emotional_significance > 0.7 or len(packet.flags) > 3
