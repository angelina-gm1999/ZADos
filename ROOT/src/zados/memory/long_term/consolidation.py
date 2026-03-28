"""
LTMM §2.1 — Memory Consolidation Engine.

Decides what gets promoted from MTMM to LTMM.

Consolidation criteria (from spec):
  - Emotional significance  (high saturation / identity relevance)
  - Repeated patterns       (concept recurs across sessions)
  - Unresolved items        (contradictions / paradoxes / unsolved questions)
  - User model updates      (preference / style signals)
  - Novel learning          (validated insights)
  - High-severity flags

Consolidation timing: called at session end or on critical-severity event.
"""
from __future__ import annotations

from typing import List

from zados.memory.long_term.store import Granularity, LTMMEntry, LTMMStore
from zados.memory.types import MemoryPacket


# Thresholds
_EMOTIONAL_SIG_THRESHOLD  = 0.6
_CRITICAL_FLAG_KEYWORDS   = {"CRITICAL", "SEVERE", "IDENTITY", "PARADOX", "UNRESOLVED"}


class MemoryConsolidationEngine:
    """
    Evaluates MTMM packets and writes qualifying ones to LTMM.

    Usage:
        engine = MemoryConsolidationEngine(ltmm_store)
        engine.consolidate(mtmm_packets)
    """

    def __init__(self, ltmm: LTMMStore) -> None:
        self._ltmm = ltmm

    def consolidate(self, packets: List[MemoryPacket]) -> List[str]:
        """
        Evaluate each packet against promotion criteria.
        Returns list of packet_ids that were promoted.
        """
        promoted = []
        for pkt in packets:
            granularity, identity_relevant = self._evaluate(pkt)
            if granularity is not None:
                entry = LTMMEntry(
                    packet=pkt,
                    granularity=granularity,
                    identity_relevant=identity_relevant,
                )
                # Mirror emotional significance onto the entry for relevance heuristics
                entry.packet.emotional_significance  # read-only access (stored on packet)
                # Re-set the relevance score field for the entry
                entry.relevance_score = self._initial_relevance(pkt)
                self._ltmm.write(entry)
                promoted.append(pkt.packet_id)
        return promoted

    # -----------------------------------------------------------------------
    # Criteria evaluation
    # -----------------------------------------------------------------------

    def _evaluate(self, pkt: MemoryPacket):
        """
        Returns (granularity, identity_relevant) if packet qualifies, else (None, False).
        """
        identity_relevant = False
        qualifies = False
        granularity = Granularity.SEMANTIC   # default

        # Criterion 1: emotional significance
        if pkt.emotional_significance >= _EMOTIONAL_SIG_THRESHOLD:
            qualifies = True

        # Criterion 2: unresolved items → must persist
        if pkt.unsolved_items_matched or pkt.paradoxes_detected > 0 or pkt.contradictions_detected > 1:
            qualifies = True
            granularity = Granularity.VERBATIM

        # Criterion 3: high-severity flags
        flag_names = {f.split(":")[0].upper() for f in pkt.flags}
        if flag_names & _CRITICAL_FLAG_KEYWORDS:
            qualifies = True
            if "IDENTITY" in flag_names:
                identity_relevant = True
                granularity = Granularity.VERBATIM

        # Criterion 4: trust weight (low trust = anomaly worth keeping)
        if pkt.trust_weight < 0.4:
            qualifies = True

        return (granularity if qualifies else None), identity_relevant

    def _initial_relevance(self, pkt: MemoryPacket) -> float:
        """New entries start with relevance 1.0, tempered by trust weight."""
        return min(1.0, 0.5 + 0.5 * pkt.trust_weight)
