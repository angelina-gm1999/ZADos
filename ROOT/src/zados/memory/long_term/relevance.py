"""
LTMM §2.4 — Memory Relevance Heuristics Engine.

Periodically scans LTMM and adjusts relevance scores.  Entries below
threshold are demoted to cold storage or flagged for purge.

Relevance formula (from spec):
  relevance(m, t) = w_recency   × recency(m, t)
                  + w_frequency × frequency(m)
                  + w_emotion   × emotional_weight(m)
                  + w_utility   × utility(m)
                  + w_coherence × coherence(m, t)   ← simplified: fixed 0.5 if not computed

  recency decays exponentially: exp(-λ × hours_since_last_access)
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional, Tuple

from zados.memory.long_term.store import LTMMEntry, LTMMStore

_THETA_COLD  = 0.15
_THETA_PURGE = 0.05

# Decay rate: relevance halves after ~168 hours (1 week)
_DECAY_LAMBDA = math.log(2) / 168.0

# Weights (sum = 1.0)
_W_RECENCY   = 0.30
_W_FREQUENCY = 0.20
_W_EMOTION   = 0.20
_W_UTILITY   = 0.20
_W_COHERENCE = 0.10


def _recency_score(entry: LTMMEntry, now: datetime) -> float:
    hours = max(0.0, (now - entry.last_accessed).total_seconds() / 3600.0)
    return math.exp(-_DECAY_LAMBDA * hours)


def _frequency_score(entry: LTMMEntry) -> float:
    # Saturates at ~10 retrievals → 1.0
    return 1.0 - math.exp(-0.23 * entry.retrieval_count)


class MemoryRelevanceHeuristicsEngine:
    """
    Scans LTMMStore and updates relevance scores.  Demotes or flags entries
    that fall below cold / purge thresholds.

    Call scan() periodically (e.g., end of session or scheduled maintenance).
    """

    def __init__(self, ltmm: LTMMStore) -> None:
        self._ltmm = ltmm

    def scan(self, now: Optional[datetime] = None) -> Tuple[List[str], List[str]]:
        """
        Update all entry relevance scores.

        Returns:
            demoted   — packet_ids moved to cold storage this scan
            purge_candidates — packet_ids eligible for purge (relevance < θ_purge)
        """
        if now is None:
            now = datetime.utcnow()

        demoted:    List[str] = []
        purgeable:  List[str] = []

        for entry in self._ltmm.get_all():
            # Identity-relevant memories are never demoted
            if entry.identity_relevant:
                entry.relevance_score = max(entry.relevance_score, 0.5)
                continue

            score = self._compute(entry, now)
            entry.relevance_score = score

            if score < _THETA_PURGE and entry.cold_storage:
                purgeable.append(entry.packet.packet_id)
            elif score < _THETA_COLD and not entry.cold_storage:
                self._ltmm.demote_to_cold(entry.packet.packet_id)
                demoted.append(entry.packet.packet_id)

        return demoted, purgeable

    def _compute(self, entry: LTMMEntry, now: datetime) -> float:
        r = _recency_score(entry, now)
        f = _frequency_score(entry)
        e = entry.packet.emotional_significance   # [0, 1]
        u = entry.utility_score                   # [0, 1]
        c = 0.5                                   # coherence placeholder

        return (
            _W_RECENCY   * r +
            _W_FREQUENCY * f +
            _W_EMOTION   * e +
            _W_UTILITY   * u +
            _W_COHERENCE * c
        )
