"""
LTMM §2.2 — Fractal Pattern Comparator.

Compares incoming consolidation candidates against existing LTMM content
to detect pattern matches across abstraction levels.

Operations:
  1. Duplicate detection  → MERGE if near-duplicate found (similarity > 0.85)
  2. Pattern reinforcement → increase weight/confidence of matching entry
  3. Pattern contradiction → flag for reconciliation
  4. Cross-level linking   → link new entry to structurally similar existing entries

Result is a FractalComparisonResult that the Memory Implementation Manager
acts upon before (or instead of) writing a new LTMMEntry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from zados.memory.long_term.store import Granularity, LTMMEntry, LTMMStore
from zados.memory.long_term.search_utils import tokenize as _tokenize, term_freq as _term_freq, cosine as _cosine

_DUP_THRESHOLD   = 0.85   # above this → duplicate, merge
_REINFORCE_THRESHOLD = 0.60   # above this → pattern reinforcement
_CONTRADICT_THRESHOLD = 0.30  # sentiment-level divergence would go here — placeholder


@dataclass
class FractalComparisonResult:
    """Outcome of comparing a candidate entry against LTMM."""
    candidate_id:    str
    action:          str            = "write"   # "write" | "merge" | "reinforce" | "flag_contradiction"
    merge_target_id: Optional[str] = None
    cross_links:     List[str]     = field(default_factory=list)
    notes:           List[str]     = field(default_factory=list)


class FractalPatternComparator:
    """
    Compares a candidate LTMMEntry against all existing LTMM entries and
    returns a FractalComparisonResult describing what action to take.
    """

    def __init__(self, ltmm: LTMMStore) -> None:
        self._ltmm = ltmm

    def compare(self, candidate: LTMMEntry) -> FractalComparisonResult:
        cand_text = f"{candidate.packet.user_message} {candidate.packet.system_response}"
        cand_vec  = _term_freq(_tokenize(cand_text))
        cand_id   = candidate.packet.packet_id

        result = FractalComparisonResult(candidate_id=cand_id)

        best_sim   = 0.0
        best_pid   = None
        cross_links: List[str] = []

        for entry in self._ltmm.get_all():
            if entry.packet.packet_id == cand_id:
                continue
            entry_text = f"{entry.packet.user_message} {entry.packet.system_response}"
            entry_vec  = _term_freq(_tokenize(entry_text))
            sim        = _cosine(cand_vec, entry_vec)

            if sim > best_sim:
                best_sim = sim
                best_pid = entry.packet.packet_id

            if sim >= _REINFORCE_THRESHOLD:
                cross_links.append(entry.packet.packet_id)

        result.cross_links = cross_links

        if best_sim >= _DUP_THRESHOLD and best_pid is not None:
            # Duplicate: merge instead of write
            result.action          = "merge"
            result.merge_target_id = best_pid
            result.notes.append(f"Near-duplicate of {best_pid} (sim={best_sim:.2f})")
            # Reinforce the existing entry
            existing = self._ltmm.get_by_id(best_pid)
            if existing:
                existing.retrieval_count += 1
                # If granularity is semantic and both are same, promote to symbolic
                if (existing.granularity == Granularity.SEMANTIC and
                        candidate.granularity == Granularity.SEMANTIC):
                    existing.granularity = Granularity.SYMBOLIC
                    result.notes.append("Promoted to symbolic (repeated pattern)")

        elif best_sim >= _REINFORCE_THRESHOLD and best_pid is not None:
            result.action = "reinforce"
            result.notes.append(f"Reinforces pattern in {best_pid} (sim={best_sim:.2f})")

        else:
            result.action = "write"

        return result
