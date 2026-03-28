"""
MTMM §2.3 — Context Processor.

Three operations on MTMM contents:
  1. Compression — progressive re-compression of older entries
  2. Validation — internal consistency cross-check
  3. Contrast   — semantic index for fast similarity search

The semantic index uses simple cosine similarity over TF-IDF-style term
vectors (no external ML model required for Phase 2 — the embedding field on
MemoryPacket is used when populated, otherwise text heuristics apply).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from zados.memory.mid_term.logger import RawInteractionLogger
from zados.memory.types import MemoryPacket


# ---------------------------------------------------------------------------
# Minimal vector similarity helpers (no external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _term_freq(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1.0
    total = max(len(tokens), 1)
    return {t: c / total for t, c in tf.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _packet_text(pkt: MemoryPacket) -> str:
    return f"{pkt.user_message} {pkt.system_response} {pkt.intention}"


# ---------------------------------------------------------------------------
# Semantic index entry
# ---------------------------------------------------------------------------

class _IndexEntry:
    __slots__ = ("turn_index", "term_vector", "packet_id")

    def __init__(self, turn_index: int, text: str, packet_id: str) -> None:
        self.turn_index  = turn_index
        self.packet_id   = packet_id
        self.term_vector = _term_freq(_tokenize(text))


# ---------------------------------------------------------------------------
# Context Processor
# ---------------------------------------------------------------------------

class ContextProcessor:
    """
    Maintains MTMM quality via compression, validation, and search indexing.
    """

    def __init__(self, logger: RawInteractionLogger) -> None:
        self._logger = logger
        self._index:  List[_IndexEntry] = []
        self._importance_cache: Dict[int, float] = {}  # turn_index → importance

    # -----------------------------------------------------------------------
    # 1. Compression — called periodically
    # -----------------------------------------------------------------------

    def compress_old_entries(self, current_turn: int, window: int = 10) -> None:
        """
        Entries older than `window` turns get re-compressed.
        High-importance entries are exempted.
        """
        for pkt in self._logger.get_all():
            age = current_turn - pkt.turn_index
            if age <= window:
                continue
            importance = self._importance_cache.get(pkt.turn_index, 0.0)
            self._logger.recompress_entry(pkt.turn_index, importance)

    # -----------------------------------------------------------------------
    # 2. Validation — internal consistency check across MTMM
    # -----------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Cross-check MTMM entries for internal consistency.
        Returns a list of inconsistency descriptions (empty = all good).
        """
        issues: List[str] = []
        entries = self._logger.get_all()

        for i, pkt_a in enumerate(entries):
            for pkt_b in entries[i + 1:]:
                # Check: contradiction count jumped but was not reflected in flags
                if pkt_b.contradictions_detected > pkt_a.contradictions_detected + 3:
                    if "CONTRADICTION_SPIKE" not in pkt_b.flags:
                        issues.append(
                            f"Turn {pkt_b.turn_index}: contradiction spike not flagged"
                        )
                # Check: trust_weight anomaly (sudden drop)
                if pkt_a.trust_weight - pkt_b.trust_weight > 0.5:
                    issues.append(
                        f"Turn {pkt_a.turn_index}→{pkt_b.turn_index}: "
                        f"trust_weight dropped sharply"
                    )
        return issues

    # -----------------------------------------------------------------------
    # 3. Contrast — semantic index
    # -----------------------------------------------------------------------

    def index_packet(self, pkt: MemoryPacket, importance: float = 0.5) -> None:
        """Add a new packet to the semantic index."""
        text = _packet_text(pkt)
        entry = _IndexEntry(pkt.turn_index, text, pkt.packet_id)
        self._index.append(entry)
        self._importance_cache[pkt.turn_index] = importance

    def search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Tuple[float, MemoryPacket]]:
        """
        Search MTMM by semantic similarity.
        Returns list of (score, packet) sorted descending by score.
        """
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, int]] = []

        for entry in self._index:
            sim = _cosine(q_vec, entry.term_vector)
            if sim > 0.0:
                scored.append((sim, entry.turn_index))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, turn_idx in scored[:limit]:
            pkt = self._logger.get_by_turn(turn_idx)
            if pkt is not None:
                results.append((sim, pkt))
        return results
