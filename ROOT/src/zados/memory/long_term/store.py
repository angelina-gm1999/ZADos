"""
LTMM persistent storage.

Backed by an in-memory dict (Phase 3 spec: "SQLite or embedded vector DB").
The architecture is backend-agnostic: the _storage dict can be swapped for
any KV / vector store without changing callers.

Each entry is stored as a LTMMEntry which wraps the MemoryPacket and adds
LTMM-specific bookkeeping (relevance score, retrieval count, granularity, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from zados.memory.types import CompressionLevel, MemoryPacket
from zados.memory.long_term.search_utils import tokenize as _tokenize, term_freq as _term_freq, cosine as _cosine


# ---------------------------------------------------------------------------
# Granularity levels (§3 Semantic & Symbolic Scalability Filter)
# ---------------------------------------------------------------------------

class Granularity(str):
    VERBATIM = "verbatim"
    SEMANTIC = "semantic"
    SYMBOLIC = "symbolic"


# ---------------------------------------------------------------------------
# LTMM entry wrapper
# ---------------------------------------------------------------------------

@dataclass
class LTMMEntry:
    packet:              MemoryPacket
    granularity:         str          = Granularity.SEMANTIC
    relevance_score:     float        = 1.0       # decays over time
    retrieval_count:     int          = 0
    last_accessed:       datetime     = field(default_factory=datetime.utcnow)
    utility_score:       float        = 0.5       # did this help when retrieved?
    cold_storage:        bool         = False
    identity_relevant:   bool         = False     # never demoted if True

    def touch(self) -> None:
        self.retrieval_count += 1
        self.last_accessed = datetime.utcnow()


def _entry_text(e: LTMMEntry) -> str:
    p = e.packet
    return f"{p.user_message} {p.system_response} {p.intention}"


# ---------------------------------------------------------------------------
# LTMM Store
# ---------------------------------------------------------------------------

_THETA_COLD  = 0.15   # below this → cold storage
_THETA_PURGE = 0.05   # below this (after extended cold) → purge candidate


class LTMMStore:
    """
    Persistent long-term memory store.

    Entries are keyed by packet_id.  The semantic index is an in-memory
    term-vector dict rebuilt lazily on write.  For production use, swap
    _storage + _index for SQLite + a vector search backend.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, LTMMEntry]         = {}   # packet_id → entry
        self._index:   Dict[str, Dict[str, float]]  = {}   # packet_id → term vector

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def write(self, entry: LTMMEntry) -> None:
        pid = entry.packet.packet_id
        self._storage[pid] = entry
        self._index[pid]   = _term_freq(_tokenize(_entry_text(entry)))

    def update_utility(self, packet_id: str, utility_delta: float) -> None:
        """Adjust utility score after a retrieval proved useful/not."""
        if packet_id in self._storage:
            e = self._storage[packet_id]
            e.utility_score = max(0.0, min(1.0, e.utility_score + utility_delta))

    # -----------------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
        include_cold: bool = False,
        internal: bool = False,
    ) -> List[Tuple[float, LTMMEntry]]:
        """Semantic search; returns (score, entry) sorted descending.

        Parameters
        ----------
        query_text : str
            Query to match against stored entries.
        limit : int
            Maximum number of results.
        include_cold : bool
            If True, include cold-storage entries.
        internal : bool
            If True, skip ``entry.touch()`` so that internal
            housekeeping queries (e.g. from MemoryContrast) do not
            inflate retrieval counts and prevent proper memory decay.
        """
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []

        for pid, t_vec in self._index.items():
            entry = self._storage[pid]
            if entry.cold_storage and not include_cold:
                continue
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, pid))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, pid in scored[:limit]:
            entry = self._storage[pid]
            if not internal:
                entry.touch()
            results.append((sim, entry))
        return results

    def get_by_id(self, packet_id: str) -> Optional[LTMMEntry]:
        return self._storage.get(packet_id)

    def get_all(self) -> List[LTMMEntry]:
        return list(self._storage.values())

    def get_active(self) -> List[LTMMEntry]:
        return [e for e in self._storage.values() if not e.cold_storage]

    # -----------------------------------------------------------------------
    # Cold storage / purge
    # -----------------------------------------------------------------------

    def demote_to_cold(self, packet_id: str) -> None:
        if packet_id in self._storage:
            entry = self._storage[packet_id]
            if entry.identity_relevant:
                return  # identity-relevant entries are never demoted
            entry.cold_storage = True

    def purge(self, packet_id: str) -> None:
        self._storage.pop(packet_id, None)
        self._index.pop(packet_id, None)

    def __len__(self) -> int:
        return len(self._storage)
