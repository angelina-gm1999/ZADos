"""
AcademicBufferStore — unsolved academic concepts.

Mirrors UnsolvedConceptsBuffer API but for knowledge-domain-specific
concepts.  Stagnated entries become REM Dream candidates.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class AcademicBufferEntry:
    """Unsolved academic concept awaiting resolution."""
    entry_id:           str        = field(default_factory=lambda: str(uuid.uuid4()))
    concept_formulation: str       = ""
    subject_category:   str        = ""
    source_engine:      str        = ""
    blocking_reason:    str        = ""
    stagnation_cycles:  int        = 0
    resolved:           bool       = False
    resolution_note:    str        = ""
    timestamp:          datetime   = field(default_factory=datetime.utcnow)
    last_checked:       datetime   = field(default_factory=datetime.utcnow)

    def tick_stagnation(self) -> None:
        self.stagnation_cycles += 1
        self.last_checked = datetime.utcnow()

    def is_dream_candidate(self, threshold: int = 5) -> bool:
        return not self.resolved and self.stagnation_cycles >= threshold


class AcademicBufferStore:
    """Buffer for unsolved academic concepts, mirroring UnsolvedConceptsBuffer API."""

    def __init__(self) -> None:
        self._entries: Dict[str, AcademicBufferEntry] = {}

    def add(self, entry: AcademicBufferEntry) -> str:
        self._entries[entry.entry_id] = entry
        return entry.entry_id

    def resolve(self, entry_id: str, note: str = "") -> None:
        if entry_id in self._entries:
            self._entries[entry_id].resolved = True
            self._entries[entry_id].resolution_note = note

    def tick_all(self) -> None:
        for e in self._entries.values():
            if not e.resolved:
                e.tick_stagnation()

    def get_dream_candidates(self, threshold: int = 5) -> List[AcademicBufferEntry]:
        return [e for e in self._entries.values() if e.is_dream_candidate(threshold)]

    def get_by_id(self, entry_id: str) -> Optional[AcademicBufferEntry]:
        return self._entries.get(entry_id)

    def get_all(self) -> List[AcademicBufferEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)
