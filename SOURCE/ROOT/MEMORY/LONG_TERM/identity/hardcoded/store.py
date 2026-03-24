"""
HardcodedStore — immutable baseline identity.

Contains system prompt fragments, axioms, and foundational values that
define ZA-DOS's identity floor.  Content is loaded once at boot and
never modified at runtime.

This store is **read-only**: no ``write()`` or ``search()`` method.
Entries are keyed by a short string ID (e.g. ``"axiom_curiosity"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HardcodedEntry:
    """Single immutable identity fragment."""
    entry_id:  str = ""
    content:   str = ""
    category:  str = ""  # "axiom" | "value" | "constraint" | "system_prompt"
    tags:      List[str] = field(default_factory=list)


class HardcodedStore:
    """
    Read-only store for immutable identity content.

    Populated once via ``load()`` during system boot.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, HardcodedEntry] = {}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def load(self, entries: List[HardcodedEntry]) -> None:
        """Populate the store (idempotent on entry_id)."""
        for e in entries:
            self._storage[e.entry_id] = e

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    def get_by_id(self, entry_id: str) -> Optional[HardcodedEntry]:
        return self._storage.get(entry_id)

    def get_all(self) -> List[HardcodedEntry]:
        return list(self._storage.values())

    def get_by_category(self, category: str) -> List[HardcodedEntry]:
        return [e for e in self._storage.values() if e.category == category]

    def __len__(self) -> int:
        return len(self._storage)
