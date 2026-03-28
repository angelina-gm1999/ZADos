"""
concept_type_registry.py
========================

ConceptTypeRegistry — queryable type system for ZADOS cognitive engines.

Every engine cluster can query which concepts from the ZA-DOS concept library
are relevant to it, get the full ConceptEntry, check dependency chains,
and retrieve atom-link specifications for AtomSpace operations.

Usage (in any engine)::

    from zados.bootstrap.concept_type_registry import ConceptTypeRegistry
    registry = ConceptTypeRegistry()  # lazy-loads on first call
    concepts = registry.get_concepts_for_cluster("detection")
    entry = registry.get_concept("contradiction")

Or use the module-level singleton::

    from zados.bootstrap.concept_type_registry import registry
    entry = registry.get_concept("exists")
"""
from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional

from zados.bootstrap.concept_library_parser import ConceptEntry


class ConceptTypeRegistry:
    """Singleton-style lazy registry. Parses the concept library on first access."""

    _instance: Optional["ConceptTypeRegistry"] = None
    _entries: Optional[List[ConceptEntry]] = None

    # -------------------------------------------------------------------
    # Singleton constructor
    # -------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "ConceptTypeRegistry":
        """Return the shared singleton instance (created on first call)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -------------------------------------------------------------------
    # Lazy loading
    # -------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self.__class__._entries is None:
            from zados.bootstrap.concept_library_parser import (
                parse_concept_library,
                get_default_library_path,
            )
            path = get_default_library_path()
            if os.path.exists(path):
                self.__class__._entries = parse_concept_library(path)
            else:
                self.__class__._entries = []

        # Rebuild indexes if not yet built or if entries were reset
        if not hasattr(self, "_by_name") or self._by_name is None:
            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Build lookup indexes after loading."""
        self._by_name: Dict[str, ConceptEntry] = {}
        self._by_alias: Dict[str, ConceptEntry] = {}

        for entry in (self._entries or []):
            key = entry.name.lower()
            self._by_name[key] = entry
            for alias in entry.aliases:
                alias_key = alias.lower().strip()
                if alias_key:
                    self._by_alias.setdefault(alias_key, entry)

    # -------------------------------------------------------------------
    # Core queries
    # -------------------------------------------------------------------

    def get_all(self) -> List[ConceptEntry]:
        """Return all parsed ConceptEntry objects."""
        self._ensure_loaded()
        return list(self._entries or [])

    def get_concept(self, name: str) -> Optional[ConceptEntry]:
        """Case-insensitive lookup by name or alias.

        Parameters
        ----------
        name : str
            The concept name or an alias.

        Returns
        -------
        ConceptEntry or None
        """
        self._ensure_loaded()
        key = name.lower().strip()
        entry = self._by_name.get(key)
        if entry is not None:
            return entry
        return self._by_alias.get(key)

    def get_concepts_for_cluster(self, cluster: str) -> List[ConceptEntry]:
        """Return all concepts that list *cluster* in ENGINE-RELEVANCE.

        Parameters
        ----------
        cluster : str
            Engine cluster name, e.g. ``"detection"``, ``"knowledge_substrate"``.
        """
        self._ensure_loaded()
        cluster_lower = cluster.lower()
        return [
            e for e in (self._entries or [])
            if any(c.lower() == cluster_lower for c in e.engine_relevance)
        ]

    def get_concepts_for_reward_domain(self, domain: str) -> List[ConceptEntry]:
        """Return all concepts that list *domain* in REWARD-DOMAIN.

        Parameters
        ----------
        domain : str
            One of ``ethics``, ``logic``, ``innovation``, ``human_attunement``.
        """
        self._ensure_loaded()
        domain_lower = domain.lower()
        return [
            e for e in (self._entries or [])
            if any(d.lower() == domain_lower for d in e.reward_domains)
        ]

    def get_concepts_for_layer(self, layer: str) -> List[ConceptEntry]:
        """Return all concepts in a given layer.

        Accepts either an exact layer (e.g. ``"1.1"``) or a layer group
        (e.g. ``"1"`` returns all 1.x concepts).

        Parameters
        ----------
        layer : str
            Layer code or group.
        """
        self._ensure_loaded()
        layer_stripped = layer.strip()
        if "." in layer_stripped:
            # Exact layer match
            return [e for e in (self._entries or []) if e.layer == layer_stripped]
        else:
            # Group match (integer prefix)
            return [e for e in (self._entries or []) if e.layer_group == layer_stripped]

    def get_high_priority(self) -> List[ConceptEntry]:
        """Return all concepts with TV-SEED == HIGH."""
        self._ensure_loaded()
        return [e for e in (self._entries or []) if e.tv_seed == "HIGH"]

    def dependency_chain(self, concept_name: str) -> List[str]:
        """BFS from *concept_name* back through DEPENDS-ON to root concepts.

        Returns a flat list of all concept names reachable via DEPENDS-ON,
        NOT including *concept_name* itself.  Root concepts (no depends_on)
        return an empty list.

        Parameters
        ----------
        concept_name : str
            Starting concept (case-insensitive).
        """
        self._ensure_loaded()
        start = self.get_concept(concept_name)
        if start is None:
            return []

        visited: set[str] = set()
        queue: deque[str] = deque(start.depends_on)
        result: List[str] = []

        while queue:
            dep_name = queue.popleft()
            dep_key = dep_name.lower()
            if dep_key in visited:
                continue
            visited.add(dep_key)
            result.append(dep_name)
            dep_entry = self.get_concept(dep_name)
            if dep_entry is not None:
                for grandparent in dep_entry.depends_on:
                    if grandparent.lower() not in visited:
                        queue.append(grandparent)

        return result

    def concept_names(self) -> List[str]:
        """Return all concept names (canonical form, as found in CONCEPT field)."""
        self._ensure_loaded()
        return [e.name for e in (self._entries or [])]

    def to_tag(self, concept_name: str) -> Optional[str]:
        """Normalize a name or alias to the canonical concept name tag.

        Returns None if no matching concept is found.

        Parameters
        ----------
        concept_name : str
            Any casing of a concept name or alias.
        """
        entry = self.get_concept(concept_name)
        return entry.name if entry is not None else None


# ---------------------------------------------------------------------------
# Module-level singleton — import directly for convenience
# ---------------------------------------------------------------------------

registry: ConceptTypeRegistry = ConceptTypeRegistry.instance()
