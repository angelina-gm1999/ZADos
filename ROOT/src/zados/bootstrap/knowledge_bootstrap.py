"""
KnowledgeBootstrap — seeds foundational knowledge into ZADOS stores at boot.

Called once during SessionOrchestrator.open_session() before the first turn.
Because the main stores (Library, KnowledgeMaps, Lessons) are in-memory and
reset each session, this runs every boot to populate the starting knowledge
baseline.  The AtomSpace (E9) is seeded only when it is empty, since its
state can be persisted to CognitoolsDataStore across sessions.

Usage
-----
    from zados.bootstrap import KnowledgeBootstrap

    result = KnowledgeBootstrap.run(memory, atomspace_engine)
    # result = {"atoms": 97, "maps": 4, "lessons": 20, "library": 2, "status": "ok"}

Adding knowledge
----------------
1. Drop a document into ROOT/knowledge_sources/books/.
2. Extract key concepts → add atoms to seeds/atomspace_seed.py.
3. Add or extend a KnowledgeMap in seeds/knowledge_map_seed.py.
4. Add validated lessons to seeds/lesson_seed.py.
5. Add a LibraryEntry to seeds/library_seed.py.
The next session open will load all accumulated seeds automatically.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class KnowledgeBootstrap:
    """Static helper — no instantiation needed."""

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def run(
        cls,
        memory: Any,
        atomspace_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Seed all knowledge stores.

        Parameters
        ----------
        memory : MemoryLayer
            The session's memory object.  Must expose:
            ``memory.ltmm.knowledge.library``      (LibraryStore)
            ``memory.ltmm.knowledge.knowledge_maps``  (KnowledgeMapStore)
            ``memory.ltmm.knowledge.lessons``          (LessonStore)
        atomspace_engine : AtomSpaceEngine, optional
            Engine 9.  If provided and empty, core cognitive ontology is seeded.

        Returns
        -------
        dict  with keys: atoms, maps, lessons, library, status
        """
        result: Dict[str, Any] = {
            "atoms": 0,
            "maps": 0,
            "lessons": 0,
            "library": 0,
            "concept_registry_size": 0,
            "status": "ok",
        }

        try:
            result["library"] = cls._seed_library(memory)
        except Exception:
            log.exception("KnowledgeBootstrap: library seed failed")
            result["status"] = "partial"

        try:
            result["maps"] = cls._seed_knowledge_maps(memory)
        except Exception:
            log.exception("KnowledgeBootstrap: knowledge_maps seed failed")
            result["status"] = "partial"

        try:
            result["lessons"] = cls._seed_lessons(memory)
        except Exception:
            log.exception("KnowledgeBootstrap: lesson seed failed")
            result["status"] = "partial"

        if atomspace_engine is not None:
            try:
                result["atoms"] = cls._seed_atomspace(atomspace_engine)
            except Exception:
                log.exception("KnowledgeBootstrap: atomspace seed failed")
                result["status"] = "partial"

        # Populate concept_registry_size after all seeding is done
        try:
            from zados.bootstrap.concept_type_registry import ConceptTypeRegistry
            reg = ConceptTypeRegistry.instance()
            reg._ensure_loaded()
            result["concept_registry_size"] = len(reg.get_all())
        except Exception:
            result["concept_registry_size"] = 0

        log.info(
            "KnowledgeBootstrap complete — atoms=%d maps=%d lessons=%d library=%d "
            "concept_registry=%d [%s]",
            result["atoms"], result["maps"], result["lessons"],
            result["library"], result["concept_registry_size"], result["status"],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal seeders                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _seed_library(memory: Any) -> int:
        from zados.bootstrap.seeds.library_seed import make_seed_library_entries
        from zados.bootstrap.seeds.identity_library_seed import make_identity_library_entries
        store = memory.knowledge.library
        entries = make_seed_library_entries() + make_identity_library_entries()
        for entry in entries:
            store.write(entry)
        return len(entries)

    @staticmethod
    def _seed_knowledge_maps(memory: Any) -> int:
        from zados.bootstrap.seeds.knowledge_map_seed import make_seed_maps
        store = memory.knowledge.knowledge_maps
        maps = make_seed_maps()
        for km in maps:
            store.write(km)
        return len(maps)

    @staticmethod
    def _seed_lessons(memory: Any) -> int:
        from zados.bootstrap.seeds.lesson_seed import make_seed_lessons
        store = memory.knowledge.lessons
        lessons = make_seed_lessons()
        for lesson in lessons:
            store.write(lesson)
        return len(lessons)

    @staticmethod
    def _seed_atomspace(engine: Any) -> int:
        """Seed only if the AtomSpace is empty (avoids double-seeding on re-open)."""
        if len(engine._atoms) > 0:
            log.debug("KnowledgeBootstrap: AtomSpace non-empty (%d atoms) — skipping seed",
                      len(engine._atoms))
            return 0
        from zados.bootstrap.seeds.atomspace_seed import seed_atomspace
        added = seed_atomspace(engine)
        log.info("KnowledgeBootstrap: AtomSpace seeded with %d atoms", added)
        return added
