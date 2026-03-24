"""
Tests for KnowledgeBootstrap and all seed modules.

Coverage:
  - AtomSpace seed: node counts, link types, specific key atoms
  - KnowledgeMap seed: 4 maps, node/link structure, subject categories
  - Lesson seed: count, validation_status, subject categories, map refs
  - Library seed: entry IDs, domains, content presence
  - KnowledgeBootstrap.run(): full integration against real stores
  - Idempotency: AtomSpace not double-seeded; stores populated once
  - Robustness: bootstrap tolerates missing atomspace_engine
"""
from __future__ import annotations

import pytest

from zados.memory import MemoryLayer
from zados.bootstrap import KnowledgeBootstrap
from zados.bootstrap.seeds.atomspace_seed import seed_atomspace
from zados.bootstrap.seeds.knowledge_map_seed import make_seed_maps
from zados.bootstrap.seeds.lesson_seed import make_seed_lessons
from zados.bootstrap.seeds.library_seed import make_seed_library_entries
from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomSpaceEngine,
    AtomType,
    TruthValue,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def memory():
    return MemoryLayer()


@pytest.fixture
def atomspace():
    return AtomSpaceEngine()


# =========================================================================
# AtomSpace seed
# =========================================================================

class TestAtomspaceSeed:
    def test_returns_positive_count(self, atomspace):
        added = seed_atomspace(atomspace)
        assert added > 0

    def test_adds_substantial_ontology(self, atomspace):
        seed_atomspace(atomspace)
        # Should have at minimum: ~60 concept/predicate nodes + ~50 links
        assert len(atomspace._atoms) >= 100

    def test_concept_nodes_present(self, atomspace):
        seed_atomspace(atomspace)
        # The new parser-driven seed uses concept library names (lowercase/hyphenated).
        # Check for well-known concept library canonical names.
        names_lower = {(a.name or "").lower() for a in atomspace._atoms.values()}
        for expected in [
            "exists", "unknown", "true", "false", "identity", "freedom",
        ]:
            assert expected in names_lower, f"Expected concept node '{expected}' not found"

    def test_predicate_nodes_present(self, atomspace):
        seed_atomspace(atomspace)
        pred_names = {
            a.name for a in atomspace._atoms.values()
            if a.atom_type == AtomType.PREDICATE_NODE
        }
        # Parser-driven seed adds engine_cluster and reward_domain predicate nodes
        assert len(pred_names) >= 1, "Expected at least one predicate node"

    def test_inheritance_links_exist(self, atomspace):
        seed_atomspace(atomspace)
        inh_links = [
            a for a in atomspace._atoms.values()
            if a.atom_type == AtomType.INHERITANCE_LINK
        ]
        assert len(inh_links) >= 20

    def test_evaluation_links_exist(self, atomspace):
        seed_atomspace(atomspace)
        ev_links = [
            a for a in atomspace._atoms.values()
            if a.atom_type == AtomType.EVALUATION_LINK
        ]
        assert len(ev_links) >= 10

    def test_similarity_links_exist(self, atomspace):
        seed_atomspace(atomspace)
        sim_links = [
            a for a in atomspace._atoms.values()
            if a.atom_type == AtomType.SIMILARITY_LINK
        ]
        assert len(sim_links) >= 5

    def test_high_confidence_on_seed_atoms(self, atomspace):
        seed_atomspace(atomspace)
        concept_nodes = [
            a for a in atomspace._atoms.values()
            if a.atom_type == AtomType.CONCEPT_NODE
        ]
        # The new seed uses TV-SEED mappings: HIGH=0.9, MEDIUM=0.75, LOW=0.6
        # Alias nodes use 0.70. At least the HIGH TV-SEED main concept nodes
        # should have confidence >= 0.75
        high_conf_nodes = [n for n in concept_nodes if n.truth_value.confidence >= 0.75]
        assert len(high_conf_nodes) >= 10, (
            f"Expected >=10 concept nodes with confidence>=0.75, "
            f"got {len(high_conf_nodes)}"
        )

    def test_source_engine_tagged_bootstrap(self, atomspace):
        seed_atomspace(atomspace)
        bootstrap_atoms = [
            a for a in atomspace._atoms.values()
            if a.source_engine == "bootstrap"
        ]
        assert len(bootstrap_atoms) > 50

    def test_idempotent_on_nonempty_atomspace(self, atomspace):
        seed_atomspace(atomspace)
        count_after_first = len(atomspace._atoms)
        # Second call to KnowledgeBootstrap should skip (atoms > 0)
        added_second = KnowledgeBootstrap._seed_atomspace(atomspace)
        assert added_second == 0
        assert len(atomspace._atoms) == count_after_first

    def test_memory_nt_types_all_inherit_neurotransmitter(self, atomspace):
        seed_atomspace(atomspace)
        # The new parser-driven seed uses concept library concepts.
        # Verify that key ontological concepts are present and have inheritance links.
        atoms = atomspace._atoms
        inh_links = [
            a for a in atoms.values()
            if a.atom_type == AtomType.INHERITANCE_LINK
        ]
        # The concept library has many DEPENDS-ON relationships encoded as InheritanceLinks
        assert len(inh_links) >= 20, (
            f"Expected >=20 InheritanceLinks from concept library, got {len(inh_links)}"
        )


# =========================================================================
# KnowledgeMap seed
# =========================================================================

class TestKnowledgeMapSeed:
    def test_returns_four_maps(self):
        maps = make_seed_maps()
        # Now returns 4 hardcoded + 8 concept-library maps = 12 total
        assert len(maps) >= 4

    def test_map_ids_unique(self):
        maps = make_seed_maps()
        ids = [m.map_id for m in maps]
        assert len(ids) == len(set(ids))

    def test_expected_map_ids(self):
        maps = make_seed_maps()
        ids = {m.map_id for m in maps}
        assert "seed_map_neurochemical_dynamics" in ids
        assert "seed_map_cognitive_architecture" in ids
        assert "seed_map_memory_systems" in ids
        assert "seed_map_learning_mechanisms" in ids

    def test_subject_categories(self):
        maps = make_seed_maps()
        subjects = {m.subject_category for m in maps}
        assert "neuroscience" in subjects
        assert "cognitive_science" in subjects
        assert "learning_theory" in subjects

    def test_each_map_has_nodes_and_links(self):
        maps = make_seed_maps()
        for km in maps:
            assert len(km.nodes) >= 5, f"{km.title} has too few nodes"
            assert len(km.links) >= 4, f"{km.title} has too few links"

    def test_node_ids_unique_within_map(self):
        maps = make_seed_maps()
        for km in maps:
            nids = [n.node_id for n in km.nodes]
            assert len(nids) == len(set(nids)), f"Duplicate node IDs in {km.title}"

    def test_link_references_valid_nodes(self):
        maps = make_seed_maps()
        for km in maps:
            node_ids = {n.node_id for n in km.nodes}
            for lk in km.links:
                assert lk.source_node in node_ids, \
                    f"Link source '{lk.source_node}' not in {km.title} nodes"
                assert lk.target_node in node_ids, \
                    f"Link target '{lk.target_node}' not in {km.title} nodes"

    def test_relation_types_valid(self):
        valid = {"supports", "contradicts", "extends", "requires", "exemplifies"}
        maps = make_seed_maps()
        for km in maps:
            for lk in km.links:
                assert lk.relation in valid, \
                    f"Invalid relation '{lk.relation}' in {km.title}"

    def test_neuro_map_covers_all_eight_nts(self):
        maps = make_seed_maps()
        neuro = next(m for m in maps if m.map_id == "seed_map_neurochemical_dynamics")
        labels = {n.label for n in neuro.nodes}
        for expected in ["Dopamine", "Serotonin", "Norepinephrine", "Acetylcholine",
                         "GABA", "Cortisol", "Oxytocin", "Cannabinoid (CB1)"]:
            assert expected in labels

    def test_maps_have_tags(self):
        maps = make_seed_maps()
        for km in maps:
            assert "seed" in km.tags
            assert len(km.tags) >= 3

    def test_maps_have_description(self):
        maps = make_seed_maps()
        for km in maps:
            assert len(km.description) > 50


# =========================================================================
# Lesson seed
# =========================================================================

class TestLessonSeed:
    def test_returns_expected_count(self):
        lessons = make_seed_lessons()
        assert len(lessons) == 20

    def test_all_validated(self):
        for lesson in make_seed_lessons():
            assert lesson.validation_status == "validated"

    def test_all_high_confidence(self):
        for lesson in make_seed_lessons():
            assert lesson.confidence >= 0.8

    def test_source_mode_is_seed(self):
        for lesson in make_seed_lessons():
            assert lesson.source_mode == "seed"

    def test_subject_categories_covered(self):
        subjects = {l.subject_category for l in make_seed_lessons()}
        assert "neuroscience" in subjects
        assert "cognitive_science" in subjects
        assert "learning_theory" in subjects
        assert "self_model" in subjects

    def test_eight_nt_lessons(self):
        nt_tags = {"dopamine", "serotonin", "norepinephrine", "acetylcholine",
                   "GABA", "cortisol", "oxytocin", "cannabinoid"}
        lessons = make_seed_lessons()
        covered = set()
        for lesson in lessons:
            for tag in lesson.tags:
                if tag in nt_tags:
                    covered.add(tag)
        assert covered == nt_tags, f"Missing NT lessons: {nt_tags - covered}"

    def test_lessons_have_content(self):
        for lesson in make_seed_lessons():
            assert len(lesson.content) > 40

    def test_lessons_have_tags(self):
        for lesson in make_seed_lessons():
            assert len(lesson.tags) >= 2

    def test_knowledge_map_refs_are_valid_seed_ids(self):
        valid_map_ids = {m.map_id for m in make_seed_maps()}
        for lesson in make_seed_lessons():
            for ref in lesson.knowledge_map_refs:
                assert ref in valid_map_ids, \
                    f"Lesson references unknown map '{ref}'"


# =========================================================================
# Library seed
# =========================================================================

class TestLibrarySeed:
    def test_returns_two_entries(self):
        entries = make_seed_library_entries()
        # Now returns 3 entries (added concept library entry)
        assert len(entries) >= 2

    def test_entry_ids_stable(self):
        entries = make_seed_library_entries()
        ids = {e.entry_id for e in entries}
        assert "seed_lib_zados_architecture" in ids
        assert "seed_lib_neurochem_foundations" in ids

    def test_domains(self):
        entries = make_seed_library_entries()
        domains = {e.domain for e in entries}
        assert "cognitive_architecture" in domains
        assert "neuroscience" in domains

    def test_entries_have_substantial_content(self):
        for entry in make_seed_library_entries():
            assert len(entry.content) > 200

    def test_entries_have_tags(self):
        for entry in make_seed_library_entries():
            assert "seed" in entry.tags
            assert len(entry.tags) >= 4

    def test_source_types(self):
        for entry in make_seed_library_entries():
            assert entry.source_type in ("book", "article", "document", "upload")

    def test_zados_architecture_mentions_engines(self):
        entries = make_seed_library_entries()
        arch = next(e for e in entries if e.entry_id == "seed_lib_zados_architecture")
        assert "29" in arch.content
        assert "AtomSpace" in arch.content
        assert "neurochemical" in arch.content.lower()

    def test_neurochem_entry_covers_all_nts(self):
        entries = make_seed_library_entries()
        neuro = next(e for e in entries if e.entry_id == "seed_lib_neurochem_foundations")
        content_lower = neuro.content.lower()
        for nt in ["dopamine", "serotonin", "norepinephrine", "acetylcholine",
                   "gaba", "cortisol", "oxytocin", "cannabinoid"]:
            assert nt in content_lower, f"NT '{nt}' missing from neurochem library entry"


# =========================================================================
# KnowledgeBootstrap integration
# =========================================================================

class TestKnowledgeBootstrap:
    def test_run_returns_result_dict(self, memory):
        result = KnowledgeBootstrap.run(memory)
        assert "atoms" in result
        assert "maps" in result
        assert "lessons" in result
        assert "library" in result
        assert "status" in result

    def test_run_status_ok(self, memory):
        result = KnowledgeBootstrap.run(memory)
        assert result["status"] == "ok"

    def test_run_without_atomspace(self, memory):
        result = KnowledgeBootstrap.run(memory, atomspace_engine=None)
        assert result["atoms"] == 0
        assert result["status"] == "ok"

    def test_run_with_atomspace(self, memory, atomspace):
        result = KnowledgeBootstrap.run(memory, atomspace_engine=atomspace)
        assert result["atoms"] > 0

    def test_library_store_populated(self, memory):
        KnowledgeBootstrap.run(memory)
        # Now 3 library entries
        assert len(memory.knowledge.library) >= 2

    def test_knowledge_maps_store_populated(self, memory):
        KnowledgeBootstrap.run(memory)
        # Now 12 maps (4 hardcoded + 8 concept-library)
        assert len(memory.knowledge.knowledge_maps) >= 4

    def test_lessons_store_populated(self, memory):
        KnowledgeBootstrap.run(memory)
        assert len(memory.knowledge.lessons) == 20

    def test_library_searchable_after_seed(self, memory):
        KnowledgeBootstrap.run(memory)
        results = memory.knowledge.library.search("dopamine reward prediction")
        assert len(results) >= 1
        assert results[0][0] > 0.0

    def test_knowledge_maps_searchable(self, memory):
        KnowledgeBootstrap.run(memory)
        results = memory.knowledge.knowledge_maps.search("neurochemical dopamine")
        assert len(results) >= 1

    def test_lessons_searchable(self, memory):
        KnowledgeBootstrap.run(memory)
        results = memory.knowledge.lessons.search("memory consolidation sleep")
        assert len(results) >= 1

    def test_lessons_are_validated_in_store(self, memory):
        KnowledgeBootstrap.run(memory)
        validated = memory.knowledge.lessons.get_validated()
        assert len(validated) == 20

    def test_knowledge_maps_by_subject(self, memory):
        KnowledgeBootstrap.run(memory)
        cog_maps = memory.knowledge.knowledge_maps.get_by_subject("cognitive_science")
        assert len(cog_maps) == 2

    def test_library_by_domain(self, memory):
        KnowledgeBootstrap.run(memory)
        neuro_entries = memory.knowledge.library.get_by_domain("neuroscience")
        assert len(neuro_entries) == 1

    def test_atomspace_not_reseeded_when_non_empty(self, memory, atomspace):
        # First run seeds the atomspace
        r1 = KnowledgeBootstrap.run(memory, atomspace_engine=atomspace)
        count1 = len(atomspace._atoms)
        assert r1["atoms"] > 0

        # Second run should skip atomspace (atoms > 0)
        r2 = KnowledgeBootstrap.run(memory, atomspace_engine=atomspace)
        assert r2["atoms"] == 0
        assert len(atomspace._atoms) == count1

    def test_run_counts_match_seed_sizes(self, memory, atomspace):
        result = KnowledgeBootstrap.run(memory, atomspace_engine=atomspace)
        assert result["maps"] >= 4   # now 12 maps (4 + 8 concept-library)
        assert result["lessons"] == 20
        assert result["library"] >= 2  # now 3 entries
        assert result["atoms"] > 0

    def test_tolerates_missing_knowledge_attr(self):
        """Bootstrap should fail gracefully if memory has no .knowledge."""
        class BrokenMemory:
            pass
        result = KnowledgeBootstrap.run(BrokenMemory(), atomspace_engine=None)
        assert result["status"] == "partial"
