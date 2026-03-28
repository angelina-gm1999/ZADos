"""
Tests for concept library parser, concept type registry, updated seeds,
and full bootstrap integration with concept library.

Covers:
  - TestConceptLibraryParser (20 tests)
  - TestConceptTypeRegistry (15 tests)
  - TestUpdatedAtomspaceSeed (10 tests)
  - TestUpdatedKnowledgeMapSeed (10 tests)
  - TestUpdatedLibrarySeed (5 tests)
  - TestFullBootstrapWithConceptLibrary (5 tests)
"""
from __future__ import annotations

import pytest

from zados.bootstrap.concept_library_parser import (
    AtomLinkSpec,
    ConceptEntry,
    parse_concept_library,
    get_default_library_path,
)
from zados.bootstrap.concept_type_registry import ConceptTypeRegistry
from zados.bootstrap.seeds.atomspace_seed import seed_atomspace
from zados.bootstrap.seeds.knowledge_map_seed import make_seed_maps
from zados.bootstrap.seeds.library_seed import make_seed_library_entries
from zados.bootstrap import KnowledgeBootstrap
from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomSpaceEngine,
    AtomType,
    TruthValue,
)
from zados.memory import MemoryLayer


# =========================================================================
# Shared fixtures
# =========================================================================

@pytest.fixture(scope="module")
def entries():
    """Parse the concept library once for the whole module."""
    path = get_default_library_path()
    return parse_concept_library(path)


@pytest.fixture(scope="module")
def reg():
    """Return the singleton registry, ensuring it is loaded."""
    r = ConceptTypeRegistry.instance()
    r._ensure_loaded()
    return r


@pytest.fixture
def atomspace():
    return AtomSpaceEngine()


@pytest.fixture
def seeded_atomspace():
    engine = AtomSpaceEngine()
    seed_atomspace(engine)
    return engine


@pytest.fixture
def memory():
    return MemoryLayer()


# =========================================================================
# TestConceptLibraryParser
# =========================================================================

class TestConceptLibraryParser:
    def test_parse_returns_nonempty_list(self, entries):
        assert len(entries) > 0

    def test_parse_returns_minimum_count(self, entries):
        assert len(entries) >= 150

    def test_concept_names_unique(self, entries):
        names = [e.name.lower() for e in entries]
        assert len(names) == len(set(names)), "Duplicate concept names found"

    def test_known_concept_exists_exists(self, entries):
        names = {e.name for e in entries}
        assert "exists" in names

    def test_known_concept_layer(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert ex.layer == "1.1"

    def test_known_concept_tv_seed(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert ex.tv_seed == "HIGH"

    def test_known_concept_engine_relevance(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert "knowledge_substrate" in ex.engine_relevance

    def test_known_concept_reward_domain(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert "logic" in ex.reward_domains

    def test_known_concept_depends_on_empty(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert ex.depends_on == [], f"Expected empty depends_on, got {ex.depends_on}"

    def test_concept_with_deps(self, entries):
        unk = next(e for e in entries if e.name == "unknown")
        assert "exists" in unk.depends_on, f"Expected 'exists' in depends_on, got {unk.depends_on}"

    def test_atom_links_parsed(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert len(ex.atom_links) >= 2, f"Expected >=2 atom links, got {ex.atom_links}"

    def test_atom_link_types_valid(self, entries):
        valid_types = {
            "InheritanceLink", "SimilarityLink", "EvaluationLink",
            "ImplicationLink", "HebbianLink", "NotLink", "ListLink",
            "AndLink", "OrLink",
        }
        for entry in entries:
            for spec in entry.atom_links:
                assert spec.link_type in valid_types, (
                    f"Unknown link type '{spec.link_type}' in concept '{entry.name}'"
                )

    def test_layer_groups_correct(self, entries):
        l11 = [e for e in entries if e.layer == "1.1"]
        for e in l11:
            assert e.layer_group == "1", (
                f"Expected layer_group '1' for layer '1.1', got '{e.layer_group}'"
            )

    def test_high_priority_concepts(self, entries):
        high = [e for e in entries if e.tv_seed == "HIGH"]
        assert len(high) >= 100, f"Expected >=100 HIGH concepts, got {len(high)}"

    def test_all_concepts_have_definition(self, entries):
        for entry in entries:
            assert entry.definition.strip(), (
                f"Concept '{entry.name}' has empty definition"
            )

    def test_all_concepts_have_layer(self, entries):
        for entry in entries:
            assert entry.layer.strip(), (
                f"Concept '{entry.name}' has empty layer"
            )

    def test_layer_1_1_concepts(self, entries):
        l11 = [e for e in entries if e.layer == "1.1"]
        assert len(l11) >= 15, f"Expected >=15 layer 1.1 concepts, got {len(l11)}"

    def test_layer_3_concepts_present(self, entries):
        l3 = [e for e in entries if e.layer_group == "3"]
        assert len(l3) >= 1, "Expected at least 1 layer 3 concept"

    def test_aliases_parsed(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert "existence" in ex.aliases, (
            f"Expected 'existence' in aliases, got {ex.aliases}"
        )

    def test_conceptual_scope_nonempty(self, entries):
        ex = next(e for e in entries if e.name == "exists")
        assert len(ex.conceptual_scope.strip()) > 0, "Expected non-empty conceptual_scope"


# =========================================================================
# TestConceptTypeRegistry
# =========================================================================

class TestConceptTypeRegistry:
    def test_registry_instance_singleton(self):
        r1 = ConceptTypeRegistry.instance()
        r2 = ConceptTypeRegistry.instance()
        assert r1 is r2

    def test_registry_loads_from_file(self, reg):
        all_entries = reg.get_all()
        assert len(all_entries) > 150

    def test_get_concept_by_name(self, reg):
        entry = reg.get_concept("exists")
        assert entry is not None
        assert isinstance(entry, ConceptEntry)
        assert entry.name == "exists"

    def test_get_concept_case_insensitive(self, reg):
        entry = reg.get_concept("EXISTS")
        assert entry is not None
        assert entry.name == "exists"

    def test_get_concept_by_alias(self, reg):
        entry = reg.get_concept("existence")
        assert entry is not None
        assert entry.name == "exists", (
            f"Expected 'exists' via alias 'existence', got '{entry.name}'"
        )

    def test_get_concepts_for_cluster(self, reg):
        results = reg.get_concepts_for_cluster("detection")
        assert len(results) >= 5, (
            f"Expected >=5 'detection' concepts, got {len(results)}"
        )

    def test_get_concepts_for_reward_domain(self, reg):
        results = reg.get_concepts_for_reward_domain("logic")
        assert len(results) >= 50, (
            f"Expected >=50 'logic' reward domain concepts, got {len(results)}"
        )

    def test_get_concepts_for_layer(self, reg):
        results = reg.get_concepts_for_layer("1.1")
        assert len(results) >= 15, (
            f"Expected >=15 layer 1.1 concepts, got {len(results)}"
        )

    def test_get_high_priority(self, reg):
        results = reg.get_high_priority()
        assert len(results) >= 100, (
            f"Expected >=100 HIGH priority concepts, got {len(results)}"
        )

    def test_dependency_chain_root(self, reg):
        chain = reg.dependency_chain("exists")
        assert chain == [], f"Root concept 'exists' should have empty chain, got {chain}"

    def test_dependency_chain_depth(self, reg):
        chain = reg.dependency_chain("unknown")
        assert "exists" in chain, (
            f"Expected 'exists' in dependency chain of 'unknown', got {chain}"
        )

    def test_concept_names_list(self, reg):
        names = reg.concept_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_to_tag_normalizes(self, reg):
        result = reg.to_tag("Existence")
        assert result == "exists", f"Expected 'exists', got '{result}'"

    def test_to_tag_unknown(self, reg):
        result = reg.to_tag("xyzzy_not_a_concept")
        assert result is None

    def test_get_all_returns_list(self, reg):
        all_entries = reg.get_all()
        assert isinstance(all_entries, list)
        assert all(isinstance(e, ConceptEntry) for e in all_entries)


# =========================================================================
# TestUpdatedAtomspaceSeed
# =========================================================================

class TestUpdatedAtomspaceSeed:
    def test_seed_count_much_larger(self, atomspace):
        added = seed_atomspace(atomspace)
        assert added >= 500, f"Expected >=500 atoms, got {added}"

    def test_concept_nodes_include_exists(self, seeded_atomspace):
        names = {a.name for a in seeded_atomspace._atoms.values() if a.name}
        assert "exists" in names, "'exists' ConceptNode not found in AtomSpace"

    def test_concept_nodes_include_freedom(self, seeded_atomspace):
        names = {a.name for a in seeded_atomspace._atoms.values() if a.name}
        assert "freedom" in names, "'freedom' ConceptNode not found in AtomSpace"

    def test_cluster_evaluation_links(self, seeded_atomspace):
        # Check that cluster concept nodes exist (e.g. cluster_detection)
        cluster_nodes = [
            a for a in seeded_atomspace._atoms.values()
            if a.name and a.name.startswith("cluster_")
        ]
        assert len(cluster_nodes) >= 1, "No cluster_* concept nodes found"
        # Also check eval links exist
        eval_links = [
            a for a in seeded_atomspace._atoms.values()
            if a.atom_type == AtomType.EVALUATION_LINK
        ]
        assert len(eval_links) >= 1, "No EvaluationLinks found for cluster tags"

    def test_reward_domain_evaluation_links(self, seeded_atomspace):
        domain_nodes = [
            a for a in seeded_atomspace._atoms.values()
            if a.name and a.name.startswith("domain_")
        ]
        assert len(domain_nodes) >= 1, "No domain_* concept nodes found"

    def test_depends_on_links_for_unknown(self, seeded_atomspace):
        # 'unknown' depends-on 'exists' → should have InheritanceLink between them
        atoms = seeded_atomspace._atoms
        unknown_ids = {
            aid for aid, a in atoms.items()
            if a.name and a.name.lower() == "unknown"
        }
        exists_ids = {
            aid for aid, a in atoms.items()
            if a.name and a.name.lower() == "exists"
        }
        assert unknown_ids, "'unknown' node not found"
        assert exists_ids, "'exists' node not found"

        inh_links = [
            a for a in atoms.values()
            if a.atom_type == AtomType.INHERITANCE_LINK
            and len(a.outgoing) >= 2
            and a.outgoing[0] in unknown_ids
            and a.outgoing[1] in exists_ids
        ]
        assert len(inh_links) >= 1, (
            "Expected InheritanceLink from 'unknown' to 'exists'"
        )

    def test_all_atoms_tagged_bootstrap(self, seeded_atomspace):
        atoms = list(seeded_atomspace._atoms.values())
        non_bootstrap = [a for a in atoms if a.source_engine != "bootstrap"]
        assert len(non_bootstrap) == 0, (
            f"{len(non_bootstrap)} atoms without source_engine='bootstrap'"
        )

    def test_idempotent_no_reseed(self, seeded_atomspace):
        count_before = len(seeded_atomspace._atoms)
        added_second = seed_atomspace(seeded_atomspace)
        assert added_second == 0, f"Expected 0 atoms on second seed, got {added_second}"
        assert len(seeded_atomspace._atoms) == count_before

    def test_tv_high_concepts_have_high_confidence(self, seeded_atomspace):
        # Concept nodes for HIGH TV-SEED entries should have confidence >= 0.85
        # (the parser maps HIGH → TruthValue(0.90, 0.90))
        # We check a few known HIGH concepts: "exists", "unknown"
        atoms = seeded_atomspace._atoms
        for expected_name in ["exists", "unknown"]:
            node = next(
                (a for a in atoms.values()
                 if a.atom_type == AtomType.CONCEPT_NODE
                 and a.name and a.name.lower() == expected_name),
                None,
            )
            assert node is not None, f"'{expected_name}' node not found"
            assert node.truth_value.confidence >= 0.85, (
                f"'{expected_name}' should have confidence >= 0.85, "
                f"got {node.truth_value.confidence}"
            )

    def test_tv_medium_concepts_have_medium_confidence(self, seeded_atomspace):
        # At least some alias nodes use _TV_ALIAS = TruthValue(0.75, 0.70)
        # Look for any node with confidence in [0.70, 0.90)
        atoms = seeded_atomspace._atoms
        medium_nodes = [
            a for a in atoms.values()
            if a.atom_type == AtomType.CONCEPT_NODE
            and 0.70 <= a.truth_value.confidence < 0.90
        ]
        assert len(medium_nodes) >= 1, (
            "Expected at least 1 concept node with medium confidence (0.70-0.90)"
        )


# =========================================================================
# TestUpdatedKnowledgeMapSeed
# =========================================================================

class TestUpdatedKnowledgeMapSeed:
    def test_returns_eight_or_more_maps(self):
        maps = make_seed_maps()
        assert len(maps) >= 8, f"Expected >=8 maps, got {len(maps)}"

    def test_layer_1_1_map_present(self):
        maps = make_seed_maps()
        ids = {m.map_id for m in maps}
        assert "layer_1_1_concepts" in ids, (
            f"'layer_1_1_concepts' not in map IDs: {ids}"
        )

    def test_layer_2_map_present(self):
        maps = make_seed_maps()
        ids = {m.map_id for m in maps}
        layer2_maps = [mid for mid in ids if "layer_2" in mid]
        assert len(layer2_maps) >= 1, (
            f"No map ID containing 'layer_2' found in: {ids}"
        )

    def test_layer_3_map_present(self):
        maps = make_seed_maps()
        ids = {m.map_id for m in maps}
        layer3_maps = [mid for mid in ids if "layer_3" in mid]
        assert len(layer3_maps) >= 1, (
            f"No map ID containing 'layer_3' found in: {ids}"
        )

    def test_each_map_has_nodes(self):
        maps = make_seed_maps()
        for km in maps:
            assert len(km.nodes) >= 3, (
                f"Map '{km.map_id}' has only {len(km.nodes)} nodes (expected >=3)"
            )

    def test_node_ids_unique_within_map(self):
        maps = make_seed_maps()
        for km in maps:
            nids = [n.node_id for n in km.nodes]
            assert len(nids) == len(set(nids)), (
                f"Duplicate node IDs in map '{km.map_id}'"
            )

    def test_layer_1_1_map_has_exists_node(self):
        maps = make_seed_maps()
        l11 = next((m for m in maps if m.map_id == "layer_1_1_concepts"), None)
        assert l11 is not None, "'layer_1_1_concepts' map not found"
        labels = {n.label for n in l11.nodes}
        assert "exists" in labels, (
            f"'exists' not in layer_1_1 nodes: {labels}"
        )

    def test_map_tags_include_concept_library(self):
        maps = make_seed_maps()
        concept_lib_maps = [
            m for m in maps if m.map_id.startswith("layer_")
        ]
        assert len(concept_lib_maps) >= 1
        for km in concept_lib_maps:
            assert "concept_library" in km.tags, (
                f"'concept_library' not in tags of map '{km.map_id}': {km.tags}"
            )

    def test_subject_categories_ontology(self):
        maps = make_seed_maps()
        cats = {m.subject_category for m in maps}
        assert "ontology" in cats, (
            f"Expected 'ontology' subject_category in maps. Found: {cats}"
        )

    def test_link_references_valid_nodes(self):
        maps = make_seed_maps()
        for km in maps:
            node_ids = {n.node_id for n in km.nodes}
            for lk in km.links:
                assert lk.source_node in node_ids, (
                    f"Link source '{lk.source_node}' not in nodes of '{km.map_id}'"
                )
                assert lk.target_node in node_ids, (
                    f"Link target '{lk.target_node}' not in nodes of '{km.map_id}'"
                )


# =========================================================================
# TestUpdatedLibrarySeed
# =========================================================================

class TestUpdatedLibrarySeed:
    def test_returns_three_entries(self):
        entries = make_seed_library_entries()
        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"

    def test_concept_library_entry_present(self):
        entries = make_seed_library_entries()
        ids = {e.entry_id for e in entries}
        assert "seed_lib_concept_library" in ids, (
            f"'seed_lib_concept_library' not in entry IDs: {ids}"
        )

    def test_concept_library_domain(self):
        entries = make_seed_library_entries()
        entry = next(e for e in entries if e.entry_id == "seed_lib_concept_library")
        assert entry.domain == "ontology"

    def test_concept_library_content_mentions_layers(self):
        entries = make_seed_library_entries()
        entry = next(e for e in entries if e.entry_id == "seed_lib_concept_library")
        content_lower = entry.content.lower()
        assert "layer" in content_lower, (
            "Expected 'layer' to appear in concept library entry content"
        )

    def test_concept_library_tags(self):
        entries = make_seed_library_entries()
        entry = next(e for e in entries if e.entry_id == "seed_lib_concept_library")
        assert "concept_library" in entry.tags, (
            f"'concept_library' not in tags: {entry.tags}"
        )


# =========================================================================
# TestFullBootstrapWithConceptLibrary
# =========================================================================

class TestFullBootstrapWithConceptLibrary:
    def test_bootstrap_run_includes_registry_size(self, memory):
        result = KnowledgeBootstrap.run(memory)
        assert "concept_registry_size" in result, (
            f"'concept_registry_size' not in result: {list(result.keys())}"
        )
        assert result["concept_registry_size"] > 0, (
            f"Expected concept_registry_size > 0, got {result['concept_registry_size']}"
        )

    def test_library_has_three_entries(self, memory):
        KnowledgeBootstrap.run(memory)
        assert len(memory.knowledge.library) == 3, (
            f"Expected 3 library entries, got {len(memory.knowledge.library)}"
        )

    def test_knowledge_maps_has_eight_plus(self, memory):
        KnowledgeBootstrap.run(memory)
        assert len(memory.knowledge.knowledge_maps) >= 8, (
            f"Expected >=8 knowledge maps, "
            f"got {len(memory.knowledge.knowledge_maps)}"
        )

    def test_concept_library_searchable(self, memory):
        KnowledgeBootstrap.run(memory)
        results = memory.knowledge.library.search("ontological primitive")
        assert len(results) >= 1, (
            "Library search for 'ontological primitive' returned no results"
        )

    def test_atomspace_seeded_with_concept_library_concepts(self, memory):
        atomspace = AtomSpaceEngine()
        KnowledgeBootstrap.run(memory, atomspace_engine=atomspace)
        assert len(atomspace._atoms) >= 500, (
            f"Expected >=500 atoms after bootstrap, got {len(atomspace._atoms)}"
        )
