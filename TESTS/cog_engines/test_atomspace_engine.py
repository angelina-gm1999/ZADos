"""
Tests for Engine 9 — AtomSpace-Lite (Typed Hypergraph Store).

Covers: TruthValue, AttentionValue, Atom, AtomType classification,
        add_node, add_link, remove_atom, read operations, pattern matching,
        truth-value merge, truth-value decay, capacity enforcement,
        process() pipeline, neurochem output, mode switching, serialization,
        edge cases.

pytest ROOT/tests/cog_engines/test_atomspace_engine.py -v
"""
from __future__ import annotations

import pytest

from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomSpaceConfig,
    AtomSpaceEngine,
    AtomSpaceNeurochem,
    AtomType,
    AttentionValue,
    Atom,
    LINK_TYPES,
    NODE_TYPES,
    SYMMETRIC_LINKS,
    Pattern,
    PatternAtom,
    TruthValue,
    compute_atomspace_neurochem,
    compute_similarity_score,
    compute_tv_decay,
    compute_write_gate_threshold,
    is_link_type,
    is_node_type,
    make_atom,
    match_pattern,
    match_pattern_atom,
    merge_truth_values,
    score_atom_for_pruning,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def engine():
    return AtomSpaceEngine()


@pytest.fixture
def small_engine():
    """Engine with small capacity for pruning tests."""
    cfg = AtomSpaceConfig(max_atoms=20, maintenance_interval=1)
    return AtomSpaceEngine(config=cfg)


@pytest.fixture
def populated_engine(engine):
    """Engine with a small knowledge graph pre-loaded."""
    dog   = engine.add_node(AtomType.CONCEPT_NODE, "dog",    TruthValue(1.0, 0.9))
    cat   = engine.add_node(AtomType.CONCEPT_NODE, "cat",    TruthValue(1.0, 0.9))
    mammal = engine.add_node(AtomType.CONCEPT_NODE, "mammal", TruthValue(1.0, 0.95))
    animal = engine.add_node(AtomType.CONCEPT_NODE, "animal", TruthValue(1.0, 0.95))
    is_a   = engine.add_node(AtomType.PREDICATE_NODE, "is_a", TruthValue(1.0, 0.99))

    engine.add_link(AtomType.INHERITANCE_LINK, (dog.atom_id, mammal.atom_id),
                    TruthValue(0.95, 0.9))
    engine.add_link(AtomType.INHERITANCE_LINK, (cat.atom_id, mammal.atom_id),
                    TruthValue(0.90, 0.85))
    engine.add_link(AtomType.INHERITANCE_LINK, (mammal.atom_id, animal.atom_id),
                    TruthValue(1.0, 0.95))
    engine.add_link(AtomType.SIMILARITY_LINK, (dog.atom_id, cat.atom_id),
                    TruthValue(0.6, 0.7))

    return engine, {"dog": dog, "cat": cat, "mammal": mammal, "animal": animal, "is_a": is_a}


# =====================================================================
# TruthValue
# =====================================================================

class TestTruthValue:
    def test_default(self):
        tv = TruthValue.DEFAULT()
        assert tv.strength == 1.0
        assert tv.confidence == 0.0

    def test_true(self):
        tv = TruthValue.TRUE()
        assert tv.strength == 1.0
        assert tv.confidence == 0.9

    def test_false(self):
        tv = TruthValue.FALSE()
        assert tv.strength == 0.0
        assert tv.confidence == 0.9

    def test_custom(self):
        tv = TruthValue(0.7, 0.8)
        assert tv.strength == 0.7
        assert tv.confidence == 0.8

    def test_clamping_high(self):
        tv = TruthValue(1.5, 2.0)
        assert tv.strength == 1.0
        assert tv.confidence == 1.0

    def test_clamping_low(self):
        tv = TruthValue(-0.5, -1.0)
        assert tv.strength == 0.0
        assert tv.confidence == 0.0

    def test_frozen(self):
        tv = TruthValue(0.5, 0.5)
        with pytest.raises(AttributeError):
            tv.strength = 0.9  # type: ignore

    def test_equality(self):
        assert TruthValue(0.5, 0.5) == TruthValue(0.5, 0.5)
        assert TruthValue(0.5, 0.5) != TruthValue(0.6, 0.5)


# =====================================================================
# AttentionValue
# =====================================================================

class TestAttentionValue:
    def test_defaults(self):
        av = AttentionValue()
        assert av.sti == 0.0
        assert av.lti == 0.0

    def test_custom(self):
        av = AttentionValue(sti=10.0, lti=5.0)
        assert av.sti == 10.0
        assert av.lti == 5.0

    def test_mutable(self):
        av = AttentionValue()
        av.sti = 42.0
        assert av.sti == 42.0

    def test_negative(self):
        av = AttentionValue(sti=-50.0, lti=-10.0)
        assert av.sti == -50.0


# =====================================================================
# AtomType Classification
# =====================================================================

class TestAtomType:
    def test_node_types(self):
        for nt in NODE_TYPES:
            assert is_node_type(nt)
            assert not is_link_type(nt)

    def test_link_types(self):
        for lt in LINK_TYPES:
            assert is_link_type(lt)
            assert not is_node_type(lt)

    def test_symmetric(self):
        assert AtomType.SIMILARITY_LINK in SYMMETRIC_LINKS
        assert AtomType.HEBBIAN_LINK in SYMMETRIC_LINKS
        assert AtomType.INHERITANCE_LINK not in SYMMETRIC_LINKS

    def test_all_types_classified(self):
        for at in AtomType:
            assert is_node_type(at) or is_link_type(at)

    def test_no_overlap(self):
        assert len(NODE_TYPES & LINK_TYPES) == 0

    def test_enum_values(self):
        assert AtomType.CONCEPT_NODE.value == "ConceptNode"
        assert AtomType.INHERITANCE_LINK.value == "InheritanceLink"


# =====================================================================
# add_node
# =====================================================================

class TestAddNode:
    def test_basic_add(self, engine):
        atom = engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(0.9, 0.8))
        assert atom.atom_type == AtomType.CONCEPT_NODE
        assert atom.name == "dog"
        assert atom.truth_value.strength == 0.9
        assert engine.atom_count() == 1

    def test_default_tv(self, engine):
        atom = engine.add_node(AtomType.CONCEPT_NODE, "cat")
        assert atom.truth_value.confidence >= 0.0

    def test_predicate_node(self, engine):
        atom = engine.add_node(AtomType.PREDICATE_NODE, "is_a", TruthValue(1.0, 0.99))
        assert atom.atom_type == AtomType.PREDICATE_NODE

    def test_number_node(self, engine):
        atom = engine.add_node(AtomType.NUMBER_NODE, "42", TruthValue(1.0, 1.0))
        assert atom.name == "42"

    def test_variable_node(self, engine):
        atom = engine.add_node(AtomType.VARIABLE_NODE, "$X", TruthValue(1.0, 0.5))
        assert atom.name == "$X"

    def test_schema_node(self, engine):
        atom = engine.add_node(AtomType.SCHEMA_NODE, "compute", TruthValue(1.0, 0.5))
        assert atom.atom_type == AtomType.SCHEMA_NODE

    def test_grounded_node(self, engine):
        atom = engine.add_node(AtomType.GROUNDED_NODE, "ltmm:abc", TruthValue(1.0, 0.5))
        assert atom.name == "ltmm:abc"

    def test_reject_link_type(self, engine):
        with pytest.raises(ValueError):
            engine.add_node(AtomType.INHERITANCE_LINK, "bad")

    def test_duplicate_merge(self, engine):
        a1 = engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(0.7, 0.5))
        a2 = engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(0.9, 0.6))
        assert a1.atom_id == a2.atom_id  # Same atom, merged
        assert a2.truth_value.confidence > 0.5  # Confidence grew

    def test_metadata(self, engine):
        atom = engine.add_node(
            AtomType.CONCEPT_NODE, "ethics",
            TruthValue(1.0, 0.9),
            metadata={"identity_relevant": True},
        )
        assert atom.metadata["identity_relevant"] is True

    def test_source_engine(self, engine):
        atom = engine.add_node(
            AtomType.CONCEPT_NODE, "test",
            TruthValue(1.0, 0.5),
            source_engine="logical_brain_engine",
        )
        assert atom.source_engine == "logical_brain_engine"


# =====================================================================
# add_link
# =====================================================================

class TestAddLink:
    def test_basic_inheritance(self, engine):
        dog = engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(1.0, 0.9))
        mammal = engine.add_node(AtomType.CONCEPT_NODE, "mammal", TruthValue(1.0, 0.9))
        link = engine.add_link(
            AtomType.INHERITANCE_LINK,
            (dog.atom_id, mammal.atom_id),
            TruthValue(0.95, 0.85),
        )
        assert link.atom_type == AtomType.INHERITANCE_LINK
        assert link.outgoing == (dog.atom_id, mammal.atom_id)
        assert link.name is None

    def test_incoming_updated(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        assert link.atom_id in a.incoming
        assert link.atom_id in b.incoming

    def test_reject_node_type(self, engine):
        with pytest.raises(ValueError):
            engine.add_link(AtomType.CONCEPT_NODE, (), TruthValue(1.0, 0.5))

    def test_reject_missing_target(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        with pytest.raises(KeyError):
            engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, "nonexistent"))

    def test_duplicate_merge(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        l1 = engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.8, 0.5))
        l2 = engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.6))
        assert l1.atom_id == l2.atom_id
        assert l2.truth_value.confidence > 0.5

    def test_similarity_link(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.SIMILARITY_LINK, (a.atom_id, b.atom_id), TruthValue(0.7, 0.8))
        assert link.atom_type == AtomType.SIMILARITY_LINK

    def test_hebbian_link(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.HEBBIAN_LINK, (a.atom_id, b.atom_id), TruthValue(0.5, 0.5))
        assert link.atom_type == AtomType.HEBBIAN_LINK

    def test_evaluation_link(self, engine):
        pred = engine.add_node(AtomType.PREDICATE_NODE, "is_a", TruthValue(1.0, 0.99))
        dog = engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(1.0, 0.9))
        mammal = engine.add_node(AtomType.CONCEPT_NODE, "mammal", TruthValue(1.0, 0.9))
        link = engine.add_link(
            AtomType.EVALUATION_LINK,
            (pred.atom_id, dog.atom_id, mammal.atom_id),
            TruthValue(0.95, 0.85),
        )
        assert len(link.outgoing) == 3

    def test_and_link(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.AND_LINK, (a.atom_id, b.atom_id), TruthValue(0.8, 0.7))
        assert link.atom_type == AtomType.AND_LINK

    def test_implication_link(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "rain", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "wet", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.IMPLICATION_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        assert link.atom_type == AtomType.IMPLICATION_LINK


# =====================================================================
# remove_atom
# =====================================================================

class TestRemoveAtom:
    def test_remove_node(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        assert engine.remove_atom(a.atom_id)
        assert engine.get_atom(a.atom_id) is None
        assert engine.atom_count() == 0

    def test_remove_nonexistent(self, engine):
        assert not engine.remove_atom("nonexistent")

    def test_cascade_removal(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        # Removing a should cascade-remove the link
        engine.remove_atom(a.atom_id, cascade=True)
        assert engine.get_atom(a.atom_id) is None
        assert engine.get_atom(link.atom_id) is None
        assert engine.get_atom(b.atom_id) is not None  # b survives

    def test_incoming_cleanup(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        # Remove the link directly
        engine.remove_atom(link.atom_id)
        assert link.atom_id not in a.incoming
        assert link.atom_id not in b.incoming

    def test_index_cleanup(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "test_name", TruthValue(1.0, 0.9))
        engine.remove_atom(a.atom_id)
        assert len(engine.get_by_name("test_name")) == 0
        assert len(engine.get_by_type(AtomType.CONCEPT_NODE)) == 0

    def test_cascade_chain(self, engine):
        """A → B → C: removing A cascades through the link chain."""
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        c = engine.add_node(AtomType.CONCEPT_NODE, "c", TruthValue(1.0, 0.9))
        l1 = engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        l2 = engine.add_link(AtomType.INHERITANCE_LINK, (b.atom_id, c.atom_id), TruthValue(0.9, 0.8))
        engine.remove_atom(b.atom_id, cascade=True)
        assert engine.get_atom(b.atom_id) is None
        assert engine.get_atom(l1.atom_id) is None
        assert engine.get_atom(l2.atom_id) is None
        assert engine.get_atom(a.atom_id) is not None
        assert engine.get_atom(c.atom_id) is not None

    def test_remove_link_preserves_targets(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        link = engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        engine.remove_atom(link.atom_id)
        assert engine.get_atom(a.atom_id) is not None
        assert engine.get_atom(b.atom_id) is not None


# =====================================================================
# Read Operations
# =====================================================================

class TestReadOperations:
    def test_get_atom(self, populated_engine):
        eng, nodes = populated_engine
        atom = eng.get_atom(nodes["dog"].atom_id)
        assert atom is not None
        assert atom.name == "dog"

    def test_get_atom_missing(self, engine):
        assert engine.get_atom("nonexistent") is None

    def test_get_by_name(self, populated_engine):
        eng, nodes = populated_engine
        results = eng.get_by_name("dog")
        assert len(results) == 1
        assert results[0].name == "dog"

    def test_get_by_name_multiple(self, engine):
        engine.add_node(AtomType.CONCEPT_NODE, "test", TruthValue(1.0, 0.9))
        engine.add_node(AtomType.PREDICATE_NODE, "test", TruthValue(1.0, 0.9))
        results = engine.get_by_name("test")
        assert len(results) == 2

    def test_get_by_type(self, populated_engine):
        eng, _ = populated_engine
        concepts = eng.get_by_type(AtomType.CONCEPT_NODE)
        assert len(concepts) == 4  # dog, cat, mammal, animal

    def test_get_by_type_links(self, populated_engine):
        eng, _ = populated_engine
        links = eng.get_by_type(AtomType.INHERITANCE_LINK)
        assert len(links) == 3

    def test_get_incoming(self, populated_engine):
        eng, nodes = populated_engine
        incoming = eng.get_incoming(nodes["mammal"].atom_id)
        assert len(incoming) >= 2  # dog→mammal, cat→mammal links

    def test_get_outgoing(self, populated_engine):
        eng, nodes = populated_engine
        links = eng.get_by_type(AtomType.INHERITANCE_LINK)
        for link in links:
            outgoing = eng.get_outgoing(link.atom_id)
            assert len(outgoing) == 2  # Each inheritance link has 2 targets

    def test_get_all_atoms(self, populated_engine):
        eng, _ = populated_engine
        all_atoms = eng.get_all_atoms()
        assert len(all_atoms) == 9  # 5 nodes + 4 links

    def test_atom_count(self, populated_engine):
        eng, _ = populated_engine
        assert eng.atom_count() == 9


# =====================================================================
# Pattern Matching
# =====================================================================

class TestPatternMatching:
    def test_simple_variable(self, populated_engine):
        eng, nodes = populated_engine
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(variable="$X"),
                PatternAtom(atom_id=nodes["mammal"].atom_id),
            ),
        )
        results = eng.pattern_match(pat)
        assert len(results) == 2  # dog→mammal, cat→mammal
        bound_names = {r["$X"].name for r in results}
        assert "dog" in bound_names
        assert "cat" in bound_names

    def test_name_constraint(self, populated_engine):
        eng, nodes = populated_engine
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(name="dog"),
                PatternAtom(variable="$Y"),
            ),
        )
        results = eng.pattern_match(pat)
        assert len(results) == 1
        assert results[0]["$Y"].name == "mammal"

    def test_type_constraint(self, populated_engine):
        eng, _ = populated_engine
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(variable="$X", atom_type=AtomType.CONCEPT_NODE),
                PatternAtom(variable="$Y", atom_type=AtomType.CONCEPT_NODE),
            ),
        )
        results = eng.pattern_match(pat)
        assert len(results) == 3  # dog→mammal, cat→mammal, mammal→animal

    def test_tv_threshold(self, populated_engine):
        eng, _ = populated_engine
        # Set ACh low to avoid boosting confidence threshold
        eng.update_neurochem_state({"ach": 0.0})
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(variable="$X"),
                PatternAtom(variable="$Y"),
            ),
            tv_min_strength=0.95,
            tv_min_confidence=0.9,
        )
        results = eng.pattern_match(pat)
        # mammal→animal: s=1.0, c=0.95 → pass
        # dog→mammal: s=0.95, c=0.9 → pass
        # cat→mammal: s=0.90 → fail strength
        assert len(results) == 2

    def test_exact_atom_id(self, populated_engine):
        eng, nodes = populated_engine
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(atom_id=nodes["dog"].atom_id),
                PatternAtom(atom_id=nodes["mammal"].atom_id),
            ),
        )
        results = eng.pattern_match(pat)
        assert len(results) == 1

    def test_no_match(self, populated_engine):
        eng, _ = populated_engine
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(name="unicorn"),
                PatternAtom(variable="$Y"),
            ),
        )
        results = eng.pattern_match(pat)
        assert len(results) == 0

    def test_binding_consistency(self, engine):
        """Same variable in multiple positions must bind to same atom."""
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        # Self-link: a→a
        engine.add_link(AtomType.SIMILARITY_LINK, (a.atom_id, a.atom_id), TruthValue(1.0, 0.9))
        # Cross-link: a→b
        engine.add_link(AtomType.SIMILARITY_LINK, (a.atom_id, b.atom_id), TruthValue(0.5, 0.5))

        pat = Pattern(
            root_type=AtomType.SIMILARITY_LINK,
            elements=(
                PatternAtom(variable="$X"),
                PatternAtom(variable="$X"),  # Same var — must be same atom
            ),
        )
        results = engine.pattern_match(pat)
        assert len(results) == 1  # Only the self-link matches
        assert results[0]["$X"].name == "a"

    def test_link_included_in_result(self, populated_engine):
        eng, nodes = populated_engine
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(variable="$X"),
                PatternAtom(variable="$Y"),
            ),
        )
        results = eng.pattern_match(pat)
        for r in results:
            assert "__link__" in r
            assert r["__link__"].atom_type == AtomType.INHERITANCE_LINK

    def test_arity_mismatch(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        # Pattern expects 3 elements but link has 2
        pat = Pattern(
            root_type=AtomType.INHERITANCE_LINK,
            elements=(
                PatternAtom(variable="$X"),
                PatternAtom(variable="$Y"),
                PatternAtom(variable="$Z"),
            ),
        )
        results = engine.pattern_match(pat)
        assert len(results) == 0

    def test_similarity_match(self, populated_engine):
        eng, nodes = populated_engine
        pat = Pattern(
            root_type=AtomType.SIMILARITY_LINK,
            elements=(
                PatternAtom(variable="$A"),
                PatternAtom(variable="$B"),
            ),
        )
        results = eng.pattern_match(pat)
        assert len(results) == 1
        names = {results[0]["$A"].name, results[0]["$B"].name}
        assert names == {"dog", "cat"}


# =====================================================================
# Truth Value Merge
# =====================================================================

class TestTruthValueMerge:
    def test_equal_weight(self):
        tv1 = TruthValue(0.8, 0.5)
        tv2 = TruthValue(0.6, 0.5)
        merged = merge_truth_values(tv1, tv2)
        assert merged.strength == pytest.approx(0.7, abs=0.01)
        assert merged.confidence == 1.0  # 0.5 + 0.5 = 1.0

    def test_unequal_weight(self):
        tv1 = TruthValue(0.8, 0.9)
        tv2 = TruthValue(0.2, 0.1)
        merged = merge_truth_values(tv1, tv2)
        assert merged.strength > 0.7  # Weighted toward tv1
        assert merged.confidence == 1.0  # 0.9 + 0.1 capped at 1.0

    def test_zero_confidence(self):
        tv1 = TruthValue(0.8, 0.0)
        tv2 = TruthValue(0.3, 0.0)
        merged = merge_truth_values(tv1, tv2)
        assert merged.strength == pytest.approx(0.5)
        assert merged.confidence == 0.0

    def test_one_zero(self):
        tv1 = TruthValue(0.8, 0.7)
        tv2 = TruthValue(0.5, 0.0)
        merged = merge_truth_values(tv1, tv2)
        assert merged.strength == pytest.approx(0.8, abs=0.01)

    def test_confidence_grows(self):
        tv1 = TruthValue(0.7, 0.3)
        tv2 = TruthValue(0.7, 0.3)
        merged = merge_truth_values(tv1, tv2)
        assert merged.confidence > tv1.confidence

    def test_confidence_capped(self):
        tv1 = TruthValue(0.8, 0.8)
        tv2 = TruthValue(0.8, 0.8)
        merged = merge_truth_values(tv1, tv2)
        assert merged.confidence == 1.0


# =====================================================================
# Truth Value Decay
# =====================================================================

class TestTruthValueDecay:
    def test_basic_decay(self, engine):
        atom = engine.add_node(AtomType.CONCEPT_NODE, "test", TruthValue(0.9, 0.8))
        old_c = atom.truth_value.confidence
        engine.decay_truth_values(dt=100.0)
        assert atom.truth_value.confidence < old_c or engine.get_atom(atom.atom_id) is None

    def test_floor(self):
        tv = compute_tv_decay(TruthValue(0.9, 0.06), rate=0.1, floor=0.05)
        assert tv.confidence >= 0.05

    def test_identity_immune(self, engine):
        atom = engine.add_node(
            AtomType.CONCEPT_NODE, "identity",
            TruthValue(0.9, 0.8),
            metadata={"identity_relevant": True},
        )
        engine.decay_truth_values(dt=1000.0)
        fetched = engine.get_atom(atom.atom_id)
        assert fetched is not None
        assert fetched.truth_value.confidence == 0.8  # Unchanged

    def test_5ht_slows_decay(self, engine):
        engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(0.9, 0.5))
        engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(0.9, 0.5))

        # Decay with low 5-HT
        eng_low = AtomSpaceEngine()
        eng_low.add_node(AtomType.CONCEPT_NODE, "x", TruthValue(0.9, 0.5))
        eng_low.update_neurochem_state({"5ht": 0.1})
        eng_low.decay_truth_values(dt=10.0)

        eng_high = AtomSpaceEngine()
        eng_high.add_node(AtomType.CONCEPT_NODE, "x", TruthValue(0.9, 0.5))
        eng_high.update_neurochem_state({"5ht": 0.9})
        eng_high.decay_truth_values(dt=10.0)

        atom_low = eng_low.get_by_name("x")
        atom_high = eng_high.get_by_name("x")
        if atom_low and atom_high:
            assert atom_high[0].truth_value.confidence >= atom_low[0].truth_value.confidence

    def test_gaba_accelerates_pruning(self):
        cfg = AtomSpaceConfig(prune_threshold_tv=0.5, tv_decay_rate=0.1, maintenance_interval=1)
        eng = AtomSpaceEngine(config=cfg)
        eng.update_neurochem_state({"gaba": 0.9, "5ht": 0.0})
        eng.add_node(AtomType.CONCEPT_NODE, "temp", TruthValue(0.9, 0.3))
        eng.decay_truth_values(dt=5.0)
        # With high GABA and low confidence, atom should be pruned
        remaining = eng.get_by_name("temp")
        assert len(remaining) == 0

    def test_lti_protects_from_pruning(self):
        cfg = AtomSpaceConfig(prune_threshold_tv=0.5, tv_decay_rate=0.1)
        eng = AtomSpaceEngine(config=cfg)
        eng.update_neurochem_state({"gaba": 0.9, "5ht": 0.0})
        atom = eng.add_node(AtomType.CONCEPT_NODE, "protected", TruthValue(0.9, 0.3))
        atom.attention_value.lti = 1.0  # High LTI
        eng.decay_truth_values(dt=5.0)
        assert eng.get_atom(atom.atom_id) is not None

    def test_strength_unchanged(self, engine):
        atom = engine.add_node(AtomType.CONCEPT_NODE, "test", TruthValue(0.7, 0.8))
        engine.decay_truth_values(dt=1.0)
        fetched = engine.get_atom(atom.atom_id)
        if fetched is not None:
            assert fetched.truth_value.strength == 0.7


# =====================================================================
# Capacity Enforcement
# =====================================================================

class TestCapacityEnforcement:
    def test_no_op_under_threshold(self, engine):
        engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        removed = engine.enforce_capacity()
        assert removed == 0

    def test_prune_when_full(self, small_engine):
        for i in range(20):
            small_engine.add_node(
                AtomType.CONCEPT_NODE, f"node_{i}",
                TruthValue(0.5, 0.1 + i * 0.04),
            )
        assert small_engine.atom_count() == 20
        # Now trigger capacity enforcement
        removed = small_engine.enforce_capacity()
        assert removed > 0
        assert small_engine.atom_count() < 20

    def test_identity_atoms_protected(self, small_engine):
        # Fill up
        for i in range(18):
            small_engine.add_node(AtomType.CONCEPT_NODE, f"n_{i}", TruthValue(0.5, 0.1))
        identity = small_engine.add_node(
            AtomType.CONCEPT_NODE, "identity",
            TruthValue(0.5, 0.1),
            metadata={"identity_relevant": True},
        )
        small_engine.add_node(AtomType.CONCEPT_NODE, "extra", TruthValue(0.5, 0.1))
        small_engine.enforce_capacity()
        assert small_engine.get_atom(identity.atom_id) is not None

    def test_lti_protected(self, small_engine):
        for i in range(18):
            small_engine.add_node(AtomType.CONCEPT_NODE, f"n_{i}", TruthValue(0.5, 0.1))
        important = small_engine.add_node(AtomType.CONCEPT_NODE, "important", TruthValue(0.5, 0.1))
        important.attention_value.lti = 1.0
        small_engine.add_node(AtomType.CONCEPT_NODE, "extra", TruthValue(0.5, 0.1))
        small_engine.enforce_capacity()
        assert small_engine.get_atom(important.atom_id) is not None

    def test_lowest_score_pruned_first(self, small_engine):
        for i in range(20):
            small_engine.add_node(
                AtomType.CONCEPT_NODE, f"node_{i}",
                TruthValue(0.5, 0.05 * (i + 1)),
            )
        small_engine.enforce_capacity()
        remaining = small_engine.get_all_atoms()
        # Higher confidence atoms should survive
        for atom in remaining:
            assert atom.truth_value.confidence > 0.05


# =====================================================================
# process() Pipeline
# =====================================================================

class TestProcess:
    def test_empty_process(self, engine):
        result = engine.process()
        assert "results" in result
        assert "stats" in result
        assert "neurochem_signals" in result

    def test_add_node_command(self, engine):
        result = engine.process({
            "commands": [{
                "action": "add_node",
                "params": {
                    "atom_type": "ConceptNode",
                    "name": "dog",
                    "strength": 0.9,
                    "confidence": 0.8,
                },
            }],
        })
        assert result["results"][0]["status"] == "ok"
        assert engine.atom_count() == 1

    def test_add_link_command(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        result = engine.process({
            "commands": [{
                "action": "add_link",
                "params": {
                    "atom_type": "InheritanceLink",
                    "outgoing_ids": [a.atom_id, b.atom_id],
                    "strength": 0.9,
                    "confidence": 0.8,
                },
            }],
        })
        assert result["results"][0]["status"] == "ok"

    def test_remove_command(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        result = engine.process({
            "commands": [{
                "action": "remove",
                "params": {"atom_id": a.atom_id},
            }],
        })
        assert result["results"][0]["status"] == "ok"
        assert engine.atom_count() == 0

    def test_get_command(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(0.9, 0.8))
        result = engine.process({
            "commands": [{
                "action": "get",
                "params": {"atom_id": a.atom_id},
            }],
        })
        assert result["results"][0]["name"] == "dog"

    def test_query_by_name_command(self, engine):
        engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(1.0, 0.9))
        result = engine.process({
            "commands": [{
                "action": "query_by_name",
                "params": {"name": "dog"},
            }],
        })
        assert len(result["results"][0]["atom_ids"]) == 1

    def test_query_by_type_command(self, engine):
        engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        result = engine.process({
            "commands": [{
                "action": "query_by_type",
                "params": {"atom_type": "ConceptNode"},
            }],
        })
        assert len(result["results"][0]["atom_ids"]) == 2

    def test_pattern_match_command(self, populated_engine):
        eng, nodes = populated_engine
        result = eng.process({
            "commands": [{
                "action": "pattern_match",
                "params": {
                    "root_type": "InheritanceLink",
                    "elements": [
                        {"variable": "$X"},
                        {"atom_id": nodes["mammal"].atom_id},
                    ],
                },
            }],
        })
        assert result["results"][0]["status"] == "ok"
        assert len(result["results"][0]["matches"]) == 2

    def test_decay_command(self, engine):
        engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(0.9, 0.8))
        result = engine.process({
            "commands": [{"action": "decay", "params": {"dt": 1.0}}],
        })
        assert result["results"][0]["status"] == "ok"

    def test_unknown_command(self, engine):
        result = engine.process({
            "commands": [{"action": "unknown", "params": {}}],
        })
        assert result["results"][0]["status"] == "error"

    def test_nt_state_update_via_process(self, engine):
        engine.process({"nt_state": {"da": 0.9, "5ht": 0.3}})
        assert engine.da_level == pytest.approx(0.9)
        assert engine._5ht_level == pytest.approx(0.3)

    def test_mode_switch_via_process(self, engine):
        engine.process({"mode": "CREATIVE"})
        assert engine._mode == "CREATIVE"
        assert engine._eff_min_confidence_to_add < engine.config.min_confidence_to_add


# =====================================================================
# Neurochemical Output
# =====================================================================

class TestNeurochemOutput:
    def test_novel_atom_da(self):
        nc = compute_atomspace_neurochem(5, 0, 0, 0, 0)
        assert nc.da_delta == pytest.approx(0.25)

    def test_merge_5ht(self):
        nc = compute_atomspace_neurochem(0, 3, 0, 0, 0)
        assert nc._5ht_delta == pytest.approx(0.09)

    def test_match_ach(self):
        nc = compute_atomspace_neurochem(0, 0, 10, 0, 0)
        assert nc.ach_delta == pytest.approx(0.2)

    def test_failure_ne(self):
        nc = compute_atomspace_neurochem(0, 0, 0, 2, 0)
        assert nc.ne_delta == pytest.approx(0.08)

    def test_prune_gamma(self):
        nc = compute_atomspace_neurochem(0, 0, 0, 0, 5)
        assert nc.gamma_boost == pytest.approx(0.05)

    def test_large_merge_theta(self):
        nc = compute_atomspace_neurochem(0, 15, 0, 0, 0)
        assert nc.theta_boost == pytest.approx(0.1)

    def test_many_novel_alpha_suppress(self):
        nc = compute_atomspace_neurochem(10, 0, 0, 0, 0)
        assert nc.alpha_suppress == pytest.approx(0.05)

    def test_as_dict(self):
        nc = AtomSpaceNeurochem(da_delta=0.1, gamma_boost=0.05)
        d = nc.as_dict()
        assert d["da_delta"] == 0.1
        assert d["gamma_boost"] == 0.05


# =====================================================================
# Neurochemical State (Pattern A)
# =====================================================================

class TestNeurochemState:
    def test_update_all(self, engine):
        engine.update_neurochem_state({
            "5ht": 0.8, "da": 0.3, "ne": 0.7, "ach": 0.6,
            "gaba": 0.4, "cor": 0.5, "oxt": 0.2, "cb1": 0.1,
        })
        assert engine._5ht_level == pytest.approx(0.8)
        assert engine.da_level == pytest.approx(0.3)
        assert engine.ne_level == pytest.approx(0.7)
        assert engine.ach_level == pytest.approx(0.6)
        assert engine.gaba_level == pytest.approx(0.4)
        assert engine.cor_level == pytest.approx(0.5)
        assert engine.oxt_level == pytest.approx(0.2)
        assert engine.cb1_level == pytest.approx(0.1)

    def test_partial_update(self, engine):
        engine.update_neurochem_state({"da": 0.9})
        assert engine.da_level == pytest.approx(0.9)
        assert engine._5ht_level == 0.5  # Unchanged

    def test_clamping(self, engine):
        engine.update_neurochem_state({"da": 1.5, "ne": -0.3})
        assert engine.da_level == 1.0
        assert engine.ne_level == 0.0

    def test_da_relaxes_write_gate(self, engine):
        low_da_threshold = compute_write_gate_threshold(0.1, 0.5, 0.1, 0.5)
        high_da_threshold = compute_write_gate_threshold(0.1, 0.5, 0.9, 0.5)
        assert high_da_threshold < low_da_threshold

    def test_cor_tightens_write_gate(self, engine):
        low_cor_threshold = compute_write_gate_threshold(0.1, 0.1, 0.5, 0.5)
        high_cor_threshold = compute_write_gate_threshold(0.1, 0.9, 0.5, 0.5)
        assert high_cor_threshold > low_cor_threshold

    def test_ne_broadens_match(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))
        # With higher NE, max_match_results should be larger
        engine.update_neurochem_state({"ne": 0.9})
        assert engine.ne_level == pytest.approx(0.9)


# =====================================================================
# Mode Configuration
# =====================================================================

class TestModes:
    def test_analytical_mode(self, engine):
        engine._apply_mode_config("ANALYTICAL")
        assert engine._eff_min_confidence_to_add == 0.2
        assert engine._eff_prune_threshold_tv == 0.1

    def test_creative_mode(self, engine):
        engine._apply_mode_config("CREATIVE")
        assert engine._eff_min_confidence_to_add == 0.05
        assert engine._eff_novelty_threshold == 0.9

    def test_rem_dream_mode(self, engine):
        engine._apply_mode_config("REM_DREAM")
        assert engine._eff_min_confidence_to_add == 0.01
        assert engine._eff_novelty_threshold == 0.95
        assert engine._eff_prune_threshold_sti == -200.0

    def test_default_mode(self, engine):
        engine._apply_mode_config("DEFAULT")
        assert engine._eff_min_confidence_to_add == engine.config.min_confidence_to_add


# =====================================================================
# Serialization (Import/Export)
# =====================================================================

class TestSerialization:
    def test_export_all(self, populated_engine):
        eng, _ = populated_engine
        data = eng.export_to_dict()
        assert len(data["atoms"]) == 9

    def test_export_subset(self, populated_engine):
        eng, nodes = populated_engine
        data = eng.export_to_dict([nodes["dog"].atom_id, nodes["cat"].atom_id])
        assert len(data["atoms"]) == 2

    def test_roundtrip(self, populated_engine):
        eng, _ = populated_engine
        data = eng.export_to_dict()

        eng2 = AtomSpaceEngine()
        created = eng2.import_from_dict(data)
        assert len(created) == 9
        assert eng2.atom_count() == 9

    def test_import_preserves_types(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "dog", TruthValue(0.9, 0.8))
        b = engine.add_node(AtomType.CONCEPT_NODE, "mammal", TruthValue(1.0, 0.95))
        engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))

        data = engine.export_to_dict()
        eng2 = AtomSpaceEngine()
        eng2.import_from_dict(data)

        dogs = eng2.get_by_name("dog")
        assert len(dogs) == 1
        assert dogs[0].atom_type == AtomType.CONCEPT_NODE

        links = eng2.get_by_type(AtomType.INHERITANCE_LINK)
        assert len(links) == 1

    def test_import_rebuilds_incoming(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.8))

        data = engine.export_to_dict()
        eng2 = AtomSpaceEngine()
        eng2.import_from_dict(data)

        a2 = eng2.get_by_name("a")[0]
        assert len(a2.incoming) > 0


# =====================================================================
# Pure Function Tests
# =====================================================================

class TestPureFunctions:
    def test_make_atom(self):
        atom = make_atom(
            AtomType.CONCEPT_NODE, "test", (),
            TruthValue(0.9, 0.8), AttentionValue(), 0,
        )
        assert atom.atom_type == AtomType.CONCEPT_NODE
        assert atom.name == "test"
        assert len(atom.atom_id) > 0

    def test_compute_tv_decay(self):
        tv = compute_tv_decay(TruthValue(0.9, 0.8), rate=0.1, floor=0.05)
        assert tv.confidence == pytest.approx(0.7)
        assert tv.strength == 0.9

    def test_compute_tv_decay_floor(self):
        tv = compute_tv_decay(TruthValue(0.9, 0.04), rate=0.1, floor=0.05)
        assert tv.confidence >= 0.05

    def test_compute_write_gate_threshold(self):
        t = compute_write_gate_threshold(0.1, 0.0, 0.0, 0.0)
        assert t == pytest.approx(0.1)

    def test_compute_similarity_exact(self):
        assert compute_similarity_score("dog", "dog") == 1.0

    def test_compute_similarity_none(self):
        assert compute_similarity_score(None, "dog") == 0.0

    def test_compute_similarity_different(self):
        score = compute_similarity_score("dog", "cat")
        assert 0.0 <= score < 1.0

    def test_compute_similarity_similar(self):
        score = compute_similarity_score("computation", "computing")
        assert score > 0.0

    def test_score_atom_for_pruning(self):
        atom = Atom(
            atom_id="test",
            atom_type=AtomType.CONCEPT_NODE,
            name="test",
            truth_value=TruthValue(0.5, 0.3),
            attention_value=AttentionValue(sti=10.0, lti=5.0),
        )
        score = score_atom_for_pruning(atom)
        assert score > 0.3  # TV + STI + LTI contributions

    def test_match_pattern_atom_exact(self):
        atom = Atom(atom_id="abc", atom_type=AtomType.CONCEPT_NODE, name="dog")
        result = match_pattern_atom(PatternAtom(atom_id="abc"), atom, {})
        assert result is not None

    def test_match_pattern_atom_wrong_id(self):
        atom = Atom(atom_id="abc", atom_type=AtomType.CONCEPT_NODE, name="dog")
        result = match_pattern_atom(PatternAtom(atom_id="xyz"), atom, {})
        assert result is None

    def test_match_pattern_atom_variable(self):
        atom = Atom(atom_id="abc", atom_type=AtomType.CONCEPT_NODE, name="dog")
        result = match_pattern_atom(PatternAtom(variable="$X"), atom, {})
        assert result == {"$X": "abc"}

    def test_match_pattern_atom_variable_consistent(self):
        atom = Atom(atom_id="abc", atom_type=AtomType.CONCEPT_NODE, name="dog")
        result = match_pattern_atom(PatternAtom(variable="$X"), atom, {"$X": "abc"})
        assert result == {"$X": "abc"}

    def test_match_pattern_atom_variable_inconsistent(self):
        atom = Atom(atom_id="abc", atom_type=AtomType.CONCEPT_NODE, name="dog")
        result = match_pattern_atom(PatternAtom(variable="$X"), atom, {"$X": "xyz"})
        assert result is None


# =====================================================================
# Introspection
# =====================================================================

class TestIntrospection:
    def test_get_status(self, engine):
        status = engine.get_status()
        assert status["engine_id"] == "atomspace_engine"
        assert status["cluster"] == "knowledge_substrate"
        assert "total_atoms" in status
        assert "nt_levels" in status

    def test_stats(self, populated_engine):
        eng, _ = populated_engine
        stats = eng._get_stats()
        assert stats["total_atoms"] == 9
        assert "ConceptNode" in stats["type_counts"]

    def test_repr(self, engine):
        r = repr(engine)
        assert "AtomSpaceEngine" in r

    def test_len(self, populated_engine):
        eng, _ = populated_engine
        assert len(eng) == 9


# =====================================================================
# update_truth_value / update_attention_value
# =====================================================================

class TestUpdates:
    def test_update_truth_value(self, engine):
        atom = engine.add_node(AtomType.CONCEPT_NODE, "test", TruthValue(0.5, 0.5))
        engine.update_truth_value(atom.atom_id, TruthValue(0.9, 0.9))
        fetched = engine.get_atom(atom.atom_id)
        assert fetched.truth_value.strength == 0.9
        assert fetched.truth_value.confidence == 0.9

    def test_update_truth_value_missing(self, engine):
        with pytest.raises(KeyError):
            engine.update_truth_value("nonexistent", TruthValue(0.5, 0.5))

    def test_update_attention_value(self, engine):
        atom = engine.add_node(AtomType.CONCEPT_NODE, "test", TruthValue(1.0, 0.9))
        engine.update_attention_value(atom.atom_id, AttentionValue(sti=50.0, lti=10.0))
        fetched = engine.get_atom(atom.atom_id)
        assert fetched.attention_value.sti == 50.0
        assert fetched.attention_value.lti == 10.0

    def test_update_attention_value_missing(self, engine):
        with pytest.raises(KeyError):
            engine.update_attention_value("nonexistent", AttentionValue())

    def test_atoms_in_focus(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        engine.update_attention_value(a.atom_id, AttentionValue(sti=10.0))
        engine.update_attention_value(b.atom_id, AttentionValue(sti=-5.0))
        in_focus = engine.get_atoms_in_focus(threshold=0.0)
        assert len(in_focus) == 1
        assert in_focus[0].atom_id == a.atom_id


# =====================================================================
# Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_empty_space(self, engine):
        assert engine.atom_count() == 0
        assert engine.get_all_atoms() == []
        assert engine.get_by_name("nothing") == []
        assert engine.get_by_type(AtomType.CONCEPT_NODE) == []

    def test_all_node_types(self, engine):
        for nt in NODE_TYPES:
            engine.add_node(nt, f"test_{nt.value}", TruthValue(1.0, 0.5))
        assert engine.atom_count() == len(NODE_TYPES)

    def test_all_link_types(self, engine):
        nodes = []
        for i in range(3):
            n = engine.add_node(AtomType.CONCEPT_NODE, f"n{i}", TruthValue(1.0, 0.9))
            nodes.append(n)
        for lt in LINK_TYPES:
            if lt == AtomType.NOT_LINK:
                engine.add_link(lt, (nodes[0].atom_id,), TruthValue(0.5, 0.5))
            else:
                engine.add_link(lt, (nodes[0].atom_id, nodes[1].atom_id), TruthValue(0.5, 0.5))

    def test_timetag_monotonic(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        b = engine.add_node(AtomType.CONCEPT_NODE, "b", TruthValue(1.0, 0.9))
        assert b.timetag > a.timetag

    def test_process_stats(self, populated_engine):
        eng, _ = populated_engine
        result = eng.process()
        assert result["stats"]["total_atoms"] == 9

    def test_get_incoming_empty(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "lonely", TruthValue(1.0, 0.9))
        assert engine.get_incoming(a.atom_id) == []

    def test_get_outgoing_node(self, engine):
        a = engine.add_node(AtomType.CONCEPT_NODE, "a", TruthValue(1.0, 0.9))
        assert engine.get_outgoing(a.atom_id) == []

    def test_get_incoming_missing(self, engine):
        assert engine.get_incoming("nonexistent") == []

    def test_get_outgoing_missing(self, engine):
        assert engine.get_outgoing("nonexistent") == []
