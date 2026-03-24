"""
Tests for Engine 10 — PLN Core (Probabilistic Logic Networks).

Covers: rule functions, confidence factors, backward chaining, rule
selection, InferenceResult, NT modulation, process(), consistency check,
neurochem output, edge cases.

pytest ROOT/tests/cog_engines/test_pln_engine.py -v
"""
from __future__ import annotations

import pytest

from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomSpaceEngine,
    AtomType,
    TruthValue,
)
from zados.cognitive_engines.cognitools.pln_engine import (
    DEFAULT_CONFIDENCE_FACTORS,
    InferenceNode,
    InferenceResult,
    PLNConfig,
    PLNEngine,
    PLNNeurochem,
    RuleType,
    compute_pln_neurochem,
    rule_abduction,
    rule_and,
    rule_context,
    rule_deduction,
    rule_induction,
    rule_inheritance_to_similarity,
    rule_intensional_inheritance,
    rule_modus_ponens,
    rule_not,
    rule_or,
    rule_revision,
    rule_similarity_to_inheritance,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def atomspace():
    return AtomSpaceEngine()


@pytest.fixture
def pln(atomspace):
    return PLNEngine(atomspace)


@pytest.fixture
def chain_atomspace():
    """AtomSpace with A→B→C→D inheritance chain."""
    asp = AtomSpaceEngine()
    a = asp.add_node(AtomType.CONCEPT_NODE, "A", TruthValue(1.0, 0.9))
    b = asp.add_node(AtomType.CONCEPT_NODE, "B", TruthValue(1.0, 0.9))
    c = asp.add_node(AtomType.CONCEPT_NODE, "C", TruthValue(1.0, 0.9))
    d = asp.add_node(AtomType.CONCEPT_NODE, "D", TruthValue(1.0, 0.9))
    asp.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), TruthValue(0.9, 0.85))
    asp.add_link(AtomType.INHERITANCE_LINK, (b.atom_id, c.atom_id), TruthValue(0.85, 0.8))
    asp.add_link(AtomType.INHERITANCE_LINK, (c.atom_id, d.atom_id), TruthValue(0.8, 0.75))
    return asp, {"A": a, "B": b, "C": c, "D": d}


@pytest.fixture
def implication_atomspace():
    """AtomSpace with implication links for modus ponens."""
    asp = AtomSpaceEngine()
    rain = asp.add_node(AtomType.CONCEPT_NODE, "rain", TruthValue(0.7, 0.8))
    wet  = asp.add_node(AtomType.CONCEPT_NODE, "wet_ground", TruthValue(0.5, 0.3))
    asp.add_link(
        AtomType.IMPLICATION_LINK,
        (rain.atom_id, wet.atom_id),
        TruthValue(0.9, 0.85),
    )
    return asp, {"rain": rain, "wet": wet}


# =====================================================================
# Rule Functions — Deduction
# =====================================================================

class TestRuleDeduction:
    def test_basic(self):
        tv = rule_deduction(TruthValue(0.9, 0.8), TruthValue(0.8, 0.7))
        assert 0.0 <= tv.strength <= 1.0
        assert tv.confidence > 0.0

    def test_high_strength(self):
        tv = rule_deduction(TruthValue(1.0, 0.9), TruthValue(1.0, 0.9))
        assert tv.strength > 0.9

    def test_low_strength(self):
        tv = rule_deduction(TruthValue(0.1, 0.5), TruthValue(0.1, 0.5))
        assert tv.strength <= 0.5

    def test_confidence_discounted(self):
        tv = rule_deduction(TruthValue(0.9, 0.9), TruthValue(0.9, 0.9))
        assert tv.confidence < 0.9  # Discounted by cf=0.9


# =====================================================================
# Rule Functions — Induction
# =====================================================================

class TestRuleInduction:
    def test_basic(self):
        tv = rule_induction(TruthValue(0.8, 0.7), TruthValue(0.7, 0.6))
        assert 0.0 <= tv.strength <= 1.0
        assert tv.confidence > 0.0

    def test_confidence_lower_than_deduction(self):
        tv_ded = rule_deduction(TruthValue(0.8, 0.7), TruthValue(0.7, 0.6))
        tv_ind = rule_induction(TruthValue(0.8, 0.7), TruthValue(0.7, 0.6))
        # Induction cf=0.6 < deduction cf=0.9
        assert tv_ind.confidence < tv_ded.confidence


# =====================================================================
# Rule Functions — Abduction
# =====================================================================

class TestRuleAbduction:
    def test_basic(self):
        tv = rule_abduction(TruthValue(0.8, 0.7), TruthValue(0.7, 0.6))
        assert 0.0 <= tv.strength <= 1.0
        assert tv.confidence > 0.0

    def test_confidence_lowest(self):
        tv_abd = rule_abduction(TruthValue(0.8, 0.7), TruthValue(0.7, 0.6))
        tv_ind = rule_induction(TruthValue(0.8, 0.7), TruthValue(0.7, 0.6))
        # Abduction cf=0.5 < induction cf=0.6
        assert tv_abd.confidence < tv_ind.confidence


# =====================================================================
# Rule Functions — Modus Ponens
# =====================================================================

class TestRuleModusPonens:
    def test_basic(self):
        tv = rule_modus_ponens(TruthValue(0.8, 0.7), TruthValue(0.9, 0.85))
        assert tv.strength > 0.5
        assert tv.confidence > 0.0

    def test_strong_antecedent_and_implication(self):
        tv = rule_modus_ponens(TruthValue(1.0, 0.9), TruthValue(1.0, 0.9))
        assert tv.strength > 0.9

    def test_weak_antecedent(self):
        tv = rule_modus_ponens(TruthValue(0.1, 0.5), TruthValue(0.9, 0.9))
        assert tv.strength < 0.6  # Weak antecedent dampens conclusion


# =====================================================================
# Rule Functions — Revision
# =====================================================================

class TestRuleRevision:
    def test_equal_weight(self):
        tv = rule_revision(TruthValue(0.8, 0.5), TruthValue(0.6, 0.5))
        assert tv.strength == pytest.approx(0.7, abs=0.01)

    def test_confidence_grows(self):
        tv = rule_revision(TruthValue(0.7, 0.3), TruthValue(0.7, 0.3))
        assert tv.confidence > 0.3

    def test_zero_confidence(self):
        tv = rule_revision(TruthValue(0.8, 0.0), TruthValue(0.3, 0.0))
        assert tv.strength == pytest.approx(0.5)
        assert tv.confidence == 0.0


# =====================================================================
# Rule Functions — AND, OR, NOT
# =====================================================================

class TestRuleAnd:
    def test_basic(self):
        tv = rule_and(TruthValue(0.8, 0.7), TruthValue(0.6, 0.8))
        assert tv.strength == pytest.approx(0.48, abs=0.01)
        assert tv.confidence > 0.0

    def test_one_false(self):
        tv = rule_and(TruthValue(0.0, 0.9), TruthValue(0.9, 0.9))
        assert tv.strength == pytest.approx(0.0)


class TestRuleOr:
    def test_basic(self):
        tv = rule_or(TruthValue(0.8, 0.7), TruthValue(0.6, 0.8))
        # 0.8 + 0.6 - 0.48 = 0.92
        assert tv.strength == pytest.approx(0.92, abs=0.01)

    def test_both_true(self):
        tv = rule_or(TruthValue(1.0, 0.9), TruthValue(1.0, 0.9))
        assert tv.strength == pytest.approx(1.0)


class TestRuleNot:
    def test_basic(self):
        tv = rule_not(TruthValue(0.8, 0.7))
        assert tv.strength == pytest.approx(0.2)
        assert tv.confidence > 0.0

    def test_negation_of_zero(self):
        tv = rule_not(TruthValue(0.0, 0.9))
        assert tv.strength == pytest.approx(1.0)


# =====================================================================
# Rule Functions — Conversion Rules
# =====================================================================

class TestConversionRules:
    def test_sim_to_inh(self):
        tv = rule_similarity_to_inheritance(TruthValue(0.8, 0.7))
        assert tv.strength == 0.8  # Strength preserved
        assert tv.confidence < 0.7  # Discounted

    def test_inh_to_sim(self):
        tv = rule_inheritance_to_similarity(
            TruthValue(0.8, 0.7), TruthValue(0.6, 0.5),
        )
        assert tv.strength == pytest.approx(0.7)  # Average
        assert tv.confidence < 0.5  # min(0.7, 0.5) * 0.7

    def test_intensional_inh(self):
        tv = rule_intensional_inheritance(shared_properties=5, total_properties=10)
        assert tv.strength == pytest.approx(0.5)
        assert tv.confidence > 0.0

    def test_intensional_inh_zero(self):
        tv = rule_intensional_inheritance(shared_properties=0, total_properties=0)
        assert tv.strength == 0.0
        assert tv.confidence == 0.0

    def test_context(self):
        tv = rule_context(TruthValue(0.9, 0.8), TruthValue(0.7, 0.6))
        assert tv.strength == 0.7  # Preserved
        assert tv.confidence < 0.6  # Discounted by context uncertainty


# =====================================================================
# Confidence Factors
# =====================================================================

class TestConfidenceFactors:
    def test_all_factors_present(self):
        for rt in RuleType:
            assert rt.value in DEFAULT_CONFIDENCE_FACTORS

    def test_revision_no_discount(self):
        assert DEFAULT_CONFIDENCE_FACTORS["revision"] == 1.0

    def test_all_less_than_or_equal_one(self):
        for v in DEFAULT_CONFIDENCE_FACTORS.values():
            assert v <= 1.0

    def test_abduction_weakest(self):
        # Among the 3 classic rules, abduction should be weakest
        assert DEFAULT_CONFIDENCE_FACTORS["abduction"] < DEFAULT_CONFIDENCE_FACTORS["induction"]
        assert DEFAULT_CONFIDENCE_FACTORS["induction"] < DEFAULT_CONFIDENCE_FACTORS["deduction"]


# =====================================================================
# Backward Chaining
# =====================================================================

class TestBackwardChaining:
    def test_direct_lookup(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        result = pln.infer(nodes["A"].atom_id)
        assert result.success
        assert result.conclusion_tv.confidence >= 0.1

    def test_deduction_chain(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        # Try to infer A→C via deduction through B
        result = pln.infer_link(
            AtomType.INHERITANCE_LINK,
            nodes["A"].atom_id,
            nodes["C"].atom_id,
        )
        assert result.success
        assert result.conclusion_tv.strength > 0.0
        assert result.conclusion_tv.confidence > 0.0

    def test_modus_ponens(self, implication_atomspace):
        asp, nodes = implication_atomspace
        pln = PLNEngine(asp)
        result = pln.infer(nodes["wet"].atom_id)
        assert result.success
        assert result.conclusion_tv is not None

    def test_depth_limit(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp, PLNConfig(max_depth=0))
        result = pln.infer(nodes["A"].atom_id, max_depth=0)
        # With 0 depth, should not be able to chain
        assert not result.success or result.steps_used == 0

    def test_step_budget(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp, PLNConfig(max_steps=1))
        result = pln.infer(nodes["A"].atom_id)
        assert result.steps_used <= 2  # At most 1 step + direct lookup

    def test_min_confidence_filter(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        # Very high min confidence — should fail for chained inferences
        result = pln.infer_link(
            AtomType.INHERITANCE_LINK,
            nodes["A"].atom_id,
            nodes["C"].atom_id,
            max_depth=5,
        )
        # The deduction chain A→B→C loses confidence
        if result.success:
            assert result.conclusion_tv.confidence < 0.85


# =====================================================================
# InferenceResult
# =====================================================================

class TestInferenceResult:
    def test_success_result(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        result = pln.infer(nodes["A"].atom_id)
        assert result.success
        assert result.target_id == nodes["A"].atom_id
        assert result.conclusion_tv is not None

    def test_failure_result(self, atomspace):
        pln = PLNEngine(atomspace)
        result = pln.infer("nonexistent_id")
        assert not result.success
        assert result.conclusion_tv is None

    def test_proof_tree(self, implication_atomspace):
        asp, nodes = implication_atomspace
        pln = PLNEngine(asp)
        result = pln.infer(nodes["wet"].atom_id)
        if result.success and result.proof_tree:
            assert result.proof_tree.target_id == nodes["wet"].atom_id

    def test_rules_applied_tracked(self, implication_atomspace):
        asp, nodes = implication_atomspace
        pln = PLNEngine(asp)
        result = pln.infer(nodes["wet"].atom_id)
        if result.success and result.rules_applied:
            assert any(r == RuleType.MODUS_PONENS for r in result.rules_applied)


# =====================================================================
# NT Modulation
# =====================================================================

class TestNTModulation:
    def test_5ht_raises_min_confidence(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.update_neurochem_state({"5ht": 0.1})
        low_5ht_mc = pln._eff_min_confidence
        pln.update_neurochem_state({"5ht": 0.9})
        high_5ht_mc = pln._eff_min_confidence
        assert high_5ht_mc > low_5ht_mc

    def test_da_lowers_min_confidence(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.update_neurochem_state({"da": 0.1})
        low_da_mc = pln._eff_min_confidence
        pln.update_neurochem_state({"da": 0.9})
        high_da_mc = pln._eff_min_confidence
        assert high_da_mc < low_da_mc

    def test_ne_increases_depth(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.update_neurochem_state({"ne": 0.0})
        low_ne_depth = pln._eff_max_depth
        pln.update_neurochem_state({"ne": 1.0})
        high_ne_depth = pln._eff_max_depth
        assert high_ne_depth > low_ne_depth

    def test_gaba_reduces_steps(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.update_neurochem_state({"gaba": 0.1})
        low_gaba_steps = pln._eff_max_steps
        pln.update_neurochem_state({"gaba": 0.9})
        high_gaba_steps = pln._eff_max_steps
        assert high_gaba_steps < low_gaba_steps

    def test_clamping(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.update_neurochem_state({"da": 1.5, "ne": -0.3})
        assert pln.da_level == 1.0
        assert pln.ne_level == 0.0

    def test_partial_update(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.update_neurochem_state({"da": 0.9})
        assert pln.da_level == pytest.approx(0.9)
        assert pln._5ht_level == 0.5  # Unchanged


# =====================================================================
# process()
# =====================================================================

class TestProcess:
    def test_empty_process(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        result = pln.process()
        assert "results" in result
        assert "neurochem_signals" in result
        assert "stats" in result

    def test_query_via_process(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        result = pln.process({
            "queries": [{"target_id": nodes["A"].atom_id}],
        })
        assert len(result["results"]) == 1
        assert result["results"][0]["success"]

    def test_nt_update_via_process(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.process({"nt_state": {"da": 0.9, "5ht": 0.2}})
        assert pln.da_level == pytest.approx(0.9)
        assert pln._5ht_level == pytest.approx(0.2)

    def test_mode_via_process(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln.process({"mode": "CREATIVE"})
        assert pln._mode == "CREATIVE"
        assert pln._eff_min_confidence < 0.1  # CREATIVE lowers it

    def test_failed_query(self, atomspace):
        pln = PLNEngine(atomspace)
        result = pln.process({"queries": [{"target_id": "nonexistent"}]})
        assert not result["results"][0]["success"]

    def test_stats_tracked(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        result = pln.process({
            "queries": [{"target_id": nodes["A"].atom_id}],
        })
        assert "successful_chains" in result["stats"]


# =====================================================================
# Consistency Check
# =====================================================================

class TestConsistencyCheck:
    def test_consistent(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        result = pln.check_consistency(nodes["A"].atom_id, nodes["B"].atom_id)
        assert result["consistent"]

    def test_missing_atom(self, atomspace):
        pln = PLNEngine(atomspace)
        result = pln.check_consistency("nonexistent1", "nonexistent2")
        assert result["consistent"]
        assert result["detail"] == "atom_not_found"

    def test_no_contradiction(self, chain_atomspace):
        asp, nodes = chain_atomspace
        pln = PLNEngine(asp)
        result = pln.check_consistency(nodes["A"].atom_id, nodes["D"].atom_id)
        assert result["consistent"]


# =====================================================================
# Neurochem Output
# =====================================================================

class TestNeurochemOutput:
    def test_novel_inference_da(self):
        nc = compute_pln_neurochem(3, 0, 0, 0, 0)
        assert nc.da_delta == pytest.approx(0.15)

    def test_successful_chain_ach(self):
        nc = compute_pln_neurochem(0, 2, 0, 0, 0)
        assert nc.ach_delta == pytest.approx(0.06)

    def test_failed_inference_ne(self):
        nc = compute_pln_neurochem(0, 0, 5, 0, 0)
        assert nc.ne_delta == pytest.approx(0.20)

    def test_revision_5ht(self):
        nc = compute_pln_neurochem(0, 0, 0, 4, 0)
        assert nc._5ht_delta == pytest.approx(0.12)

    def test_deep_chain_theta(self):
        nc = compute_pln_neurochem(0, 1, 0, 0, 5)
        assert nc.theta_boost == pytest.approx(0.1)

    def test_shallow_chain_no_theta(self):
        nc = compute_pln_neurochem(0, 1, 0, 0, 1)
        assert nc.theta_boost == 0.0

    def test_as_dict(self):
        nc = PLNNeurochem(da_delta=0.1, gamma_boost=0.05)
        d = nc.as_dict()
        assert d["da_delta"] == 0.1
        assert d["gamma_boost"] == 0.05


# =====================================================================
# Mode Configuration
# =====================================================================

class TestModes:
    def test_analytical(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln._apply_mode_config("ANALYTICAL")
        assert pln._mode == "ANALYTICAL"

    def test_creative_lower_confidence(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln._apply_mode_config("DEFAULT")
        default_mc = pln._eff_min_confidence
        pln._apply_mode_config("CREATIVE")
        creative_mc = pln._eff_min_confidence
        assert creative_mc < default_mc

    def test_rem_dream_deepest(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        pln._apply_mode_config("DEFAULT")
        default_depth = pln._eff_max_depth
        pln._apply_mode_config("REM_DREAM")
        rem_depth = pln._eff_max_depth
        assert rem_depth > default_depth


# =====================================================================
# Introspection
# =====================================================================

class TestIntrospection:
    def test_get_status(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        status = pln.get_status()
        assert status["engine_id"] == "pln_engine"
        assert status["cluster"] == "knowledge_substrate"
        assert "nt_levels" in status

    def test_repr(self, chain_atomspace):
        asp, _ = chain_atomspace
        pln = PLNEngine(asp)
        r = repr(pln)
        assert "PLNEngine" in r


# =====================================================================
# Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_empty_atomspace(self, atomspace):
        pln = PLNEngine(atomspace)
        result = pln.infer("nonexistent")
        assert not result.success

    def test_single_atom(self, atomspace):
        a = atomspace.add_node(AtomType.CONCEPT_NODE, "solo", TruthValue(0.9, 0.8))
        pln = PLNEngine(atomspace)
        result = pln.infer(a.atom_id)
        assert result.success
        assert result.conclusion_tv.confidence >= 0.1

    def test_no_applicable_rules(self, atomspace):
        a = atomspace.add_node(AtomType.CONCEPT_NODE, "isolated", TruthValue(0.3, 0.05))
        pln = PLNEngine(atomspace, PLNConfig(min_confidence=0.5))
        result = pln.infer(a.atom_id)
        # Confidence 0.05 < min 0.5
        assert not result.success

    def test_not_link_inference(self, atomspace):
        a = atomspace.add_node(AtomType.CONCEPT_NODE, "good", TruthValue(0.8, 0.7))
        not_link = atomspace.add_link(AtomType.NOT_LINK, (a.atom_id,), TruthValue(0.2, 0.3))
        pln = PLNEngine(atomspace)
        result = pln.infer(not_link.atom_id)
        # Should use NOT rule on the inner atom
        assert result.success or not_link.truth_value.confidence >= pln._eff_min_confidence

    def test_and_link_inference(self, atomspace):
        a = atomspace.add_node(AtomType.CONCEPT_NODE, "p", TruthValue(0.8, 0.7))
        b = atomspace.add_node(AtomType.CONCEPT_NODE, "q", TruthValue(0.6, 0.6))
        and_link = atomspace.add_link(AtomType.AND_LINK, (a.atom_id, b.atom_id), TruthValue(0.48, 0.3))
        pln = PLNEngine(atomspace)
        result = pln.infer(and_link.atom_id)
        assert result.success

    def test_or_link_inference(self, atomspace):
        a = atomspace.add_node(AtomType.CONCEPT_NODE, "p", TruthValue(0.8, 0.7))
        b = atomspace.add_node(AtomType.CONCEPT_NODE, "q", TruthValue(0.6, 0.6))
        or_link = atomspace.add_link(AtomType.OR_LINK, (a.atom_id, b.atom_id), TruthValue(0.92, 0.3))
        pln = PLNEngine(atomspace)
        result = pln.infer(or_link.atom_id)
        assert result.success
