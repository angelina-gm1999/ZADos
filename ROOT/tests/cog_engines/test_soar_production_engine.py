"""
Tests for Engine 3 — SOAR Production Rule Engine.

Covers:
  - Data structures (WME, Production, Operator, Preference, Impasse)
  - Working memory operations (add, remove, indexing)
  - Condition matching and variable binding
  - Production firing and quiescence
  - Operator proposal and preference collection
  - NT preference bias modulation
  - Preference resolution (SOAR semantics)
  - Impasse detection and delegation
  - Operator application
  - Chunking (learning)
  - Input/output link population and extraction
  - Neurochemical signal computation
  - Full decision cycle integration
  - Mode-dependent configuration
  - Introspection and status
  - Edge cases
"""
import pytest
import numpy as np

from zados.cognitive_engines.py_engines.soar_production_engine import (
    # Enums
    Support,
    ProductionType,
    ConditionTest,
    PrefType,
    ImpasseType,
    # Frozen data structures
    WME,
    Condition,
    Action,
    Production,
    Operator,
    Preference,
    Impasse,
    # Config & State
    SOARConfig,
    SOARState,
    # I/O
    SOARInput,
    SOARResult,
    SOARNeurochem,
    # Pure functions
    _clamp,
    _mode_key,
    _cfg,
    create_wme,
    add_wme_to_state,
    remove_wme_from_state,
    _add_wme_quick,
    _test_value,
    match_condition,
    match_production,
    find_matching_productions,
    fire_production,
    elaborate_until_quiescence,
    propose_operators,
    collect_comparison_preferences,
    apply_nt_preference_bias,
    resolve_preferences,
    detect_impasse_delegation,
    apply_operator_productions,
    populate_input_link,
    extract_output_link,
    decay_o_supported,
    decay_chunk_activations,
    create_chunk,
    compute_soar_neurochem,
    # Engine
    SOARProductionEngine,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def state():
    return SOARState()


@pytest.fixture
def config():
    return SOARConfig()


@pytest.fixture
def engine():
    return SOARProductionEngine(rng_seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# =====================================================================
# Test Data Structures
# =====================================================================

class TestWME:
    def test_wme_creation(self):
        w = WME("I1", "^type", "test", 0, Support.I_SUPPORTED)
        assert w.identifier == "I1"
        assert w.attribute == "^type"
        assert w.value == "test"
        assert w.timetag == 0
        assert w.support == Support.I_SUPPORTED
        assert w.source_operator is None

    def test_wme_frozen(self):
        w = WME("I1", "^type", "test", 0, Support.I_SUPPORTED)
        with pytest.raises(AttributeError):
            w.value = "changed"

    def test_wme_with_source_operator(self):
        w = WME("I1", "^val", 0.5, 1, Support.O_SUPPORTED, source_operator="op_1")
        assert w.source_operator == "op_1"


class TestProduction:
    def test_production_creation(self):
        cond = Condition("<s>", "^type", ConditionTest.EQ, "test")
        act = Action("add", "<s>", "^result", "done")
        prod = Production("test_rule", (cond,), (act,), ProductionType.ELABORATION)
        assert prod.name == "test_rule"
        assert prod.prod_type == ProductionType.ELABORATION
        assert prod.activation == 1.0
        assert not prod.learned

    def test_learned_production(self):
        prod = Production(
            "chunk_0", (), (), ProductionType.COMPARISON,
            activation=0.8, learned=True, chunk_source="impasse_tie_0",
        )
        assert prod.learned
        assert prod.chunk_source == "impasse_tie_0"


class TestOperator:
    def test_operator_creation(self):
        op = Operator("op_1", "investigate", {"target": "E1"}, "rule_1")
        assert op.op_id == "op_1"
        assert op.name == "investigate"
        assert op.parameters["target"] == "E1"


class TestPreference:
    def test_acceptable_preference(self):
        p = Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)
        assert p.pref_type == PrefType.ACCEPTABLE
        assert p.reference_id is None

    def test_better_preference(self):
        p = Preference("op_1", PrefType.BETTER, reference_id="op_2", strength=0.8)
        assert p.pref_type == PrefType.BETTER
        assert p.reference_id == "op_2"


class TestImpasse:
    def test_no_impasse(self):
        imp = Impasse(ImpasseType.NONE)
        assert imp.impasse_type == ImpasseType.NONE
        assert imp.tied_operators == ()

    def test_tie_impasse(self):
        imp = Impasse(ImpasseType.TIE, tied_operators=("op_1", "op_2"))
        assert imp.impasse_type == ImpasseType.TIE
        assert len(imp.tied_operators) == 2


# =====================================================================
# Test Working Memory Operations
# =====================================================================

class TestAddRemoveWME:
    def test_add_wme(self, state):
        wme = _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        assert wme.timetag in state.wm
        assert wme.timetag in state.index_by_id["I1"]
        assert wme.timetag in state.index_by_attr["^type"]
        assert wme.timetag in state.index_by_id_attr[("I1", "^type")]

    def test_remove_wme(self, state):
        wme = _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        tt = wme.timetag
        removed = remove_wme_from_state(state, tt)
        assert removed is not None
        assert removed.timetag == tt
        assert tt not in state.wm
        assert "I1" not in state.index_by_id
        assert "^type" not in state.index_by_attr

    def test_remove_nonexistent(self, state):
        result = remove_wme_from_state(state, 999)
        assert result is None

    def test_multiple_wmes_same_identifier(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        _add_wme_quick(state, "I1", "^value", 42, Support.I_SUPPORTED)
        assert len(state.index_by_id["I1"]) == 2
        assert len(state.index_by_id_attr[("I1", "^type")]) == 1
        assert len(state.index_by_id_attr[("I1", "^value")]) == 1


class TestWMIndexing:
    def test_index_by_id(self, state):
        _add_wme_quick(state, "I1", "^a", 1, Support.I_SUPPORTED)
        _add_wme_quick(state, "I1", "^b", 2, Support.I_SUPPORTED)
        _add_wme_quick(state, "I2", "^a", 3, Support.I_SUPPORTED)
        assert len(state.index_by_id["I1"]) == 2
        assert len(state.index_by_id["I2"]) == 1

    def test_index_by_attr(self, state):
        _add_wme_quick(state, "I1", "^type", "x", Support.I_SUPPORTED)
        _add_wme_quick(state, "I2", "^type", "y", Support.I_SUPPORTED)
        assert len(state.index_by_attr["^type"]) == 2

    def test_index_cleanup_on_remove(self, state):
        w1 = _add_wme_quick(state, "I1", "^a", 1, Support.I_SUPPORTED)
        w2 = _add_wme_quick(state, "I1", "^b", 2, Support.I_SUPPORTED)
        remove_wme_from_state(state, w1.timetag)
        assert len(state.index_by_id["I1"]) == 1
        assert ("I1", "^a") not in state.index_by_id_attr


# =====================================================================
# Test Condition Matching
# =====================================================================

class TestConditionMatching:
    def test_eq_match(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        results = match_condition(cond, state, {})
        assert len(results) == 1

    def test_eq_no_match(self, state):
        _add_wme_quick(state, "I1", "^type", "other", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        results = match_condition(cond, state, {})
        assert len(results) == 0

    def test_gt_match(self, state):
        _add_wme_quick(state, "I1", "^score", 0.8, Support.I_SUPPORTED)
        cond = Condition("I1", "^score", ConditionTest.GT, 0.5)
        results = match_condition(cond, state, {})
        assert len(results) == 1

    def test_lt_match(self, state):
        _add_wme_quick(state, "I1", "^score", 0.2, Support.I_SUPPORTED)
        cond = Condition("I1", "^score", ConditionTest.LT, 0.5)
        results = match_condition(cond, state, {})
        assert len(results) == 1

    def test_exists_match(self, state):
        _add_wme_quick(state, "I1", "^type", "anything", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EXISTS)
        results = match_condition(cond, state, {})
        assert len(results) == 1

    def test_neq_match(self, state):
        _add_wme_quick(state, "I1", "^type", "other", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.NEQ, "test")
        results = match_condition(cond, state, {})
        assert len(results) == 1

    def test_gte_match(self, state):
        _add_wme_quick(state, "I1", "^val", 0.5, Support.I_SUPPORTED)
        cond = Condition("I1", "^val", ConditionTest.GTE, 0.5)
        results = match_condition(cond, state, {})
        assert len(results) == 1

    def test_lte_match(self, state):
        _add_wme_quick(state, "I1", "^val", 0.5, Support.I_SUPPORTED)
        cond = Condition("I1", "^val", ConditionTest.LTE, 0.5)
        results = match_condition(cond, state, {})
        assert len(results) == 1


class TestVariableBinding:
    def test_variable_identifier_binding(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        cond = Condition("<s>", "^type", ConditionTest.EQ, "test")
        results = match_condition(cond, state, {})
        assert len(results) >= 1
        assert results[0]["<s>"] == "I1"

    def test_value_binding(self, state):
        _add_wme_quick(state, "I1", "^score", 0.85, Support.I_SUPPORTED)
        cond = Condition("I1", "^score", ConditionTest.EXISTS, binding_var="<v>")
        results = match_condition(cond, state, {})
        assert len(results) == 1
        assert results[0]["<v>"] == 0.85

    def test_pre_bound_variable(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        _add_wme_quick(state, "I2", "^type", "test", Support.I_SUPPORTED)
        cond = Condition("<s>", "^type", ConditionTest.EQ, "test")
        results = match_condition(cond, state, {"<s>": "I1"})
        assert len(results) == 1
        assert results[0]["<s>"] == "I1"


# =====================================================================
# Test Production Matching & Firing
# =====================================================================

class TestProductionMatching:
    def test_single_condition_match(self, state):
        _add_wme_quick(state, "I1", "^type", "engine_output", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "engine_output")
        prod = Production("rule1", (cond,), (), ProductionType.ELABORATION)
        bindings = match_production(prod, state, 0.5)
        assert bindings is not None

    def test_multi_condition_match(self, state):
        _add_wme_quick(state, "I1", "^type", "engine_output", Support.I_SUPPORTED)
        _add_wme_quick(state, "I1", "^flag-count", 3, Support.I_SUPPORTED)
        cond1 = Condition("I1", "^type", ConditionTest.EQ, "engine_output")
        cond2 = Condition("I1", "^flag-count", ConditionTest.GT, 0)
        prod = Production("rule1", (cond1, cond2), (), ProductionType.ELABORATION)
        bindings = match_production(prod, state, 0.5)
        assert bindings is not None

    def test_no_match(self, state):
        _add_wme_quick(state, "I1", "^type", "other", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "engine_output")
        prod = Production("rule1", (cond,), (), ProductionType.ELABORATION)
        bindings = match_production(prod, state, 0.5)
        assert bindings is None

    def test_activation_threshold(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        prod = Production("rule1", (cond,), (), ProductionType.ELABORATION, activation=0.3)
        assert match_production(prod, state, 0.5) is None
        assert match_production(prod, state, 0.2) is not None


class TestProductionFiring:
    def test_add_action(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        act = Action("add", "I1", "^result", "done")
        prod = Production("rule1", (cond,), (act,), ProductionType.ELABORATION)
        bindings = match_production(prod, state, 0.5)
        new_wmes, _, _ = fire_production(prod, bindings, state)
        assert len(new_wmes) == 1
        assert new_wmes[0].attribute == "^result"
        assert new_wmes[0].value == "done"

    def test_remove_action(self, state):
        w = _add_wme_quick(state, "I1", "^temp", "x", Support.I_SUPPORTED)
        cond = Condition("I1", "^temp", ConditionTest.EXISTS)
        act = Action("remove", "I1", "^temp")
        prod = Production("rule1", (cond,), (act,), ProductionType.ELABORATION)
        bindings = match_production(prod, state, 0.5)
        fire_production(prod, bindings, state)
        assert w.timetag not in state.wm

    def test_propose_action(self, state):
        _add_wme_quick(state, "I1", "^need", "investigate", Support.I_SUPPORTED)
        cond = Condition("I1", "^need", ConditionTest.EQ, "investigate")
        act = Action("propose", "I1", operator_name="investigate_op",
                     value={"target": "E1", "urgency": 0.7}, strength=1.0)
        prod = Production("propose1", (cond,), (act,), ProductionType.PROPOSAL)
        bindings = match_production(prod, state, 0.5)
        _, new_ops, new_prefs = fire_production(prod, bindings, state)
        assert len(new_ops) == 1
        assert new_ops[0].name == "investigate_op"
        assert len(new_prefs) == 1
        assert new_prefs[0].pref_type == PrefType.ACCEPTABLE


class TestQuiescence:
    def test_single_round_quiescence(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        act = Action("add", "I1", "^elaborated", True)
        prod = Production("elab1", (cond,), (act,), ProductionType.ELABORATION)
        state.productions["elab1"] = prod
        rounds = elaborate_until_quiescence(state, 10, 0.5)
        assert rounds == 1

    def test_quiescence_prevents_refire(self, state):
        _add_wme_quick(state, "I1", "^type", "test", Support.I_SUPPORTED)
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        act = Action("add", "I1", "^elaborated", True)
        prod = Production("elab1", (cond,), (act,), ProductionType.ELABORATION)
        state.productions["elab1"] = prod
        rounds = elaborate_until_quiescence(state, 10, 0.5)
        # Should fire once then reach quiescence
        assert rounds == 1
        assert "elab1" in state.fired_this_cycle


# =====================================================================
# Test Operator Proposal
# =====================================================================

class TestOperatorProposal:
    def test_propose_from_production(self, state):
        _add_wme_quick(state, "I1", "^alert", True, Support.I_SUPPORTED)
        cond = Condition("I1", "^alert", ConditionTest.EQ, True)
        act = Action("propose", "I1", operator_name="handle_alert",
                     value={"urgency": 0.8}, strength=1.0)
        prod = Production("prop1", (cond,), (act,), ProductionType.PROPOSAL)
        state.productions["prop1"] = prod
        propose_operators(state, 0.5)
        assert len(state.proposed_operators) == 1
        assert len(state.preferences) == 1
        op = list(state.proposed_operators.values())[0]
        assert op.name == "handle_alert"

    def test_multiple_proposals(self, state):
        _add_wme_quick(state, "I1", "^alert", True, Support.I_SUPPORTED)
        _add_wme_quick(state, "I2", "^need", "deepen", Support.I_SUPPORTED)
        cond1 = Condition("I1", "^alert", ConditionTest.EQ, True)
        act1 = Action("propose", "I1", operator_name="handle_alert",
                      value={"urgency": 0.8}, strength=1.0)
        prod1 = Production("prop1", (cond1,), (act1,), ProductionType.PROPOSAL)

        cond2 = Condition("I2", "^need", ConditionTest.EQ, "deepen")
        act2 = Action("propose", "I2", operator_name="deepen_analysis",
                      value={"familiarity": 0.9}, strength=0.7)
        prod2 = Production("prop2", (cond2,), (act2,), ProductionType.PROPOSAL)

        state.productions["prop1"] = prod1
        state.productions["prop2"] = prod2
        propose_operators(state, 0.5)
        assert len(state.proposed_operators) == 2


class TestPreferenceCollection:
    def test_comparison_adds_prefs(self, state):
        # Set up two operators in WM
        _add_wme_quick(state, "I1", "^type", "operator", Support.I_SUPPORTED)
        _add_wme_quick(state, "I1", "^name", "op_a", Support.I_SUPPORTED)
        _add_wme_quick(state, "I2", "^type", "operator", Support.I_SUPPORTED)
        _add_wme_quick(state, "I2", "^name", "op_b", Support.I_SUPPORTED)

        # Pre-populate proposed operators
        state.proposed_operators["op_a"] = Operator("op_a", "A", {}, "p1")
        state.proposed_operators["op_b"] = Operator("op_b", "B", {}, "p2")
        state.preferences = [
            Preference("op_a", PrefType.ACCEPTABLE),
            Preference("op_b", PrefType.ACCEPTABLE),
        ]

        # Comparison production: prefer op_a over op_b
        cond = Condition("I1", "^type", ConditionTest.EQ, "operator")
        act = Action("prefer", "op_a", pref_type="better",
                     reference_var="op_b", strength=0.9)
        prod = Production("comp1", (cond,), (act,), ProductionType.COMPARISON)
        state.productions["comp1"] = prod
        collect_comparison_preferences(state, 0.5)
        # Original 2 + 1 new comparison preference
        assert len(state.preferences) == 3


# =====================================================================
# Test NT Preference Bias
# =====================================================================

class TestNTPreferenceBias:
    def test_da_boosts_novel(self, state, config):
        state.da_level = 0.9  # High DA
        op = Operator("op_1", "explore", {"novelty": 0.8}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert result[0].strength > 1.0  # Boosted

    def test_da_no_effect_on_familiar(self, state, config):
        state.da_level = 0.9
        op = Operator("op_1", "safe", {"novelty": 0.2}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert abs(result[0].strength - 1.0) < 0.1  # Barely changed

    def test_sht_boosts_familiar(self, state, config):
        state._5ht_level = 0.9  # High 5-HT
        op = Operator("op_1", "safe", {"familiarity": 0.8}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert result[0].strength > 1.0

    def test_cortisol_suppresses_risky(self, state, config):
        state.cor_level = 0.8  # High cortisol
        op = Operator("op_1", "risky", {"risk": 0.9}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert result[0].strength < 1.0  # Suppressed

    def test_gaba_narrows_low_confidence(self, state, config):
        state.gaba_level = 0.9  # High GABA
        op = Operator("op_1", "unsure", {"confidence": 0.2}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert result[0].strength < 1.0

    def test_ne_boosts_urgent(self, state, config):
        state.ne_level = 0.9  # High NE
        op = Operator("op_1", "rush", {"urgency": 0.8}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert result[0].strength > 1.0

    def test_oxt_boosts_social(self, state, config):
        state.oxt_level = 0.9
        op = Operator("op_1", "empathize", {"social": 0.8}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert result[0].strength > 1.0

    def test_cb1_boosts_unconventional(self, state, config):
        state.cb1_level = 0.9
        op = Operator("op_1", "wild_idea", {"unconventional": 0.8}, "p1")
        state.proposed_operators["op_1"] = op
        state.preferences = [Preference("op_1", PrefType.ACCEPTABLE, strength=1.0)]
        result = apply_nt_preference_bias(state, config, "normal")
        assert result[0].strength > 1.0


# =====================================================================
# Test Preference Resolution
# =====================================================================

class TestPreferenceResolution:
    def test_single_acceptable(self, rng):
        ops = {"op_1": Operator("op_1", "A", {}, "p1")}
        prefs = [Preference("op_1", PrefType.ACCEPTABLE)]
        winner, imp, _, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert winner == "op_1"
        assert imp == ImpasseType.NONE

    def test_prohibit_removes(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
            "op_2": Operator("op_2", "B", {}, "p2"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.ACCEPTABLE),
            Preference("op_1", PrefType.PROHIBIT),
        ]
        winner, imp, _, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert winner == "op_2"
        assert imp == ImpasseType.NONE

    def test_require_filters(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
            "op_2": Operator("op_2", "B", {}, "p2"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.ACCEPTABLE),
            Preference("op_1", PrefType.REQUIRE),
        ]
        winner, imp, _, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert winner == "op_1"

    def test_better_dominance(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
            "op_2": Operator("op_2", "B", {}, "p2"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.ACCEPTABLE),
            Preference("op_1", PrefType.BETTER, reference_id="op_2", strength=1.0),
        ]
        winner, imp, _, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert winner == "op_1"

    def test_tie_impasse(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
            "op_2": Operator("op_2", "B", {}, "p2"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.ACCEPTABLE),
        ]
        winner, imp, tied, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert imp == ImpasseType.TIE
        assert len(tied) == 2
        assert winner is None

    def test_indifferent_breaks_tie(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
            "op_2": Operator("op_2", "B", {}, "p2"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.ACCEPTABLE),
            Preference("op_1", PrefType.INDIFFERENT, reference_id="op_2"),
        ]
        winner, imp, _, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert imp == ImpasseType.NONE
        assert winner in ("op_1", "op_2")

    def test_all_prohibited_conflict(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_1", PrefType.PROHIBIT),
        ]
        winner, imp, _, conflict_prefs = resolve_preferences(prefs, ops, 0.01, rng)
        assert imp == ImpasseType.CONFLICT
        assert winner is None

    def test_no_operators_state_no_change(self, rng):
        winner, imp, _, _ = resolve_preferences([], {}, 0.01, rng)
        assert imp == ImpasseType.STATE_NO_CHANGE
        assert winner is None

    def test_best_preference(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
            "op_2": Operator("op_2", "B", {}, "p2"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.ACCEPTABLE),
            Preference("op_1", PrefType.BEST, strength=1.0),
        ]
        winner, imp, _, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert winner == "op_1"
        assert imp == ImpasseType.NONE


class TestOperatorSelection:
    def test_select_single_winner(self, rng):
        ops = {
            "op_1": Operator("op_1", "A", {}, "p1"),
            "op_2": Operator("op_2", "B", {}, "p2"),
        }
        prefs = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.ACCEPTABLE),
            Preference("op_1", PrefType.BETTER, reference_id="op_2"),
        ]
        winner, imp, _, _ = resolve_preferences(prefs, ops, 0.01, rng)
        assert winner == "op_1"
        assert imp == ImpasseType.NONE


# =====================================================================
# Test Impasse Detection & Delegation
# =====================================================================

class TestImpasseDetection:
    def test_tie_delegation(self):
        target = detect_impasse_delegation(ImpasseType.TIE)
        assert target == "simulation_brain_engine"

    def test_conflict_delegation(self):
        target = detect_impasse_delegation(ImpasseType.CONFLICT)
        assert target == "contradiction_detection_engine"

    def test_no_change_delegation(self):
        target = detect_impasse_delegation(ImpasseType.NO_CHANGE)
        assert target == "simulated_opposition_engine"

    def test_state_no_change_delegation(self):
        target = detect_impasse_delegation(ImpasseType.STATE_NO_CHANGE)
        assert target == "uncertainty_pattern_engine"

    def test_none_no_delegation(self):
        target = detect_impasse_delegation(ImpasseType.NONE)
        assert target is None


# =====================================================================
# Test Input/Output Link
# =====================================================================

class TestInputLinkPopulation:
    def test_engine_outputs_populated(self, state):
        inp = SOARInput(
            engine_outputs={
                "contradiction_detection_engine": {"flag_count": 3, "confidence": 0.85},
            },
        )
        populate_input_link(state, inp)
        # Should have WMEs for type, engine-id, flag_count, confidence
        assert len(state.wm) >= 4

    def test_nt_state_populated(self, state):
        inp = SOARInput(
            nt_state={"da": 0.7, "ne": 0.3},
        )
        populate_input_link(state, inp)
        # Each NT: type + nt + level = 3 WMEs per NT, 2 NTs = 6
        assert len(state.wm) == 6

    def test_reward_scores_populated(self, state):
        inp = SOARInput(
            reward_scores={"logic": 0.8, "ethics": 0.9},
        )
        populate_input_link(state, inp)
        # Each reward: type + domain + score = 3 WMEs per domain, 2 domains = 6
        assert len(state.wm) == 6

    def test_cleanup_previous_input(self, state):
        inp1 = SOARInput(nt_state={"da": 0.5})
        populate_input_link(state, inp1)
        size1 = len(state.wm)
        inp2 = SOARInput(nt_state={"ne": 0.3})
        populate_input_link(state, inp2)
        size2 = len(state.wm)
        assert size2 == size1  # Old input removed, new input added

    def test_memory_contrast_populated(self, state):
        inp = SOARInput(
            memory_contrast={"similarity": 0.7, "contradicts_memory": False},
        )
        populate_input_link(state, inp)
        assert len(state.wm) >= 3  # type + 2 attrs


class TestOutputLinkExtraction:
    def test_extract_commands(self, state):
        state.output_link_id = "output-link"
        _add_wme_quick(state, "output-link", "^action", "investigate", Support.O_SUPPORTED)
        _add_wme_quick(state, "output-link", "^target", "E1", Support.O_SUPPORTED)
        commands = extract_output_link(state)
        assert len(commands) == 2

    def test_empty_output(self, state):
        commands = extract_output_link(state)
        assert len(commands) == 0


# =====================================================================
# Test O-Supported Decay
# =====================================================================

class TestOSupportedDecay:
    def test_decay_removes_wmes(self, state):
        _add_wme_quick(state, "I1", "^result", "x", Support.O_SUPPORTED, source_operator="op_1")
        _add_wme_quick(state, "I2", "^result", "y", Support.O_SUPPORTED, source_operator="op_1")
        _add_wme_quick(state, "I3", "^fact", "z", Support.I_SUPPORTED)
        count = decay_o_supported(state, "op_1")
        assert count == 2
        assert len(state.wm) == 1  # Only i-supported remains

    def test_decay_preserves_other_ops(self, state):
        _add_wme_quick(state, "I1", "^r1", "x", Support.O_SUPPORTED, source_operator="op_1")
        _add_wme_quick(state, "I2", "^r2", "y", Support.O_SUPPORTED, source_operator="op_2")
        count = decay_o_supported(state, "op_1")
        assert count == 1
        assert len(state.wm) == 1

    def test_decay_none_operator(self, state):
        _add_wme_quick(state, "I1", "^r1", "x", Support.O_SUPPORTED, source_operator="op_1")
        count = decay_o_supported(state, None)
        assert count == 0
        assert len(state.wm) == 1


# =====================================================================
# Test Chunking (Learning)
# =====================================================================

class TestChunking:
    def test_create_chunk_success(self, config):
        impasse = Impasse(ImpasseType.TIE, tied_operators=("op_1", "op_2"))
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        act = Action("prefer", "op_1", pref_type="better", reference_var="op_2")
        chunk = create_chunk(impasse, (cond,), (act,), 0.8, config, "normal", 0)
        assert chunk is not None
        assert chunk.name == "chunk_0"
        assert chunk.learned is True
        assert chunk.prod_type == ProductionType.COMPARISON

    def test_chunk_below_threshold(self, config):
        impasse = Impasse(ImpasseType.TIE)
        chunk = create_chunk(impasse, (), (), 0.1, config, "normal", 0)
        assert chunk is None  # Below chunk_confidence_min=0.6

    def test_chunk_type_from_impasse(self, config):
        for imp_type, expected_prod_type in [
            (ImpasseType.TIE, ProductionType.COMPARISON),
            (ImpasseType.CONFLICT, ProductionType.ELABORATION),
            (ImpasseType.NO_CHANGE, ProductionType.APPLICATION),
            (ImpasseType.STATE_NO_CHANGE, ProductionType.PROPOSAL),
        ]:
            impasse = Impasse(imp_type)
            cond = Condition("I1", "^x", ConditionTest.EXISTS)
            act = Action("add", "I1", "^y", "z")
            chunk = create_chunk(impasse, (cond,), (act,), 0.9, config, "normal", 0)
            assert chunk is not None
            assert chunk.prod_type == expected_prod_type

    def test_chunk_max_conditions(self, config):
        impasse = Impasse(ImpasseType.TIE)
        conds = tuple(
            Condition(f"I{i}", "^x", ConditionTest.EXISTS)
            for i in range(20)
        )
        act = Action("add", "I1", "^y", "z")
        chunk = create_chunk(impasse, conds, (act,), 0.9, config, "normal", 0)
        assert chunk is not None
        assert len(chunk.conditions) == config.max_chunk_conditions  # Limited to 8


class TestChunkActivationDecay:
    def test_decay_reduces_activation(self, state):
        prod = Production("chunk_0", (), (), ProductionType.COMPARISON,
                          activation=0.8, learned=True)
        state.productions["chunk_0"] = prod
        decay_chunk_activations(state, 0.1, 0.5)
        assert state.productions["chunk_0"].activation == pytest.approx(0.7)

    def test_non_learned_not_decayed(self, state):
        prod = Production("rule1", (), (), ProductionType.ELABORATION,
                          activation=0.8, learned=False)
        state.productions["rule1"] = prod
        decay_chunk_activations(state, 0.1, 0.5)
        assert state.productions["rule1"].activation == 0.8


class TestChunkExport:
    def test_learn_from_resolution(self, engine):
        impasse = Impasse(ImpasseType.TIE, tied_operators=("op_1", "op_2"))
        cond = Condition("I1", "^type", ConditionTest.EQ, "test")
        act = Action("prefer", "op_1", pref_type="better", reference_var="op_2")
        chunk = engine.learn_from_resolution(impasse, (cond,), (act,), 0.8)
        assert chunk is not None
        assert chunk.name in engine.list_productions()
        assert engine._state.total_chunks_learned == 1


# =====================================================================
# Test Neurochemical Signals
# =====================================================================

class TestNeurochem:
    def test_successful_selection_da(self, state, config):
        state.selected_operator = "op_1"
        state.proposed_operators["op_1"] = Operator(
            "op_1", "test", {"confidence": 0.9}, "p1",
        )
        impasse = Impasse(ImpasseType.NONE)
        nc = compute_soar_neurochem(state, impasse, 0, 5, config)
        assert nc.delta_da > 0
        assert nc.beta_boost > 0

    def test_impasse_ne_cor(self, state, config):
        impasse = Impasse(ImpasseType.CONFLICT)
        nc = compute_soar_neurochem(state, impasse, 0, 0, config)
        assert nc.delta_ne > 0
        assert nc.delta_cor >= 0

    def test_chunk_learned_5ht(self, state, config):
        impasse = Impasse(ImpasseType.NONE)
        nc = compute_soar_neurochem(state, impasse, 2, 0, config)
        assert nc.delta_5ht > 0
        assert nc.theta_gamma_boost > 0

    def test_preference_suppression_gaba(self, state, config):
        state.preferences = [
            Preference("op_1", PrefType.ACCEPTABLE),
            Preference("op_2", PrefType.PROHIBIT),
        ]
        impasse = Impasse(ImpasseType.NONE)
        nc = compute_soar_neurochem(state, impasse, 0, 0, config)
        assert nc.delta_gaba > 0

    def test_wm_load_alpha_suppress(self, state, config):
        # Fill WM
        for i in range(100):
            _add_wme_quick(state, f"I{i}", "^x", i, Support.I_SUPPORTED)
        impasse = Impasse(ImpasseType.NONE)
        nc = compute_soar_neurochem(state, impasse, 0, 0, config)
        assert nc.alpha_suppress > 0
        assert nc.delta_ach > 0

    def test_neurochem_clamped(self, state, config):
        # Edge case: huge WM
        for i in range(600):
            _add_wme_quick(state, f"I{i}", "^x", i, Support.I_SUPPORTED)
        impasse = Impasse(ImpasseType.NONE)
        nc = compute_soar_neurochem(state, impasse, 0, 0, config)
        assert nc.alpha_suppress <= 1.0
        assert nc.delta_ach <= 1.0


# =====================================================================
# Test Full Decision Cycle
# =====================================================================

class TestFullCycle:
    def test_empty_cycle(self, engine):
        inp = SOARInput()
        result = engine.process(inp)
        assert result.selected_operator is None
        assert result.impasse.impasse_type == ImpasseType.STATE_NO_CHANGE
        assert result.cycle_count == 1
        assert result.engine_id == "soar_production_engine"

    def test_cycle_with_proposal_and_selection(self, engine):
        # Add an elaboration that sets up a fact
        _add_wme_quick(engine._state, "CONST", "^active", True, Support.I_SUPPORTED)

        # Add a proposal production
        cond = Condition("CONST", "^active", ConditionTest.EQ, True)
        act = Action("propose", "CONST", operator_name="default_action",
                     value={"confidence": 0.7}, strength=1.0)
        engine.add_production(Production("propose_default", (cond,), (act,),
                                          ProductionType.PROPOSAL))

        # Add an application production so the cycle completes without NO_CHANGE
        app_cond = Condition("CONST", "^active", ConditionTest.EQ, True)
        app_act = Action("add", "output-link", "^action", "execute",
                         support=Support.O_SUPPORTED)
        engine.add_production(Production("apply_default", (app_cond,), (app_act,),
                                          ProductionType.APPLICATION))

        inp = SOARInput()
        result = engine.process(inp)
        assert result.selected_operator is not None
        assert result.selected_operator.name == "default_action"
        assert result.impasse.impasse_type == ImpasseType.NONE

    def test_cycle_with_tie(self, engine):
        _add_wme_quick(engine._state, "CONST", "^active", True, Support.I_SUPPORTED)

        # Two proposal productions with no comparison
        cond = Condition("CONST", "^active", ConditionTest.EQ, True)
        act1 = Action("propose", "CONST", operator_name="action_a",
                      value={"confidence": 0.5}, strength=1.0)
        act2 = Action("propose", "CONST", operator_name="action_b",
                      value={"confidence": 0.5}, strength=1.0)
        engine.add_production(Production("prop_a", (cond,), (act1,), ProductionType.PROPOSAL))
        engine.add_production(Production("prop_b", (cond,), (act2,), ProductionType.PROPOSAL))

        inp = SOARInput()
        result = engine.process(inp)
        assert result.impasse.impasse_type == ImpasseType.TIE
        assert len(result.delegation_requests) > 0
        assert result.delegation_requests[0]["target_engine"] == "simulation_brain_engine"

    def test_cycle_increments_counter(self, engine):
        inp = SOARInput()
        engine.process(inp)
        engine.process(inp)
        engine.process(inp)
        assert engine._state.cycle_count == 3

    def test_cycle_with_input_data(self, engine):
        inp = SOARInput(
            engine_outputs={"contradiction_detection_engine": {"flag_count": 2}},
            nt_state={"da": 0.8, "ne": 0.3},
            reward_scores={"logic": 0.9},
        )
        result = engine.process(inp)
        # WM should have been populated with input data
        assert result.wm_size > 0

    def test_nt_state_updates_levels(self, engine):
        engine.update_neurochem_state({"da": 0.9, "ne": 0.2, "5ht": 0.7})
        assert engine._state.da_level == pytest.approx(0.9)
        assert engine._state.ne_level == pytest.approx(0.2)
        assert engine._state._5ht_level == pytest.approx(0.7)


# =====================================================================
# Test Mode-Dependent Configuration
# =====================================================================

class TestModes:
    def test_configure_normal(self, engine):
        engine.configure("NORMAL")
        assert engine._mode == "normal"

    def test_configure_dev(self, engine):
        engine.configure("DEV")
        assert engine._mode == "dev"

    def test_configure_rem_dream(self, engine):
        engine.configure("REM_DREAM")
        assert engine._mode == "rem_dream"

    def test_mode_affects_threshold(self, engine, config):
        # Dev mode has lower threshold (0.5 vs 0.7 normal)
        assert _cfg(config.matching_threshold, "dev") < _cfg(config.matching_threshold, "normal")

    def test_mode_affects_max_substates(self, engine, config):
        # Reflective has more substates
        assert _cfg(config.max_substates, "reflective") > _cfg(config.max_substates, "normal")

    def test_mode_affects_exploration_weight(self, engine, config):
        # REM_DREAM has highest DA exploration weight
        assert _cfg(config.da_exploration_weight, "rem_dream") > _cfg(config.da_exploration_weight, "normal")

    def test_unknown_mode_falls_back(self, engine):
        engine.configure("UNKNOWN_MODE")
        assert engine._mode == "normal"


# =====================================================================
# Test Introspection & Status
# =====================================================================

class TestIntrospection:
    def test_get_status(self, engine):
        status = engine.get_status()
        assert status["engine_id"] == "soar_production_engine"
        assert status["cluster"] == "executive_control"
        assert status["mode"] == "normal"
        assert status["cycle_count"] == 0
        assert "da_level" in status

    def test_introspect(self, engine):
        info = engine.introspect()
        assert "proposed_operators" in info
        assert "preferences" in info
        assert "current_impasse" in info
        assert "learned_productions" in info

    def test_status_after_cycle(self, engine):
        engine.process(SOARInput())
        status = engine.get_status()
        assert status["cycle_count"] == 1


# =====================================================================
# Test Production Management
# =====================================================================

class TestProductionManagement:
    def test_add_production(self, engine):
        prod = Production("rule1", (), (), ProductionType.ELABORATION)
        engine.add_production(prod)
        assert "rule1" in engine.list_productions()
        assert engine.get_production("rule1") is prod

    def test_remove_production(self, engine):
        prod = Production("rule1", (), (), ProductionType.ELABORATION)
        engine.add_production(prod)
        removed = engine.remove_production("rule1")
        assert removed is prod
        assert "rule1" not in engine.list_productions()

    def test_remove_nonexistent(self, engine):
        result = engine.remove_production("nonexistent")
        assert result is None

    def test_list_productions(self, engine):
        engine.add_production(Production("r1", (), (), ProductionType.ELABORATION))
        engine.add_production(Production("r2", (), (), ProductionType.PROPOSAL))
        names = engine.list_productions()
        assert "r1" in names
        assert "r2" in names


# =====================================================================
# Test Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_nt_clamp(self, engine):
        engine.update_neurochem_state({"da": 2.0, "ne": -1.0})
        assert engine._state.da_level == 1.0
        assert engine._state.ne_level == 0.0

    def test_empty_production_memory(self, engine):
        result = engine.process(SOARInput())
        assert result.selected_operator is None

    def test_large_wm(self, engine):
        for i in range(200):
            _add_wme_quick(engine._state, f"I{i}", "^val", i, Support.I_SUPPORTED)
        result = engine.process(SOARInput())
        assert result.wm_size > 0

    def test_clamp_utility(self):
        assert _clamp(1.5) == 1.0
        assert _clamp(-0.5) == 0.0
        assert _clamp(0.5) == 0.5
        assert _clamp(0.5, 0.0, 2.0) == 0.5

    def test_mode_key_normalization(self):
        assert _mode_key("NORMAL") == "normal"
        assert _mode_key("REM_DREAM") == "rem_dream"
        assert _mode_key("invalid") == "normal"

    def test_test_value_function(self):
        assert _test_value(ConditionTest.EQ, "a", "a")
        assert not _test_value(ConditionTest.EQ, "a", "b")
        assert _test_value(ConditionTest.GT, 5, 3)
        assert not _test_value(ConditionTest.GT, 3, 5)
        assert _test_value(ConditionTest.EXISTS, "anything", None)
        assert _test_value(ConditionTest.NEQ, "a", "b")
        assert _test_value(ConditionTest.GTE, 5, 5)
        assert _test_value(ConditionTest.LTE, 5, 5)

    def test_processing_time_tracked(self, engine):
        result = engine.process(SOARInput())
        assert result.processing_time_ms >= 0.0

    def test_neurochem_all_nts(self, engine):
        engine.update_neurochem_state({
            "da": 0.1, "ne": 0.2, "5ht": 0.3, "ach": 0.4,
            "cor": 0.5, "gaba": 0.6, "oxt": 0.7, "cb1": 0.8,
        })
        s = engine._state
        assert s.da_level == pytest.approx(0.1)
        assert s.ne_level == pytest.approx(0.2)
        assert s._5ht_level == pytest.approx(0.3)
        assert s.ach_level == pytest.approx(0.4)
        assert s.cor_level == pytest.approx(0.5)
        assert s.gaba_level == pytest.approx(0.6)
        assert s.oxt_level == pytest.approx(0.7)
        assert s.cb1_level == pytest.approx(0.8)

    def test_result_frozen(self, engine):
        result = engine.process(SOARInput())
        with pytest.raises(AttributeError):
            result.cycle_count = 999

    def test_cfg_unknown_mode_fallback(self, config):
        val = _cfg(config.matching_threshold, "nonexistent_mode")
        assert val == 0.70  # Falls back to normal
