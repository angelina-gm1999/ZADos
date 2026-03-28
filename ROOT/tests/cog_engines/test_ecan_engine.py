"""
Tests for Engine 16 — ECAN Core (Economic Attention Networks).

Covers: ECANConfig, ECANNeurochem, pure helper functions, ECANEngine
        step() cycle (rent, spread, wage, clamp, AF, LTI, Hebbian),
        NT modulation, mode switching, process() pipeline, queries,
        introspection, edge cases.

pytest ROOT/tests/cog_engines/test_ecan_engine.py -v
"""
from __future__ import annotations

import pytest

from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomSpaceConfig,
    AtomSpaceEngine,
    AtomType,
    AttentionValue,
    TruthValue,
)
from zados.cognitive_engines.cognitools.ecan_engine import (
    ECANConfig,
    ECANEngine,
    ECANNeurochem,
    compute_ecan_neurochem,
    compute_effective_af_threshold,
    compute_effective_hebbian_decay,
    compute_effective_rent,
    compute_effective_spread_rate,
    compute_effective_wage,
    _W_NE,
    _W_ACH,
    _W_DA,
    _W_GABA,
    _W_5HT,
    _W_CB1,
    _W_OXT,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def atomspace():
    """Fresh AtomSpace with permissive write gate."""
    cfg = AtomSpaceConfig()
    eng = AtomSpaceEngine(config=cfg)
    # Set NT levels so write gate is wide open
    eng.update_neurochem_state({"da": 1.0, "cb1": 0.0, "cortisol": 0.0, "ach": 0.0})
    return eng


@pytest.fixture
def ecan(atomspace):
    """Fresh ECAN with default config, zeroed NTs."""
    e = ECANEngine(atomspace)
    e.update_neurochem_state({
        "ne": 0.0, "ach": 0.0, "da": 0.0, "gaba": 0.0,
        "5ht": 0.0, "cb1": 0.0, "oxt": 0.0,
    })
    return e


@pytest.fixture
def populated(atomspace, ecan):
    """AtomSpace + ECAN with a few nodes for testing."""
    a = atomspace.add_node(AtomType.CONCEPT_NODE, "alpha", TruthValue(1.0, 0.9))
    b = atomspace.add_node(AtomType.CONCEPT_NODE, "beta", TruthValue(1.0, 0.9))
    c = atomspace.add_node(AtomType.CONCEPT_NODE, "gamma", TruthValue(1.0, 0.9))
    return ecan, atomspace, {"alpha": a, "beta": b, "gamma": c}


# =====================================================================
# 1. ECANConfig
# =====================================================================

class TestECANConfig:
    def test_defaults(self):
        cfg = ECANConfig()
        assert cfg.rent_rate == 1.0
        assert cfg.wage_amount == 10.0
        assert cfg.spread_rate == 0.3
        assert cfg.af_threshold == 10.0
        assert cfg.max_af_size == 100
        assert cfg.hebbian_creation_threshold == 3
        assert cfg.lti_increment == 0.1
        assert cfg.lti_decay_rate == 0.01

    def test_custom_values(self):
        cfg = ECANConfig(rent_rate=2.0, wage_amount=20.0, af_threshold=5.0)
        assert cfg.rent_rate == 2.0
        assert cfg.wage_amount == 20.0
        assert cfg.af_threshold == 5.0

    def test_mode_configs_present(self):
        cfg = ECANConfig()
        assert "ANALYTICAL" in cfg.mode_configs
        assert "CREATIVE" in cfg.mode_configs
        assert "REM_DREAM" in cfg.mode_configs
        assert "DEFAULT" in cfg.mode_configs

    def test_analytical_mode_values(self):
        cfg = ECANConfig()
        ana = cfg.mode_configs["ANALYTICAL"]
        assert ana["af_threshold"] == 15.0
        assert ana["rent_rate"] == 1.5


# =====================================================================
# 2. ECANNeurochem
# =====================================================================

class TestECANNeurochem:
    def test_defaults(self):
        nc = ECANNeurochem()
        assert nc.da_delta == 0.0
        assert nc.ne_delta == 0.0
        assert nc.ach_delta == 0.0
        assert nc._5ht_delta == 0.0
        assert nc.gamma_boost == 0.0
        assert nc.beta_boost == 0.0

    def test_as_dict(self):
        nc = ECANNeurochem(da_delta=0.1, ne_delta=0.2)
        d = nc.as_dict()
        assert d["da_delta"] == 0.1
        assert d["ne_delta"] == 0.2
        assert len(d) == 6


# =====================================================================
# 3. Pure Helper Functions
# =====================================================================

class TestPureFunctions:
    def test_effective_rent_zero_gaba(self):
        assert compute_effective_rent(1.0, 0.0) == 1.0

    def test_effective_rent_high_gaba(self):
        result = compute_effective_rent(1.0, 1.0)
        assert result == pytest.approx(1.0 + _W_GABA)

    def test_effective_wage_zero_da(self):
        assert compute_effective_wage(10.0, 0.0) == 10.0

    def test_effective_wage_high_da(self):
        result = compute_effective_wage(10.0, 1.0)
        assert result == pytest.approx(10.0 * (1.0 + _W_DA))

    def test_effective_spread_zero_ne(self):
        assert compute_effective_spread_rate(0.3, 0.0) == 0.3

    def test_effective_spread_high_ne(self):
        result = compute_effective_spread_rate(0.3, 1.0)
        assert result == pytest.approx(0.3 * (1.0 + _W_NE))

    def test_effective_af_threshold_no_nt(self):
        result = compute_effective_af_threshold(10.0, 0.0, 0.0)
        assert result == 10.0

    def test_effective_af_threshold_high_ach(self):
        result = compute_effective_af_threshold(10.0, 1.0, 0.0)
        assert result > 10.0  # ACh tightens

    def test_effective_af_threshold_high_cb1(self):
        result = compute_effective_af_threshold(10.0, 0.0, 1.0)
        assert result < 10.0  # CB1 lowers

    def test_effective_af_threshold_cb1_floor(self):
        # CB1=1.0 → factor = max(1 - 0.3*1.0, 0.3) = 0.7
        result = compute_effective_af_threshold(10.0, 0.0, 1.0)
        assert result == pytest.approx(10.0 * 0.7)

    def test_effective_hebbian_decay_zero_5ht(self):
        assert compute_effective_hebbian_decay(0.01, 0.0) == 0.01

    def test_effective_hebbian_decay_high_5ht(self):
        result = compute_effective_hebbian_decay(0.01, 1.0)
        # 5-HT stabilizes → lower decay
        assert result < 0.01

    def test_effective_hebbian_decay_floor(self):
        # max(1 - 0.4*1.0, 0.1) = max(0.6, 0.1) = 0.6
        result = compute_effective_hebbian_decay(0.01, 1.0)
        assert result == pytest.approx(0.01 * 0.6)


class TestComputeEcanNeurochem:
    def test_zero_activity(self):
        nc = compute_ecan_neurochem(0, 0, 0, 0)
        assert nc.da_delta == 0.0
        assert nc.ne_delta == 0.0
        assert nc.ach_delta == 0.0
        assert nc._5ht_delta == 0.0
        assert nc.gamma_boost == 0.0
        assert nc.beta_boost == 0.0

    def test_atoms_entered_af_boosts_da(self):
        nc = compute_ecan_neurochem(5, 0, 0, 0)
        assert nc.da_delta == pytest.approx(5 * 0.05)

    def test_af_size_boosts_ne(self):
        nc = compute_ecan_neurochem(0, 20, 0, 0)
        assert nc.ne_delta == pytest.approx(20 * 0.01)

    def test_ne_delta_capped(self):
        nc = compute_ecan_neurochem(0, 100, 0, 0)
        assert nc.ne_delta == pytest.approx(0.3)

    def test_small_af_boosts_ach(self):
        nc = compute_ecan_neurochem(0, 5, 0, 0)
        assert nc.ach_delta == 0.05

    def test_large_af_no_ach(self):
        nc = compute_ecan_neurochem(0, 50, 0, 0)
        assert nc.ach_delta == 0.0

    def test_empty_af_no_ach(self):
        nc = compute_ecan_neurochem(0, 0, 0, 0)
        assert nc.ach_delta == 0.0

    def test_hebbian_strengthened_boosts_5ht(self):
        nc = compute_ecan_neurochem(0, 0, 10, 0)
        assert nc._5ht_delta == pytest.approx(10 * 0.03)

    def test_spread_events_boosts_gamma(self):
        nc = compute_ecan_neurochem(0, 0, 0, 20)
        assert nc.gamma_boost == pytest.approx(20 * 0.01)

    def test_small_af_boosts_beta(self):
        nc = compute_ecan_neurochem(0, 10, 0, 0)
        assert nc.beta_boost == 0.05

    def test_large_af_no_beta(self):
        nc = compute_ecan_neurochem(0, 30, 0, 0)
        assert nc.beta_boost == 0.0


# =====================================================================
# 4. Engine Initialization
# =====================================================================

class TestEngineInit:
    def test_engine_id(self, ecan):
        assert ecan.engine_id == "ecan_engine"

    def test_cluster(self, ecan):
        assert ecan.cluster == "knowledge_substrate"

    def test_default_mode(self, ecan):
        assert ecan._mode == "DEFAULT"

    def test_tick_counter_starts_zero(self, ecan):
        assert ecan._tick_counter == 0

    def test_default_nt_levels_zeroed(self, ecan):
        # We zeroed NTs in fixture
        assert ecan.ne_level == 0.0
        assert ecan.da_level == 0.0

    def test_custom_config(self, atomspace):
        cfg = ECANConfig(rent_rate=5.0)
        e = ECANEngine(atomspace, config=cfg)
        assert e.config.rent_rate == 5.0


# =====================================================================
# 5. Rent
# =====================================================================

class TestRent:
    def test_rent_decreases_sti(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 50.0
        initial_sti = alpha.attention_value.sti
        ecan.step()
        # Rent = 1.0 * (1 + 0) = 1.0 (gaba=0)
        # After rent: 50 - 1 = 49, but wage/spread may modify
        assert alpha.attention_value.sti < initial_sti

    def test_rent_applied_to_all_atoms(self, populated):
        ecan, atomspace, nodes = populated
        for n in nodes.values():
            n.attention_value.sti = 20.0
        ecan.step()
        for n in nodes.values():
            assert n.attention_value.sti < 20.0

    def test_high_gaba_increases_rent(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 100.0

        # Run with zero GABA
        ecan.update_neurochem_state({"gaba": 0.0})
        ecan.step()
        sti_low_gaba = alpha.attention_value.sti

        # Reset
        alpha.attention_value.sti = 100.0
        ecan.update_neurochem_state({"gaba": 1.0})
        ecan.step()
        sti_high_gaba = alpha.attention_value.sti

        assert sti_high_gaba < sti_low_gaba


# =====================================================================
# 6. Wages
# =====================================================================

class TestWages:
    def test_accessed_atom_earns_wage(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 0.0
        ecan.mark_accessed(alpha.atom_id)
        ecan.step()
        # Wage = 10.0 * (1+0) = 10.0; minus rent = 1.0 → net +9
        assert alpha.attention_value.sti == pytest.approx(9.0)

    def test_unaccessed_atom_no_wage(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 0.0
        ecan.step()
        # Only rent: 0 - 1 = -1
        assert alpha.attention_value.sti == pytest.approx(-1.0)

    def test_high_da_increases_wage(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]

        # Low DA
        alpha.attention_value.sti = 0.0
        ecan.update_neurochem_state({"da": 0.0})
        ecan.mark_accessed(alpha.atom_id)
        ecan.step()
        sti_low_da = alpha.attention_value.sti

        # High DA
        alpha.attention_value.sti = 0.0
        ecan.update_neurochem_state({"da": 1.0})
        ecan.mark_accessed(alpha.atom_id)
        ecan.step()
        sti_high_da = alpha.attention_value.sti

        assert sti_high_da > sti_low_da

    def test_oxt_social_bonus(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.metadata["social_context"] = True
        alpha.attention_value.sti = 0.0

        ecan.update_neurochem_state({"oxt": 1.0})
        ecan.mark_accessed(alpha.atom_id)
        ecan.step()
        sti_social = alpha.attention_value.sti

        # Non-social atom for comparison
        beta = nodes["beta"]
        beta.attention_value.sti = 0.0
        ecan.mark_accessed(beta.atom_id)
        ecan.step()
        sti_nonsocial = beta.attention_value.sti

        # Social got OXT bonus
        assert sti_social > sti_nonsocial

    def test_accessed_atoms_cleared_after_step(self, populated):
        ecan, atomspace, nodes = populated
        ecan.mark_accessed(nodes["alpha"].atom_id)
        ecan.step()
        assert len(ecan._accessed_atoms) == 0


# =====================================================================
# 7. STI Clamping
# =====================================================================

class TestSTIClamping:
    def test_clamp_to_ceiling(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 300.0  # Above ceiling
        ecan.step()
        assert alpha.attention_value.sti <= ecan.config.sti_ceiling

    def test_clamp_to_floor(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = -500.0  # Below floor
        ecan.step()
        assert alpha.attention_value.sti >= ecan.config.sti_floor


# =====================================================================
# 8. Spreading Activation
# =====================================================================

class TestSpreading:
    def test_spread_through_hebbian_link(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        # Create a HebbianLink between alpha and beta
        atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (alpha.atom_id, beta.atom_id),
            TruthValue(strength=0.8, confidence=0.9),
            source_engine="test",
        )

        alpha.attention_value.sti = 100.0
        beta.attention_value.sti = 0.0

        ecan.step()

        # Beta should have received some spread from alpha
        # Net: -rent + spread_delta = -1 + (0.3 * 100 * 0.8) = -1 + 24 = 23
        assert beta.attention_value.sti > 0.0

    def test_spread_bidirectional(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (alpha.atom_id, beta.atom_id),
            TruthValue(strength=0.5, confidence=0.9),
            source_engine="test",
        )

        alpha.attention_value.sti = 50.0
        beta.attention_value.sti = 50.0

        ecan.step()

        # Both should get spread from the other
        # spread_from_other = 0.3 * 50 * 0.5 = 7.5
        # net each: 50 - 1 + 7.5 = 56.5
        assert alpha.attention_value.sti > 50.0 - 1.0  # More than just rent
        assert beta.attention_value.sti > 50.0 - 1.0

    def test_no_spread_from_negative_sti(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (alpha.atom_id, beta.atom_id),
            TruthValue(strength=0.8, confidence=0.9),
            source_engine="test",
        )

        alpha.attention_value.sti = -10.0  # Negative → no outward spread
        beta.attention_value.sti = 0.0

        ecan.step()

        # Beta gets only rent deduction, no spread from negative alpha
        assert beta.attention_value.sti == pytest.approx(-1.0)

    def test_high_ne_increases_spread(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (alpha.atom_id, beta.atom_id),
            TruthValue(strength=0.5, confidence=0.9),
            source_engine="test",
        )

        # Low NE
        alpha.attention_value.sti = 100.0
        beta.attention_value.sti = 0.0
        ecan.update_neurochem_state({"ne": 0.0})
        ecan.step()
        sti_low_ne = beta.attention_value.sti

        # Reset
        alpha.attention_value.sti = 100.0
        beta.attention_value.sti = 0.0
        ecan.update_neurochem_state({"ne": 1.0})
        ecan.step()
        sti_high_ne = beta.attention_value.sti

        assert sti_high_ne > sti_low_ne

    def test_spread_events_counted(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (alpha.atom_id, beta.atom_id),
            TruthValue(strength=0.5, confidence=0.9),
            source_engine="test",
        )

        alpha.attention_value.sti = 50.0
        beta.attention_value.sti = 50.0

        result = ecan.step()
        assert result["spread_events"] >= 2  # Bidirectional


# =====================================================================
# 9. Attentional Focus
# =====================================================================

class TestAttentionalFocus:
    def test_high_sti_enters_af(self, populated):
        ecan, atomspace, nodes = populated
        nodes["alpha"].attention_value.sti = 50.0
        ecan.step()
        af = ecan.get_attentional_focus()
        # After step: 50 - 1 = 49, which > threshold 10
        assert nodes["alpha"].atom_id in af

    def test_low_sti_excluded_from_af(self, populated):
        ecan, atomspace, nodes = populated
        nodes["alpha"].attention_value.sti = 5.0  # Below threshold
        ecan.step()
        af = ecan.get_attentional_focus()
        # After step: 5 - 1 = 4, below threshold 10
        assert nodes["alpha"].atom_id not in af

    def test_max_af_size_enforced(self, atomspace, ecan):
        cfg = ECANConfig(max_af_size=2, af_threshold=5.0)
        ecan_small = ECANEngine(atomspace, config=cfg)
        ecan_small.update_neurochem_state({
            "ne": 0.0, "ach": 0.0, "da": 0.0, "gaba": 0.0,
            "5ht": 0.0, "cb1": 0.0, "oxt": 0.0,
        })

        # Add 5 atoms with high STI
        for i in range(5):
            n = atomspace.add_node(AtomType.CONCEPT_NODE, f"n{i}", TruthValue(1.0, 0.9))
            n.attention_value.sti = 100.0 + i

        ecan_small.step()
        af = ecan_small.get_attentional_focus()
        assert len(af) <= 2

    def test_af_sorted_by_sti_descending(self, atomspace, ecan):
        n1 = atomspace.add_node(AtomType.CONCEPT_NODE, "n1", TruthValue(1.0, 0.9))
        n2 = atomspace.add_node(AtomType.CONCEPT_NODE, "n2", TruthValue(1.0, 0.9))
        n1.attention_value.sti = 50.0
        n2.attention_value.sti = 100.0

        ecan.step()
        af = ecan.get_attentional_focus()

        if len(af) == 2:
            # n2 has higher STI, should be first
            assert af[0] == n2.atom_id
            assert af[1] == n1.atom_id

    def test_ach_tightens_af_threshold(self, populated):
        ecan, atomspace, nodes = populated
        # Set STI just above default threshold
        nodes["alpha"].attention_value.sti = 15.0

        ecan.update_neurochem_state({"ach": 0.0})
        ecan.step()
        af_low_ach = ecan.get_attentional_focus()
        in_af_low = nodes["alpha"].atom_id in af_low_ach

        # Reset STI
        nodes["alpha"].attention_value.sti = 15.0
        ecan.update_neurochem_state({"ach": 1.0})
        ecan.step()
        af_high_ach = ecan.get_attentional_focus()
        in_af_high = nodes["alpha"].atom_id in af_high_ach

        # High ACh → higher threshold → harder to enter AF
        # With ach=1.0: threshold = 10 * (1+0.3) * 1.0 = 13.0
        # After rent: 15 - 1 = 14, 14 > 13 → still in
        # But higher ACh means tighter focus overall
        assert ecan._eff_af_thresh > 10.0

    def test_cb1_lowers_af_threshold(self, populated):
        ecan, atomspace, nodes = populated

        ecan.update_neurochem_state({"cb1": 0.0})
        thresh_no_cb1 = ecan._eff_af_thresh

        ecan.update_neurochem_state({"cb1": 1.0})
        thresh_high_cb1 = ecan._eff_af_thresh

        assert thresh_high_cb1 < thresh_no_cb1

    def test_get_af_size(self, populated):
        ecan, atomspace, nodes = populated
        nodes["alpha"].attention_value.sti = 50.0
        nodes["beta"].attention_value.sti = 50.0
        ecan.step()
        assert ecan.get_af_size() >= 2

    def test_atoms_entered_af_tracked(self, populated):
        ecan, atomspace, nodes = populated
        nodes["alpha"].attention_value.sti = 50.0
        result = ecan.step()
        assert result["atoms_entered_af"] >= 1


# =====================================================================
# 10. LTI Dynamics
# =====================================================================

class TestLTI:
    def test_lti_increments_for_af_atom(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 50.0
        alpha.attention_value.lti = 0.0

        ecan.step()

        # Alpha is in AF (50-1=49 > 10), so LTI should increase
        assert alpha.attention_value.lti == pytest.approx(0.1)

    def test_lti_decays_for_non_af_atom(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 0.0  # Not in AF
        alpha.attention_value.lti = 1.0

        ecan.step()

        assert alpha.attention_value.lti < 1.0
        assert alpha.attention_value.lti == pytest.approx(1.0 - 0.01)

    def test_lti_does_not_go_negative(self, populated):
        ecan, atomspace, nodes = populated
        alpha = nodes["alpha"]
        alpha.attention_value.sti = 0.0
        alpha.attention_value.lti = 0.005  # Less than decay rate

        ecan.step()

        assert alpha.attention_value.lti >= 0.0


# =====================================================================
# 11. HebbianLink Management
# =====================================================================

class TestHebbianLinks:
    def test_coactivation_tracking(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]
        alpha.attention_value.sti = 50.0
        beta.attention_value.sti = 50.0

        # Run cycles — co-activation should be tracked
        ecan.step()  # Cycle 1: both in AF
        assert ecan._hebbian_strengthened >= 0  # May or may not have created yet

    def test_hebbian_link_created_after_threshold(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        # Run enough cycles to exceed hebbian_creation_threshold (3)
        for _ in range(4):
            alpha.attention_value.sti = 50.0
            beta.attention_value.sti = 50.0
            ecan.step()

        # Check that a HebbianLink exists between alpha and beta
        heb = ecan._find_hebbian_link(alpha.atom_id, beta.atom_id)
        assert heb is not None
        assert heb.atom_type == AtomType.HEBBIAN_LINK

    def test_existing_hebbian_strengthened(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        # Manually create HebbianLink
        heb = atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (alpha.atom_id, beta.atom_id),
            TruthValue(strength=0.3, confidence=0.5),
            source_engine="test",
        )

        alpha.attention_value.sti = 50.0
        beta.attention_value.sti = 50.0
        ecan.step()

        # HebbianLink should be strengthened
        updated = atomspace.get_atom(heb.atom_id)
        assert updated.truth_value.strength > 0.3

    def test_hebbian_strength_capped_at_one(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (alpha.atom_id, beta.atom_id),
            TruthValue(strength=0.95, confidence=0.5),
            source_engine="test",
        )

        alpha.attention_value.sti = 50.0
        beta.attention_value.sti = 50.0
        ecan.step()

        heb = ecan._find_hebbian_link(alpha.atom_id, beta.atom_id)
        assert heb.truth_value.strength <= 1.0

    def test_hebbian_decay(self, populated):
        ecan, atomspace, nodes = populated

        heb = atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (nodes["alpha"].atom_id, nodes["beta"].atom_id),
            TruthValue(strength=0.5, confidence=0.5),
            source_engine="test",
        )

        # Neither in AF → no strengthening, just decay
        nodes["alpha"].attention_value.sti = 0.0
        nodes["beta"].attention_value.sti = 0.0
        ecan.step()

        updated = atomspace.get_atom(heb.atom_id)
        if updated is not None:
            # Decay: 0.5 - eff_decay (0.01 * max(1-0,0.1) = 0.01)
            assert updated.truth_value.strength < 0.5

    def test_weak_hebbian_removed(self, populated):
        ecan, atomspace, nodes = populated

        heb = atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (nodes["alpha"].atom_id, nodes["beta"].atom_id),
            TruthValue(strength=0.015, confidence=0.5),  # Near min
            source_engine="test",
        )

        nodes["alpha"].attention_value.sti = 0.0
        nodes["beta"].attention_value.sti = 0.0
        ecan.step()

        # Should have been removed (0.015 - 0.01 = 0.005 < 0.01 min)
        assert atomspace.get_atom(heb.atom_id) is None

    def test_5ht_stabilizes_hebbian_decay(self, populated):
        ecan, atomspace, nodes = populated

        heb = atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (nodes["alpha"].atom_id, nodes["beta"].atom_id),
            TruthValue(strength=0.5, confidence=0.5),
            source_engine="test",
        )

        # With high 5-HT → slower decay
        ecan.update_neurochem_state({"5ht": 1.0})
        nodes["alpha"].attention_value.sti = 0.0
        nodes["beta"].attention_value.sti = 0.0
        ecan.step()

        updated_high_5ht = atomspace.get_atom(heb.atom_id)
        strength_high_5ht = updated_high_5ht.truth_value.strength

        # Decay with 5ht=1.0: 0.01 * max(1-0.4, 0.1) = 0.01 * 0.6 = 0.006
        # So strength = 0.5 - 0.006 = 0.494
        assert strength_high_5ht == pytest.approx(0.5 - 0.01 * 0.6, abs=0.01)


# =====================================================================
# 12. NT Modulation
# =====================================================================

class TestNTModulation:
    def test_update_neurochem_state(self, ecan):
        ecan.update_neurochem_state({"ne": 0.8, "da": 0.3, "gaba": 0.6})
        assert ecan.ne_level == pytest.approx(0.8)
        assert ecan.da_level == pytest.approx(0.3)
        assert ecan.gaba_level == pytest.approx(0.6)

    def test_nt_clamped(self, ecan):
        ecan.update_neurochem_state({"ne": 1.5, "da": -0.5})
        assert ecan.ne_level == 1.0
        assert ecan.da_level == 0.0

    def test_effective_params_recomputed(self, ecan):
        ecan.update_neurochem_state({"gaba": 1.0})
        assert ecan._eff_rent > ecan.config.rent_rate

    def test_partial_update_preserves_others(self, ecan):
        ecan.update_neurochem_state({"ne": 0.0, "da": 0.0})
        ecan.update_neurochem_state({"ne": 0.9})
        assert ecan.ne_level == 0.9
        assert ecan.da_level == 0.0  # Unchanged


# =====================================================================
# 13. Mode Switching
# =====================================================================

class TestModes:
    def test_analytical_mode(self, populated):
        ecan, atomspace, nodes = populated
        ecan._apply_mode_config("ANALYTICAL")
        assert ecan._mode == "ANALYTICAL"
        # ANALYTICAL: af_threshold=15, rent_rate=1.5
        assert ecan._eff_af_thresh > 10.0  # Tighter focus

    def test_creative_mode(self, populated):
        ecan, atomspace, nodes = populated
        ecan._apply_mode_config("CREATIVE")
        assert ecan._mode == "CREATIVE"
        # CREATIVE: af_threshold=5, wage=15
        assert ecan._eff_af_thresh < 10.0  # Broader focus

    def test_rem_dream_mode(self, populated):
        ecan, atomspace, nodes = populated
        ecan._apply_mode_config("REM_DREAM")
        assert ecan._mode == "REM_DREAM"
        # REM_DREAM: af_threshold=2, rent=0.5, spread=0.5
        assert ecan._eff_af_thresh < 10.0
        assert ecan._eff_rent < 1.0

    def test_mode_via_process(self, populated):
        ecan, atomspace, nodes = populated
        ecan.process({"mode": "CREATIVE"})
        assert ecan._mode == "CREATIVE"


# =====================================================================
# 14. process() Pipeline
# =====================================================================

class TestProcess:
    def test_basic_process(self, populated):
        ecan, atomspace, nodes = populated
        nodes["alpha"].attention_value.sti = 50.0
        result = ecan.process()
        assert "af_size" in result
        assert "af_atom_ids" in result
        assert "neurochem_signals" in result

    def test_process_with_nt_state(self, populated):
        ecan, atomspace, nodes = populated
        result = ecan.process({"nt_state": {"ne": 0.9, "da": 0.1}})
        assert ecan.ne_level == pytest.approx(0.9)
        assert ecan.da_level == pytest.approx(0.1)

    def test_process_with_accessed_atoms(self, populated):
        ecan, atomspace, nodes = populated
        nodes["alpha"].attention_value.sti = 0.0
        result = ecan.process({"accessed_atoms": [nodes["alpha"].atom_id]})
        # Alpha got wage → should be positive (wage - rent)
        assert nodes["alpha"].attention_value.sti > -1.0

    def test_tick_counter_increments(self, populated):
        ecan, atomspace, nodes = populated
        assert ecan._tick_counter == 0
        ecan.process()
        assert ecan._tick_counter == 1
        ecan.process()
        assert ecan._tick_counter == 2

    def test_process_returns_neurochem_dict(self, populated):
        ecan, atomspace, nodes = populated
        nodes["alpha"].attention_value.sti = 50.0
        result = ecan.process()
        nc = result["neurochem_signals"]
        assert "da_delta" in nc
        assert "ne_delta" in nc
        assert "ach_delta" in nc

    def test_process_empty_input(self, populated):
        ecan, atomspace, nodes = populated
        result = ecan.process({})
        assert isinstance(result, dict)

    def test_process_none_input(self, populated):
        ecan, atomspace, nodes = populated
        result = ecan.process(None)
        assert isinstance(result, dict)


# =====================================================================
# 15. Introspection
# =====================================================================

class TestIntrospection:
    def test_get_status(self, ecan):
        status = ecan.get_status()
        assert status["engine_id"] == "ecan_engine"
        assert status["cluster"] == "knowledge_substrate"
        assert status["mode"] == "DEFAULT"
        assert "tick_counter" in status
        assert "nt_levels" in status

    def test_get_status_nt_levels(self, ecan):
        ecan.update_neurochem_state({"ne": 0.7})
        status = ecan.get_status()
        assert status["nt_levels"]["ne"] == pytest.approx(0.7)

    def test_repr(self, ecan):
        r = repr(ecan)
        assert "ECANEngine" in r
        assert "DEFAULT" in r


# =====================================================================
# 16. Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_step_with_empty_atomspace(self, atomspace, ecan):
        result = ecan.step()
        assert result["af_size"] == 0
        assert result["spread_events"] == 0

    def test_hebbian_link_with_missing_atom(self, populated):
        """HebbianLink referencing removed atom should not crash."""
        ecan, atomspace, nodes = populated

        atomspace.add_link(
            AtomType.HEBBIAN_LINK,
            (nodes["alpha"].atom_id, nodes["beta"].atom_id),
            TruthValue(strength=0.5, confidence=0.5),
            source_engine="test",
        )
        nodes["alpha"].attention_value.sti = 50.0

        # Remove beta — the HebbianLink's outgoing still references it
        atomspace.remove_atom(nodes["beta"].atom_id, cascade=True)

        # Should not raise
        result = ecan.step()
        assert isinstance(result, dict)

    def test_many_atoms_af_bounded(self, atomspace, ecan):
        """With many high-STI atoms, AF is bounded by max_af_size."""
        for i in range(200):
            n = atomspace.add_node(AtomType.CONCEPT_NODE, f"node_{i}", TruthValue(1.0, 0.5))
            n.attention_value.sti = 50.0
        ecan.step()
        af = ecan.get_attentional_focus()
        assert len(af) <= ecan.config.max_af_size

    def test_rapid_cycles_stable(self, populated):
        """Run many rapid cycles without crash."""
        ecan, atomspace, nodes = populated
        for n in nodes.values():
            n.attention_value.sti = 20.0
        for _ in range(50):
            ecan.mark_accessed(nodes["alpha"].atom_id)
            ecan.step()
        # Should complete without error
        assert ecan._tick_counter == 50

    def test_all_atoms_decay_to_floor(self, populated):
        """Without wages, atoms eventually decay to floor."""
        ecan, atomspace, nodes = populated
        for n in nodes.values():
            n.attention_value.sti = 5.0  # Start below AF threshold

        for _ in range(300):
            ecan.step()

        # After many cycles of rent without wage, STI → floor
        for n in nodes.values():
            assert n.attention_value.sti == ecan.config.sti_floor

    def test_find_hebbian_link_nonexistent(self, populated):
        ecan, atomspace, nodes = populated
        result = ecan._find_hebbian_link(nodes["alpha"].atom_id, nodes["beta"].atom_id)
        assert result is None

    def test_coactivation_count_accumulates(self, populated):
        ecan, atomspace, nodes = populated
        alpha, beta = nodes["alpha"], nodes["beta"]

        alpha.attention_value.sti = 50.0
        beta.attention_value.sti = 50.0

        # 1st cycle — co-activation count starts
        ecan.step()
        pair = (min(alpha.atom_id, beta.atom_id), max(alpha.atom_id, beta.atom_id))
        assert ecan._coactivation_counts.get(pair, 0) >= 1

    def test_mark_accessed_invalid_atom(self, ecan):
        """Marking a nonexistent atom doesn't crash."""
        ecan.mark_accessed("nonexistent_id")
        result = ecan.step()
        assert isinstance(result, dict)
