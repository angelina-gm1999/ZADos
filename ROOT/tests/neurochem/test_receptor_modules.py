"""
Tests for per-receptor-family modules.

Phase 13: Verifies that every receptor family module has consistent structure,
valid pharmacodynamic specs, correct effective signaling computation,
and emotion plasticity hooks.
"""

import pytest

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec
from zados.neurochem.receptors.receptor_registry import (
    ReceptorModuleRegistry,
    register_all_receptor_modules,
)
from zados.neurochem.receptors.dopamine_receptors import DopamineReceptors
from zados.neurochem.receptors.serotonin_receptors import SerotoninReceptors
from zados.neurochem.receptors.norepinephrine_receptors import NorepinephrineReceptors
from zados.neurochem.receptors.acetylcholine_receptors import AcetylcholineReceptors
from zados.neurochem.receptors.oxytocin_receptors import OxytocinReceptors
from zados.neurochem.receptors.opioid_receptors import OpioidReceptors
from zados.neurochem.receptors.cannabinoid_receptors import CannabinoidReceptors
from zados.neurochem.receptors.crh_receptors import CRHReceptors
from zados.neurochem.receptors.gaba_receptors import GABAReceptors
from zados.neurochem.receptors.glutamate_receptors import GlutamateReceptors
from zados.neurochem.neurotransmitters.configs import (
    NT_RECEPTOR_MAP,
    DEFAULT_RECEPTOR_CONFIGS,
)


# All receptor family module classes
ALL_RECEPTOR_MODULES = [
    DopamineReceptors,
    SerotoninReceptors,
    NorepinephrineReceptors,
    AcetylcholineReceptors,
    OxytocinReceptors,
    OpioidReceptors,
    CannabinoidReceptors,
    CRHReceptors,
    GABAReceptors,
    GlutamateReceptors,
]

# NTs that have receptor dynamics (all except cortisol)
NTS_WITH_RECEPTORS = [nt for nt, recs in NT_RECEPTOR_MAP.items() if len(recs) > 0]


@pytest.fixture(
    params=ALL_RECEPTOR_MODULES,
    ids=[c.__name__ for c in ALL_RECEPTOR_MODULES],
)
def receptor_module(request):
    """Parametrized fixture providing each receptor family module instance."""
    return request.param()


# =====================================================================
# ReceptorSpec Validation
# =====================================================================

class TestReceptorSpec:
    """Test ReceptorSpec dataclass behavior."""

    def test_valid_excitatory(self):
        spec = ReceptorSpec(receptor_id="TEST_1", signaling_type="excitatory")
        assert spec.signaling_type == "excitatory"

    def test_valid_inhibitory(self):
        spec = ReceptorSpec(receptor_id="TEST_2", signaling_type="inhibitory")
        assert spec.signaling_type == "inhibitory"

    def test_valid_modulatory(self):
        spec = ReceptorSpec(receptor_id="TEST_3", signaling_type="modulatory")
        assert spec.signaling_type == "modulatory"

    def test_invalid_signaling_type_raises(self):
        with pytest.raises(ValueError, match="Invalid signaling_type"):
            ReceptorSpec(receptor_id="TEST_BAD", signaling_type="invalid")

    def test_default_ionotropic_false(self):
        spec = ReceptorSpec(receptor_id="TEST_4")
        assert spec.ionotropic is False

    def test_default_weight_one(self):
        spec = ReceptorSpec(receptor_id="TEST_5")
        assert spec.effective_signaling_weight == 1.0

    def test_default_empty_plasticity(self):
        spec = ReceptorSpec(receptor_id="TEST_6")
        assert spec.emotion_plasticity_rules == {}

    def test_frozen(self):
        spec = ReceptorSpec(receptor_id="TEST_7")
        with pytest.raises(AttributeError):
            spec.receptor_id = "CHANGED"

    def test_plasticity_rules_preserved(self):
        rules = {"joy": {"sigma_delta": 0.1}}
        spec = ReceptorSpec(
            receptor_id="TEST_8",
            emotion_plasticity_rules=rules,
        )
        assert spec.emotion_plasticity_rules["joy"]["sigma_delta"] == 0.1


# =====================================================================
# Structure Tests (parametrized across all receptor families)
# =====================================================================

class TestReceptorModuleStructure:
    """Every receptor family module must have consistent structure."""

    def test_is_receptor_family_module(self, receptor_module):
        assert isinstance(receptor_module, ReceptorFamilyModule)

    def test_has_parent_nt(self, receptor_module):
        assert isinstance(receptor_module.parent_nt, str)
        assert len(receptor_module.parent_nt) > 0

    def test_parent_nt_in_nt_receptor_map(self, receptor_module):
        """Parent NT must be a key in NT_RECEPTOR_MAP."""
        assert receptor_module.parent_nt in NT_RECEPTOR_MAP, (
            f"{receptor_module.__class__.__name__}.parent_nt = "
            f"{receptor_module.parent_nt!r} not found in NT_RECEPTOR_MAP"
        )

    def test_has_receptor_specs(self, receptor_module):
        specs = receptor_module.receptor_specs
        assert isinstance(specs, dict)
        assert len(specs) > 0

    def test_specs_match_configs(self, receptor_module):
        """Every receptor_id in specs must exist in DEFAULT_RECEPTOR_CONFIGS."""
        for receptor_id in receptor_module.receptor_specs:
            assert receptor_id in DEFAULT_RECEPTOR_CONFIGS, (
                f"Receptor {receptor_id} from "
                f"{receptor_module.__class__.__name__} "
                f"not in DEFAULT_RECEPTOR_CONFIGS"
            )

    def test_specs_match_nt_receptor_map(self, receptor_module):
        """Receptor IDs should match NT_RECEPTOR_MAP for this parent NT."""
        expected = set(NT_RECEPTOR_MAP[receptor_module.parent_nt])
        actual = set(receptor_module.receptor_specs.keys())
        assert actual == expected, (
            f"{receptor_module.__class__.__name__}: "
            f"expected {expected}, got {actual}"
        )

    def test_all_specs_are_receptor_spec(self, receptor_module):
        for rid, spec in receptor_module.receptor_specs.items():
            assert isinstance(spec, ReceptorSpec), (
                f"{rid} is {type(spec)}, expected ReceptorSpec"
            )

    def test_receptor_id_matches_key(self, receptor_module):
        """Each spec's receptor_id must match its dict key."""
        for key, spec in receptor_module.receptor_specs.items():
            assert spec.receptor_id == key, (
                f"Key {key} != spec.receptor_id {spec.receptor_id}"
            )

    def test_signaling_weights_positive(self, receptor_module):
        for rid, spec in receptor_module.receptor_specs.items():
            assert spec.effective_signaling_weight > 0.0, (
                f"{rid} has non-positive weight "
                f"{spec.effective_signaling_weight}"
            )

    def test_signaling_weights_bounded(self, receptor_module):
        """Weights should be in (0, 2] range."""
        for rid, spec in receptor_module.receptor_specs.items():
            assert 0.0 < spec.effective_signaling_weight <= 2.0, (
                f"{rid} weight {spec.effective_signaling_weight} out of range"
            )

    def test_get_receptor_ids_sorted(self, receptor_module):
        ids = receptor_module.get_receptor_ids()
        assert ids == sorted(ids)
        assert len(ids) == len(receptor_module.receptor_specs)


# =====================================================================
# Pharmacodynamic Type Tests
# =====================================================================

class TestReceptorPharmacology:
    """Test biologically-accurate pharmacodynamic classifications."""

    def test_da_all_metabotropic(self):
        module = DopamineReceptors()
        for rid, spec in module.receptor_specs.items():
            assert spec.ionotropic is False, (
                f"DA {rid} should be metabotropic (GPCR)"
            )

    def test_da_d1_d5_excitatory(self):
        module = DopamineReceptors()
        assert module.receptor_specs["DA_D1"].signaling_type == "excitatory"
        assert module.receptor_specs["DA_D5"].signaling_type == "excitatory"

    def test_da_d2_d3_inhibitory(self):
        module = DopamineReceptors()
        assert module.receptor_specs["DA_D2"].signaling_type == "inhibitory"
        assert module.receptor_specs["DA_D3"].signaling_type == "inhibitory"

    def test_5ht_3_ionotropic(self):
        """5-HT3 is the only ionotropic serotonin receptor."""
        module = SerotoninReceptors()
        assert module.receptor_specs["5HT_3"].ionotropic is True
        # Others are metabotropic
        for rid in ["5HT_1A", "5HT_1B", "5HT_2A", "5HT_2C"]:
            assert module.receptor_specs[rid].ionotropic is False

    def test_5ht_1a_1b_inhibitory(self):
        module = SerotoninReceptors()
        assert module.receptor_specs["5HT_1A"].signaling_type == "inhibitory"
        assert module.receptor_specs["5HT_1B"].signaling_type == "inhibitory"

    def test_5ht_2a_excitatory(self):
        module = SerotoninReceptors()
        assert module.receptor_specs["5HT_2A"].signaling_type == "excitatory"

    def test_ach_nicotinic_ionotropic(self):
        module = AcetylcholineReceptors()
        assert module.receptor_specs["ACh_nicotinic"].ionotropic is True

    def test_ach_muscarinic_metabotropic(self):
        module = AcetylcholineReceptors()
        assert module.receptor_specs["ACh_muscarinic"].ionotropic is False

    def test_gaba_a_ionotropic(self):
        module = GABAReceptors()
        assert module.receptor_specs["GABA_A"].ionotropic is True

    def test_gaba_b_metabotropic(self):
        module = GABAReceptors()
        assert module.receptor_specs["GABA_B"].ionotropic is False

    def test_gaba_both_inhibitory(self):
        module = GABAReceptors()
        assert module.receptor_specs["GABA_A"].signaling_type == "inhibitory"
        assert module.receptor_specs["GABA_B"].signaling_type == "inhibitory"

    def test_glu_nmda_ampa_kainate_ionotropic(self):
        module = GlutamateReceptors()
        assert module.receptor_specs["GLU_NMDA"].ionotropic is True
        assert module.receptor_specs["GLU_AMPA"].ionotropic is True
        assert module.receptor_specs["GLU_KAINATE"].ionotropic is True

    def test_glu_mglur_metabotropic(self):
        module = GlutamateReceptors()
        assert module.receptor_specs["GLU_mGluR"].ionotropic is False

    def test_glu_excitatory(self):
        module = GlutamateReceptors()
        assert module.receptor_specs["GLU_NMDA"].signaling_type == "excitatory"
        assert module.receptor_specs["GLU_AMPA"].signaling_type == "excitatory"

    def test_mor_inhibitory(self):
        module = OpioidReceptors()
        assert module.receptor_specs["MOR_mu"].signaling_type == "inhibitory"

    def test_cb1_inhibitory(self):
        module = CannabinoidReceptors()
        assert module.receptor_specs["CB1"].signaling_type == "inhibitory"

    def test_oxtr_excitatory(self):
        module = OxytocinReceptors()
        assert module.receptor_specs["OXTR"].signaling_type == "excitatory"

    def test_crh_r1_excitatory(self):
        module = CRHReceptors()
        assert module.receptor_specs["CRH_R1"].signaling_type == "excitatory"

    def test_ne_alpha1_excitatory(self):
        module = NorepinephrineReceptors()
        assert module.receptor_specs["NE_alpha1"].signaling_type == "excitatory"

    def test_ne_alpha2_inhibitory(self):
        module = NorepinephrineReceptors()
        assert module.receptor_specs["NE_alpha2"].signaling_type == "inhibitory"


# =====================================================================
# Effective Signaling Proxy Tests
# =====================================================================

class TestEffectiveSignaling:
    """Test A_ij = rho * sigma * g(chi) * S * w computation."""

    def test_active_state_full_params(self, receptor_module):
        """Full density, sensitivity, saturation in ACTIVE state."""
        for rid in receptor_module.get_receptor_ids():
            a_ij = receptor_module.compute_effective_signaling(
                receptor_id=rid,
                rho=1.0,
                sigma=1.0,
                functional_state="ACTIVE",
                saturation=1.0,
            )
            spec = receptor_module.receptor_specs[rid]
            # g(ACTIVE) = 1.0, so A_ij = 1 * 1 * 1 * 1 * w
            assert abs(a_ij - spec.effective_signaling_weight) < 1e-9

    def test_zero_saturation(self, receptor_module):
        """Zero saturation should give zero signaling."""
        for rid in receptor_module.get_receptor_ids():
            a_ij = receptor_module.compute_effective_signaling(
                receptor_id=rid,
                rho=1.0,
                sigma=1.0,
                functional_state="ACTIVE",
                saturation=0.0,
            )
            assert a_ij == 0.0

    def test_zero_density(self, receptor_module):
        """Zero receptor density should give zero signaling."""
        for rid in receptor_module.get_receptor_ids():
            a_ij = receptor_module.compute_effective_signaling(
                receptor_id=rid,
                rho=0.0,
                sigma=1.0,
                functional_state="ACTIVE",
                saturation=1.0,
            )
            assert a_ij == 0.0

    def test_desensitized_reduces_signaling(self, receptor_module):
        """DESENSITIZED state should reduce signaling vs ACTIVE."""
        for rid in receptor_module.get_receptor_ids():
            active = receptor_module.compute_effective_signaling(
                rid, 0.8, 0.7, "ACTIVE", 0.5,
            )
            desensitized = receptor_module.compute_effective_signaling(
                rid, 0.8, 0.7, "DESENSITIZED", 0.5,
            )
            assert desensitized < active

    def test_internalized_minimal_signaling(self, receptor_module):
        """INTERNALIZED state should give very low signaling."""
        for rid in receptor_module.get_receptor_ids():
            active = receptor_module.compute_effective_signaling(
                rid, 0.8, 0.7, "ACTIVE", 0.5,
            )
            internalized = receptor_module.compute_effective_signaling(
                rid, 0.8, 0.7, "INTERNALIZED", 0.5,
            )
            assert internalized < active * 0.2  # g(INTERNALIZED) = 0.1

    def test_upregulated_exceeds_active(self, receptor_module):
        """UPREGULATED state should exceed ACTIVE signaling."""
        for rid in receptor_module.get_receptor_ids():
            active = receptor_module.compute_effective_signaling(
                rid, 0.8, 0.7, "ACTIVE", 0.5,
            )
            upregulated = receptor_module.compute_effective_signaling(
                rid, 0.8, 0.7, "UPREGULATED", 0.5,
            )
            assert upregulated > active

    def test_signaling_non_negative(self, receptor_module):
        """Effective signaling should always be non-negative."""
        for rid in receptor_module.get_receptor_ids():
            for state in ["ACTIVE", "DESENSITIZED", "INTERNALIZED", "UPREGULATED"]:
                a_ij = receptor_module.compute_effective_signaling(
                    rid, 0.5, 0.5, state, 0.5,
                )
                assert a_ij >= 0.0

    def test_unknown_receptor_uses_default_weight(self, receptor_module):
        """Unknown receptor_id should use weight=1.0 fallback."""
        a_ij = receptor_module.compute_effective_signaling(
            receptor_id="NONEXISTENT_RECEPTOR",
            rho=1.0,
            sigma=1.0,
            functional_state="ACTIVE",
            saturation=1.0,
        )
        # g(ACTIVE) = 1.0, weight fallback = 1.0 → A_ij = 1.0
        assert abs(a_ij - 1.0) < 1e-9


# =====================================================================
# Emotion Plasticity Tests
# =====================================================================

class TestEmotionPlasticity:
    """Test emotion-specific plasticity hooks."""

    def test_at_least_one_receptor_has_plasticity(self, receptor_module):
        """Each family should have at least one receptor with plasticity rules."""
        has_plasticity = False
        for rid, spec in receptor_module.receptor_specs.items():
            if len(spec.emotion_plasticity_rules) > 0:
                has_plasticity = True
                break
        assert has_plasticity, (
            f"{receptor_module.__class__.__name__} has no emotion plasticity rules"
        )

    def test_get_emotion_plasticity_returns_dict(self, receptor_module):
        """Valid emotion/receptor pairs should return a dict."""
        for rid, spec in receptor_module.receptor_specs.items():
            for emotion_id in spec.emotion_plasticity_rules:
                result = receptor_module.get_emotion_plasticity(rid, emotion_id)
                assert isinstance(result, dict)
                assert len(result) > 0

    def test_get_emotion_plasticity_unknown_emotion(self, receptor_module):
        """Unknown emotion should return None."""
        for rid in receptor_module.get_receptor_ids():
            result = receptor_module.get_emotion_plasticity(
                rid, "completely_made_up_emotion",
            )
            assert result is None

    def test_get_emotion_plasticity_unknown_receptor(self, receptor_module):
        """Unknown receptor should return None."""
        result = receptor_module.get_emotion_plasticity(
            "NONEXISTENT", "joy",
        )
        assert result is None

    def test_plasticity_deltas_are_small(self, receptor_module):
        """Plasticity deltas should be small adjustments (|delta| < 0.5)."""
        for rid, spec in receptor_module.receptor_specs.items():
            for emotion_id, deltas in spec.emotion_plasticity_rules.items():
                for param, value in deltas.items():
                    assert abs(value) < 0.5, (
                        f"{rid}/{emotion_id}/{param} = {value} "
                        f"exceeds ±0.5 bound"
                    )


# =====================================================================
# Specific Family Tests
# =====================================================================

class TestDopamineReceptorsSpecific:
    """DA-specific receptor tests."""

    def test_five_subtypes(self):
        module = DopamineReceptors()
        assert len(module.receptor_specs) == 5

    def test_d4_novelty_plasticity(self):
        """D4 should have curiosity plasticity (novelty-seeking receptor)."""
        module = DopamineReceptors()
        result = module.get_emotion_plasticity("DA_D4", "curiosity")
        assert result is not None
        assert result["sigma_delta"] > 0  # Curiosity increases D4 sensitivity


class TestSerotoninReceptorsSpecific:
    """5-HT-specific receptor tests."""

    def test_five_subtypes(self):
        module = SerotoninReceptors()
        assert len(module.receptor_specs) == 5

    def test_2a_openness_plasticity(self):
        """5HT-2A should respond to openness (flexibility receptor)."""
        module = SerotoninReceptors()
        result = module.get_emotion_plasticity("5HT_2A", "openness")
        assert result is not None
        assert result["sigma_delta"] > 0


class TestGABAReceptorsSpecific:
    """GABA-specific receptor tests."""

    def test_two_subtypes(self):
        module = GABAReceptors()
        assert len(module.receptor_specs) == 2

    def test_gaba_a_anxiety_reduces_sensitivity(self):
        """Anxiety should reduce GABA_A sensitivity (disinhibition)."""
        module = GABAReceptors()
        result = module.get_emotion_plasticity("GABA_A", "anxiety")
        assert result is not None
        assert result["sigma_delta"] < 0


class TestGlutamateReceptorsSpecific:
    """GLU-specific receptor tests."""

    def test_four_subtypes(self):
        module = GlutamateReceptors()
        assert len(module.receptor_specs) == 4

    def test_nmda_learning_plasticity(self):
        """NMDA should have learning plasticity (coincidence detector)."""
        module = GlutamateReceptors()
        result = module.get_emotion_plasticity("GLU_NMDA", "learning")
        assert result is not None
        assert result["sigma_delta"] > 0


# =====================================================================
# Registry Tests
# =====================================================================

class TestReceptorModuleRegistry:
    """Test receptor module registration."""

    def test_register_all(self):
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()
        assert ReceptorModuleRegistry.count() == 11

    def test_cortisol_not_registered(self):
        """Cortisol has no receptor dynamics."""
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()
        assert not ReceptorModuleRegistry.is_registered("cortisol")
        ReceptorModuleRegistry.clear()

    def test_all_nts_with_receptors_registered(self):
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()
        for nt_name in NTS_WITH_RECEPTORS:
            assert ReceptorModuleRegistry.is_registered(nt_name), (
                f"No receptor module for NT: {nt_name}"
            )
        ReceptorModuleRegistry.clear()

    def test_get_returns_correct_module(self):
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()
        da_module = ReceptorModuleRegistry.get("DA")
        assert isinstance(da_module, DopamineReceptors)
        assert da_module.parent_nt == "DA"
        ReceptorModuleRegistry.clear()

    def test_get_nonexistent_returns_none(self):
        ReceptorModuleRegistry.clear()
        assert ReceptorModuleRegistry.get("NONEXISTENT") is None

    def test_clear(self):
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()
        assert ReceptorModuleRegistry.count() == 11
        ReceptorModuleRegistry.clear()
        assert ReceptorModuleRegistry.count() == 0

    def test_registered_names(self):
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()
        names = ReceptorModuleRegistry.registered_names()
        assert names == sorted(names)
        assert len(names) == 11
        ReceptorModuleRegistry.clear()

    def test_get_all(self):
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()
        all_modules = ReceptorModuleRegistry.get_all()
        assert len(all_modules) == 11
        for nt_name, module in all_modules.items():
            assert module.parent_nt == nt_name
        ReceptorModuleRegistry.clear()


# =====================================================================
# Cross-Family Completeness Tests
# =====================================================================

class TestCrossFamilyCompleteness:
    """Verify all 25 receptors are covered across all families."""

    def test_all_receptors_covered(self):
        """Every receptor in DEFAULT_RECEPTOR_CONFIGS should appear in
        exactly one receptor family module."""
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()

        covered = set()
        for nt_name, module in ReceptorModuleRegistry.get_all().items():
            for rid in module.get_receptor_ids():
                assert rid not in covered, f"Receptor {rid} appears in multiple modules"
                covered.add(rid)

        # All receptors with parent NT that has a receptor module should be covered
        expected = set()
        for nt_name in NTS_WITH_RECEPTORS:
            for rid in NT_RECEPTOR_MAP[nt_name]:
                expected.add(rid)

        assert covered == expected, (
            f"Missing: {expected - covered}, Extra: {covered - expected}"
        )
        ReceptorModuleRegistry.clear()

    def test_total_receptor_count(self):
        """Should cover all 30 receptors (all from NT_RECEPTOR_MAP)."""
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()

        total = 0
        for module in ReceptorModuleRegistry.get_all().values():
            total += len(module.receptor_specs)

        # Count expected from NT_RECEPTOR_MAP (excluding cortisol which has [])
        expected = sum(len(recs) for recs in NT_RECEPTOR_MAP.values())
        assert total == expected
        ReceptorModuleRegistry.clear()

    def test_parent_nt_configs_match(self):
        """Each receptor's parent_nt from config should match module's parent_nt."""
        ReceptorModuleRegistry.clear()
        register_all_receptor_modules()

        for module in ReceptorModuleRegistry.get_all().values():
            for rid in module.get_receptor_ids():
                config_parent = DEFAULT_RECEPTOR_CONFIGS[rid].get("parent_nt")
                if config_parent:
                    assert config_parent == module.parent_nt, (
                        f"{rid}: config parent_nt={config_parent}, "
                        f"module parent_nt={module.parent_nt}"
                    )
        ReceptorModuleRegistry.clear()
