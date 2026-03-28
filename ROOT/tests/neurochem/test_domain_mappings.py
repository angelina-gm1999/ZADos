"""
Tests for domain-to-NT mapping modules.

Phase 14: Verifies that each domain mapping produces valid NT signals,
handles missing subscores gracefully, and targets the correct NTs.
"""

import pytest

from zados.neurochem.domains.base import DomainNTMapping, NTSignalMapping
from zados.neurochem.domains.innovation import InnovationMapping
from zados.neurochem.domains.logic import LogicMapping
from zados.neurochem.domains.human_attunement import HumanAttunementMapping
from zados.neurochem.domains.ethics import EthicsMapping
from zados.neurochem.neurotransmitters.configs import DEFAULT_NT_CONFIGS


ALL_DOMAIN_CLASSES = [
    InnovationMapping,
    LogicMapping,
    HumanAttunementMapping,
    EthicsMapping,
]

# All valid NT names
VALID_NTS = set(DEFAULT_NT_CONFIGS.keys())


@pytest.fixture(
    params=ALL_DOMAIN_CLASSES,
    ids=[c.__name__ for c in ALL_DOMAIN_CLASSES],
)
def domain(request):
    """Parametrized fixture providing each domain mapping instance."""
    return request.param()


# =====================================================================
# NTSignalMapping Unit Tests
# =====================================================================

class TestNTSignalMapping:
    """Test NTSignalMapping dataclass and computation."""

    def test_basic_compute(self):
        m = NTSignalMapping(nt_name="DA", signal_key="novelty", weight=0.5)
        assert abs(m.compute(0.8) - 0.4) < 1e-9

    def test_compute_invert(self):
        m = NTSignalMapping(
            nt_name="NE", signal_key="precision",
            weight=1.0, invert=True,
        )
        assert abs(m.compute(0.3) - 0.7) < 1e-9

    def test_compute_with_offset(self):
        m = NTSignalMapping(
            nt_name="DA", signal_key="rpe",
            weight=0.5, offset=-0.25,
        )
        # 0.5 * 1.0 + (-0.25) = 0.25
        assert abs(m.compute(1.0) - 0.25) < 1e-9

    def test_compute_invert_with_offset(self):
        m = NTSignalMapping(
            nt_name="DA", signal_key="rpe",
            weight=1.0, invert=True, offset=-0.5,
        )
        # 1.0 * (1.0 - 0.6) + (-0.5) = 0.4 - 0.5 = -0.1
        assert abs(m.compute(0.6) - (-0.1)) < 1e-9

    def test_compute_zero_input(self):
        m = NTSignalMapping(nt_name="DA", signal_key="novelty", weight=0.5)
        assert m.compute(0.0) == 0.0

    def test_frozen(self):
        m = NTSignalMapping(nt_name="DA", signal_key="novelty")
        with pytest.raises(AttributeError):
            m.nt_name = "NE"


# =====================================================================
# Structure Tests (parametrized across all domains)
# =====================================================================

class TestDomainStructure:
    """Every domain mapping must have consistent structure."""

    def test_is_domain_mapping(self, domain):
        assert isinstance(domain, DomainNTMapping)

    def test_has_domain_name(self, domain):
        assert isinstance(domain.domain_name, str)
        assert len(domain.domain_name) > 0

    def test_has_target_nts(self, domain):
        assert isinstance(domain.target_nts, list)
        assert len(domain.target_nts) >= 1

    def test_target_nts_valid(self, domain):
        """All target NTs must exist in DEFAULT_NT_CONFIGS."""
        for nt_name in domain.target_nts:
            assert nt_name in VALID_NTS, (
                f"{domain.__class__.__name__} targets unknown NT: {nt_name}"
            )

    def test_has_signal_mappings(self, domain):
        assert isinstance(domain.signal_mappings, dict)
        assert len(domain.signal_mappings) > 0

    def test_signal_mappings_target_valid_nts(self, domain):
        """All NT targets in mappings must be valid NTs."""
        for subscore, mappings in domain.signal_mappings.items():
            for m in mappings:
                assert m.nt_name in VALID_NTS, (
                    f"{domain.domain_name}/{subscore} targets "
                    f"unknown NT: {m.nt_name}"
                )

    def test_signal_mappings_target_listed_nts(self, domain):
        """All NTs in mappings should be in target_nts list."""
        target_set = set(domain.target_nts)
        for subscore, mappings in domain.signal_mappings.items():
            for m in mappings:
                assert m.nt_name in target_set, (
                    f"{domain.domain_name}/{subscore} targets "
                    f"{m.nt_name} which is not in target_nts"
                )

    def test_all_target_nts_have_mappings(self, domain):
        """Every listed target NT should have at least one mapping."""
        mapped_nts = set()
        for mappings in domain.signal_mappings.values():
            for m in mappings:
                mapped_nts.add(m.nt_name)
        for nt in domain.target_nts:
            assert nt in mapped_nts, (
                f"{domain.domain_name}: {nt} listed as target but has no mappings"
            )


# =====================================================================
# map_subscores Tests
# =====================================================================

class TestMapSubscores:
    """Test the map_subscores method."""

    def test_empty_subscores(self, domain):
        """Empty subscores should return empty dict."""
        result = domain.map_subscores({})
        assert isinstance(result, dict)
        # May return empty dict or dict with some defaults

    def test_dict_subscores(self, domain):
        """Subscores as dicts with 'score' key should work."""
        # Build subscores with all keys set to 0.5
        subscores = {}
        for key in domain.signal_mappings:
            subscores[key] = {"score": 0.5}
        result = domain.map_subscores(subscores)
        assert isinstance(result, dict)
        # Should have entries for target NTs
        for nt in result:
            assert nt in VALID_NTS

    def test_float_subscores(self, domain):
        """Subscores as raw floats should work."""
        subscores = {key: 0.7 for key in domain.signal_mappings}
        result = domain.map_subscores(subscores)
        assert isinstance(result, dict)

    def test_high_subscores_produce_signals(self, domain):
        """High subscores should produce non-zero signals."""
        subscores = {key: {"score": 0.9} for key in domain.signal_mappings}
        result = domain.map_subscores(subscores)
        # At least one NT should have signals
        total_signals = sum(
            sum(abs(v) for v in signals.values())
            for signals in result.values()
        )
        assert total_signals > 0.0

    def test_unknown_subscores_ignored(self, domain):
        """Subscores not in signal_mappings should be ignored."""
        result = domain.map_subscores({
            "completely_unknown_subscore": 0.9,
        })
        # Should produce no signals from unknown subscore
        total = sum(
            sum(abs(v) for v in signals.values())
            for signals in result.values()
        )
        assert total == 0.0

    def test_partial_subscores(self, domain):
        """Should work with only some subscores provided."""
        first_key = next(iter(domain.signal_mappings))
        result = domain.map_subscores({first_key: 0.8})
        assert isinstance(result, dict)

    def test_get_mappings_for_nt(self, domain):
        """get_mappings_for_nt should return relevant mappings."""
        for nt in domain.target_nts:
            mappings = domain.get_mappings_for_nt(nt)
            assert len(mappings) > 0
            for m in mappings:
                assert m.nt_name == nt


# =====================================================================
# Innovation-Specific Tests
# =====================================================================

class TestInnovationMapping:
    """Innovation domain specific tests."""

    def test_domain_name(self):
        m = InnovationMapping()
        assert m.domain_name == "innovation"

    def test_targets_da_and_cb1(self):
        m = InnovationMapping()
        assert set(m.target_nts) == {"DA", "CB1"}

    def test_novelty_maps_to_da(self):
        m = InnovationMapping()
        result = m.map_subscores({
            "novelty_generation": {"score": 0.8},
        })
        assert "DA" in result
        assert "novelty" in result["DA"]
        assert result["DA"]["novelty"] > 0.0

    def test_rpe_centered_around_zero(self):
        """RPE signals should be centered around 0 at score=0.5."""
        m = InnovationMapping()
        result = m.map_subscores({
            "exploration_drive": {"score": 0.5},
        })
        # At score=0.5: 0.5 * 0.5 + (-0.25) = 0.0
        assert "DA" in result
        assert abs(result["DA"]["rpe"]) < 0.01

    def test_pattern_divergence_maps_to_cb1(self):
        m = InnovationMapping()
        result = m.map_subscores({
            "pattern_divergence": {"score": 0.7},
        })
        assert "CB1" in result
        assert "flexibility" in result["CB1"]
        assert result["CB1"]["flexibility"] > 0.0


# =====================================================================
# Logic-Specific Tests
# =====================================================================

class TestLogicMapping:
    """Logic domain specific tests."""

    def test_domain_name(self):
        m = LogicMapping()
        assert m.domain_name == "logic"

    def test_targets_ne_ach_glu(self):
        m = LogicMapping()
        assert set(m.target_nts) == {"NE", "ACh", "GLU"}

    def test_low_consistency_high_precision(self):
        """Low internal consistency should drive high NE precision."""
        m = LogicMapping()
        result = m.map_subscores({
            "internal_consistency": {"score": 0.2},  # Low
        })
        assert "NE" in result
        assert "precision" in result["NE"]
        assert result["NE"]["precision"] > 0.0

    def test_low_consistency_high_contradiction(self):
        """Low consistency should drive high contradiction signal."""
        m = LogicMapping()
        result = m.map_subscores({
            "internal_consistency": {"score": 0.1},  # Very low
        })
        assert "NE" in result
        assert result["NE"]["contradiction"] > 0.5

    def test_inferential_rigor_maps_to_ach_and_glu(self):
        m = LogicMapping()
        result = m.map_subscores({
            "inferential_rigor": {"score": 0.8},
        })
        assert "ACh" in result
        assert "attention_demand" in result["ACh"]
        assert "GLU" in result
        assert "integration_demand" in result["GLU"]


# =====================================================================
# Human Attunement-Specific Tests
# =====================================================================

class TestHumanAttunementMapping:
    """Human Attunement domain specific tests."""

    def test_domain_name(self):
        m = HumanAttunementMapping()
        assert m.domain_name == "human_attunement"

    def test_targets_oxt_5ht_mor(self):
        m = HumanAttunementMapping()
        assert set(m.target_nts) == {"OXT", "5HT", "MOR"}

    def test_empathy_maps_to_oxt(self):
        m = HumanAttunementMapping()
        result = m.map_subscores({
            "empathetic_inference": {"score": 0.8},
        })
        assert "OXT" in result
        assert "empathy" in result["OXT"]
        assert result["OXT"]["empathy"] > 0.0

    def test_emotional_resonance_maps_to_5ht_and_mor(self):
        m = HumanAttunementMapping()
        result = m.map_subscores({
            "emotional_resonance": {"score": 0.7},
        })
        assert "5HT" in result
        assert "mood_stability" in result["5HT"]
        assert "MOR" in result
        assert "hedonic_tone" in result["MOR"]

    def test_comfort_provision_maps_to_mor(self):
        m = HumanAttunementMapping()
        result = m.map_subscores({
            "comfort_provision": {"score": 0.9},
        })
        assert "MOR" in result
        assert result["MOR"]["comfort"] > 0.5


# =====================================================================
# Ethics-Specific Tests
# =====================================================================

class TestEthicsMapping:
    """Ethics domain specific tests."""

    def test_domain_name(self):
        m = EthicsMapping()
        assert m.domain_name == "ethics"

    def test_targets_gaba_cortisol_crh(self):
        m = EthicsMapping()
        assert set(m.target_nts) == {"GABA", "cortisol", "CRH"}

    def test_failure_awareness_maps_to_gaba(self):
        m = EthicsMapping()
        result = m.map_subscores({
            "failure_mode_awareness": {"score": 0.8},
        })
        assert "GABA" in result
        assert "inhibition" in result["GABA"]
        assert result["GABA"]["inhibition"] > 0.0

    def test_low_risk_score_high_boundary_proximity(self):
        """Low risk amplification score → high boundary proximity."""
        m = EthicsMapping()
        result = m.map_subscores({
            "downstream_risk_amplification": {"score": 0.1},  # Low score = bad
        })
        assert "GABA" in result
        assert result["GABA"]["boundary_proximity"] > 0.5

    def test_stakes_maps_to_cortisol_and_crh(self):
        m = EthicsMapping()
        result = m.map_subscores({
            "stakes_assessment": {"score": 0.9},  # High stakes
        })
        assert "cortisol" in result
        assert "stress_level" in result["cortisol"]
        assert "CRH" in result
        assert "acute_stress" in result["CRH"]

    def test_low_consistency_high_pressure(self):
        """Low ethical consistency → high CRH pressure scaling."""
        m = EthicsMapping()
        result = m.map_subscores({
            "ethical_consistency": {"score": 0.2},  # Low
        })
        assert "CRH" in result
        assert result["CRH"]["pressure_scaling"] > 0.4


# =====================================================================
# Cross-Domain Completeness
# =====================================================================

class TestCrossDomainCompleteness:
    """Verify that all 11 NTs are reachable from domain mappings."""

    def test_all_domains_cover_11_nts(self):
        """The 4 domains together should cover all 11 NTs via
        direct targeting or emotion_drive."""
        covered = set()
        for cls in ALL_DOMAIN_CLASSES:
            domain = cls()
            covered.update(domain.target_nts)

        # These are the NTs directly targeted by domain mappings
        directly_covered = covered

        # NTs not directly covered should still receive signals via
        # emotion_drive (every NT module accepts it)
        # For now, verify the key ones are covered
        expected_direct = {"DA", "CB1", "NE", "ACh", "GLU", "OXT", "5HT", "MOR",
                           "GABA", "cortisol", "CRH"}
        assert directly_covered == expected_direct

    def test_no_duplicate_domain_names(self):
        names = [cls().domain_name for cls in ALL_DOMAIN_CLASSES]
        assert len(names) == len(set(names))
