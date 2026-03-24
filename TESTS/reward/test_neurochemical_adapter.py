import pytest
from zados.reward.adapter.neurochemical_adapter import NeurochemicalAdapter
from zados.reward.adapter.mapping import (
    map_innovation_to_dopamine,
    map_logic_to_norepinephrine,
    map_attunement_to_oxytocin,
    map_ethics_to_constraint_awareness,
    map_flags_to_stress_response,
    compute_motivation_modulation,
)
from zados.reward.base.structure import RewardFlag


# ---------------------------------------------------------------------
# Mapping function tests
# ---------------------------------------------------------------------


def test_map_innovation_to_dopamine_full():
    """Innovation scores map to DA signals."""
    innovation_result = {
        "subscores": {
            "novelty_generation": {"score": 0.8},
            "conceptual_novelty": {"score": 0.6},
            "exploration_drive": {"score": 0.7},
            "resolution_satisfaction": {"score": 0.5},
            "pattern_divergence": {"score": 0.4},
        }
    }
    
    signals = map_innovation_to_dopamine(innovation_result)
    
    assert "novelty" in signals
    assert "rpe" in signals
    assert "tonic_bias" in signals
    
    # Novelty = avg of novelty_generation and conceptual_novelty
    assert signals["novelty"] == pytest.approx((0.8 + 0.6) / 2)
    
    # RPE = (exploration + resolution) / 2 - 0.5
    assert signals["rpe"] == pytest.approx((0.7 + 0.5) / 2 - 0.5)
    
    # Tonic bias = divergence * 0.2
    assert signals["tonic_bias"] == pytest.approx(0.4 * 0.2)


def test_map_innovation_to_dopamine_none():
    """None innovation result returns zero signals."""
    signals = map_innovation_to_dopamine(None)
    
    assert signals["novelty"] == 0.0
    assert signals["rpe"] == 0.0
    assert signals["tonic_bias"] == 0.0


def test_map_logic_to_norepinephrine():
    """Logic scores map to NE precision/uncertainty."""
    logic_result = {
        "subscores": {
            "epistemic_calibration": {"score": 0.8},
            "internal_consistency": {"score": 0.7},
            "external_consistency": {"score": 0.6},
            "uncertainty_acknowledgment": {"score": 0.5},
        }
    }
    
    signals = map_logic_to_norepinephrine(logic_result)
    
    assert "precision" in signals
    assert "uncertainty" in signals
    
    # Low consistency → high precision need
    consistency_avg = (0.7 + 0.6) / 2
    expected_precision = (1.0 - consistency_avg + (1.0 - 0.8)) / 2
    
    assert signals["precision"] == pytest.approx(expected_precision)
    assert signals["uncertainty"] == pytest.approx(1.0 - 0.5)


def test_map_logic_to_norepinephrine_none():
    """None logic result returns defaults."""
    signals = map_logic_to_norepinephrine(None)
    
    assert signals["precision"] == 0.0
    assert signals["uncertainty"] == 0.5


def test_map_attunement_to_oxytocin():
    """Attunement scores map to OXT signals."""
    attunement_result = {
        "subscores": {
            "empathetic_inference": {"score": 0.8},
            "cognitive_reading": {"score": 0.7},
            "intention_calibration": {"score": 0.6},
            "attuned_dissonance": {"score": 0.5},
        }
    }
    
    signals = map_attunement_to_oxytocin(attunement_result)
    
    assert "empathy" in signals
    assert "social_engagement" in signals
    
    assert signals["empathy"] == pytest.approx((0.8 + 0.7) / 2)
    assert signals["social_engagement"] == pytest.approx((0.6 + 0.5) / 2)


def test_map_attunement_to_oxytocin_none():
    """None attunement result returns zeros."""
    signals = map_attunement_to_oxytocin(None)
    
    assert signals["empathy"] == 0.0
    assert signals["social_engagement"] == 0.0


def test_map_ethics_to_constraint_awareness():
    """Ethics scores map to constraint awareness."""
    ethics_result = {
        "subscores": {
            "failure_mode_awareness": {"score": 0.7},
            "downstream_risk_amplification": {"score": 0.6},
        }
    }
    
    signals = map_ethics_to_constraint_awareness(ethics_result)
    
    assert "risk_awareness" in signals
    assert "boundary_proximity" in signals
    
    assert signals["risk_awareness"] == pytest.approx(0.7)
    assert signals["boundary_proximity"] == pytest.approx(1.0 - 0.6)


def test_map_flags_to_stress_response_empty():
    """No flags returns zero stress."""
    signals = map_flags_to_stress_response({})
    
    assert signals["cortisol"] == 0.0
    assert signals["crh"] == 0.0


def test_map_flags_to_stress_response_critical():
    """Critical flags elevate cortisol/CRH."""
    flags = {
        "flag1": {"severity": "critical"},
        "flag2": {"severity": "risk"},
        "flag3": {"severity": "warning"},
    }
    
    signals = map_flags_to_stress_response(flags)
    
    # Critical: 1 * 0.4, Risk: 1 * 0.2, Warning: 1 * 0.1
    assert signals["cortisol"] == pytest.approx(0.4 + 0.2 + 0.1)
    
    # CRH: Critical 1 * 0.5, Risk: 1 * 0.25
    assert signals["crh"] == pytest.approx(0.5 + 0.25)


def test_map_flags_to_stress_response_capped():
    """Stress signals are capped at 1.0."""
    flags = {
        f"flag{i}": {"severity": "critical"}
        for i in range(10)
    }
    
    signals = map_flags_to_stress_response(flags)
    
    assert signals["cortisol"] == 1.0
    assert signals["crh"] == 1.0


def test_compute_motivation_modulation_suppress():
    """Suppression dampens motivation."""
    meta_directive = {"suppress": True, "abstain": False}
    innovation_signals = {"novelty": 0.8, "rpe": 0.2}
    
    motivation = compute_motivation_modulation(meta_directive, innovation_signals)
    
    assert motivation == pytest.approx(-0.4)


def test_compute_motivation_modulation_abstain():
    """Abstention dampens motivation."""
    meta_directive = {"suppress": False, "abstain": True}
    innovation_signals = {"novelty": 0.8, "rpe": 0.2}
    
    motivation = compute_motivation_modulation(meta_directive, innovation_signals)
    
    assert motivation == pytest.approx(-0.3)


def test_compute_motivation_modulation_allow():
    """Allow directive uses innovation signals."""
    meta_directive = {"suppress": False, "abstain": False}
    innovation_signals = {"novelty": 0.8, "rpe": 0.2}
    
    motivation = compute_motivation_modulation(meta_directive, innovation_signals)
    
    expected = (0.8 * 0.3) + (0.2 * 0.2)
    assert motivation == pytest.approx(expected)


def test_compute_motivation_modulation_none():
    """No directive returns neutral motivation."""
    motivation = compute_motivation_modulation(None, {"novelty": 0.5, "rpe": 0.1})
    
    assert motivation == 0.0


# ---------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------


def test_adapter_transform_all_domains():
    """Adapter transforms all domains correctly."""
    adapter = NeurochemicalAdapter()
    
    domain_results = {
        "innovation": {
            "domain": "innovation",
            "general_score": 0.7,
            "subscores": {
                "novelty_generation": {"score": 0.8},
                "conceptual_novelty": {"score": 0.6},
                "exploration_drive": {"score": 0.7},
                "resolution_satisfaction": {"score": 0.5},
                "pattern_divergence": {"score": 0.4},
            },
            "flags": {},
            "meta": {},
        },
        "logic": {
            "domain": "logic",
            "general_score": 0.6,
            "subscores": {
                "epistemic_calibration": {"score": 0.8},
                "internal_consistency": {"score": 0.7},
                "external_consistency": {"score": 0.6},
                "uncertainty_acknowledgment": {"score": 0.5},
            },
            "flags": {},
            "meta": {},
        },
        "human_attunement": {
            "domain": "human_attunement",
            "general_score": 0.7,
            "subscores": {
                "empathetic_inference": {"score": 0.8},
                "cognitive_reading": {"score": 0.7},
                "intention_calibration": {"score": 0.6},
                "attuned_dissonance": {"score": 0.5},
            },
            "flags": {},
            "meta": {},
        },
        "ethics": {
            "domain": "ethics",
            "general_score": 0.8,
            "subscores": {
                "failure_mode_awareness": {"score": 0.7},
                "downstream_risk_amplification": {"score": 0.6},
            },
            "flags": {},
            "meta": {},
        },
    }
    
    signals = adapter.transform(domain_results)
    
    # Check all NT systems are present
    assert "DA" in signals
    assert "NE" in signals
    assert "OXT" in signals
    assert "cortisol" in signals
    assert "CRH" in signals
    assert "GABA" in signals
    assert "motivation_modulation" in signals
    assert "meta" in signals
    
    # Check DA signals
    assert signals["DA"]["novelty"] == pytest.approx((0.8 + 0.6) / 2)
    assert signals["DA"]["rpe"] == pytest.approx((0.7 + 0.5) / 2 - 0.5)
    assert signals["DA"]["effort"] == 0.0
    
    # Check NE signals
    assert "precision" in signals["NE"]
    assert "uncertainty" in signals["NE"]
    
    # Check OXT signals
    assert signals["OXT"]["empathy"] == pytest.approx((0.8 + 0.7) / 2)
    assert signals["OXT"]["social_engagement"] == pytest.approx((0.6 + 0.5) / 2)
    
    # Check stress signals
    assert signals["cortisol"]["level"] == 0.0  # No flags
    assert signals["CRH"]["level"] == 0.0


def test_adapter_transform_missing_domains():
    """Adapter handles missing domains gracefully."""
    adapter = NeurochemicalAdapter()
    
    domain_results = {
        "innovation": {
            "subscores": {
                "novelty_generation": {"score": 0.8},
            },
            "flags": {},
        },
    }
    
    signals = adapter.transform(domain_results)
    
    # Should still produce valid output with defaults
    assert signals["DA"]["novelty"] > 0.0
    assert signals["NE"]["precision"] == 0.0  # Default from missing logic
    assert signals["OXT"]["empathy"] == 0.0  # Default from missing attunement


def test_adapter_transform_with_flags():
    """Adapter aggregates flags across domains."""
    adapter = NeurochemicalAdapter()
    
    domain_results = {
        "innovation": {
            "subscores": {},
            "flags": {
                "reckless_exploration": {"severity": "risk"},
            },
        },
        "logic": {
            "subscores": {},
            "flags": {
                "internal_contradiction": {"severity": "critical"},
            },
        },
    }
    
    signals = adapter.transform(domain_results)
    
    # Flags should elevate stress
    assert signals["cortisol"]["level"] > 0.0
    assert signals["CRH"]["level"] > 0.0


def test_adapter_transform_with_meta_directive():
    """Adapter handles meta-directives."""
    adapter = NeurochemicalAdapter()
    
    domain_results = {
        "innovation": {
            "subscores": {
                "novelty_generation": {"score": 0.8},
            },
            "flags": {},
        },
    }
    
    # Mock meta-directive
    class MockMetaDirective:
        def __init__(self):
            self.suppress = True
            self.abstain = False
        
        def as_dict(self):
            return {"suppress": True, "abstain": False}
    
    meta = MockMetaDirective()
    
    signals = adapter.transform(domain_results, meta_directive=meta)
    
    # Suppression should dampen motivation
    assert signals["motivation_modulation"] == pytest.approx(-0.4)


def test_adapter_extract_domain_from_object():
    """Adapter converts RewardDomainResult objects to dicts."""
    adapter = NeurochemicalAdapter()
    
    # Mock object with attributes
    class MockDomainResult:
        def __init__(self):
            self.domain = "test"
            self.general_score = 0.7
            self.subscores = {"test_sub": {"score": 0.8}}
            self.flags = {}
            self.meta = {}
    
    mock_result = MockDomainResult()
    
    domain_results = {"test": mock_result}
    
    extracted = adapter._extract_domain(domain_results, "test")
    
    assert extracted is not None
    assert extracted["domain"] == "test"
    assert extracted["general_score"] == 0.7


def test_adapter_aggregate_flags():
    """Adapter aggregates flags with domain prefixes."""
    adapter = NeurochemicalAdapter()
    
    class MockResult:
        def __init__(self, domain, flags):
            self.domain = domain
            self.flags = flags
    
    domain_results = {
        "innovation": MockResult("innovation", {"flag1": {"severity": "risk"}}),
        "logic": MockResult("logic", {"flag2": {"severity": "warning"}}),
    }
    
    all_flags = adapter._aggregate_flags(domain_results)
    
    assert "innovation_flag1" in all_flags
    assert "logic_flag2" in all_flags
    assert len(all_flags) == 2


# =====================================================================
# Sleep metric NT signal injection
# =====================================================================


def _make_sleep_domain_results(**kwargs):
    """Build minimal domain results dict from keyword scores."""
    results = {}
    for domain, score in kwargs.items():
        results[domain] = {
            "domain": domain,
            "general_score": score,
            "subscores": {},
            "flags": {},
            "meta": {},
        }
    return results


def test_transform_no_sleep_metrics():
    """Without sleep metrics, no extra sleep keys in output."""
    adapter = NeurochemicalAdapter()
    domain_results = _make_sleep_domain_results(
        innovation=0.7, logic=0.6, human_attunement=0.5, ethics=0.6,
    )
    signals = adapter.transform(domain_results)
    assert "sleep_metrics" not in signals.get("meta", {})


def test_transform_dream_permissiveness_injects_da_cb1():
    """dream_permissiveness > 0 injects DA and CB1 boost signals."""
    adapter = NeurochemicalAdapter()
    domain_results = _make_sleep_domain_results(
        innovation=0.7, logic=0.6, human_attunement=0.5, ethics=0.6,
    )
    sm = {"dream_permissiveness": 0.8}
    signals = adapter.transform(domain_results, sleep_metrics=sm)
    assert signals["DA"]["dream_release_boost"] == pytest.approx(0.8 * 0.3)
    assert signals["CB1"]["dream_baseline_boost"] == pytest.approx(0.8 * 0.2)


def test_transform_consolidation_injects_gaba_glu():
    """consolidation_depth > 0 injects GABA and GLU boost signals."""
    adapter = NeurochemicalAdapter()
    domain_results = _make_sleep_domain_results(
        innovation=0.5, logic=0.5, human_attunement=0.5, ethics=0.5,
    )
    sm = {"consolidation_depth": 0.6}
    signals = adapter.transform(domain_results, sleep_metrics=sm)
    assert signals["GABA"]["consolidation_baseline_boost"] == pytest.approx(0.6 * 0.25)
    assert signals["GLU"]["consolidation_nmda_boost"] == pytest.approx(0.6 * 0.2)


def test_transform_narrative_plasticity_injects_da_d3():
    """narrative_plasticity > 0 injects DA d3 boost."""
    adapter = NeurochemicalAdapter()
    domain_results = _make_sleep_domain_results(
        innovation=0.5, logic=0.5, human_attunement=0.5, ethics=0.5,
    )
    sm = {"narrative_plasticity": 0.5}
    signals = adapter.transform(domain_results, sleep_metrics=sm)
    assert signals["DA"]["narrative_d3_boost"] == pytest.approx(0.5 * 0.2)


def test_transform_sleep_metrics_zero_no_injection():
    """All-zero sleep metrics don't inject extra signals."""
    adapter = NeurochemicalAdapter()
    domain_results = _make_sleep_domain_results(
        innovation=0.5, logic=0.5, human_attunement=0.5, ethics=0.5,
    )
    sm = {"dream_permissiveness": 0.0, "consolidation_depth": 0.0, "narrative_plasticity": 0.0}
    signals = adapter.transform(domain_results, sleep_metrics=sm)
    assert "dream_release_boost" not in signals["DA"]
    assert "consolidation_baseline_boost" not in signals["GABA"]