"""
Tests for Engine 27 -- Neurochemical Homeostatic Engine
=======================================================
Covers: enums, config, NT bounds, cognitive load estimation,
bound checking, correction computation, health status, neurochem
coupling, engine pipeline, mode tolerance, edge cases.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
    BoundViolation,
    CognitiveLoadEstimate,
    CorrectionType,
    HealthStatus,
    HomeostaticConfig,
    HomeostaticInput,
    HomeostaticResult,
    NTViolationRecord,
    NeurochemicalHomeostaticEngine,
    _NT_BOUNDS,
    check_nt_bounds,
    compute_cognitive_load,
    compute_correction,
    compute_health_status,
    compute_homeostatic_neurochem,
    sigmoid,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Fixtures
# =====================================================================

RNG = np.random.default_rng(42)
CFG = HomeostaticConfig()


# =====================================================================
# Enums
# =====================================================================


class TestEnums:
    def test_bound_violations(self):
        assert len(BoundViolation) == 4

    def test_correction_types(self):
        assert len(CorrectionType) == 3

    def test_health_statuses(self):
        assert len(HealthStatus) == 4


# =====================================================================
# Config
# =====================================================================


class TestConfig:
    def test_defaults(self):
        assert CFG.overload_threshold == 0.85
        assert CFG.w_urgency > CFG.w_symbolic  # urgency highest weight

    def test_frozen(self):
        with pytest.raises(AttributeError):
            CFG.overload_threshold = 0.99


# =====================================================================
# NT Bounds registry
# =====================================================================


class TestNTBounds:
    def test_all_12_nts(self):
        assert len(_NT_BOUNDS) == 12

    def test_bounds_ordered(self):
        for nt, (low, baseline, high, crit_low, crit_high) in _NT_BOUNDS.items():
            assert crit_low <= low <= baseline <= high <= crit_high, f"{nt} bounds out of order"

    def test_baseline_in_range(self):
        for nt, (low, baseline, high, _, _) in _NT_BOUNDS.items():
            assert low <= baseline <= high


# =====================================================================
# Sigmoid
# =====================================================================


class TestSigmoid:
    def test_zero(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert sigmoid(20.0) > 0.99

    def test_large_negative(self):
        assert sigmoid(-20.0) < 0.01

    def test_monotonic(self):
        assert sigmoid(1.0) > sigmoid(0.0) > sigmoid(-1.0)


# =====================================================================
# Cognitive load estimation
# =====================================================================


class TestCognitiveLoad:
    def test_zero_inputs(self):
        load = compute_cognitive_load(0.0, 0.0, 0.0, 0.0, CFG)
        assert load == pytest.approx(0.5)  # sigmoid(0) = 0.5

    def test_high_urgency_overload(self):
        load = compute_cognitive_load(0.5, 0.5, 2.0, 0.0, CFG)
        assert load > 0.85  # Should be near overload

    def test_urgency_dominant(self):
        load_urg = compute_cognitive_load(0.0, 0.0, 1.0, 0.0, CFG)
        load_sym = compute_cognitive_load(1.0, 0.0, 0.0, 0.0, CFG)
        # Urgency has weight 1.5 vs symbolic 1.0
        assert load_urg > load_sym

    def test_epsilon_modifier(self):
        load_base = compute_cognitive_load(0.5, 0.5, 0.5, 0.0, CFG)
        load_spike = compute_cognitive_load(0.5, 0.5, 0.5, 1.0, CFG)
        assert load_spike > load_base


# =====================================================================
# Bound checking
# =====================================================================


class TestBoundChecking:
    def test_normal_within_bounds(self):
        violation, low, high, baseline = check_nt_bounds("DA", 0.40, OperationalMode.NORMAL, CFG)
        assert violation == BoundViolation.NONE

    def test_elevated(self):
        violation, _, _, _ = check_nt_bounds("DA", 0.80, OperationalMode.NORMAL, CFG)
        assert violation == BoundViolation.ELEVATED

    def test_depleted(self):
        violation, _, _, _ = check_nt_bounds("DA", 0.07, OperationalMode.NORMAL, CFG)
        assert violation == BoundViolation.DEPLETED

    def test_critical_high(self):
        violation, _, _, _ = check_nt_bounds("DA", 0.95, OperationalMode.NORMAL, CFG)
        assert violation == BoundViolation.CRITICAL

    def test_critical_low(self):
        violation, _, _, _ = check_nt_bounds("DA", 0.02, OperationalMode.NORMAL, CFG)
        assert violation == BoundViolation.CRITICAL

    def test_dream_mode_wider_bounds(self):
        # In dream mode, high bound is expanded
        violation_normal, _, high_n, _ = check_nt_bounds("DA", 0.80, OperationalMode.NORMAL, CFG)
        violation_dream, _, high_d, _ = check_nt_bounds("DA", 0.80, OperationalMode.REM_DREAM, CFG)
        assert high_d > high_n
        # 0.80 might be elevated in normal but ok in dream
        if violation_normal == BoundViolation.ELEVATED:
            assert violation_dream in (BoundViolation.NONE, BoundViolation.ELEVATED)

    def test_dev_mode_tighter_bounds(self):
        _, _, high_n, _ = check_nt_bounds("DA", 0.5, OperationalMode.NORMAL, CFG)
        _, _, high_d, _ = check_nt_bounds("DA", 0.5, OperationalMode.DEV, CFG)
        # Dev has tolerance 0.90 → tighter high bound
        assert high_d <= high_n

    def test_unknown_nt_uses_defaults(self):
        violation, low, high, baseline = check_nt_bounds("UNKNOWN", 0.40, OperationalMode.NORMAL, CFG)
        assert violation == BoundViolation.NONE


# =====================================================================
# Correction computation
# =====================================================================


class TestCorrection:
    def test_no_violation_no_correction(self):
        ctype, delta = compute_correction(0.40, 0.40, BoundViolation.NONE, 0, CFG)
        assert ctype == CorrectionType.GRADUAL
        assert delta == 0.0

    def test_gradual_correction(self):
        ctype, delta = compute_correction(0.80, 0.40, BoundViolation.ELEVATED, 1, CFG)
        assert ctype == CorrectionType.GRADUAL
        assert delta < 0  # Pull down

    def test_aggressive_after_threshold(self):
        ctype, delta = compute_correction(0.80, 0.40, BoundViolation.ELEVATED,
                                          CFG.cycles_to_aggressive, CFG)
        assert ctype == CorrectionType.AGGRESSIVE
        assert abs(delta) > abs(compute_correction(0.80, 0.40, BoundViolation.ELEVATED, 1, CFG)[1])

    def test_hard_reset_critical(self):
        ctype, delta = compute_correction(0.95, 0.40, BoundViolation.CRITICAL, 1, CFG)
        assert ctype == CorrectionType.HARD_RESET
        assert abs(delta) > 0.3

    def test_hard_reset_after_long_violation(self):
        ctype, _ = compute_correction(0.80, 0.40, BoundViolation.ELEVATED,
                                      CFG.cycles_to_hard_reset, CFG)
        assert ctype == CorrectionType.HARD_RESET

    def test_depleted_correction_positive(self):
        ctype, delta = compute_correction(0.05, 0.40, BoundViolation.DEPLETED, 1, CFG)
        assert delta > 0  # Pull up


# =====================================================================
# Health status
# =====================================================================


class TestHealthStatus:
    def test_healthy(self):
        load = CognitiveLoadEstimate(l_cog=0.5, overloaded=False)
        assert compute_health_status([], load) == HealthStatus.HEALTHY

    def test_stressed(self):
        load = CognitiveLoadEstimate(l_cog=0.5, overloaded=False)
        violations = [NTViolationRecord(violation_type=BoundViolation.ELEVATED)]
        assert compute_health_status(violations, load) == HealthStatus.STRESSED

    def test_overloaded(self):
        load = CognitiveLoadEstimate(l_cog=0.9, overloaded=True)
        assert compute_health_status([], load) == HealthStatus.OVERLOADED

    def test_critical(self):
        load = CognitiveLoadEstimate(l_cog=0.5, overloaded=False)
        violations = [NTViolationRecord(violation_type=BoundViolation.CRITICAL)]
        assert compute_health_status(violations, load) == HealthStatus.CRITICAL

    def test_many_violations_critical(self):
        load = CognitiveLoadEstimate(l_cog=0.5, overloaded=False)
        violations = [NTViolationRecord(violation_type=BoundViolation.ELEVATED)] * 5
        assert compute_health_status(violations, load) == HealthStatus.CRITICAL


# =====================================================================
# Neurochemical coupling
# =====================================================================


class TestNeurochemCoupling:
    def test_overloaded_gaba_burst(self):
        load = CognitiveLoadEstimate(l_cog=0.9, overloaded=True)
        sig = compute_homeostatic_neurochem(load, [], CFG, np.random.default_rng(42))
        assert sig.delta_gaba > 0.0

    def test_not_overloaded_no_gaba(self):
        load = CognitiveLoadEstimate(l_cog=0.5, overloaded=False)
        sig = compute_homeostatic_neurochem(load, [], CFG, np.random.default_rng(42))
        assert sig.delta_gaba == 0.0

    def test_critical_violation_ne(self):
        load = CognitiveLoadEstimate(l_cog=0.5, overloaded=False)
        violations = [NTViolationRecord(violation_type=BoundViolation.CRITICAL)]
        for seed in range(20):
            sig = compute_homeostatic_neurochem(load, violations, CFG, np.random.default_rng(seed))
            if sig.delta_ne > 0.0:
                break
        assert sig.delta_ne > 0.0

    def test_chronic_violation_cor(self):
        load = CognitiveLoadEstimate(l_cog=0.5, overloaded=False)
        violations = [NTViolationRecord(violation_type=BoundViolation.ELEVATED, consecutive_cycles=6)]
        sig = compute_homeostatic_neurochem(load, violations, CFG, np.random.default_rng(42))
        assert sig.delta_cor > 0.0

    def test_moderate_load_5ht1a_cb1(self):
        load = CognitiveLoadEstimate(l_cog=0.7, overloaded=False)
        sig = compute_homeostatic_neurochem(load, [], CFG, np.random.default_rng(42))
        assert sig.delta_5ht1a > 0.0
        assert sig.delta_cb1 > 0.0


# =====================================================================
# Engine pipeline
# =====================================================================


class TestEngineBasic:
    def setup_method(self):
        self.engine = NeurochemicalHomeostaticEngine(rng=np.random.default_rng(42))

    def test_empty_input(self):
        result = self.engine.process(HomeostaticInput())
        assert isinstance(result, HomeostaticResult)
        assert result.health_status == HealthStatus.HEALTHY
        assert len(result.violations) == 0

    def test_normal_concentrations(self):
        concentrations = {nt: bounds[1] for nt, bounds in _NT_BOUNDS.items()}
        result = self.engine.process(HomeostaticInput(nt_concentrations=concentrations))
        assert result.health_status == HealthStatus.HEALTHY
        assert len(result.violations) == 0

    def test_elevated_nt(self):
        concentrations = {"da": 0.85, "5ht": 0.45}
        result = self.engine.process(HomeostaticInput(nt_concentrations=concentrations))
        assert len(result.violations) >= 1
        da_violations = [v for v in result.violations if v.nt_name == "da"]
        assert len(da_violations) == 1
        assert da_violations[0].violation_type == BoundViolation.ELEVATED

    def test_overload_detection(self):
        result = self.engine.process(HomeostaticInput(
            symbolic_saturation=1.0,
            emotional_saturation=1.0,
            urgency_saturation=2.0,
        ))
        assert result.cognitive_load.overloaded
        assert result.health_status in {HealthStatus.OVERLOADED, HealthStatus.CRITICAL}

    def test_corrections_emitted(self):
        concentrations = {"da": 0.95}  # Critical
        result = self.engine.process(HomeostaticInput(nt_concentrations=concentrations))
        assert "da" in result.corrections
        assert result.corrections["da"] != 0.0

    def test_processing_time(self):
        result = self.engine.process(HomeostaticInput())
        assert result.processing_time_ms >= 0.0


# =====================================================================
# Engine -- violation escalation
# =====================================================================


class TestViolationEscalation:
    def test_sustained_violation_escalates(self):
        engine = NeurochemicalHomeostaticEngine(rng=np.random.default_rng(42))
        # Run multiple cycles with elevated DA
        for i in range(CFG.cycles_to_aggressive + 1):
            result = engine.process(HomeostaticInput(nt_concentrations={"da": 0.85}))

        da_violations = [v for v in result.violations if v.nt_name == "da"]
        assert len(da_violations) == 1
        assert da_violations[0].correction_type in {CorrectionType.AGGRESSIVE, CorrectionType.HARD_RESET}

    def test_violation_counter_resets(self):
        engine = NeurochemicalHomeostaticEngine(rng=np.random.default_rng(42))
        # Elevated
        engine.process(HomeostaticInput(nt_concentrations={"da": 0.85}))
        engine.process(HomeostaticInput(nt_concentrations={"da": 0.85}))
        # Back to normal
        engine.process(HomeostaticInput(nt_concentrations={"da": 0.40}))
        # Check counter reset
        status = engine.get_status()
        assert status["active_violations"].get("da", 0) == 0


# =====================================================================
# Engine -- configure + status
# =====================================================================


class TestEngineStatus:
    def test_get_status(self):
        engine = NeurochemicalHomeostaticEngine()
        status = engine.get_status()
        assert status["engine_id"] == "neurochemical_homeostatic_engine"
        assert status["cycle_count"] == 0

    def test_configure_mode(self):
        engine = NeurochemicalHomeostaticEngine()
        engine.configure(OperationalMode.DEV)
        assert engine.get_status()["mode"] == "dev"

    def test_cycle_count(self):
        engine = NeurochemicalHomeostaticEngine()
        engine.process(HomeostaticInput())
        engine.process(HomeostaticInput())
        assert engine.get_status()["cycle_count"] == 2


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_all_critical(self):
        concentrations = {nt: 0.01 for nt in _NT_BOUNDS}
        engine = NeurochemicalHomeostaticEngine(rng=np.random.default_rng(42))
        result = engine.process(HomeostaticInput(nt_concentrations=concentrations))
        assert result.health_status == HealthStatus.CRITICAL
        assert len(result.violations) == len(_NT_BOUNDS)

    def test_unknown_nt(self):
        engine = NeurochemicalHomeostaticEngine(rng=np.random.default_rng(42))
        result = engine.process(HomeostaticInput(nt_concentrations={"UNKNOWN_NT": 0.40}))
        assert len(result.violations) == 0

    def test_cognitive_load_fields(self):
        engine = NeurochemicalHomeostaticEngine(rng=np.random.default_rng(42))
        result = engine.process(HomeostaticInput(
            symbolic_saturation=0.3,
            emotional_saturation=0.4,
            urgency_saturation=0.2,
            dynamic_modifier=0.1,
        ))
        cl = result.cognitive_load
        assert cl.s_symbolic == pytest.approx(0.3)
        assert cl.s_emotional == pytest.approx(0.4)
        assert cl.s_urgency == pytest.approx(0.2)
        assert cl.epsilon == pytest.approx(0.1)
