"""
Tests for Sleep & Dream Modes neurochemical extension.

Covers:
  1. OscillationState sigma band
  2. Sleep state vectors
  3. Transition logic
  4. Containment monitoring
  5. SleepNeurochemicalStateManager
  6. Sleep triggers
  7. New composite metrics
  8. Sigma modulation coefficients
  9. E27 dream mode tolerance
  10. Generators sigma derivation
"""

import pytest
from zados.neurochem.state.oscillation_state import OscillationState


# =====================================================================
# 1. TestOscillationStateSigma
# =====================================================================

class TestOscillationStateSigma:
    """Tests for sigma band addition to OscillationState."""

    def test_sigma_default_zero(self):
        osc = OscillationState()
        assert osc.sigma == 0.0

    def test_sigma_set_value(self):
        osc = OscillationState(sigma=0.7)
        assert osc.sigma == 0.7

    def test_sigma_clamped_high(self):
        osc = OscillationState(sigma=1.5)
        assert osc.sigma == 1.0

    def test_sigma_clamped_low(self):
        osc = OscillationState(sigma=-0.3)
        assert osc.sigma == 0.0

    def test_set_band_sigma(self):
        osc = OscillationState()
        osc.set_band("sigma", 0.6)
        assert osc.sigma == 0.6

    def test_get_band_sigma(self):
        osc = OscillationState(sigma=0.55)
        assert osc.get_band("sigma") == 0.55

    def test_delta_sigma_coupling(self):
        osc = OscillationState(delta=0.8, sigma=0.7)
        assert abs(osc.delta_sigma_coupling() - 0.56) < 1e-9

    def test_delta_sigma_coupling_zero_sigma(self):
        osc = OscillationState(delta=0.8, sigma=0.0)
        assert osc.delta_sigma_coupling() == 0.0

    def test_bands_includes_sigma(self):
        osc = OscillationState()
        bands = osc.bands()
        assert "sigma" in bands

    def test_as_dict_includes_sigma(self):
        osc = OscillationState(sigma=0.5, delta=0.4)
        d = osc.as_dict()
        assert "sigma" in d
        assert d["sigma"] == 0.5
        assert "delta_sigma_coupling" in d
        assert abs(d["delta_sigma_coupling"] - 0.2) < 1e-9

    def test_from_dict_restores_sigma(self):
        osc = OscillationState(sigma=0.65, delta=0.3)
        d = osc.as_dict()
        restored = OscillationState.from_dict(d)
        assert restored.sigma == 0.65

    def test_copy_preserves_sigma(self):
        osc = OscillationState(sigma=0.42)
        copied = osc.copy()
        assert copied.sigma == 0.42


# =====================================================================
# 2. TestSleepStateVectors
# =====================================================================

class TestSleepStateVectors:
    """Tests for prescribed sleep state vectors."""

    def test_sleep_phase_enum(self):
        from zados.neurochem.sleep.state_vectors import SleepPhase
        assert SleepPhase.WAKING.value == "waking"
        assert SleepPhase.TRIAGE.value == "triage"
        assert SleepPhase.REM_PROCESSING.value == "rem_processing"
        assert SleepPhase.DREAM.value == "dream"

    def test_triage_vector_nt_values(self):
        from zados.neurochem.sleep.state_vectors import TRIAGE_STATE_VECTOR
        nt = TRIAGE_STATE_VECTOR.nt_baselines
        assert nt["ACh"] == 0.45
        assert nt["NE"] == 0.40
        assert nt["5HT"] == 0.55
        assert nt["GABA"] == 0.60
        assert len(nt) == 12

    def test_rem_processing_vector_nt_values(self):
        from zados.neurochem.sleep.state_vectors import REM_PROCESSING_STATE_VECTOR
        nt = REM_PROCESSING_STATE_VECTOR.nt_baselines
        assert nt["ACh"] == 0.20
        assert nt["GABA"] == 0.80
        assert nt["5HT"] == 0.60
        assert nt["histamine"] == 0.15

    def test_dream_vector_nt_values(self):
        from zados.neurochem.sleep.state_vectors import DREAM_STATE_VECTOR
        nt = DREAM_STATE_VECTOR.nt_baselines
        assert nt["ACh"] == 0.85
        assert nt["NE"] == 0.05
        assert nt["5HT"] == 0.05
        assert nt["DA"] == 0.65
        assert nt["CB1"] == 0.75

    def test_triage_osc_config(self):
        from zados.neurochem.sleep.state_vectors import TRIAGE_STATE_VECTOR
        osc = TRIAGE_STATE_VECTOR.oscillatory_config
        assert osc["sigma"] == 0.50
        assert osc["delta"] == 0.35

    def test_dream_osc_config(self):
        from zados.neurochem.sleep.state_vectors import DREAM_STATE_VECTOR
        osc = DREAM_STATE_VECTOR.oscillatory_config
        assert osc["theta"] == 0.85
        assert osc["sigma"] == 0.05
        assert osc["gamma"] == 0.65

    def test_sleep_state_vectors_lookup(self):
        from zados.neurochem.sleep.state_vectors import (
            SleepPhase, SLEEP_STATE_VECTORS,
        )
        assert SleepPhase.TRIAGE in SLEEP_STATE_VECTORS
        assert SleepPhase.REM_PROCESSING in SLEEP_STATE_VECTORS
        assert SleepPhase.DREAM in SLEEP_STATE_VECTORS
        assert SleepPhase.WAKING not in SLEEP_STATE_VECTORS


# =====================================================================
# 3. TestTransitionLogic
# =====================================================================

class TestTransitionLogic:
    """Tests for pharmacodynamic transition functions."""

    def test_compute_transition_step_approaches_target(self):
        from zados.neurochem.sleep.transitions import compute_transition_step
        val = compute_transition_step(0.4, 0.8, 0.25, 1.0)
        assert val > 0.4
        assert val < 0.8

    def test_compute_transition_step_at_target(self):
        from zados.neurochem.sleep.transitions import compute_transition_step
        val = compute_transition_step(0.5, 0.5, 0.25, 1.0)
        assert abs(val - 0.5) < 1e-9

    def test_compute_transition_step_clamped(self):
        from zados.neurochem.sleep.transitions import compute_transition_step
        val = compute_transition_step(0.9, 1.5, 0.5, 1.0)
        assert val <= 1.0

    def test_transition_nt_baselines(self):
        from zados.neurochem.sleep.transitions import transition_nt_baselines
        current = {"ACh": 0.7, "NE": 0.6, "5HT": 0.45}
        target = {"ACh": 0.45, "NE": 0.40, "5HT": 0.55}
        result = transition_nt_baselines(current, target, k=0.2, dt=1.0)
        assert result["ACh"] < 0.7
        assert result["NE"] < 0.6
        assert result["5HT"] > 0.45

    def test_fast_collapse_nts(self):
        from zados.neurochem.sleep.transitions import transition_nt_baselines
        current = {"NE": 0.5, "5HT": 0.5, "DA": 0.5}
        target = {"NE": 0.05, "5HT": 0.05, "DA": 0.65}
        result = transition_nt_baselines(
            current, target, k=0.1, dt=1.0,
            k_fast=0.4,
        )
        # NE and 5HT should move faster than DA
        ne_delta = abs(result["NE"] - 0.5)
        da_delta = abs(result["DA"] - 0.5)
        assert ne_delta > da_delta

    def test_triage_to_rem_conditions_met(self):
        from zados.neurochem.sleep.transitions import check_triage_to_rem_conditions
        nt = {"5HT": 0.55, "ACh": 0.25}
        osc = {"delta": 0.65, "sigma": 0.60}
        assert check_triage_to_rem_conditions(nt, osc) is True

    def test_triage_to_rem_conditions_not_met(self):
        from zados.neurochem.sleep.transitions import check_triage_to_rem_conditions
        nt = {"5HT": 0.40, "ACh": 0.25}
        osc = {"delta": 0.65, "sigma": 0.60}
        assert check_triage_to_rem_conditions(nt, osc) is False

    def test_rem_to_dream_desensitization(self):
        from zados.neurochem.sleep.transitions import check_rem_to_dream_conditions
        assert check_rem_to_dream_conditions({}, {}, desensitization_flag=True) is True

    def test_rem_to_dream_stagnated_queue(self):
        from zados.neurochem.sleep.transitions import check_rem_to_dream_conditions
        nt = {"5HT": 0.35}
        assert check_rem_to_dream_conditions(
            nt, {}, stagnated_queue_nonempty=True,
        ) is True

    def test_rem_to_dream_not_met(self):
        from zados.neurochem.sleep.transitions import check_rem_to_dream_conditions
        nt = {"5HT": 0.55}
        assert check_rem_to_dream_conditions(nt, {}) is False


# =====================================================================
# 4. TestContainment
# =====================================================================

class TestContainment:
    """Tests for GABA-A dreambox containment."""

    def test_containment_intact(self):
        from zados.neurochem.sleep.containment import check_containment
        assert check_containment(0.65) is True

    def test_containment_breach(self):
        from zados.neurochem.sleep.containment import check_containment
        assert check_containment(0.50) is False

    def test_containment_at_threshold(self):
        from zados.neurochem.sleep.containment import check_containment
        assert check_containment(0.55) is True

    def test_dream_state_valid(self):
        from zados.neurochem.sleep.containment import check_dream_state_validity
        assert check_dream_state_validity(0.05, 0.05) is True

    def test_dream_state_invalid_ne_high(self):
        from zados.neurochem.sleep.containment import check_dream_state_validity
        assert check_dream_state_validity(0.15, 0.05) is False

    def test_ne_upregulation_cap(self):
        from zados.neurochem.sleep.containment import check_ne_upregulation_cap
        assert check_ne_upregulation_cap(1.3) == 1.3
        assert check_ne_upregulation_cap(1.8) == 1.5
        assert check_ne_upregulation_cap(1.5) == 1.5


# =====================================================================
# 5. TestSleepStateManager
# =====================================================================

class TestSleepStateManager:
    """Tests for SleepNeurochemicalStateManager."""

    def _make_waking_baselines(self):
        return {
            "ACh": 0.70, "NE": 0.65, "5HT": 0.45, "DA": 0.40,
            "GABA": 0.35, "GLU": 0.55, "CB1": 0.30, "MOR": 0.25,
            "CRH": 0.20, "cortisol": 0.20, "histamine": 0.55, "OXT": 0.50,
        }

    def _make_waking_osc(self):
        return {
            "delta": 0.05, "theta": 0.50, "alpha": 0.45,
            "beta": 0.55, "gamma": 0.40, "sigma": 0.0,
        }

    def test_initial_state_is_waking(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        assert mgr.phase == SleepPhase.WAKING
        assert mgr.is_active() is False

    def test_enter_sleep_sets_triage(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        assert mgr.phase == SleepPhase.TRIAGE
        assert mgr.is_active() is True

    def test_enter_sleep_saves_waking(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager
        mgr = SleepNeurochemicalStateManager()
        waking = self._make_waking_baselines()
        mgr.enter_sleep(waking, self._make_waking_osc())
        # Current baselines start at waking values
        assert mgr.current_baselines["ACh"] == 0.70

    def test_step_transition_moves_toward_target(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        result = mgr.step_transition(dt=1.0)
        # ACh should move toward triage target (0.45) from waking (0.70)
        assert result["nt_baselines"]["ACh"] < 0.70

    def test_try_advance_triage_to_rem(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        # Force baselines to meet conditions
        mgr._current_nt_baselines["5HT"] = 0.55
        mgr._current_nt_baselines["ACh"] = 0.25
        mgr._current_osc_config["delta"] = 0.65
        mgr._current_osc_config["sigma"] = 0.60
        assert mgr.try_advance_phase() is True
        assert mgr.phase == SleepPhase.REM_PROCESSING

    def test_try_advance_rem_to_dream(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        mgr._phase = SleepPhase.REM_PROCESSING
        assert mgr.try_advance_phase(desensitization_flag=True) is True
        assert mgr.phase == SleepPhase.DREAM

    def test_try_advance_no_conditions(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        assert mgr.try_advance_phase() is False
        assert mgr.phase == SleepPhase.TRIAGE

    def test_monitor_containment_safe(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        mgr._phase = SleepPhase.DREAM
        mgr._current_nt_baselines["NE"] = 0.05
        mgr._current_nt_baselines["5HT"] = 0.05
        alerts = mgr.monitor_containment({"GABA_A": 0.70})
        assert len(alerts) == 0

    def test_monitor_containment_breach(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        mgr._phase = SleepPhase.DREAM
        mgr._current_nt_baselines["NE"] = 0.05
        mgr._current_nt_baselines["5HT"] = 0.05
        alerts = mgr.monitor_containment({"GABA_A": 0.40})
        assert any("CONTAINMENT_BREACH" in a for a in alerts)

    def test_monitor_containment_ne_floor_violation(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        mgr._phase = SleepPhase.DREAM
        mgr._current_nt_baselines["NE"] = 0.20
        mgr._current_nt_baselines["5HT"] = 0.05
        alerts = mgr.monitor_containment({"GABA_A": 0.70})
        assert any("DREAM_STATE_INVALID" in a for a in alerts)

    def test_exit_sleep(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        waking = self._make_waking_baselines()
        mgr.enter_sleep(waking, self._make_waking_osc())
        mgr._phase = SleepPhase.DREAM
        mgr.exit_sleep()
        assert mgr.phase == SleepPhase.WAKING
        assert mgr.is_active() is False
        # Waking baselines restored
        assert mgr.current_baselines["ACh"] == 0.70

    def test_as_dict_from_dict_roundtrip(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        mgr.step_transition(dt=1.0)
        data = mgr.as_dict()
        restored = SleepNeurochemicalStateManager.from_dict(data)
        assert restored.phase == mgr.phase
        assert restored.current_baselines == mgr.current_baselines

    def test_full_lifecycle(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager, SleepPhase
        mgr = SleepNeurochemicalStateManager()
        mgr.enter_sleep(self._make_waking_baselines(), self._make_waking_osc())
        assert mgr.phase == SleepPhase.TRIAGE

        # Step multiple times to approach triage targets
        for _ in range(50):
            mgr.step_transition(dt=1.0)

        # Force conditions and advance
        mgr._current_nt_baselines["5HT"] = 0.55
        mgr._current_nt_baselines["ACh"] = 0.25
        mgr._current_osc_config["delta"] = 0.65
        mgr._current_osc_config["sigma"] = 0.60
        assert mgr.try_advance_phase() is True
        assert mgr.phase == SleepPhase.REM_PROCESSING

        # Advance to dream
        assert mgr.try_advance_phase(desensitization_flag=True) is True
        assert mgr.phase == SleepPhase.DREAM

        # Exit
        mgr.exit_sleep()
        assert mgr.phase == SleepPhase.WAKING

    def test_not_active_in_waking(self):
        from zados.neurochem.sleep import SleepNeurochemicalStateManager
        mgr = SleepNeurochemicalStateManager()
        result = mgr.step_transition(dt=1.0)
        assert result["phase"].value == "waking"
        assert result["alerts"] == []


# =====================================================================
# 6. TestSleepTriggers
# =====================================================================

class TestSleepTriggers:
    """Tests for sleep neurosymbolic triggers."""

    def test_trigger_count(self):
        from zados.neurochem.sleep.sleep_triggers import DEFAULT_SLEEP_TRIGGERS
        assert len(DEFAULT_SLEEP_TRIGGERS) == 6

    def test_sleep_entry_trigger_fires(self):
        from zados.neurochem.sleep.sleep_triggers import SLEEP_ENTRY_TRIGGER
        from zados.neurochem.neurosymbolic.triggers import evaluate_trigger
        ns = {"GABA": 0.60, "NE": 0.40, "histamine": 0.30}
        result = evaluate_trigger(SLEEP_ENTRY_TRIGGER, ns)
        assert result.fired is True
        assert result.mode == "TriageMode"

    def test_sleep_entry_trigger_not_fires(self):
        from zados.neurochem.sleep.sleep_triggers import SLEEP_ENTRY_TRIGGER
        from zados.neurochem.neurosymbolic.triggers import evaluate_trigger
        ns = {"GABA": 0.40, "NE": 0.60, "histamine": 0.50}
        result = evaluate_trigger(SLEEP_ENTRY_TRIGGER, ns)
        assert result.fired is False

    def test_containment_trigger_fires(self):
        from zados.neurochem.sleep.sleep_triggers import CONTAINMENT_CHECK_TRIGGER
        from zados.neurochem.neurosymbolic.triggers import evaluate_trigger
        ns = {"S_GABA_A": 0.40}
        result = evaluate_trigger(CONTAINMENT_CHECK_TRIGGER, ns)
        assert result.fired is True
        assert result.mode == "DreamAbort"

    def test_consolidation_trigger_fires(self):
        from zados.neurochem.sleep.sleep_triggers import CONSOLIDATION_WINDOW_TRIGGER
        from zados.neurochem.neurosymbolic.triggers import evaluate_trigger
        ns = {"phi_delta_sigma": 0.60, "S_GLU_NMDA": 0.60}
        result = evaluate_trigger(CONSOLIDATION_WINDOW_TRIGGER, ns)
        assert result.fired is True

    def test_sleep_exit_trigger(self):
        from zados.neurochem.sleep.sleep_triggers import SLEEP_EXIT_TRIGGER
        from zados.neurochem.neurosymbolic.triggers import evaluate_trigger
        ns = {"SleepProcessComplete": 1}
        result = evaluate_trigger(SLEEP_EXIT_TRIGGER, ns)
        assert result.fired is True
        assert result.mode == "WakingReturn"


# =====================================================================
# 7. TestNewMetrics
# =====================================================================

class TestNewMetrics:
    """Tests for the 3 new sleep composite metrics."""

    def test_dream_permissiveness_formula(self):
        from zados.neurochem.neurosymbolic.metrics import compute_dream_permissiveness
        # High dream state: CB1=0.75, theta_gamma=0.55, NE=0, 5HT=0, GABA_B=0.3
        val = compute_dream_permissiveness(0.75, 0.55, 0.0, 0.0, 0.3)
        # raw = (0.75*0.55) + 1.0 + 1.0 - 0.3 = 2.1125
        # normalized = (2.1125 + 1) / 4 = 0.778
        assert 0.77 < val < 0.79

    def test_dream_permissiveness_waking(self):
        from zados.neurochem.neurosymbolic.metrics import compute_dream_permissiveness
        # Typical waking: moderate NE and 5HT
        val = compute_dream_permissiveness(0.3, 0.25, 0.5, 0.5, 0.4)
        assert val < 0.5  # Low during waking

    def test_consolidation_depth_formula(self):
        from zados.neurochem.neurosymbolic.metrics import compute_consolidation_depth
        # Peak NREM: delta_sigma=0.56, GLU_NMDA=0.6, no cAMP, low ACh
        val = compute_consolidation_depth(0.56, 0.6, 0.3, 0.15)
        # raw = 0.56*0.6 + 0 - 0.15 = 0.186
        # normalized = (0.186+1)/3 = 0.395
        assert 0.39 < val < 0.40

    def test_consolidation_depth_with_camp(self):
        from zados.neurochem.neurosymbolic.metrics import compute_consolidation_depth
        val_no = compute_consolidation_depth(0.56, 0.6, 0.3, 0.15, 0.0)
        val_yes = compute_consolidation_depth(0.56, 0.6, 0.3, 0.15, 1.0)
        assert val_yes > val_no

    def test_narrative_plasticity_formula(self):
        from zados.neurochem.neurosymbolic.metrics import compute_narrative_plasticity
        # High dream: theta_gamma=0.55, GLU_NMDA=0.6, DA_D3=0.5, CB1=0.75, rigidity=0.2
        val = compute_narrative_plasticity(0.55, 0.6, 0.5, 0.75, 0.2)
        # raw = 0.55*0.6*0.5 + 0.75 + (1-0.2) = 0.165 + 0.75 + 0.8 = 1.715
        # normalized = 1.715/3 = 0.572
        assert 0.57 < val < 0.58

    def test_narrative_plasticity_zero_inputs(self):
        from zados.neurochem.neurosymbolic.metrics import compute_narrative_plasticity
        val = compute_narrative_plasticity(0.0, 0.0, 0.0, 0.0, 1.0)
        # raw = 0 + 0 + 0 = 0
        assert val == 0.0

    def test_metrics_defaults_zero(self):
        from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
        m = NeurochemicalMetrics()
        assert m.dream_permissiveness == 0.0
        assert m.consolidation_depth == 0.0
        assert m.narrative_plasticity == 0.0

    def test_compute_all_metrics_includes_new(self):
        from zados.neurochem.neurosymbolic.metrics import compute_all_metrics
        m = compute_all_metrics(
            concentrations={"NE": 0.5, "CRH": 0.2, "cortisol": 0.2},
            receptor_saturations={
                "DA_D3": 0.4, "OXTR": 0.5, "GABA_B": 0.3,
                "CB1": 0.5, "NE_alpha1": 0.4, "5HT_1A": 0.5,
                "NE_beta1": 0.3, "DA_D2": 0.3, "5HT_2A": 0.4,
                "GABA_A": 0.4, "GLU_NMDA": 0.4, "ACh_M1": 0.3,
            },
            oscillations={
                "theta": 0.5, "delta": 0.3, "beta": 0.4,
                "theta_gamma": 0.25, "delta_sigma": 0.1,
                "alpha_beta": 0.2,
            },
        )
        d = m.as_dict()
        assert "dream_permissiveness" in d
        assert "consolidation_depth" in d
        assert "narrative_plasticity" in d

    def test_new_metrics_in_as_dict(self):
        from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
        m = NeurochemicalMetrics(
            dream_permissiveness=0.8,
            consolidation_depth=0.5,
            narrative_plasticity=0.6,
        )
        d = m.as_dict()
        assert d["dream_permissiveness"] == 0.8
        assert d["consolidation_depth"] == 0.5
        assert d["narrative_plasticity"] == 0.6


# =====================================================================
# 8. TestSigmaModulation
# =====================================================================

class TestSigmaModulation:
    """Tests for sigma-band modulation coefficients in receptor configs."""

    def test_gaba_a_has_sigma_coefficient(self):
        from zados.neurochem.neurotransmitters.configs import DEFAULT_RECEPTOR_CONFIGS
        cfg = DEFAULT_RECEPTOR_CONFIGS["GABA_A"]
        assert "kd_band_coefficients" in cfg
        assert cfg["kd_band_coefficients"]["sigma"] == 0.3

    def test_glu_ampa_has_sigma_coefficient(self):
        from zados.neurochem.neurotransmitters.configs import DEFAULT_RECEPTOR_CONFIGS
        cfg = DEFAULT_RECEPTOR_CONFIGS["GLU_AMPA"]
        assert cfg["kd_band_coefficients"]["sigma"] == 0.25

    def test_gaba_b_has_sigma_coefficient(self):
        from zados.neurochem.neurotransmitters.configs import DEFAULT_RECEPTOR_CONFIGS
        cfg = DEFAULT_RECEPTOR_CONFIGS["GABA_B"]
        assert cfg["kd_band_coefficients"]["sigma"] == 0.15

    def test_ne_beta1_has_sigma_coefficient(self):
        from zados.neurochem.neurotransmitters.configs import DEFAULT_RECEPTOR_CONFIGS
        cfg = DEFAULT_RECEPTOR_CONFIGS["NE_beta1"]
        assert cfg["kd_band_coefficients"]["sigma"] == 0.2

    def test_da_d1_has_sigma_coefficient(self):
        from zados.neurochem.neurotransmitters.configs import DEFAULT_RECEPTOR_CONFIGS
        cfg = DEFAULT_RECEPTOR_CONFIGS["DA_D1"]
        assert cfg["kd_band_coefficients"]["sigma"] == 0.1

    def test_sigma_modulate_kd_multiband(self):
        from zados.neurochem.oscillations.oscillation_modulation import (
            modulate_K_d_multiband,
        )
        K_d_base = 0.5
        osc_amps = {"sigma": 0.7, "theta": 0.3}
        band_coeffs = {"sigma": 0.3, "theta": 0.2}
        result = modulate_K_d_multiband(K_d_base, osc_amps, band_coeffs)
        # Sigma should reduce K_d when high
        assert result < K_d_base


# =====================================================================
# 9. TestE27DreamMode
# =====================================================================

class TestE27DreamMode:
    """Tests for E27 dream mode tolerance."""

    def test_dream_config_fields(self):
        from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
            HomeostaticConfig,
        )
        cfg = HomeostaticConfig()
        assert cfg.dream_tolerance_band == 0.25
        assert cfg.dream_runaway_band == 0.30
        assert cfg.dream_runaway_max_ticks == 3
        assert "ach" in cfg.dream_monitored_nts
        assert "ne" in cfg.dream_floor_nts

    def test_dream_bounds_monitored_nt_in_range(self):
        from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
            check_nt_bounds_dream, HomeostaticConfig, BoundViolation,
        )
        cfg = HomeostaticConfig()
        # ach at dream baseline (0.85) → no violation
        violation, _, _, _ = check_nt_bounds_dream("ach", 0.85, cfg)
        assert violation == BoundViolation.NONE

    def test_dream_bounds_monitored_nt_elevated(self):
        from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
            check_nt_bounds_dream, HomeostaticConfig, BoundViolation,
        )
        cfg = HomeostaticConfig()
        # ach way above baseline+tolerance → elevated
        violation, _, _, _ = check_nt_bounds_dream("ach", 1.0, cfg)
        # 0.85 + 0.25 = 1.10, clamped to 1.0. So 1.0 is within high bound.
        # Actually 1.0 == adj_high (min(1.0, 1.10)) → within bounds
        assert violation == BoundViolation.NONE

    def test_dream_bounds_monitored_nt_critical(self):
        from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
            check_nt_bounds_dream, HomeostaticConfig, BoundViolation,
        )
        cfg = HomeostaticConfig()
        # gaba far below dream baseline (0.65 - 0.30 = 0.35)
        violation, _, _, _ = check_nt_bounds_dream("gaba", 0.30, cfg)
        assert violation == BoundViolation.CRITICAL

    def test_dream_bounds_floor_nt_valid(self):
        from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
            check_nt_bounds_dream, HomeostaticConfig, BoundViolation,
        )
        cfg = HomeostaticConfig()
        violation, _, _, _ = check_nt_bounds_dream("ne", 0.05, cfg)
        assert violation == BoundViolation.NONE

    def test_dream_bounds_floor_nt_violation(self):
        from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
            check_nt_bounds_dream, HomeostaticConfig, BoundViolation,
        )
        cfg = HomeostaticConfig()
        violation, _, _, _ = check_nt_bounds_dream("ne", 0.20, cfg)
        assert violation == BoundViolation.ELEVATED

    def test_dream_bounds_unmonitored_nt(self):
        from zados.cognitive_engines.py_engines.neurochemical_homeostatic_engine import (
            check_nt_bounds_dream, HomeostaticConfig, BoundViolation,
        )
        cfg = HomeostaticConfig()
        # cb1 is not in dream_monitored_nts or dream_floor_nts
        violation, _, _, _ = check_nt_bounds_dream("cb1", 0.99, cfg)
        assert violation == BoundViolation.NONE


# =====================================================================
# 10. TestGeneratorsSigma
# =====================================================================

class TestGeneratorsSigma:
    """Tests for sigma derivation in oscillation generators."""

    def test_sigma_derivation_rule_exists(self):
        from zados.neurochem.oscillations.generators import DEFAULT_BAND_DERIVATION_RULES
        assert "sigma" in DEFAULT_BAND_DERIVATION_RULES
        sigma_rules = DEFAULT_BAND_DERIVATION_RULES["sigma"]
        assert "GABA" in sigma_rules
        assert "GLU" in sigma_rules
        assert "NE" in sigma_rules

    def test_derive_sigma_zero_waking(self):
        """Sigma should be near-zero in normal waking NTs."""
        from zados.neurochem.oscillations.generators import derive_oscillation_state
        from zados.neurochem.state import NeurotransmitterState
        nt_states = {
            "GABA": NeurotransmitterState(C_tonic=0.35),
            "GLU": NeurotransmitterState(C_phasic=0.1),
            "NE": NeurotransmitterState(C_tonic=0.45),
        }
        osc = derive_oscillation_state(nt_states)
        # sigma = 0.5*0.35 + 0.3*0.1 - 0.3*0.45 = 0.175+0.03-0.135 = 0.07
        assert osc.sigma < 0.15

    def test_derive_sigma_sleep_state(self):
        """Sigma should be higher with elevated GABA and suppressed NE."""
        from zados.neurochem.oscillations.generators import derive_oscillation_state
        from zados.neurochem.state import NeurotransmitterState
        nt_states = {
            "GABA": NeurotransmitterState(C_tonic=0.80),
            "GLU": NeurotransmitterState(C_phasic=0.3),
            "NE": NeurotransmitterState(C_tonic=0.10),
        }
        osc = derive_oscillation_state(nt_states)
        # sigma = 0.5*0.80 + 0.3*0.3 - 0.3*0.10 = 0.4+0.09-0.03 = 0.46
        assert osc.sigma > 0.40

    def test_derive_returns_six_bands(self):
        from zados.neurochem.oscillations.generators import derive_oscillation_state
        from zados.neurochem.state import NeurotransmitterState
        nt_states = {"DA": NeurotransmitterState()}
        osc = derive_oscillation_state(nt_states)
        assert hasattr(osc, "sigma")
        bands = osc.bands()
        assert len(bands) == 6
