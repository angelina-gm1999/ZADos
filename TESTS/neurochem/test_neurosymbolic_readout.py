import pytest
from zados.neurochem.state import (
    NeurotransmitterState,
    ReceptorState,
    OscillationState,
)
from zados.neurochem.neurosymbolic.readout import (
    compute_receptor_saturation,
    extract_concentrations,
    extract_receptor_saturations,
    extract_oscillation_amplitudes,
    compute_neurosymbolic_readout,
    format_metrics_summary,
    identify_dominant_metrics,
    identify_suppressed_metrics,
)


def test_compute_receptor_saturation():
    """Receptor saturation computation."""
    # Half saturation at C = K_d
    assert compute_receptor_saturation(0.5, 0.5) == pytest.approx(0.5)
    
    # High concentration
    assert compute_receptor_saturation(10.0, 1.0) > 0.9
    
    # Low concentration
    assert compute_receptor_saturation(0.1, 1.0) < 0.2


def test_extract_concentrations():
    """Can extract concentrations from NT states."""
    nt_states = {
        "DA": NeurotransmitterState(C_tonic=0.6, C_phasic=0.2),
        "NE": NeurotransmitterState(C_tonic=0.5),
    }
    
    concentrations = extract_concentrations(nt_states)
    
    assert concentrations["DA"] == pytest.approx(0.8)
    assert concentrations["NE"] == pytest.approx(0.5)


def test_extract_receptor_saturations():
    """Can extract receptor saturations."""
    nt_states = {
        "DA": NeurotransmitterState(C_tonic=0.6),
    }
    receptor_states = {
        "DA_D1": ReceptorState(receptor_id="DA_D1"),
        "DA_D2": ReceptorState(receptor_id="DA_D2"),
    }
    
    saturations = extract_receptor_saturations(receptor_states, nt_states)
    
    assert "DA_D1" in saturations
    assert "DA_D2" in saturations
    assert 0.0 <= saturations["DA_D1"] <= 1.0


def test_extract_receptor_saturations_with_config():
    """Can extract saturations with custom K_d values."""
    nt_states = {
        "DA": NeurotransmitterState(C_tonic=0.5),
    }
    receptor_states = {
        "DA_D1": ReceptorState(receptor_id="DA_D1"),
    }
    receptor_configs = {
        "DA_D1": {"K_d": 0.5},
    }
    
    saturations = extract_receptor_saturations(
        receptor_states,
        nt_states,
        receptor_configs,
    )
    
    # C=0.5, K_d=0.5 → saturation = 0.5
    assert saturations["DA_D1"] == pytest.approx(0.5)


def test_extract_receptor_saturations_missing_nt():
    """Handles missing neurotransmitter gracefully."""
    nt_states = {}
    receptor_states = {
        "DA_D1": ReceptorState(receptor_id="DA_D1"),
    }
    
    saturations = extract_receptor_saturations(receptor_states, nt_states)
    
    assert saturations["DA_D1"] == 0.0


def test_extract_oscillation_amplitudes():
    """Can extract oscillation amplitudes."""
    osc_state = OscillationState(delta=0.2, theta=0.5, gamma=0.8)
    
    amplitudes = extract_oscillation_amplitudes(osc_state)
    
    assert amplitudes["delta"] == 0.2
    assert amplitudes["theta"] == 0.5
    assert amplitudes["gamma"] == 0.8


def test_compute_neurosymbolic_readout():
    """Full readout produces valid metrics."""
    nt_states = {
        "DA": NeurotransmitterState(C_tonic=0.6),
        "NE": NeurotransmitterState(C_tonic=0.5),
        "cortisol": NeurotransmitterState(C_tonic=0.3),
    }
    receptor_states = {
        "DA_D3": ReceptorState(receptor_id="DA_D3"),
        "DA_D2": ReceptorState(receptor_id="DA_D2"),
        "OXTR": ReceptorState(receptor_id="OXTR"),
        "GABA_B": ReceptorState(receptor_id="GABA_B"),
        "NE_beta1": ReceptorState(receptor_id="NE_beta1"),
    }
    osc_state = OscillationState(theta=0.5, delta=0.2, beta=0.4)
    
    metrics = compute_neurosymbolic_readout(nt_states, receptor_states, osc_state)
    
    # All metrics should be bounded
    assert 0.0 <= metrics.motivation <= 1.0
    assert 0.0 <= metrics.empathy <= 1.0
    assert 0.0 <= metrics.cognitive_rigidity <= 1.0
    assert 0.0 <= metrics.fatigue <= 1.0


def test_compute_neurosymbolic_readout_with_empty_state():
    """Readout handles empty state gracefully."""
    metrics = compute_neurosymbolic_readout({}, {}, OscillationState())
    
    assert 0.0 <= metrics.motivation <= 1.0


def test_format_metrics_summary():
    """Metrics can be formatted as summary string."""
    from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
    
    metrics = NeurochemicalMetrics(
        motivation=0.7,
        empathy=0.5,
        fatigue=0.3,
    )
    
    summary = format_metrics_summary(metrics)
    
    assert "Motivation" in summary
    assert "0.700" in summary
    assert "Empathy" in summary


def test_identify_dominant_metrics():
    """Can identify dominant metrics above threshold."""
    from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
    
    metrics = NeurochemicalMetrics(
        motivation=0.8,
        empathy=0.9,
        fatigue=0.2,
    )
    
    dominant = identify_dominant_metrics(metrics, threshold=0.7)
    
    assert "motivation" in dominant
    assert "empathy" in dominant
    assert "fatigue" not in dominant


def test_identify_suppressed_metrics():
    """Can identify suppressed metrics below threshold."""
    from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
    
    metrics = NeurochemicalMetrics(
        motivation=0.8,
        fatigue=0.2,
        anxiety=0.1,
    )
    
    suppressed = identify_suppressed_metrics(metrics, threshold=0.3)
    
    assert "fatigue" in suppressed
    assert "anxiety" in suppressed
    assert "motivation" not in suppressed


def test_identify_dominant_empty():
    """Dominant identification returns empty list if none above threshold."""
    from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
    
    metrics = NeurochemicalMetrics()  # All 0.5
    
    dominant = identify_dominant_metrics(metrics, threshold=0.9)
    
    assert dominant == []


def test_identify_suppressed_empty():
    """Suppressed identification returns empty list if none below threshold."""
    from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics

    metrics = NeurochemicalMetrics(
        motivation=0.8,
        empathy=0.9,
        fatigue=0.7,
        dream_permissiveness=0.5,
        consolidation_depth=0.5,
        narrative_plasticity=0.5,
    )

    suppressed = identify_suppressed_metrics(metrics, threshold=0.1)

    assert suppressed == []


# ---------------------------------------------------------------------------
# Phase 34: compute_full_readout tests
# ---------------------------------------------------------------------------

from zados.neurochem.neurosymbolic.readout import compute_full_readout
from zados.neurochem.neurosymbolic.state_expressions import StateTerm, StateDefinition
from zados.neurochem.neurosymbolic.triggers import TriggerDefinition


class TestComputeFullReadout:
    """Tests for the extended compute_full_readout function."""

    def test_metrics_only(self):
        """No extras → just metrics + raw state dicts."""
        result = compute_full_readout(
            neurotransmitter_states={},
            receptor_states={},
            oscillation_state=OscillationState(),
        )
        assert "metrics" in result
        assert "concentrations" in result
        assert "saturations" in result
        assert "oscillations" in result
        assert "states" not in result
        assert "trigger_results" not in result

    def test_with_state_definitions(self):
        """STATE definitions are evaluated."""
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=0.6),
        }
        receptor_states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1"),
        }
        osc = OscillationState(theta=0.4)

        defns = [
            StateDefinition(
                name="TestState",
                terms=(StateTerm(weight=1.0, variable="C_DA"),),
            ),
        ]
        result = compute_full_readout(
            nt_states, receptor_states, osc,
            state_definitions=defns,
        )
        assert "states" in result
        assert "TestState" in result["states"]
        assert result["states"]["TestState"] == pytest.approx(0.6)

    def test_with_triggers(self):
        """Triggers are evaluated."""
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=0.8),
        }
        osc = OscillationState(beta=0.7)

        triggers = [
            TriggerDefinition(
                condition_str="beta>0.6",
                actions=("INT(D2)",),
                activate_mode="LogicMode",
            ),
        ]
        result = compute_full_readout(
            nt_states, {}, osc,
            triggers=triggers,
        )
        assert "trigger_results" in result
        assert len(result["trigger_results"]) == 1
        assert result["trigger_results"][0].fired is True
        assert result["trigger_results"][0].mode == "LogicMode"

    def test_full_integration(self):
        """Everything together: metrics + states + triggers."""
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=0.6, C_phasic=0.2),
            "NE": NeurotransmitterState(C_tonic=0.5),
        }
        receptor_states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1"),
            "DA_D2": ReceptorState(receptor_id="DA_D2"),
        }
        osc = OscillationState(theta=0.5, beta=0.4)

        defns = [
            StateDefinition(
                name="Focus",
                terms=(
                    StateTerm(weight=0.5, variable="C_DA"),
                    StateTerm(weight=0.3, variable="C_NE"),
                ),
            ),
        ]
        triggers = [
            TriggerDefinition(
                condition_str="DA>0.5",
                actions=(),
                activate_mode="ActiveMode",
            ),
        ]
        result = compute_full_readout(
            nt_states, receptor_states, osc,
            state_definitions=defns,
            triggers=triggers,
        )
        assert "metrics" in result
        assert "states" in result
        assert "trigger_results" in result
        assert result["states"]["Focus"] == pytest.approx(0.5 * 0.8 + 0.3 * 0.5)
        assert result["trigger_results"][0].fired is True


# ---------------------------------------------------------------------------
# Phase 37: Mode hooks integration with compute_full_readout
# ---------------------------------------------------------------------------

from zados.neurochem.neurosymbolic.mode_hooks import (
    ModeHookDefinition,
    DEFAULT_MODE_HOOKS,
)


class TestComputeFullReadoutModeHooks:
    """Tests for mode_hooks parameter in compute_full_readout."""

    def test_no_mode_hooks(self):
        """Existing behavior unchanged when mode_hooks not provided."""
        result = compute_full_readout(
            neurotransmitter_states={},
            receptor_states={},
            oscillation_state=OscillationState(),
        )
        assert "mode_selection" not in result

    def test_with_mode_hooks_in_result(self):
        """mode_selection appears in result when mode_hooks provided."""
        hooks = [
            ModeHookDefinition(
                name="TestMode", condition_str="M_hat>0.5", priority_tier=3,
            ),
        ]
        result = compute_full_readout(
            neurotransmitter_states={},
            receptor_states={},
            oscillation_state=OscillationState(),
            mode_hooks=hooks,
        )
        assert "mode_selection" in result

    def test_mode_selection_fires(self):
        """Mode fires when a custom hook condition is met by computed metrics."""
        hooks = [
            ModeHookDefinition(
                name="AlwaysFires",
                condition_str="M_hat>=0.0",
                priority_tier=3,
                composite_gate="1.0*M_hat",
            ),
        ]
        result = compute_full_readout(
            neurotransmitter_states={},
            receptor_states={},
            oscillation_state=OscillationState(),
            mode_hooks=hooks,
        )
        assert "mode_selection" in result
        ms = result["mode_selection"]
        assert ms.active_mode == "AlwaysFires"