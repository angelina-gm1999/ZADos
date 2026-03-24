from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState, OscillationState, ReceptorFunctionalState


def test_engine_initializes():
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    assert engine.current_time == 0.0
    assert engine.dt == 0.01


def test_engine_add_neurotransmitter():
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    state = engine.registry.get_neurotransmitter("DA")
    assert state is not None


def test_engine_step_no_signals():
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    engine.step()  # No signals, should not crash
    assert engine.current_time == 0.01


def test_engine_step_with_signals():
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})

    signals = {
        "DA": {"novelty": 0.8, "rpe": 0.5, "effort": 0.2}
    }
    engine.step(signals)

    state = engine.registry.get_neurotransmitter("DA")
    assert 0.0 <= state.C_tonic <= 1.0
    assert 0.0 <= state.C_phasic <= 1.0


def test_engine_multiple_steps_accumulate_time():
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})

    for _ in range(10):
        engine.step()

    assert abs(engine.current_time - 0.1) < 1e-9


def test_engine_fatigue_accumulates():
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})

    initial_F = engine.registry.get_neurotransmitter("DA").F

    for _ in range(100):
        engine.step()

    new_F = engine.registry.get_neurotransmitter("DA").F
    assert new_F > initial_F


def test_engine_neurosymbolic_readout():
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    engine.set_oscillation_state(OscillationState())

    engine.step({"DA": {"novelty": 0.5, "rpe": 0.3, "effort": 0.1}})
    readout = engine.get_neurosymbolic_readout()

    assert "motivation" in readout
    assert "fatigue" in readout


def test_engine_receptor_updates_on_step():
    """Receptors update their state when engine.step() is called."""
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    engine.add_receptor("DA_D1", config={"K_d": 0.5, "parent_nt": "DA"})

    initial = engine.registry.get_receptor("DA_D1")
    initial_trace = initial.exposure_trace

    for _ in range(100):
        engine.step({"DA": {"novelty": 0.5, "rpe": 0.3, "effort": 0.1}})

    updated = engine.registry.get_receptor("DA_D1")
    assert updated.exposure_trace > initial_trace
    assert updated.time_in_state > 0


def test_engine_receptor_desensitization():
    """Receptor eventually desensitizes under sustained high concentration."""
    engine = NeurochemicalEngine(dt=0.1, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.9})
    engine.add_receptor("DA_D1", config={
        "K_d": 0.2,
        "parent_nt": "DA",
        "thresholds": {"t0_desens": 2.0, "theta_desens": 0.5},
    })

    for _ in range(100):
        engine.step({"DA": {"novelty": 0.8, "rpe": 0.5, "effort": 0.3}})

    state = engine.registry.get_receptor("DA_D1")
    # With K_d=0.2 and high C, saturation > 0.5 threshold
    # After 10 seconds (100 * 0.1), should have transitioned
    assert state.chi != ReceptorFunctionalState.ACTIVE or state.sigma < 1.0