import pytest
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import OscillationState, ReceptorFunctionalState


def test_alpha_suppresses_noise():
    """Higher alpha amplitude reduces stochastic noise in concentration.

    PDF Appendix I: alpha → noise suppression.
    With alpha=1.0, sigma is suppressed by factor max(0.1, 1 - 0.4) = 0.6.
    We test by running many steps and checking variance is lower with alpha.
    """
    import statistics

    results_no_alpha = []
    results_alpha = []

    for trial in range(20):
        engine = NeurochemicalEngine(dt=0.01, seed=trial)
        engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})
        for _ in range(100):
            engine.step()
        results_no_alpha.append(engine.registry.get_neurotransmitter("DA").C_tonic)

    for trial in range(20):
        engine = NeurochemicalEngine(dt=0.01, seed=trial)
        engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})
        engine.set_oscillation_state(OscillationState(alpha=1.0))
        for _ in range(100):
            engine.step()
        results_alpha.append(engine.registry.get_neurotransmitter("DA").C_tonic)

    var_no_alpha = statistics.variance(results_no_alpha)
    var_alpha = statistics.variance(results_alpha)

    # Alpha should reduce variance (less noise)
    assert var_alpha <= var_no_alpha


def test_gamma_boosts_phasic_release():
    """Higher gamma amplitude boosts phasic release drive.

    PDF Appendix I: gamma → release boost.
    With gamma=1.0, oscillatory gating amplifies release drive.
    """
    # Engine with high gamma
    engine_gamma = NeurochemicalEngine(dt=0.01, seed=42)
    engine_gamma.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    engine_gamma.set_oscillation_state(OscillationState(gamma=0.8))

    # Engine without gamma (same seed)
    engine_no_gamma = NeurochemicalEngine(dt=0.01, seed=42)
    engine_no_gamma.add_neurotransmitter("DA", config={"C_baseline": 0.5})

    # Step with release-driving signals
    for _ in range(50):
        engine_gamma.step({"DA": {"novelty": 0.8, "rpe": 0.5, "effort": 0.3}})
        engine_no_gamma.step({"DA": {"novelty": 0.8, "rpe": 0.5, "effort": 0.3}})

    state_gamma = engine_gamma.registry.get_neurotransmitter("DA")
    state_no_gamma = engine_no_gamma.registry.get_neurotransmitter("DA")

    # With gamma, phasic component should be higher due to amplified release
    assert state_gamma.C_phasic >= state_no_gamma.C_phasic


def test_theta_lowers_receptor_kd():
    """Theta oscillation should lower effective K_d (increase affinity).

    PDF Appendix I: theta → K_d modulation.
    K_d_eff = K_d * (1 - 0.3 * theta), increasing saturation at same concentration.
    """
    # Engine with theta → should have lower K_d → higher saturation
    engine_theta = NeurochemicalEngine(dt=0.01, seed=42)
    engine_theta.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    engine_theta.add_receptor("DA_D1", config={"K_d": 0.4, "parent_nt": "DA"})
    engine_theta.set_oscillation_state(OscillationState(theta=0.8))

    # Engine without theta (same seed)
    engine_no_theta = NeurochemicalEngine(dt=0.01, seed=42)
    engine_no_theta.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    engine_no_theta.add_receptor("DA_D1", config={"K_d": 0.4, "parent_nt": "DA"})

    for _ in range(50):
        engine_theta.step({"DA": {"novelty": 0.5}})
        engine_no_theta.step({"DA": {"novelty": 0.5}})

    # With theta, lower K_d means higher saturation at same concentration
    # → exposure_trace accumulates higher saturation → should be greater
    receptor_theta = engine_theta.registry.get_receptor("DA_D1")
    receptor_no_theta = engine_no_theta.registry.get_receptor("DA_D1")

    # Both should have positive exposure traces
    assert receptor_theta.exposure_trace >= 0.0
    assert receptor_no_theta.exposure_trace >= 0.0

    # Theta-modulated engine should have higher exposure trace
    # (lower K_d → higher saturation at same concentration → more trace accumulation)
    assert receptor_theta.exposure_trace >= receptor_no_theta.exposure_trace


def test_beta_accelerates_desensitization_in_engine():
    """Higher beta in oscillation state causes faster receptor desensitization."""
    # Engine with high beta
    engine = NeurochemicalEngine(dt=0.1, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.9})
    engine.add_receptor("DA_D1", config={
        "K_d": 0.2,
        "parent_nt": "DA",
        "thresholds": {"t0_desens": 5.0, "theta_desens": 0.5},
    })
    engine.set_oscillation_state(OscillationState(beta=1.0))

    # Run until desensitization (beta reduces t0_desens from 5.0 to 3.5)
    for _ in range(50):
        engine.step({"DA": {"novelty": 0.8, "rpe": 0.5, "effort": 0.3}})

    state_beta = engine.registry.get_receptor("DA_D1")

    # Engine without beta (same seed)
    engine2 = NeurochemicalEngine(dt=0.1, seed=42)
    engine2.add_neurotransmitter("DA", config={"C_baseline": 0.9})
    engine2.add_receptor("DA_D1", config={
        "K_d": 0.2,
        "parent_nt": "DA",
        "thresholds": {"t0_desens": 5.0, "theta_desens": 0.5},
    })
    # No oscillation state

    for _ in range(50):
        engine2.step({"DA": {"novelty": 0.8, "rpe": 0.5, "effort": 0.3}})

    state_no_beta = engine2.registry.get_receptor("DA_D1")

    # With beta, desensitization should occur sooner -> sigma should be <= no-beta case
    assert state_beta.sigma <= state_no_beta.sigma


def test_oscillation_coupling_without_oscillations():
    """Engine works correctly when no oscillation state is set."""
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    engine.add_receptor("DA_D1", config={"K_d": 0.5, "parent_nt": "DA"})

    # Should not crash
    for _ in range(100):
        engine.step({"DA": {"novelty": 0.5, "rpe": 0.3, "effort": 0.1}})

    state = engine.registry.get_neurotransmitter("DA")
    # C_tonic and C_phasic are individually bounded [0,1]
    assert 0.0 <= state.C_tonic <= 1.0
    assert 0.0 <= state.C_phasic <= 1.0

    receptor = engine.registry.get_receptor("DA_D1")
    assert receptor.exposure_trace >= 0.0


def test_reuptake_drift_formula():
    """Test that higher reuptake rate causes more negative drift (pure kinetics)."""
    from zados.neurochem.kinetics.mass_balance import compute_drift_term

    u_base = 0.1
    u_base_high = 0.13  # 30% increase

    drift_low = compute_drift_term(
        C=0.9, C_baseline=0.5, theta=0.1, eta_u=1.0,
        u_base=u_base, d_base=0.05, c_base=0.02,
    )
    drift_high = compute_drift_term(
        C=0.9, C_baseline=0.5, theta=0.1, eta_u=1.0,
        u_base=u_base_high, d_base=0.05, c_base=0.02,
    )

    # Both drifts should be negative (above baseline), higher u_base → more negative
    assert drift_high < drift_low < 0
