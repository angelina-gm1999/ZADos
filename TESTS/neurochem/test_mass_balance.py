import pytest
import math
from zados.neurochem.kinetics.mass_balance import (
    compute_reuptake_loss,
    compute_degradation_loss,
    compute_clearance_loss,
    compute_total_loss,
    compute_drift_term,
    compute_diffusion_term,
    compute_mass_balance_drift,
    compute_effective_reversion_rate,
)


def test_compute_reuptake_loss():
    """Reuptake loss is proportional to concentration and transporter efficiency."""
    loss = compute_reuptake_loss(C=0.5, eta_u=0.8, u_base=0.1)
    expected = 0.1 * 0.8 * 0.5
    assert loss == pytest.approx(expected)


def test_compute_reuptake_loss_zero_efficiency():
    """Zero transporter efficiency means no reuptake."""
    loss = compute_reuptake_loss(C=0.5, eta_u=0.0, u_base=0.1)
    assert loss == 0.0


def test_compute_degradation_loss():
    """Degradation is first-order kinetics."""
    loss = compute_degradation_loss(C=0.6, d_base=0.05)
    expected = 0.05 * 0.6
    assert loss == pytest.approx(expected)


def test_compute_clearance_loss():
    """Clearance is proportional to concentration."""
    loss = compute_clearance_loss(C=0.4, c_base=0.02)
    expected = 0.02 * 0.4
    assert loss == pytest.approx(expected)


def test_compute_total_loss():
    """Total loss is sum of all loss mechanisms."""
    C = 0.5
    eta_u = 0.8
    u_base = 0.1
    d_base = 0.05
    c_base = 0.02
    
    total = compute_total_loss(C, eta_u, u_base, d_base, c_base)
    
    L_u = u_base * eta_u * C
    L_d = d_base * C
    L_c = c_base * C
    expected = L_u + L_d + L_c
    
    assert total == pytest.approx(expected)


def test_compute_drift_term_above_baseline():
    """Concentration above baseline produces negative drift (mean reversion down)."""
    drift = compute_drift_term(
        C=0.8,
        C_baseline=0.5,
        theta=0.2,
        eta_u=1.0,
    )
    # Negative drift because C > baseline and losses also negative
    assert drift < 0.0


def test_compute_drift_term_below_baseline():
    """Concentration below baseline can produce positive drift if reversion stronger than loss."""
    drift = compute_drift_term(
        C=0.2,
        C_baseline=0.5,
        theta=0.5,
        eta_u=0.1,
        u_base=0.01,
        d_base=0.01,
        c_base=0.01,
    )
    # Positive reversion should dominate small losses
    assert drift > 0.0


def test_compute_drift_term_at_baseline():
    """At baseline, only loss terms contribute (negative drift)."""
    drift = compute_drift_term(
        C=0.5,
        C_baseline=0.5,
        theta=0.2,
        eta_u=1.0,
    )
    # Should be negative due to losses
    assert drift < 0.0


def test_compute_diffusion_term_multiplicative():
    """Multiplicative diffusion scales with √C."""
    C = 0.64
    sigma = 0.1
    
    diffusion = compute_diffusion_term(C, sigma, multiplicative=True)
    expected = sigma * math.sqrt(C)
    
    assert diffusion == pytest.approx(expected)


def test_compute_diffusion_term_additive():
    """Additive diffusion is constant."""
    diffusion = compute_diffusion_term(C=0.5, sigma=0.1, multiplicative=False)
    assert diffusion == pytest.approx(0.1)


def test_compute_diffusion_term_zero_concentration():
    """Multiplicative diffusion goes to zero as C → 0."""
    diffusion = compute_diffusion_term(C=0.0, sigma=0.1, multiplicative=True)
    assert diffusion == 0.0


def test_compute_mass_balance_drift():
    """Separate drift for tonic and phasic components."""
    drift_tonic, drift_phasic = compute_mass_balance_drift(
        C_tonic=0.5,
        C_phasic=0.3,
        C_baseline=0.5,
        theta_tonic=0.1,
        theta_phasic=1.0,
        eta_u=0.8,
    )
    
    # Tonic at baseline → should have negative drift (losses only)
    assert drift_tonic < 0.0
    
    # Phasic above zero with high reversion → strong negative drift
    assert drift_phasic < 0.0
    assert abs(drift_phasic) > abs(drift_tonic)  # Phasic decays faster


def test_compute_mass_balance_drift_phasic_decay():
    """Phasic component decays toward zero, not baseline."""
    drift_tonic, drift_phasic = compute_mass_balance_drift(
        C_tonic=0.5,
        C_phasic=0.5,
        C_baseline=0.5,
        theta_tonic=0.1,
        theta_phasic=1.0,
        eta_u=0.8,
    )
    
    # Phasic should decay more aggressively
    assert abs(drift_phasic) > abs(drift_tonic)


def test_compute_effective_reversion_rate_no_fatigue():
    """No fatigue means full reversion rate."""
    theta_eff = compute_effective_reversion_rate(
        theta_base=0.5,
        fatigue=0.0,
        fatigue_scaling=0.5,
    )
    assert theta_eff == pytest.approx(0.5)


def test_compute_effective_reversion_rate_high_fatigue():
    """High fatigue reduces reversion rate."""
    theta_eff = compute_effective_reversion_rate(
        theta_base=0.5,
        fatigue=1.0,
        fatigue_scaling=0.5,
    )
    expected = 0.5 * (1.0 - 0.5 * 1.0)
    assert theta_eff == pytest.approx(expected)


def test_compute_effective_reversion_rate_bounded():
    """Effective reversion rate is non-negative even with extreme fatigue."""
    theta_eff = compute_effective_reversion_rate(
        theta_base=0.2,
        fatigue=1.0,
        fatigue_scaling=1.0,
    )
    assert theta_eff >= 0.0


def test_loss_functions_positive():
    """All loss functions produce non-negative values."""
    assert compute_reuptake_loss(0.5, 0.8) >= 0.0
    assert compute_degradation_loss(0.5) >= 0.0
    assert compute_clearance_loss(0.5) >= 0.0
    assert compute_total_loss(0.5, 0.8) >= 0.0


def test_diffusion_nonnegative():
    """Diffusion term is always non-negative."""
    assert compute_diffusion_term(0.5, 0.1, multiplicative=True) >= 0.0
    assert compute_diffusion_term(0.5, 0.1, multiplicative=False) >= 0.0
    assert compute_diffusion_term(0.0, 0.1, multiplicative=True) >= 0.0