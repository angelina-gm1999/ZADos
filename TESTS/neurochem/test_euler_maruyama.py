import pytest
import math
from zados.neurochem.stochastic_modulation.euler_maruyama import (
    euler_maruyama_step,
    euler_maruyama_step_bounded,
    integrate_sde,
    generate_brownian_increments,
    compute_local_truncation_error,
    check_stability_condition,
)


def test_euler_maruyama_step_deterministic():
    """EM step with zero noise is deterministic Euler step."""
    X = 0.5
    drift = 0.1
    diffusion = 0.05
    dt = 0.1
    dW = 0.0
    
    X_next = euler_maruyama_step(X, drift, diffusion, dt, dW)
    expected = X + drift * dt
    
    assert X_next == pytest.approx(expected)


def test_euler_maruyama_step_with_noise():
    """EM step with noise includes stochastic term."""
    X = 0.5
    drift = 0.1
    diffusion = 0.05
    dt = 0.1
    dW = 0.3
    
    X_next = euler_maruyama_step(X, drift, diffusion, dt, dW)
    expected = X + drift * dt + diffusion * dW
    
    assert X_next == pytest.approx(expected)


def test_euler_maruyama_step_bounded_clamping():
    """Bounded EM step clamps to boundaries."""
    X = 0.9
    drift = 0.5  # Would push above 1.0
    diffusion = 0.0
    dt = 0.5
    
    X_next = euler_maruyama_step_bounded(
        X, drift, diffusion, dt,
        lower_bound=0.0,
        upper_bound=1.0,
        reflection=False,
    )
    
    assert X_next == 1.0


def test_euler_maruyama_step_bounded_lower():
    """Bounded EM step prevents going below lower bound."""
    X = 0.1
    drift = -0.5  # Would push below 0.0
    diffusion = 0.0
    dt = 0.5
    
    X_next = euler_maruyama_step_bounded(
        X, drift, diffusion, dt,
        lower_bound=0.0,
        upper_bound=1.0,
        reflection=False,
    )
    
    assert X_next == 0.0


def test_euler_maruyama_step_bounded_within():
    """Bounded EM step unchanged when within bounds."""
    X = 0.5
    drift = 0.1
    diffusion = 0.0
    dt = 0.1
    
    X_next = euler_maruyama_step_bounded(
        X, drift, diffusion, dt,
        lower_bound=0.0,
        upper_bound=1.0,
        reflection=False,
    )
    
    expected = X + drift * dt
    assert X_next == pytest.approx(expected)


def test_integrate_sde_deterministic():
    """Integration with zero diffusion produces deterministic trajectory."""
    def drift_fn(X, t):
        return 0.1
    
    def diffusion_fn(X, t):
        return 0.0
    
    time_points, trajectory = integrate_sde(
        X0=0.5,
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        t0=0.0,
        t_final=1.0,
        dt=0.1,
        seed=42,
    )
    
    # Should have ~11 points (0.0, 0.1, ..., 1.0)
    assert len(time_points) >= 10
    assert len(trajectory) == len(time_points)
    
    # Trajectory should increase linearly
    assert trajectory[-1] > trajectory[0]


def test_integrate_sde_bounded():
    """Integration respects bounds."""
    def drift_fn(X, t):
        return 1.0  # Strong upward drift
    
    def diffusion_fn(X, t):
        return 0.0
    
    time_points, trajectory = integrate_sde(
        X0=0.5,
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        t0=0.0,
        t_final=2.0,
        dt=0.1,
        lower_bound=0.0,
        upper_bound=1.0,
        seed=42,
    )
    
    # All values should be in bounds
    assert all(0.0 <= x <= 1.0 for x in trajectory)
    
    # Should saturate at upper bound
    assert trajectory[-1] == 1.0


def test_integrate_sde_mean_reversion():
    """Integration with mean-reverting drift."""
    C_baseline = 0.5
    theta = 0.5
    
    def drift_fn(X, t):
        return -theta * (X - C_baseline)
    
    def diffusion_fn(X, t):
        return 0.01
    
    time_points, trajectory = integrate_sde(
        X0=0.1,  # Start below baseline
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        t0=0.0,
        t_final=10.0,
        dt=0.1,
        seed=42,
    )
    
    # Should converge toward baseline
    assert trajectory[-1] > trajectory[0]
    assert abs(trajectory[-1] - C_baseline) < 0.2


def test_generate_brownian_increments_statistics():
    """Brownian increments have correct mean and variance."""
    n_steps = 10000
    dt = 0.1
    
    increments = generate_brownian_increments(n_steps, dt, seed=42)
    
    # Mean should be ~0
    mean = sum(increments) / len(increments)
    assert abs(mean) < 0.02
    
    # Variance should be ~dt
    variance = sum((x - mean)**2 for x in increments) / len(increments)
    assert abs(variance - dt) < 0.01


def test_generate_brownian_increments_reproducible():
    """Same seed produces same increments."""
    n_steps = 10
    dt = 0.1
    
    inc1 = generate_brownian_increments(n_steps, dt, seed=42)
    inc2 = generate_brownian_increments(n_steps, dt, seed=42)
    
    assert inc1 == inc2


def test_compute_local_truncation_error():
    """Local truncation error scales with σ√dt."""
    diffusion = 0.1
    dt = 0.01
    
    error = compute_local_truncation_error(diffusion, dt)
    expected = diffusion * math.sqrt(dt)
    
    assert error == pytest.approx(expected)


def test_compute_local_truncation_error_zero_diffusion():
    """Zero diffusion means zero error."""
    error = compute_local_truncation_error(0.0, 0.1)
    assert error == 0.0


def test_check_stability_condition_stable():
    """Stable configuration passes stability check."""
    stable = check_stability_condition(
        drift=0.1,
        diffusion=0.05,
        dt=0.01,
        X=0.5,
    )
    assert stable is True


def test_check_stability_condition_unstable_drift():
    """Large drift with large dt fails stability."""
    stable = check_stability_condition(
        drift=10.0,
        diffusion=0.05,
        dt=0.1,
        X=0.5,
    )
    assert stable is False


def test_check_stability_condition_unstable_diffusion():
    """Large diffusion with large dt fails stability."""
    stable = check_stability_condition(
        drift=0.1,
        diffusion=10.0,
        dt=0.1,
        X=0.5,
    )
    assert stable is False


def test_check_stability_condition_zero_state():
    """Zero state is considered stable."""
    stable = check_stability_condition(
        drift=1.0,
        diffusion=1.0,
        dt=0.1,
        X=0.0,
    )
    assert stable is True


def test_euler_maruyama_step_bounded_reflection():
    """Reflection mode bounces off boundaries."""
    X = 0.95
    drift = 0.2  # Would go above 1.0
    diffusion = 0.0
    dt = 0.5
    
    X_next = euler_maruyama_step_bounded(
        X, drift, diffusion, dt,
        lower_bound=0.0,
        upper_bound=1.0,
        reflection=True,
    )
    
    # Should reflect back from boundary
    assert 0.0 <= X_next <= 1.0