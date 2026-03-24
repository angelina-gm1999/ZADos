import pytest
import math
from zados.neurochem.kinetics.release_drives import (
    compute_novelty_drive,
    compute_rpe_drive,
    compute_effort_drive,
    compute_combined_release_drive,
    apply_fatigue_gating,
    apply_oscillatory_gating,
    compute_phasic_burst_amplitude,
    compute_adaptive_threshold,
    update_recent_activity_trace,
)


def test_compute_novelty_drive_above_threshold():
    """Novelty above threshold triggers drive."""
    drive = compute_novelty_drive(stimulus_novelty=0.6, sensitivity=1.0, threshold=0.3)
    expected = 1.0 * (0.6 - 0.3)
    assert drive == pytest.approx(expected)


def test_compute_novelty_drive_below_threshold():
    """Novelty below threshold produces no drive."""
    drive = compute_novelty_drive(stimulus_novelty=0.2, sensitivity=1.0, threshold=0.3)
    assert drive == 0.0


def test_compute_novelty_drive_at_threshold():
    """Novelty exactly at threshold produces no drive."""
    drive = compute_novelty_drive(stimulus_novelty=0.3, sensitivity=1.0, threshold=0.3)
    assert drive == 0.0


def test_compute_novelty_drive_sensitivity():
    """Sensitivity scales novelty drive."""
    drive = compute_novelty_drive(stimulus_novelty=0.6, sensitivity=2.0, threshold=0.3)
    expected = 2.0 * (0.6 - 0.3)
    assert drive == pytest.approx(expected)


def test_compute_rpe_drive_positive():
    """Positive RPE produces positive drive."""
    drive = compute_rpe_drive(reward_prediction_error=0.5, gain=1.0)
    assert drive == pytest.approx(0.5)


def test_compute_rpe_drive_negative():
    """Negative RPE produces negative drive."""
    drive = compute_rpe_drive(reward_prediction_error=-0.3, gain=1.0)
    assert drive == pytest.approx(-0.3)


def test_compute_rpe_drive_zero():
    """Zero RPE produces no drive."""
    drive = compute_rpe_drive(reward_prediction_error=0.0, gain=1.0)
    assert drive == 0.0


def test_compute_rpe_drive_gain():
    """Gain scales RPE drive."""
    drive = compute_rpe_drive(reward_prediction_error=0.5, gain=2.0)
    assert drive == pytest.approx(1.0)


def test_compute_effort_drive_above_threshold():
    """Task demand above threshold triggers drive."""
    drive = compute_effort_drive(task_demand=0.5, willingness=1.0, threshold=0.2)
    expected = 1.0 * (0.5 - 0.2)
    assert drive == pytest.approx(expected)


def test_compute_effort_drive_below_threshold():
    """Task demand below threshold produces no drive."""
    drive = compute_effort_drive(task_demand=0.1, willingness=1.0, threshold=0.2)
    assert drive == 0.0


def test_compute_effort_drive_low_willingness():
    """Low willingness reduces effort drive."""
    drive = compute_effort_drive(task_demand=0.5, willingness=0.5, threshold=0.2)
    expected = 0.5 * (0.5 - 0.2)
    assert drive == pytest.approx(expected)


def test_compute_combined_release_drive():
    """Combined drive sums all components."""
    combined = compute_combined_release_drive(
        novelty_drive=0.3,
        rpe_drive=0.2,
        effort_drive=0.1,
        baseline_release=0.05,
    )
    expected = 0.05 + 0.3 + 0.2 + 0.1
    assert combined == pytest.approx(expected)


def test_compute_combined_release_drive_negative_rpe():
    """Combined drive can be reduced by negative RPE."""
    combined = compute_combined_release_drive(
        novelty_drive=0.3,
        rpe_drive=-0.4,
        effort_drive=0.0,
        baseline_release=0.0,
    )
    expected = 0.3 - 0.4
    assert combined == pytest.approx(expected)


def test_apply_fatigue_gating_below_threshold():
    """Fatigue below threshold does not gate release."""
    gated = apply_fatigue_gating(
        release_drive=1.0,
        fatigue=0.5,
        fatigue_threshold=0.7,
    )
    assert gated == pytest.approx(1.0)


def test_apply_fatigue_gating_above_threshold():
    """Fatigue above threshold suppresses release."""
    gated = apply_fatigue_gating(
        release_drive=1.0,
        fatigue=0.9,
        fatigue_threshold=0.7,
        suppression_factor=0.5,
    )
    assert gated < 1.0


def test_apply_fatigue_gating_extreme():
    """Extreme fatigue strongly suppresses release."""
    gated = apply_fatigue_gating(
        release_drive=1.0,
        fatigue=1.0,
        fatigue_threshold=0.7,
        suppression_factor=1.0,
    )
    assert gated < 0.5


def test_apply_oscillatory_gating_no_oscillation():
    """No oscillation means baseline release."""
    gated = apply_oscillatory_gating(
        release_drive=1.0,
        oscillation_amplitude=0.0,
        band_preference=1.0,
    )
    assert gated == pytest.approx(1.0)


def test_apply_oscillatory_gating_enhancement():
    """Oscillation enhances release when band is active."""
    gated = apply_oscillatory_gating(
        release_drive=1.0,
        oscillation_amplitude=0.5,
        band_preference=1.0,
    )
    expected = 1.0 * (1.0 + 1.0 * 0.5)
    assert gated == pytest.approx(expected)


def test_compute_phasic_burst_amplitude_zero_drive():
    """Zero drive produces no burst."""
    amplitude = compute_phasic_burst_amplitude(release_drive=0.0)
    assert amplitude == 0.0


def test_compute_phasic_burst_amplitude_negative_drive():
    """Negative drive produces no burst."""
    amplitude = compute_phasic_burst_amplitude(release_drive=-0.5)
    assert amplitude == 0.0


def test_compute_phasic_burst_amplitude_low_drive():
    """Low drive produces small burst."""
    amplitude = compute_phasic_burst_amplitude(
        release_drive=0.1,
        receptor_sensitivity=1.0,
        max_burst=1.0,
    )
    assert 0.0 < amplitude < 0.2


def test_compute_phasic_burst_amplitude_high_drive():
    """High drive saturates near max burst."""
    amplitude = compute_phasic_burst_amplitude(
        release_drive=10.0,
        receptor_sensitivity=1.0,
        max_burst=1.0,
    )
    assert amplitude > 0.99


def test_compute_phasic_burst_amplitude_bounded():
    """Burst amplitude never exceeds max_burst."""
    amplitude = compute_phasic_burst_amplitude(
        release_drive=100.0,
        receptor_sensitivity=1.0,
        max_burst=1.0,
    )
    assert amplitude <= 1.0


def test_compute_adaptive_threshold_no_activity():
    """No recent activity means baseline threshold."""
    threshold = compute_adaptive_threshold(
        baseline_threshold=0.3,
        recent_activity=0.0,
        adaptation_rate=0.1,
    )
    assert threshold == pytest.approx(0.3)


def test_compute_adaptive_threshold_high_activity():
    """High recent activity raises threshold."""
    threshold = compute_adaptive_threshold(
        baseline_threshold=0.3,
        recent_activity=1.0,
        adaptation_rate=0.1,
    )
    expected = 0.3 + 0.1 * 1.0
    assert threshold == pytest.approx(expected)


def test_update_recent_activity_trace_decay():
    """Activity trace decays exponentially."""
    updated = update_recent_activity_trace(
        current_trace=1.0,
        current_drive=0.0,
        dt=1.0,
        tau=10.0,
    )
    expected = 1.0 * math.exp(-1.0 / 10.0)
    assert updated == pytest.approx(expected)


def test_update_recent_activity_trace_accumulation():
    """Activity trace accumulates new drive."""
    updated = update_recent_activity_trace(
        current_trace=0.0,
        current_drive=1.0,
        dt=0.1,
        tau=10.0,
    )
    assert updated > 0.0


def test_update_recent_activity_trace_combined():
    """Activity trace combines decay and accumulation."""
    updated = update_recent_activity_trace(
        current_trace=1.0,
        current_drive=0.5,
        dt=1.0,
        tau=10.0,
    )
    decay = 1.0 * math.exp(-1.0 / 10.0)
    accumulation = 0.5 * 1.0
    expected = decay + accumulation
    assert updated == pytest.approx(expected)


def test_all_drives_nonnegative():
    """Novelty and effort drives are always non-negative."""
    assert compute_novelty_drive(0.8, 1.0, 0.3) >= 0.0
    assert compute_novelty_drive(0.1, 1.0, 0.3) >= 0.0
    assert compute_effort_drive(0.8, 1.0, 0.2) >= 0.0
    assert compute_effort_drive(0.1, 1.0, 0.2) >= 0.0


def test_burst_amplitude_nonnegative():
    """Burst amplitude is always non-negative."""
    assert compute_phasic_burst_amplitude(0.5) >= 0.0
    assert compute_phasic_burst_amplitude(0.0) >= 0.0
    assert compute_phasic_burst_amplitude(-0.5) >= 0.0