import pytest
from zados.neurochem.neurosymbolic.metrics import (
    NeurochemicalMetrics,
    compute_motivation,
    compute_empathy,
    compute_cognitive_rigidity,
    compute_fatigue,
    compute_precision,
    compute_openness,
    compute_anxiety,
    compute_social_engagement,
    compute_all_metrics,
)


def test_metrics_initialization():
    """Metrics object initializes with defaults."""
    metrics = NeurochemicalMetrics()
    assert metrics.motivation == 0.5
    assert metrics.empathy == 0.5
    assert metrics.fatigue == 0.5


def test_metrics_as_dict():
    """Metrics can be exported to dictionary."""
    metrics = NeurochemicalMetrics(motivation=0.7, empathy=0.3)
    d = metrics.as_dict()
    assert d["motivation"] == 0.7
    assert d["empathy"] == 0.3


def test_compute_motivation_baseline():
    """Motivation computation at baseline."""
    motivation = compute_motivation(S_DA_D3=0.5, S_OXT=0.5, S_GABA_B=0.5)
    assert 0.0 <= motivation <= 1.0


def test_compute_motivation_high():
    """High DA-D3 and OXT increase motivation."""
    motivation = compute_motivation(S_DA_D3=1.0, S_OXT=1.0, S_GABA_B=0.0)
    assert motivation > 0.5


def test_compute_motivation_low():
    """High GABA-B decreases motivation."""
    motivation = compute_motivation(S_DA_D3=0.0, S_OXT=0.0, S_GABA_B=1.0)
    assert motivation < 0.5


def test_compute_empathy():
    """Empathy is product of OXTR, theta, and 5-HT1A."""
    empathy = compute_empathy(S_OXTR=0.8, phi_theta=0.5, S_5HT1A=0.6)
    expected = 0.8 * 0.5 * 0.6
    assert empathy == pytest.approx(expected)


def test_compute_empathy_bounded():
    """Empathy stays in [0, 1]."""
    empathy = compute_empathy(S_OXTR=1.0, phi_theta=1.0, S_5HT1A=1.0)
    assert empathy == 1.0


def test_compute_cognitive_rigidity_baseline():
    """Cognitive rigidity at baseline."""
    rigidity = compute_cognitive_rigidity(S_NE_beta1=0.5, S_DA_D2=0.5, S_CB1=0.5)
    assert 0.0 <= rigidity <= 1.0


def test_compute_cognitive_rigidity_high():
    """High NE and DA-D2 increase rigidity."""
    rigidity = compute_cognitive_rigidity(S_NE_beta1=1.0, S_DA_D2=1.0, S_CB1=0.0)
    assert rigidity > 0.5


def test_compute_cognitive_rigidity_low():
    """High CB1 decreases rigidity."""
    rigidity = compute_cognitive_rigidity(S_NE_beta1=0.0, S_DA_D2=0.0, S_CB1=1.0)
    assert rigidity < 0.5


def test_compute_fatigue():
    """Fatigue increases with GABA-B and delta."""
    fatigue = compute_fatigue(S_GABA_B=0.8, phi_delta=0.6)
    expected = (0.8 + 0.6) / 2.0
    assert fatigue == pytest.approx(expected)


def test_compute_fatigue_bounded():
    """Fatigue stays in [0, 1]."""
    fatigue = compute_fatigue(S_GABA_B=1.0, phi_delta=1.0)
    assert fatigue == 1.0


def test_compute_precision():
    """Precision increases with NE, DA-D2, and beta."""
    precision = compute_precision(S_NE_beta1=0.6, S_DA_D2=0.8, phi_beta=0.5)
    expected = (0.6 + 0.8) * 0.5 / 2.0
    assert precision == pytest.approx(expected)


def test_compute_openness_baseline():
    """Openness at baseline."""
    openness = compute_openness(S_5HT2A=0.5, S_DA_D3=0.5, S_5HT1A=0.5)
    assert 0.0 <= openness <= 1.0


def test_compute_openness_high():
    """High 5-HT2A and DA-D3 increase openness."""
    openness = compute_openness(S_5HT2A=1.0, S_DA_D3=1.0, S_5HT1A=0.0)
    assert openness > 0.5


def test_compute_anxiety_baseline():
    """Anxiety at baseline."""
    anxiety = compute_anxiety(C_NE=0.5, C_CRH=0.5, C_cortisol=0.5, S_GABA_A=0.5)
    assert 0.0 <= anxiety <= 1.0


def test_compute_anxiety_high():
    """High stress hormones increase anxiety."""
    anxiety = compute_anxiety(C_NE=1.0, C_CRH=1.0, C_cortisol=1.0, S_GABA_A=0.0)
    assert anxiety > 0.5


def test_compute_anxiety_low():
    """High GABA-A decreases anxiety."""
    anxiety = compute_anxiety(C_NE=0.0, C_CRH=0.0, C_cortisol=0.0, S_GABA_A=1.0)
    assert anxiety < 0.5


def test_compute_social_engagement_baseline():
    """Social engagement at baseline."""
    engagement = compute_social_engagement(S_OXTR=0.5, S_DA_D3=0.5, C_cortisol=0.5)
    assert 0.0 <= engagement <= 1.0


def test_compute_social_engagement_high():
    """High OXT and DA-D3 increase social engagement."""
    engagement = compute_social_engagement(S_OXTR=1.0, S_DA_D3=1.0, C_cortisol=0.0)
    assert engagement > 0.5


def test_compute_all_metrics_with_full_state():
    """compute_all_metrics produces valid metrics from full state."""
    concentrations = {
        "DA": 0.6,
        "NE": 0.5,
        "CRH": 0.3,
        "cortisol": 0.4,
    }
    receptor_saturations = {
        "DA_D3": 0.7,
        "DA_D2": 0.5,
        "OXTR": 0.8,
        "GABA_B": 0.3,
        "GABA_A": 0.6,
        "NE_beta1": 0.4,
        "CB1": 0.5,
        "5HT_1A": 0.6,
        "5HT_2A": 0.7,
    }
    oscillations = {
        "delta": 0.2,
        "theta": 0.6,
        "beta": 0.5,
    }
    
    metrics = compute_all_metrics(concentrations, receptor_saturations, oscillations)
    
    # All metrics should be bounded
    assert 0.0 <= metrics.motivation <= 1.0
    assert 0.0 <= metrics.empathy <= 1.0
    assert 0.0 <= metrics.cognitive_rigidity <= 1.0
    assert 0.0 <= metrics.fatigue <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.openness <= 1.0
    assert 0.0 <= metrics.anxiety <= 1.0
    assert 0.0 <= metrics.social_engagement <= 1.0


def test_compute_all_metrics_with_partial_state():
    """compute_all_metrics handles missing keys gracefully."""
    concentrations = {"DA": 0.5}
    receptor_saturations = {"DA_D3": 0.7}
    oscillations = {"theta": 0.5}
    
    metrics = compute_all_metrics(concentrations, receptor_saturations, oscillations)
    
    # Should not crash, all metrics should be bounded
    assert 0.0 <= metrics.motivation <= 1.0
    assert 0.0 <= metrics.empathy <= 1.0


def test_compute_all_metrics_with_empty_state():
    """compute_all_metrics handles empty state."""
    metrics = compute_all_metrics({}, {}, {})
    
    # Should return valid (if low) metrics
    assert 0.0 <= metrics.motivation <= 1.0
    assert metrics.empathy == 0.0  # All zeros → product is zero


def test_all_metrics_bounded():
    """All individual metric functions produce bounded outputs."""
    # Test with extreme values
    assert 0.0 <= compute_motivation(1.0, 1.0, 0.0) <= 1.0
    assert 0.0 <= compute_empathy(1.0, 1.0, 1.0) <= 1.0
    assert 0.0 <= compute_cognitive_rigidity(1.0, 1.0, 0.0) <= 1.0
    assert 0.0 <= compute_fatigue(1.0, 1.0) <= 1.0
    assert 0.0 <= compute_precision(1.0, 1.0, 1.0) <= 1.0
    assert 0.0 <= compute_openness(1.0, 1.0, 0.0) <= 1.0
    assert 0.0 <= compute_anxiety(1.0, 1.0, 1.0, 0.0) <= 1.0
    assert 0.0 <= compute_social_engagement(1.0, 1.0, 0.0) <= 1.0