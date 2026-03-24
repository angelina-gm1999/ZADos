from zados.neurochem.kinetics.reuptake import reuptake as reuptake_function


def test_reuptake_no_fatigue():
    result = reuptake_function(C=1.0, F=0.0, ku0=0.5, gamma=0.4)
    assert abs(result - 0.5) < 1e-6


def test_reuptake_with_fatigue():
    result = reuptake_function(C=1.0, F=0.5, ku0=0.5, gamma=0.4)
    expected = 0.5 * (1.0 - 0.4 * 0.5)  # = 0.5 * 0.8 = 0.4
    assert abs(result - expected) < 1e-6


def test_reuptake_fatigue_caps_at_zero():
    result = reuptake_function(C=1.0, F=5.0, ku0=0.5, gamma=1.0)
    # Fatigue scaling goes negative → clipped to 0
    assert abs(result - 0.0) < 1e-6
