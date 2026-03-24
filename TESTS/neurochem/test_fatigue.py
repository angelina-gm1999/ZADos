from zados.neurochem.kinetics.fatigue import fatigue as fatigue_update


def test_fatigue_accumulates():
    F0 = 0.0
    C = 1.0
    F1 = fatigue_update(F0, C, epsilon=0.01, decay=0.0)
    assert F1 > F0


def test_fatigue_decays_without_C():
    F0 = 1.0
    C = 0.0
    F1 = fatigue_update(F0, C, epsilon=0.01, decay=0.1)
    assert F1 < F0


def test_fatigue_clips_to_zero():
    F0 = 0.0
    C = 0.0
    F1 = fatigue_update(F0, C, epsilon=0.0, decay=0.1)
    assert F1 == 0.0
