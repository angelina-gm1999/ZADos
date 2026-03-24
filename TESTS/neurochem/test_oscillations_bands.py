from zados.neurochem.oscillations.bands import OscillationState


def test_default_bands():
    osc = OscillationState()
    assert set(osc.as_dict().keys()) == {"delta", "theta", "alpha", "beta", "gamma"}


def test_set_and_get_band():
    osc = OscillationState()
    osc.set("alpha", 0.7)
    assert abs(osc.get("alpha") - 0.7) < 1e-6


def test_normalization():
    osc = OscillationState({
        "delta": 1.0,
        "theta": 1.0,
        "alpha": 2.0,
        "beta": 0.0,
        "gamma": 0.0
    })
    osc.normalize()
    d = osc.as_dict()
    total = sum(d.values())
    assert abs(total - 1.0) < 1e-6
    assert abs(d["alpha"] - 0.5) < 1e-3
