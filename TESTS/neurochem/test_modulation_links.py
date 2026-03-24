from zados.neurochem.oscillations.modulation_links import modulate_parameters


def test_modulation_scaling():
    base_params = {"βrew": 1.0, "βnov": 1.0, "R0": 1.0}
    osc = {"beta": 0.4, "gamma": 0.6, "alpha": 0.5}

    modulated = modulate_parameters(base_params, osc)

    assert abs(modulated["βrew"] - 1.3) < 1e-6  # 1.0 * (1 + 0.5 * 0.6)
    assert abs(modulated["βnov"] - 1.12) < 1e-6  # 1.0 * (1 + 0.3 * 0.4)
    assert abs(modulated["R0"] - 0.9) < 1e-6     # 1.0 * (1 - 0.2 * 0.5)


def test_missing_bands_safe():
    base_params = {"βrew": 1.0}
    osc = {}  # no band data

    modulated = modulate_parameters(base_params, osc)
    assert modulated["βrew"] == 1.0  # unchanged
