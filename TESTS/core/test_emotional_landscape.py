"""Tests for core.processes.emotional_landscape — EmotionalPreset definitions."""

from zados.core.processes.emotional_landscape import (
    MODE_EMOTIONAL_PRESETS,
    MODE_OSCILLATORY_REGIMES,
    get_emotional_preset,
)
from zados.core.types import EmotionalPreset


# ------------------------------------------------------------------
# All 12 presets present (M1-M5 + 7 pipeline presets)
# ------------------------------------------------------------------

EXPECTED_PRESET_KEYS = {
    "M1", "M2", "M3", "M4", "M5",
    "Homework", "Reflective", "SleepTriage", "SleepREM", "SleepDream",
    "Regular", "SelfReflective",
}


def test_all_presets_present():
    assert set(MODE_EMOTIONAL_PRESETS.keys()) == EXPECTED_PRESET_KEYS


def test_preset_count():
    assert len(MODE_EMOTIONAL_PRESETS) == 12


# ------------------------------------------------------------------
# All presets are EmotionalPreset instances
# ------------------------------------------------------------------

def test_all_presets_are_emotional_presets():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        assert isinstance(preset, EmotionalPreset), (
            f"MODE_EMOTIONAL_PRESETS['{key}'] is {type(preset).__name__}, "
            f"not EmotionalPreset"
        )


# ------------------------------------------------------------------
# All presets have required fields
# ------------------------------------------------------------------

def test_all_presets_have_nt_adjustments():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        assert isinstance(preset.nt_adjustments, dict), f"'{key}' missing nt_adjustments"
        assert len(preset.nt_adjustments) > 0, f"'{key}' has empty nt_adjustments"


def test_all_presets_have_oscillatory_bias():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        assert isinstance(preset.oscillatory_bias, dict), f"'{key}' missing oscillatory_bias"


def test_all_presets_have_reward_weight_overrides():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        assert isinstance(preset.reward_weight_overrides, dict), (
            f"'{key}' missing reward_weight_overrides"
        )


def test_all_presets_have_domain_weight_overrides():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        assert isinstance(preset.domain_weight_overrides, dict), (
            f"'{key}' missing domain_weight_overrides"
        )


def test_all_presets_have_risk_emotions():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        assert isinstance(preset.risk_emotions, list), f"'{key}' missing risk_emotions"
        assert len(preset.risk_emotions) > 0, f"'{key}' has empty risk_emotions"


def test_all_presets_have_risk_thresholds():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        assert isinstance(preset.risk_thresholds, dict), f"'{key}' missing risk_thresholds"


# ------------------------------------------------------------------
# Oscillatory regimes match presets
# ------------------------------------------------------------------

def test_oscillatory_regimes_match_presets():
    assert set(MODE_OSCILLATORY_REGIMES.keys()) == EXPECTED_PRESET_KEYS


# ------------------------------------------------------------------
# get_emotional_preset() API
# ------------------------------------------------------------------

def test_get_emotional_preset_known():
    for key in EXPECTED_PRESET_KEYS:
        preset = get_emotional_preset(key)
        assert preset is not None, f"get_emotional_preset('{key}') returned None"
        assert isinstance(preset, EmotionalPreset)


def test_get_emotional_preset_unknown():
    assert get_emotional_preset("NonExistentMode") is None


# ------------------------------------------------------------------
# New pipeline presets — spot checks
# ------------------------------------------------------------------

def test_homework_preset_nt_adjustments():
    p = MODE_EMOTIONAL_PRESETS["Homework"]
    assert "ACh" in p.nt_adjustments
    assert "GABA" in p.nt_adjustments
    assert p.nt_adjustments["ACh"]["preset_drive"] > 0


def test_sleep_dream_preset_high_da():
    p = MODE_EMOTIONAL_PRESETS["SleepDream"]
    assert p.nt_adjustments["DA"]["preset_drive"] > 0
    # NE should be suppressed in dream state
    assert p.nt_adjustments["NE"]["preset_drive"] < 0


def test_sleep_triage_preset_lowered_arousal():
    p = MODE_EMOTIONAL_PRESETS["SleepTriage"]
    assert p.nt_adjustments["NE"]["preset_drive"] < 0
    assert p.nt_adjustments["histamine"]["preset_drive"] < 0


def test_regular_preset_balanced():
    p = MODE_EMOTIONAL_PRESETS["Regular"]
    # Regular preset should have mild adjustments
    for nt, signals in p.nt_adjustments.items():
        for key, val in signals.items():
            assert abs(val) <= 0.2, (
                f"Regular preset {nt}.{key} = {val} is too strong"
            )


def test_self_reflective_preset():
    p = MODE_EMOTIONAL_PRESETS["SelfReflective"]
    assert "5HT" in p.nt_adjustments
    assert "ACh" in p.nt_adjustments
    assert p.nt_adjustments["5HT"]["preset_drive"] > 0


def test_reflective_preset_reduced_vigilance():
    p = MODE_EMOTIONAL_PRESETS["Reflective"]
    assert p.nt_adjustments["NE"]["preset_drive"] < 0


def test_sleep_rem_preset_deep_inhibition():
    p = MODE_EMOTIONAL_PRESETS["SleepREM"]
    assert p.nt_adjustments["GABA"]["preset_drive"] > 0.1
    assert p.nt_adjustments["NE"]["preset_drive"] < -0.1


# ------------------------------------------------------------------
# Domain weight overrides sum to ~1.0
# ------------------------------------------------------------------

def test_domain_weight_overrides_sum_approximately():
    for key, preset in MODE_EMOTIONAL_PRESETS.items():
        total = sum(preset.domain_weight_overrides.values())
        assert 0.9 <= total <= 1.1, (
            f"'{key}' domain_weight_overrides sum = {total:.2f}, expected ~1.0"
        )
