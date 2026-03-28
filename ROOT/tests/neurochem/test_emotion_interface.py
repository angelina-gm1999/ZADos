"""
Tests for emotion layer → neurochemical signal interface.

Phase 16: Verifies emotion profile conversion, recipe coverage,
and integration with NT module signal keys.
"""

import pytest

from zados.neurochem.utils.emotion_interface import (
    emotion_profile_to_signals,
    get_emotion_ids,
    get_emotion_recipe,
    get_emotions_affecting_nt,
    EmotionNTRecipe,
    DEFAULT_EMOTION_RECIPES,
)
from zados.neurochem.neurotransmitters.configs import DEFAULT_NT_CONFIGS


# =====================================================================
# EmotionNTRecipe Tests
# =====================================================================

class TestEmotionNTRecipe:
    """Test EmotionNTRecipe dataclass."""

    def test_basic_recipe(self):
        recipe = EmotionNTRecipe(
            emotion_id="test",
            nt_drives={"DA": {"emotion_drive": 0.5}},
        )
        assert recipe.emotion_id == "test"
        assert recipe.nt_drives["DA"]["emotion_drive"] == 0.5

    def test_frozen(self):
        recipe = EmotionNTRecipe(emotion_id="test")
        with pytest.raises(AttributeError):
            recipe.emotion_id = "changed"

    def test_default_empty_drives(self):
        recipe = EmotionNTRecipe(emotion_id="empty")
        assert recipe.nt_drives == {}


# =====================================================================
# Default Recipes Tests
# =====================================================================

class TestDefaultRecipes:
    """Test default emotion recipe collection."""

    def test_has_core_emotions(self):
        """Should include core emotional states."""
        expected = {
            "joy", "curiosity", "anxiety", "calm",
            "empathy", "focus", "sadness", "anger",
            "trust", "surprise", "contentment", "fear",
        }
        actual = set(DEFAULT_EMOTION_RECIPES.keys())
        assert expected.issubset(actual)

    def test_all_recipes_target_valid_nts(self):
        """Every NT targeted by a recipe must exist in configs."""
        valid_nts = set(DEFAULT_NT_CONFIGS.keys())
        for emotion_id, recipe in DEFAULT_EMOTION_RECIPES.items():
            for nt_name in recipe.nt_drives:
                assert nt_name in valid_nts, (
                    f"Emotion {emotion_id} targets unknown NT: {nt_name}"
                )

    def test_all_recipes_have_emotion_drive(self):
        """Every recipe should include emotion_drive for at least one NT."""
        for emotion_id, recipe in DEFAULT_EMOTION_RECIPES.items():
            has_drive = False
            for drives in recipe.nt_drives.values():
                if "emotion_drive" in drives:
                    has_drive = True
                    break
            assert has_drive, (
                f"Emotion {emotion_id} has no emotion_drive signal"
            )

    def test_joy_drives_da_and_5ht(self):
        recipe = DEFAULT_EMOTION_RECIPES["joy"]
        assert "DA" in recipe.nt_drives
        assert "5HT" in recipe.nt_drives

    def test_anxiety_drives_ne_and_crh(self):
        recipe = DEFAULT_EMOTION_RECIPES["anxiety"]
        assert "NE" in recipe.nt_drives
        assert "CRH" in recipe.nt_drives

    def test_fear_reduces_gaba(self):
        recipe = DEFAULT_EMOTION_RECIPES["fear"]
        assert recipe.nt_drives["GABA"]["emotion_drive"] < 0

    def test_calm_reduces_ne(self):
        recipe = DEFAULT_EMOTION_RECIPES["calm"]
        assert recipe.nt_drives["NE"]["emotion_drive"] < 0

    def test_recipes_have_descriptions(self):
        for emotion_id, recipe in DEFAULT_EMOTION_RECIPES.items():
            assert len(recipe.description) > 0, (
                f"Emotion {emotion_id} has no description"
            )


# =====================================================================
# emotion_profile_to_signals Tests
# =====================================================================

class TestEmotionProfileToSignals:
    """Test emotion profile conversion."""

    def test_single_emotion(self):
        signals = emotion_profile_to_signals({"joy": 1.0})
        assert "DA" in signals
        assert "emotion_drive" in signals["DA"]
        assert signals["DA"]["emotion_drive"] > 0.0

    def test_empty_profile(self):
        signals = emotion_profile_to_signals({})
        assert signals == {}

    def test_unknown_emotion_ignored(self):
        signals = emotion_profile_to_signals({"made_up_emotion": 1.0})
        assert signals == {}

    def test_strength_scaling(self):
        """Higher strength should produce stronger signals."""
        weak = emotion_profile_to_signals({"joy": 0.2})
        strong = emotion_profile_to_signals({"joy": 0.8})
        assert strong["DA"]["emotion_drive"] > weak["DA"]["emotion_drive"]

    def test_zero_strength(self):
        signals = emotion_profile_to_signals({"joy": 0.0})
        for nt_signals in signals.values():
            for value in nt_signals.values():
                assert value == 0.0

    def test_multiple_emotions_combine(self):
        """Multiple emotions targeting same NT should combine."""
        signals = emotion_profile_to_signals({
            "joy": 0.5,      # DA emotion_drive: 0.8 * 0.5 = 0.4
            "curiosity": 0.5,  # DA emotion_drive: 0.7 * 0.5 = 0.35
        })
        # DA emotion_drive should be sum: 0.4 + 0.35 = 0.75
        assert "DA" in signals
        assert signals["DA"]["emotion_drive"] > 0.7

    def test_opposing_emotions(self):
        """Emotions with opposing drives should partially cancel."""
        signals = emotion_profile_to_signals({
            "anxiety": 0.5,  # NE: +0.7 * 0.5 = 0.35
            "calm": 0.5,     # NE: -0.3 * 0.5 = -0.15
        })
        # NE should be net positive (0.35 - 0.15 = 0.20)
        assert signals["NE"]["emotion_drive"] > 0.0
        # But less than anxiety alone
        anxiety_only = emotion_profile_to_signals({"anxiety": 0.5})
        assert signals["NE"]["emotion_drive"] < anxiety_only["NE"]["emotion_drive"]

    def test_curiosity_adds_novelty_signal(self):
        """Curiosity should add novelty signal to DA."""
        signals = emotion_profile_to_signals({"curiosity": 0.8})
        assert "novelty" in signals["DA"]
        assert signals["DA"]["novelty"] > 0.0

    def test_custom_recipes(self):
        custom = {
            "test_emotion": EmotionNTRecipe(
                emotion_id="test_emotion",
                nt_drives={"GLU": {"emotion_drive": 0.9}},
            ),
        }
        signals = emotion_profile_to_signals(
            {"test_emotion": 1.0},
            recipes=custom,
        )
        assert "GLU" in signals
        assert abs(signals["GLU"]["emotion_drive"] - 0.9) < 1e-9

    def test_negative_strength(self):
        """Negative strength should invert signals."""
        signals = emotion_profile_to_signals({"joy": -0.5})
        assert signals["DA"]["emotion_drive"] < 0.0


# =====================================================================
# Helper Function Tests
# =====================================================================

class TestHelperFunctions:
    """Test utility helper functions."""

    def test_get_emotion_ids(self):
        ids = get_emotion_ids()
        assert isinstance(ids, list)
        assert ids == sorted(ids)
        assert "joy" in ids
        assert "anxiety" in ids

    def test_get_emotion_recipe(self):
        recipe = get_emotion_recipe("joy")
        assert recipe is not None
        assert recipe.emotion_id == "joy"

    def test_get_emotion_recipe_unknown(self):
        assert get_emotion_recipe("nonexistent") is None

    def test_get_emotions_affecting_nt(self):
        da_emotions = get_emotions_affecting_nt("DA")
        assert "joy" in da_emotions
        assert "curiosity" in da_emotions

    def test_all_nts_have_at_least_one_emotion(self):
        """Every NT should be affected by at least one emotion."""
        for nt_name in DEFAULT_NT_CONFIGS:
            emotions = get_emotions_affecting_nt(nt_name)
            assert len(emotions) >= 1, (
                f"No emotions affect {nt_name}"
            )
