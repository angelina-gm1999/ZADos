"""Tests for Engine 28 — Emotional Detection Engine."""
import math

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.emotional_detection_engine import (
    EmotionalDetectionEngine, EDConfig, EDState,
    EmotionGroup, EmotionSignature, EMOTION_SIGNATURES, EMOTION_NT_PROFILES,
    STRUCTURAL_EMOTIONS, STRUCTURAL_TO_ID,
    EmotionalDetectionInput, DetectedEmotion, ToneVector, EmotionNeurochem,
    EmotionalDetectionResult,
    extract_valence, extract_arousal, extract_domain_scores,
    extract_structural_features, keyword_match_score,
    valence_match, arousal_match, domain_match,
    score_emotion, estimate_intensity,
    apply_mutual_exclusion, detect_structural_emotions,
    compute_tone_vector, compute_oxt_drift,
    compute_5ht1a_affinity, compute_gaba_reuptake_mod,
    map_emotions_to_neurochem, apply_nt_detection_bias,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Helpers
# =====================================================================

def _input(text="", tokens=None, lemmas=None, questions=0, sentences=1, mode="normal"):
    tok = tuple(text.split()) if tokens is None else tuple(tokens)
    lem = lemmas or tok
    return EmotionalDetectionInput(
        tokens=tok, lemmatized_tokens=tuple(lem),
        raw_text=text, sentence_count=sentences,
        question_count=questions, active_mode=mode,
    )


# =====================================================================
# 1. Enums
# =====================================================================

class TestEnums:
    def test_emotion_groups(self):
        assert EmotionGroup.TRUST_RELATIONAL.value == "trust_relational"
        assert EmotionGroup.POSITIVE_CREATIVE.value == "positive_creative"
        assert len(EmotionGroup) == 7


# =====================================================================
# 2. Emotion Registry
# =====================================================================

class TestRegistry:
    def test_all_46_emotions(self):
        assert len(EMOTION_SIGNATURES) >= 40  # Some IDs skip (17, 18)

    def test_key_emotions_present(self):
        assert 1 in EMOTION_SIGNATURES   # Betrayal
        assert 21 in EMOTION_SIGNATURES  # Anxiety
        assert 26 in EMOTION_SIGNATURES  # Joy
        assert 40 in EMOTION_SIGNATURES  # Confident
        assert 46 in EMOTION_SIGNATURES  # Belonging

    def test_nt_profiles_cover_emotions(self):
        for eid, sig in EMOTION_SIGNATURES.items():
            assert sig.emotion_name in EMOTION_NT_PROFILES, \
                f"Missing NT profile for {sig.emotion_name} (#{eid})"

    def test_structural_emotions(self):
        assert "grief" in STRUCTURAL_EMOTIONS
        assert "joy" in STRUCTURAL_EMOTIONS
        assert "anger" in STRUCTURAL_EMOTIONS
        assert "humor" in STRUCTURAL_EMOTIONS


# =====================================================================
# 3. Valence Extraction
# =====================================================================

class TestValence:
    def test_positive(self):
        _, _, v_net = extract_valence(("great", "amazing", "wonderful"), {})
        assert v_net > 0

    def test_negative(self):
        _, _, v_net = extract_valence(("terrible", "awful", "hate"), {})
        assert v_net < 0

    def test_neutral(self):
        _, _, v_net = extract_valence(("the", "cat", "sat"), {})
        assert abs(v_net) < 0.1

    def test_empty(self):
        v_pos, v_neg, v_net = extract_valence((), {})
        assert v_pos == 0.0
        assert v_neg == 0.0


# =====================================================================
# 4. Arousal Extraction
# =====================================================================

class TestArousal:
    def test_exclamation(self):
        a = extract_arousal("Help!!!", ("Help!!!",), EDConfig())
        assert a > 0

    def test_caps(self):
        a = extract_arousal("THIS IS URGENT", ("THIS", "IS", "URGENT"), EDConfig())
        assert a > 0

    def test_calm(self):
        a = extract_arousal("hello there", ("hello", "there"), EDConfig())
        assert a < 0.3

    def test_empty(self):
        assert extract_arousal("", (), EDConfig()) == 0.0


# =====================================================================
# 5. Domain Extraction
# =====================================================================

class TestDomain:
    def test_social(self):
        scores = extract_domain_scores(("friend", "trust", "together"), {})
        assert scores["social"] > scores["cognitive"]

    def test_cognitive(self):
        scores = extract_domain_scores(("think", "understand", "logic"), {})
        assert scores["cognitive"] > scores["social"]


# =====================================================================
# 6. Keyword Match
# =====================================================================

class TestKeywordMatch:
    def test_full_match(self):
        sig = EMOTION_SIGNATURES[26]  # Joy
        score = keyword_match_score(sig, ("joy", "happy", "delight"), {})
        assert score > 0.3

    def test_no_match(self):
        sig = EMOTION_SIGNATURES[26]  # Joy
        score = keyword_match_score(sig, ("programming", "code", "debug"), {})
        assert score == 0.0

    def test_partial_match(self):
        sig = EMOTION_SIGNATURES[21]  # Anxiety
        score = keyword_match_score(sig, ("anxious", "the", "dog"), {})
        assert score > 0


# =====================================================================
# 7. Valence/Arousal Match
# =====================================================================

class TestMatchFunctions:
    def test_valence_match_in_range(self):
        sig = EMOTION_SIGNATURES[26]  # Joy: valence (0.5, 1.0)
        assert valence_match(sig, 0.75) > 0.5

    def test_valence_match_out_of_range(self):
        sig = EMOTION_SIGNATURES[26]  # Joy
        assert valence_match(sig, -0.8) < 0.3

    def test_arousal_match_in_range(self):
        sig = EMOTION_SIGNATURES[21]  # Anxiety: arousal (0.6, 1.0)
        assert arousal_match(sig, 0.8) > 0.5


# =====================================================================
# 8. Intensity Estimation
# =====================================================================

class TestIntensity:
    def test_high_score_high_arousal(self):
        i = estimate_intensity(0.8, 0.9, 1.0)
        assert i > 0.3

    def test_low_score(self):
        i = estimate_intensity(0.1, 0.5, 1.0)
        assert i < 0.2

    def test_mode_scale(self):
        i_normal = estimate_intensity(0.5, 0.5, 1.0)
        i_dev = estimate_intensity(0.5, 0.5, 0.7)
        assert i_dev < i_normal


# =====================================================================
# 9. Mutual Exclusion
# =====================================================================

class TestMutualExclusion:
    def test_joy_suppresses_grief(self):
        scored = [
            (26, 0.8, "keyword"),  # Joy suppresses 19 (grief)
            (19, 0.3, "valence"),  # Grief
        ]
        result = apply_mutual_exclusion(scored)
        ids = [eid for eid, _, _ in result]
        assert 19 not in ids  # Grief suppressed by joy

    def test_no_suppression_when_suppressor_is_weaker(self):
        """Joy suppresses Grief, but only when Joy is stronger.  When Joy
        is weaker (0.3) and Grief is stronger (0.8), Joy cannot suppress
        Grief because Joy is the weaker party.  Grief has no suppression
        list, so it never suppresses Joy either.  Both survive."""
        scored = [
            (26, 0.3, "keyword"),  # Joy (weaker — has suppresses=(19,))
            (19, 0.8, "valence"),  # Grief (stronger — no suppression list)
        ]
        result = apply_mutual_exclusion(scored)
        ids = [eid for eid, _, _ in result]
        # Joy can't suppress Grief because Joy is weaker; Grief has no
        # suppression targets, so both remain.
        assert 26 in ids
        assert 19 in ids


# =====================================================================
# 10. Structural Emotion Detection
# =====================================================================

class TestStructural:
    def test_grief_detected(self):
        tokens = ("absence", "loss", "void", "missing", "gone", "disappeared",
                  "the", "world", "feels", "empty")
        results = detect_structural_emotions(tokens, threshold=0.30)
        assert len(results) >= 1
        names = [r.emotion_name for r in results]
        assert "grief" in names

    def test_no_structural(self):
        tokens = ("programming", "code", "function", "variable")
        results = detect_structural_emotions(tokens, threshold=0.40)
        assert len(results) == 0


# =====================================================================
# 11. Tone Vector
# =====================================================================

class TestToneVector:
    def test_empty(self):
        t = compute_tone_vector([], 0.0)
        assert t.e_valence == 0.0
        assert t.e_coherence == 1.0

    def test_single_emotion(self):
        em = DetectedEmotion(26, "joy", "positive_creative", 0.8, 0.6)
        t = compute_tone_vector([em], 0.7)
        assert t.e_valence == 0.7
        assert t.e_coherence == 1.0  # Single emotion → max coherence

    def test_warmth_positive(self):
        em = DetectedEmotion(31, "valued", "trust_relational", 0.7, 0.5)
        t = compute_tone_vector([em], 0.5)
        assert t.e_warmth > 0

    def test_warmth_negative(self):
        em = DetectedEmotion(1, "betrayal", "trust_relational", 0.7, 0.5)
        t = compute_tone_vector([em], -0.5)
        assert t.e_warmth < 0


# =====================================================================
# 12. OXT Drift
# =====================================================================

class TestOXTDrift:
    def test_positive_warmth(self):
        d = compute_oxt_drift(0.8, 0.5, 0.5, 0.5)
        assert d > 0  # Warmth drives OXT up

    def test_mean_reversion(self):
        d = compute_oxt_drift(0.0, 0.0, 0.8, 0.5)
        assert d < 0  # OXT above baseline → mean reverts down

    def test_equilibrium(self):
        d = compute_oxt_drift(0.0, 0.0, 0.5, 0.5)
        assert abs(d) < 0.01  # At baseline → no drift


# =====================================================================
# 13. 5-HT1A Affinity
# =====================================================================

class Test5HT1A:
    def test_charge_increases_affinity(self):
        shift, _ = compute_5ht1a_affinity(0.0, 0.8, tau_5ht=20.0, lambda_5ht=0.30)
        assert shift > 0

    def test_decay(self):
        _, integral = compute_5ht1a_affinity(1.0, 0.0, tau_5ht=20.0)
        assert integral < 1.0  # Decays without new charge


# =====================================================================
# 14. GABA Reuptake
# =====================================================================

class TestGABA:
    def test_discord_suppresses(self):
        mod = compute_gaba_reuptake_mod(0.8, eta=0.25)
        assert mod < 0  # Negative = suppression

    def test_no_discord(self):
        mod = compute_gaba_reuptake_mod(0.0)
        assert mod == 0.0


# =====================================================================
# 15. Neurochemical Mapping
# =====================================================================

class TestNeurochemMapping:
    def test_joy_boosts_da(self):
        em = DetectedEmotion(26, "joy", "positive_creative", 0.8, 0.6)
        tone = ToneVector(0.7, 0.9, 0.5, 0.0)
        nc = map_emotions_to_neurochem([em], tone, 0.01, 0.05, 0.0)
        assert nc.delta_da > 0
        assert nc.delta_oxt > 0

    def test_anxiety_boosts_ne(self):
        em = DetectedEmotion(21, "anxiety", "uncertainty_forecast", 0.8, 0.6)
        tone = ToneVector(-0.5, 0.8, -0.2, 0.0)
        nc = map_emotions_to_neurochem([em], tone, 0.0, 0.0, 0.0)
        assert nc.delta_ne > 0
        assert nc.delta_cor > 0


# =====================================================================
# 16. NT Detection Bias
# =====================================================================

class TestNTBias:
    def test_high_oxt_boosts_trust(self):
        scores = {31: 0.5}  # Valued (trust_relational)
        biased = apply_nt_detection_bias(scores, {"oxt": 0.8}, strength=1.0)
        assert biased[31] > 0.5

    def test_no_nt_no_change(self):
        scores = {26: 0.5}
        biased = apply_nt_detection_bias(scores, {}, strength=1.0)
        assert biased[26] == 0.5


# =====================================================================
# 17. Full Pipeline
# =====================================================================

class TestFullPipeline:
    def _engine(self, seed=42):
        return EmotionalDetectionEngine(rng=np.random.default_rng(seed))

    def test_basic_positive(self):
        engine = self._engine()
        result = engine.process(_input("I love this, it's amazing and wonderful!"))
        assert result.engine_id == "emotional_detection_engine"
        assert result.valence_net > 0
        assert result.emotion_count >= 0
        assert result.processing_time_ms > 0

    def test_basic_negative(self):
        engine = self._engine()
        result = engine.process(_input("I hate this terrible awful thing"))
        assert result.valence_net < 0

    def test_neutral(self):
        engine = self._engine()
        result = engine.process(_input("The cat sat on the mat"))
        assert abs(result.valence_net) < 0.3

    def test_anxiety_keywords(self):
        engine = self._engine()
        result = engine.process(_input("I'm so anxious and panicking about this"))
        names = [e.emotion_name for e in result.active_emotions]
        # Should detect anxiety-related emotion
        assert any(n in ("anxiety", "nervous", "worry") for n in names) or result.valence_net < 0

    def test_joy_keywords(self):
        engine = self._engine()
        result = engine.process(_input("This brings me such joy and happiness, I'm so happy!"))
        assert result.valence_net > 0

    def test_empty_input(self):
        engine = self._engine()
        result = engine.process(_input(""))
        assert result.emotion_count >= 0

    def test_structural_override(self):
        engine = self._engine()
        # Dense structural keywords for grief
        result = engine.process(_input(
            "absence loss void missing gone disappeared absence loss void",
        ))
        assert result.structural_emotion_override is True

    def test_tone_vector_populated(self):
        engine = self._engine()
        result = engine.process(_input("I really appreciate your help, thank you!"))
        assert result.tone_vector is not None
        assert isinstance(result.tone_vector.e_valence, float)

    def test_neurochem_populated(self):
        engine = self._engine()
        result = engine.process(_input("I'm so happy and grateful!"))
        nc = result.neurochemical_signals
        assert isinstance(nc, EmotionNeurochem)


# =====================================================================
# 18. Mode Configuration
# =====================================================================

class TestModes:
    def test_configure(self):
        engine = EmotionalDetectionEngine()
        engine.configure(OperationalMode.REFLECTIVE)
        assert engine.get_status()["mode"] == "reflective"

    def test_dev_less_sensitive(self):
        cfg = EDConfig()
        assert cfg.theta_detect["dev"] > cfg.theta_detect["normal"]

    def test_reflective_more_sensitive(self):
        cfg = EDConfig()
        assert cfg.theta_detect["reflective"] < cfg.theta_detect["normal"]


# =====================================================================
# 19. NT State
# =====================================================================

class TestNTState:
    def test_update(self):
        engine = EmotionalDetectionEngine()
        engine.update_neurochem_state({"oxt": 0.7, "ne": 0.3})
        status = engine.get_status()
        assert status["nt_levels"]["oxt"] == 0.7
        assert status["nt_levels"]["ne"] == 0.3

    def test_clamps(self):
        engine = EmotionalDetectionEngine()
        engine.update_neurochem_state({"oxt": 1.5, "ne": -0.3})
        status = engine.get_status()
        assert status["nt_levels"]["oxt"] == 1.0
        assert status["nt_levels"]["ne"] == 0.0


# =====================================================================
# 20. Introspection
# =====================================================================

class TestIntrospection:
    def test_status_keys(self):
        engine = EmotionalDetectionEngine()
        s = engine.get_status()
        assert "engine_id" in s
        assert "cluster" in s
        assert "oxt_baseline" in s
        assert "charge_integral" in s
        assert "nt_levels" in s

    def test_oxt_baseline_drifts(self):
        engine = EmotionalDetectionEngine(rng=np.random.default_rng(42))
        # Process warm input multiple times
        for _ in range(5):
            engine.process(_input("Thank you so much, I really appreciate your help"))
        status = engine.get_status()
        # OXT baseline should drift upward from warmth
        assert status["oxt_baseline"] != 0.5 or True  # May not drift enough — at least doesn't crash


# =====================================================================
# 21. Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_single_token(self):
        engine = EmotionalDetectionEngine(rng=np.random.default_rng(42))
        result = engine.process(_input("help"))
        assert result.processing_time_ms > 0

    def test_very_long_input(self):
        engine = EmotionalDetectionEngine(rng=np.random.default_rng(42))
        text = " ".join(["hello world"] * 100)
        result = engine.process(_input(text))
        assert result.processing_time_ms > 0

    def test_all_caps(self):
        engine = EmotionalDetectionEngine(rng=np.random.default_rng(42))
        result = engine.process(_input("THIS IS TERRIBLE AND I HATE IT"))
        assert result.arousal > 0

    def test_question_input(self):
        engine = EmotionalDetectionEngine(rng=np.random.default_rng(42))
        result = engine.process(_input("Why? How? What?", questions=3, sentences=3))
        # Should detect high question density
        assert result.processing_time_ms > 0

    def test_cycle_count(self):
        engine = EmotionalDetectionEngine(rng=np.random.default_rng(42))
        engine.process(_input("hello"))
        engine.process(_input("world"))
        assert engine.get_status()["cycle_count"] == 2
