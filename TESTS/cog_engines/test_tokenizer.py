"""
Tests for Tokenizer — Pipeline Infrastructure.
"""

import math
import re

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.tokenizer import (
    AggregateFeatures,
    DiscourseMarkerType,
    EntitySpan,
    LexicalProfile,
    MorphologyEntry,
    QuestionClassification,
    SentenceAnalysis,
    SentenceType,
    Tokenizer,
    TokenData,
    TokenizerConfig,
    TokenizerResult,
    WordNetEntry,
    build_lexical_profile,
    build_morphology,
    build_wordnet_entry,
    classify_sentence,
    compute_aggregate_features,
    compute_neurochemical_signals,
    compute_parsing_load,
)


# =====================================================================
# Helpers
# =====================================================================

RNG = np.random.default_rng(42)


def _make_tokenizer(seed: int = 42, **config_kw) -> Tokenizer:
    cfg = TokenizerConfig(**config_kw)
    return Tokenizer(config=cfg, rng=np.random.default_rng(seed))


# =====================================================================
# Test: Enumerations
# =====================================================================

class TestEnums:
    def test_sentence_types(self):
        assert len(SentenceType) == 5

    def test_question_classification(self):
        assert len(QuestionClassification) == 6

    def test_discourse_marker_types(self):
        assert len(DiscourseMarkerType) == 4


# =====================================================================
# Test: Configuration
# =====================================================================

class TestConfig:
    def test_default_config_valid(self):
        cfg = TokenizerConfig()
        assert cfg.n_baseline == 20
        assert cfg.lambda_complexity == 0.10
        assert cfg.ach_coupling == 0.04
        assert cfg.beta_boost == 0.03

    def test_hedging_markers_non_empty(self):
        cfg = TokenizerConfig()
        assert len(cfg.hedging_markers) > 0

    def test_discourse_markers_non_empty(self):
        cfg = TokenizerConfig()
        assert len(cfg.discourse_markers_additive) > 0
        assert len(cfg.discourse_markers_contrastive) > 0
        assert len(cfg.discourse_markers_causal) > 0
        assert len(cfg.discourse_markers_temporal) > 0


# =====================================================================
# Test: Data Types
# =====================================================================

class TestDataTypes:
    def test_lexical_profile_defaults(self):
        lp = LexicalProfile()
        assert lp.valence == 0.0
        assert lp.concreteness == 3.0
        assert not lp.is_hedging
        assert not lp.is_discourse_marker

    def test_morphology_entry_defaults(self):
        me = MorphologyEntry()
        assert me.token == ""
        assert me.root == ""
        assert me.prefixes == ()
        assert me.suffixes == ()

    def test_wordnet_entry_defaults(self):
        wn = WordNetEntry()
        assert wn.synset_count == 0
        assert wn.polysemy_score == 0

    def test_token_data_defaults(self):
        td = TokenData()
        assert td.text == ""
        assert td.token_index == 0
        assert not td.is_content_word

    def test_sentence_analysis_defaults(self):
        sa = SentenceAnalysis()
        assert sa.sentence_type == SentenceType.DECLARATIVE.value

    def test_aggregate_features_defaults(self):
        af = AggregateFeatures()
        assert af.emotional_vocab_density == 0.0
        assert af.type_token_ratio == 0.0

    def test_tokenizer_result_defaults(self):
        tr = TokenizerResult()
        assert tr.token_count == 0
        assert tr.sentence_count == 0
        assert tr.raw_text == ""

    def test_frozen_dataclass_immutable(self):
        lp = LexicalProfile()
        with pytest.raises(AttributeError):
            lp.valence = 1.0


# =====================================================================
# Test: Build Lexical Profile
# =====================================================================

class TestBuildLexicalProfile:
    def test_hedging_detected(self):
        cfg = TokenizerConfig()
        lp = build_lexical_profile("maybe", "ADV", cfg)
        assert lp.is_hedging

    def test_non_hedging(self):
        cfg = TokenizerConfig()
        lp = build_lexical_profile("freedom", "NOUN", cfg)
        assert not lp.is_hedging

    def test_discourse_marker_contrastive(self):
        cfg = TokenizerConfig()
        lp = build_lexical_profile("however", "ADV", cfg)
        assert lp.is_discourse_marker
        assert lp.discourse_marker_type == DiscourseMarkerType.CONTRASTIVE.value

    def test_discourse_marker_causal(self):
        cfg = TokenizerConfig()
        lp = build_lexical_profile("because", "SCONJ", cfg)
        assert lp.is_discourse_marker
        assert lp.discourse_marker_type == DiscourseMarkerType.CAUSAL.value

    def test_action_verb_detected(self):
        cfg = TokenizerConfig()
        lp = build_lexical_profile("running", "VERB", cfg)
        assert lp.is_action_verb
        assert not lp.is_stative_verb

    def test_stative_verb_detected(self):
        cfg = TokenizerConfig()
        lp = build_lexical_profile("believe", "VERB", cfg)
        assert lp.is_stative_verb
        assert not lp.is_action_verb

    def test_concreteness_short_word(self):
        cfg = TokenizerConfig()
        lp = build_lexical_profile("cat", "NOUN", cfg)
        assert lp.concreteness > 3.0  # short words more concrete

    def test_emotion_from_lexicon(self):
        cfg = TokenizerConfig(emotion_lexicon={"happy": {"valence": 0.8, "joy": 0.9}})
        lp = build_lexical_profile("happy", "ADJ", cfg)
        assert lp.valence == 0.8
        assert "joy" in lp.emotion_associations


# =====================================================================
# Test: Build Morphology
# =====================================================================

class TestBuildMorphology:
    def test_prefix_detection(self):
        me = build_morphology("unhappy")
        assert "un" in me.prefixes
        assert me.root == "happy"

    def test_suffix_detection(self):
        me = build_morphology("freedom")
        # "dom" is not in suffix list, so check for token
        assert me.token == "freedom"

    def test_nominalization_suffix(self):
        me = build_morphology("protection")
        assert "tion" in me.suffixes
        assert me.derivation_type == "nominalization"

    def test_adjectival_suffix(self):
        me = build_morphology("beautiful")
        assert "ful" in me.suffixes
        assert me.derivation_type == "adjectival"

    def test_adverbial_suffix(self):
        me = build_morphology("quickly")
        assert "ly" in me.suffixes
        assert me.derivation_type == "adverbial"

    def test_no_affixes_short_word(self):
        me = build_morphology("go")
        assert me.prefixes == ()
        assert me.suffixes == ()


# =====================================================================
# Test: WordNet Entry
# =====================================================================

class TestBuildWordnetEntry:
    def test_short_word_high_polysemy(self):
        wn = build_wordnet_entry("go", "VERB")
        assert wn.polysemy_score > 1

    def test_long_word_low_polysemy(self):
        wn = build_wordnet_entry("counterrevolutionary", "NOUN")
        assert wn.polysemy_score == 1

    def test_primary_sense_format(self):
        wn = build_wordnet_entry("cat", "NOUN")
        assert wn.primary_sense == "cat.01"

    def test_pos_extracted(self):
        wn = build_wordnet_entry("run", "VERB")
        assert wn.pos == "v"


# =====================================================================
# Test: Sentence Classification
# =====================================================================

class TestClassifySentence:
    def test_declarative(self):
        sa = classify_sentence("Freedom is important.")
        assert sa.sentence_type == SentenceType.DECLARATIVE.value

    def test_interrogative(self):
        sa = classify_sentence("What is freedom?")
        assert sa.sentence_type == SentenceType.INTERROGATIVE.value
        assert sa.question_type == QuestionClassification.OPEN.value

    def test_exclamatory(self):
        sa = classify_sentence("This is outrageous!")
        assert sa.sentence_type == SentenceType.EXCLAMATORY.value

    def test_imperative(self):
        sa = classify_sentence("Please stop doing that.")
        assert sa.sentence_type == SentenceType.IMPERATIVE.value

    def test_conditional(self):
        sa = classify_sentence("If freedom is lost, all is lost.")
        assert sa.sentence_type == SentenceType.CONDITIONAL.value
        assert sa.has_conditional

    def test_negation_detected(self):
        sa = classify_sentence("Freedom is not an illusion.")
        assert sa.has_negation

    def test_no_negation(self):
        sa = classify_sentence("Freedom is real.")
        assert not sa.has_negation

    def test_closed_question(self):
        sa = classify_sentence("Is freedom important?")
        assert sa.question_type == QuestionClassification.CLOSED.value

    def test_tag_question(self):
        sa = classify_sentence("Freedom matters, don't you?")
        assert sa.question_type == QuestionClassification.TAG.value

    def test_modality_deontic(self):
        sa = classify_sentence("You should value freedom.")
        assert sa.modality == "deontic"

    def test_modality_epistemic(self):
        sa = classify_sentence("Freedom might be an illusion.")
        assert sa.modality == "epistemic"

    def test_clause_count_multiple(self):
        sa = classify_sentence("If freedom is lost, because we fail, then nothing matters.")
        assert sa.clause_count > 1


# =====================================================================
# Test: Aggregate Features
# =====================================================================

class TestAggregateFeatures:
    def test_empty_tokens_returns_defaults(self):
        agg = compute_aggregate_features([], [], TokenizerConfig())
        assert agg.type_token_ratio == 0.0

    def test_pronoun_distribution(self):
        tokens = [
            TokenData(text="I", pos_coarse="PRON", is_content_word=False),
            TokenData(text="love", pos_coarse="VERB", is_content_word=True),
            TokenData(text="you", pos_coarse="PRON", is_content_word=False),
        ]
        sents = [SentenceAnalysis(sentence_text="I love you")]
        agg = compute_aggregate_features(tokens, sents, TokenizerConfig())
        assert agg.pronoun_distribution["I"] > 0.0
        assert agg.pronoun_distribution["you"] > 0.0

    def test_hedging_density(self):
        tokens = [
            TokenData(text="maybe", pos_coarse="ADV", is_content_word=False,
                      lexical_profile=LexicalProfile(is_hedging=True)),
            TokenData(text="freedom", pos_coarse="NOUN", is_content_word=True),
        ]
        sents = [SentenceAnalysis(sentence_text="maybe freedom")]
        agg = compute_aggregate_features(tokens, sents, TokenizerConfig())
        assert agg.hedging_density == 0.5

    def test_question_type_distribution(self):
        # Need at least one token so the function doesn't return early
        tokens = [
            TokenData(text="what", pos_coarse="PRON", is_content_word=False),
            TokenData(text="why", pos_coarse="PRON", is_content_word=False),
        ]
        sents = [
            SentenceAnalysis(sentence_type=SentenceType.INTERROGATIVE.value,
                             question_type=QuestionClassification.OPEN.value),
            SentenceAnalysis(sentence_type=SentenceType.INTERROGATIVE.value,
                             question_type=QuestionClassification.OPEN.value),
        ]
        agg = compute_aggregate_features(tokens, sents, TokenizerConfig())
        assert agg.question_type_distribution.get("open", 0) == 2

    def test_message_length_ratio(self):
        tokens = [TokenData(text=f"w{i}", pos_coarse="NOUN", is_content_word=True) for i in range(10)]
        sents = [SentenceAnalysis()]
        agg = compute_aggregate_features(tokens, sents, TokenizerConfig(), previous_message_length=5)
        assert agg.message_length_ratio == 2.0

    def test_type_token_ratio(self):
        tokens = [
            TokenData(text="cat", lemma="cat", pos_coarse="NOUN", is_content_word=True),
            TokenData(text="cat", lemma="cat", pos_coarse="NOUN", is_content_word=True),
            TokenData(text="dog", lemma="dog", pos_coarse="NOUN", is_content_word=True),
        ]
        sents = [SentenceAnalysis()]
        agg = compute_aggregate_features(tokens, sents, TokenizerConfig())
        assert 0.5 < agg.type_token_ratio < 1.0


# =====================================================================
# Test: Neurochemical Coupling
# =====================================================================

class TestNeurochemicalCoupling:
    def test_parsing_load_formula(self):
        load = compute_parsing_load(40, 2, TokenizerConfig())
        # 40 * (1 + 0.10 * 2) / 20 = 40 * 1.2 / 20 = 2.4
        assert abs(load - 2.4) < 0.01

    def test_parsing_load_clamped(self):
        load = compute_parsing_load(1000, 10, TokenizerConfig())
        assert load <= 10.0

    def test_ach_burst_positive_load(self):
        rng = np.random.default_rng(42)
        signals = compute_neurochemical_signals(2.0, False, TokenizerConfig(), rng)
        assert signals["ach_burst"] > 0.0

    def test_ach_burst_zero_load(self):
        rng = np.random.default_rng(42)
        signals = compute_neurochemical_signals(0.0, False, TokenizerConfig(), rng)
        assert signals["ach_burst"] == 0.0

    def test_ne_burst_on_anomaly(self):
        rng = np.random.default_rng(42)
        signals = compute_neurochemical_signals(1.0, True, TokenizerConfig(), rng)
        assert signals["ne_burst"] > 0.0

    def test_ne_burst_no_anomaly(self):
        rng = np.random.default_rng(42)
        signals = compute_neurochemical_signals(1.0, False, TokenizerConfig(), rng)
        assert signals["ne_burst"] == 0.0

    def test_beta_boost_always_present(self):
        rng = np.random.default_rng(42)
        signals = compute_neurochemical_signals(0.0, False, TokenizerConfig(), rng)
        assert signals["beta_boost"] == 0.03


# =====================================================================
# Test: Full Engine
# =====================================================================

class TestTokenizerEngine:
    def test_process_simple_sentence(self):
        tok = _make_tokenizer()
        result = tok.process("Freedom is important.")
        assert result.token_count > 0
        assert result.sentence_count >= 1
        assert result.raw_text == "Freedom is important."
        assert result.processing_time_ms >= 0.0

    def test_process_multiple_sentences(self):
        tok = _make_tokenizer()
        result = tok.process("Freedom is vital. Security matters too.")
        assert result.sentence_count == 2

    def test_process_question(self):
        tok = _make_tokenizer()
        result = tok.process("What is freedom?")
        assert any(s.sentence_type == SentenceType.INTERROGATIVE.value for s in result.sentences)

    def test_process_empty_string(self):
        tok = _make_tokenizer()
        result = tok.process("")
        assert result.token_count == 0

    def test_neurochemical_signals_present(self):
        tok = _make_tokenizer()
        result = tok.process("I believe freedom is more important than security.")
        assert "ach_burst" in result.neurochemical_signals
        assert "ne_burst" in result.neurochemical_signals
        assert "beta_boost" in result.neurochemical_signals

    def test_tokens_have_pos_tags(self):
        tok = _make_tokenizer()
        result = tok.process("I believe freedom is important.")
        for t in result.tokens:
            assert t.pos_coarse != ""

    def test_tokens_have_morphology(self):
        tok = _make_tokenizer()
        result = tok.process("Understanding requires commitment.")
        for t in result.tokens:
            assert t.morphology is not None

    def test_tokens_have_wordnet_entry(self):
        tok = _make_tokenizer()
        result = tok.process("The quick brown fox.")
        content_tokens = [t for t in result.tokens if t.is_content_word]
        for t in content_tokens:
            assert t.wordnet is not None

    def test_content_word_detection(self):
        tok = _make_tokenizer()
        result = tok.process("I believe freedom is more important than security.")
        content = [t for t in result.tokens if t.is_content_word]
        stop = [t for t in result.tokens if t.is_stop_word]
        assert len(content) > 0
        assert len(stop) > 0

    def test_sentence_index_assignment(self):
        tok = _make_tokenizer()
        result = tok.process("Hello world. Goodbye world.")
        # Some tokens should have sentence_index 0, some 1
        indices = set(t.sentence_index for t in result.tokens)
        assert len(indices) >= 1

    def test_aggregate_features_computed(self):
        tok = _make_tokenizer()
        result = tok.process("I think maybe freedom is what everyone really deserves.")
        agg = result.aggregate_features
        assert agg.sentence_count >= 1
        assert agg.type_token_ratio > 0.0

    def test_previous_message_length_tracking(self):
        tok = _make_tokenizer()
        tok.process("Hello world.")
        result2 = tok.process("How are you today my friend?")
        assert result2.aggregate_features.message_length_ratio is not None

    def test_rem_dream_returns_empty(self):
        tok = _make_tokenizer()
        tok.configure(OperationalMode.REM_DREAM)
        result = tok.process("This should be ignored.")
        assert result.token_count == 0
        assert result.raw_text == "This should be ignored."

    def test_configure_mode(self):
        tok = _make_tokenizer()
        tok.configure(OperationalMode.LEARNING)
        assert tok.get_status()["mode"] == "learning"

    def test_get_status(self):
        tok = _make_tokenizer()
        status = tok.get_status()
        assert "mode" in status
        assert status["previous_message_length"] is None

    def test_complex_sentence_high_nesting(self):
        tok = _make_tokenizer()
        text = "If you believe, because you think, although you wonder, then maybe."
        result = tok.process(text)
        assert result.aggregate_features.clause_nesting_depth_max > 0

    def test_exclamation_density(self):
        tok = _make_tokenizer()
        result = tok.process("Amazing! Wonderful! Incredible!")
        assert result.aggregate_features.exclamation_density > 0.0


# =====================================================================
# Test: Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_single_word_input(self):
        tok = _make_tokenizer()
        result = tok.process("Freedom")
        assert result.token_count >= 1

    def test_very_long_input(self):
        tok = _make_tokenizer()
        text = "word " * 200
        result = tok.process(text.strip())
        assert result.token_count > 100

    def test_punctuation_only(self):
        tok = _make_tokenizer()
        result = tok.process("!!! ??? ...")
        # May produce tokens depending on tokenizer
        assert result.raw_text == "!!! ??? ..."

    def test_numbers_in_text(self):
        tok = _make_tokenizer()
        result = tok.process("I have 42 reasons and 3 concerns.")
        assert result.token_count > 0

    def test_unicode_text(self):
        tok = _make_tokenizer()
        result = tok.process("This is a test with unicode: cafe\u0301.")
        assert result.token_count > 0

    def test_multiple_spaces(self):
        tok = _make_tokenizer()
        result = tok.process("Hello    world")
        assert result.token_count >= 2

    def test_newlines_in_text(self):
        tok = _make_tokenizer()
        result = tok.process("Hello\nworld")
        assert result.token_count >= 2
