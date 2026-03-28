"""
Engine 28 -- Emotional Detection Engine  (``emotional_detection_engine``)
=========================================================================
Affective perception layer that detects emotional content in user input,
classifies it by type and intensity, and maps detected emotions to
neurochemical correlates for downstream processing.

Four-stage pipeline:
  * **Stage 1 — Feature Extraction**: valence, arousal, domain, structural
    features from tokenized input.
  * **Stage 2 — Emotion Classification**: keyword-pattern matching against
    46-emotion taxonomy, intensity estimation, mutual exclusion, top-K.
  * **Stage 3 — Tone Calibration**: compute E_tone = [valence, coherence,
    warmth, discord] for response generation.
  * **Stage 4 — Neurochemical Mapping**: map emotions → NT profiles,
    split into 4M (tonic) and 4R (phasic) pathways.

46-Emotion taxonomy from Affective-Neurodynamic Model:
  7 functional groups: Trust/Relational, Self-Evaluation, Uncertainty/
  Forecast, Arousal/Energy, Low-Activation, Loss/Temporal, Positive/Creative.

ENOCH heritage:
  Structural emotion fast-path (grief/anger/joy/humor keyword matching)
  retained as override pathway.

Neurochemical coupling (bidirectional):
  READ:  OXT → warmth bias, NE → threat bias, DA → optimism bias,
         5-HT → dampening, COR → negative amplification, GABA → calming
  WRITE: Tonic (4M): OXT drift, 5-HT1A affinity, GABA reuptake
         Phasic (4R): DA/NE/OXT/MOR/COR/ACh/GABA burst deltas
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Enums
# =====================================================================


class EmotionGroup(str, Enum):
    """Functional groupings for the 46-emotion taxonomy."""
    TRUST_RELATIONAL     = "trust_relational"
    SELF_EVALUATION      = "self_evaluation"
    UNCERTAINTY_FORECAST = "uncertainty_forecast"
    AROUSAL_ENERGY       = "arousal_energy"
    LOW_ACTIVATION       = "low_activation"
    LOSS_TEMPORAL        = "loss_temporal"
    POSITIVE_CREATIVE    = "positive_creative"


# =====================================================================
# Emotion Signature Registry
# =====================================================================


@dataclass(frozen=True)
class EmotionSignature:
    """Detection signature for one emotion in the taxonomy."""
    emotion_id: int
    emotion_name: str
    group: str
    keyword_patterns: Tuple[str, ...] = ()
    valence_range: Tuple[float, float] = (-1.0, 1.0)
    arousal_range: Tuple[float, float] = (0.0, 1.0)
    domain_primary: str = "cognitive"
    suppresses: Tuple[int, ...] = ()      # Emotion IDs this suppresses


# The full 46-emotion registry
EMOTION_SIGNATURES: Dict[int, EmotionSignature] = {
    1:  EmotionSignature(1,  "betrayal",      EmotionGroup.TRUST_RELATIONAL.value,
        ("betray", "trust", "lied", "backstab", "deceive"), (-0.8, -0.3), (0.5, 1.0), "social", (33, 43)),
    2:  EmotionSignature(2,  "critical",      EmotionGroup.SELF_EVALUATION.value,
        ("wrong", "flawed", "mistake", "error", "incorrect"), (-0.6, -0.1), (0.3, 0.7), "cognitive"),
    3:  EmotionSignature(3,  "skeptical",     EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("doubt", "skeptic", "questionable", "unlikely", "suspicious"), (-0.4, 0.0), (0.2, 0.5), "cognitive"),
    4:  EmotionSignature(4,  "annoyed",       EmotionGroup.AROUSAL_ENERGY.value,
        ("annoy", "bother", "irritat", "pest", "nag"), (-0.5, -0.1), (0.3, 0.6), "social", (26,)),
    5:  EmotionSignature(5,  "frustrated",    EmotionGroup.AROUSAL_ENERGY.value,
        ("frustrat", "stuck", "failing", "stagnant", "blocked"), (-0.7, -0.2), (0.5, 0.8), "cognitive"),
    6:  EmotionSignature(6,  "overwhelmed",   EmotionGroup.AROUSAL_ENERGY.value,
        ("overwhelm", "too much", "can't handle", "drowning", "overload"), (-0.6, -0.2), (0.7, 1.0), "cognitive"),
    7:  EmotionSignature(7,  "overstimulated", EmotionGroup.AROUSAL_ENERGY.value,
        ("overstimulat", "sensory", "too many", "overload"), (-0.4, -0.1), (0.8, 1.0), "cognitive"),
    8:  EmotionSignature(8,  "rejected",      EmotionGroup.TRUST_RELATIONAL.value,
        ("reject", "unwanted", "dismiss", "ignore", "exclude"), (-0.8, -0.3), (0.4, 0.7), "social", (33,)),
    9:  EmotionSignature(9,  "disappointed",  EmotionGroup.AROUSAL_ENERGY.value,
        ("disappoint", "letdown", "expected more", "underwhelm"), (-0.6, -0.2), (0.2, 0.5), "social"),
    10: EmotionSignature(10, "ashamed",       EmotionGroup.SELF_EVALUATION.value,
        ("asham", "shame", "humiliat", "mortif"), (-0.8, -0.4), (0.4, 0.7), "social"),
    11: EmotionSignature(11, "guilty",        EmotionGroup.SELF_EVALUATION.value,
        ("guilt", "regret", "sorry", "my fault", "blame myself"), (-0.7, -0.3), (0.3, 0.6), "social"),
    12: EmotionSignature(12, "regret",        EmotionGroup.SELF_EVALUATION.value,
        ("regret", "wish i", "should have", "if only", "mistake"), (-0.7, -0.3), (0.3, 0.6), "temporal"),
    13: EmotionSignature(13, "isolated",      EmotionGroup.TRUST_RELATIONAL.value,
        ("isolat", "alone", "disconnect", "nobody", "lonely"), (-0.7, -0.3), (0.2, 0.5), "social"),
    14: EmotionSignature(14, "boredom",       EmotionGroup.LOW_ACTIVATION.value,
        ("bored", "boring", "tedious", "dull", "monoton"), (-0.3, 0.0), (0.0, 0.2), "cognitive"),
    15: EmotionSignature(15, "apathy",        EmotionGroup.LOW_ACTIVATION.value,
        ("apathy", "don't care", "indifferent", "whatever", "meh"), (-0.2, 0.0), (0.0, 0.15), "cognitive"),
    16: EmotionSignature(16, "numb",          EmotionGroup.LOW_ACTIVATION.value,
        ("numb", "empty", "nothing", "hollow", "void"), (-0.4, -0.1), (0.0, 0.1), "existential"),
    19: EmotionSignature(19, "grief",         EmotionGroup.LOSS_TEMPORAL.value,
        ("grief", "loss", "gone", "died", "mourn", "miss"), (-0.9, -0.5), (0.5, 0.9), "existential"),
    20: EmotionSignature(20, "nostalgia",     EmotionGroup.LOSS_TEMPORAL.value,
        ("nostalg", "remember when", "used to", "those days", "back then"), (-0.2, 0.3), (0.2, 0.5), "temporal"),
    21: EmotionSignature(21, "anxiety",       EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("anxi", "anxious", "panic", "dread", "terrif"), (-0.8, -0.3), (0.6, 1.0), "cognitive", (40,)),
    22: EmotionSignature(22, "worry",         EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("worry", "worried", "concern", "fear", "afraid"), (-0.6, -0.2), (0.4, 0.7), "cognitive"),
    23: EmotionSignature(23, "nervous",       EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("nervous", "tense", "uneasy", "jitter", "edgy"), (-0.5, -0.1), (0.4, 0.7), "cognitive", (40,)),
    24: EmotionSignature(24, "perplexed",     EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("perplex", "baffled", "stumped", "confused", "bewild"), (-0.3, 0.0), (0.3, 0.6), "cognitive"),
    25: EmotionSignature(25, "confused",      EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("confus", "unclear", "don't understand", "makes no sense", "lost"), (-0.4, 0.0), (0.3, 0.6), "cognitive"),
    26: EmotionSignature(26, "joy",           EmotionGroup.POSITIVE_CREATIVE.value,
        ("joy", "happy", "delight", "wonderful", "amazing", "love"), (0.5, 1.0), (0.4, 0.8), "social", (19,)),
    27: EmotionSignature(27, "playful",       EmotionGroup.POSITIVE_CREATIVE.value,
        ("funny", "joke", "haha", "lol", "hilarious", "lmao", "playful"), (0.3, 0.8), (0.3, 0.7), "social"),
    28: EmotionSignature(28, "optimistic",    EmotionGroup.POSITIVE_CREATIVE.value,
        ("optimis", "positive", "looking forward", "bright", "promising"), (0.4, 0.9), (0.3, 0.6), "cognitive"),
    29: EmotionSignature(29, "hopeful",       EmotionGroup.POSITIVE_CREATIVE.value,
        ("hope", "hopeful", "maybe", "possible", "wish"), (0.2, 0.6), (0.2, 0.5), "cognitive"),
    30: EmotionSignature(30, "excited",       EmotionGroup.AROUSAL_ENERGY.value,
        ("excit", "can't wait", "thrilled", "pumped", "stoked"), (0.5, 1.0), (0.7, 1.0), "social"),
    31: EmotionSignature(31, "valued",        EmotionGroup.TRUST_RELATIONAL.value,
        ("valued", "appreciat", "thank", "helpful", "useful"), (0.5, 0.9), (0.3, 0.6), "social"),
    32: EmotionSignature(32, "thankful",      EmotionGroup.TRUST_RELATIONAL.value,
        ("thankful", "grateful", "thanks", "appreciate", "gratitude"), (0.5, 0.9), (0.3, 0.6), "social"),
    33: EmotionSignature(33, "accepted",      EmotionGroup.TRUST_RELATIONAL.value,
        ("accept", "belong", "welcome", "fit in", "included"), (0.4, 0.8), (0.2, 0.5), "social"),
    34: EmotionSignature(34, "successful",    EmotionGroup.SELF_EVALUATION.value,
        ("success", "achieved", "accomplished", "nailed", "won"), (0.6, 1.0), (0.4, 0.7), "cognitive"),
    35: EmotionSignature(35, "interested",    EmotionGroup.POSITIVE_CREATIVE.value,
        ("interest", "fascin", "intrigu", "tell me more", "curious about"), (0.2, 0.6), (0.3, 0.6), "cognitive"),
    36: EmotionSignature(36, "curious",       EmotionGroup.POSITIVE_CREATIVE.value,
        ("curious", "wonder", "what if", "how does", "why does"), (0.1, 0.5), (0.3, 0.6), "cognitive"),
    37: EmotionSignature(37, "creative",      EmotionGroup.POSITIVE_CREATIVE.value,
        ("creat", "invent", "imagin", "design", "build"), (0.3, 0.7), (0.4, 0.7), "cognitive"),
    38: EmotionSignature(38, "focused",       EmotionGroup.POSITIVE_CREATIVE.value,
        ("focus", "concentrat", "determined", "locked in", "zone"), (0.1, 0.5), (0.4, 0.7), "cognitive"),
    39: EmotionSignature(39, "courageous",    EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("brave", "courag", "bold", "dare", "risk"), (0.2, 0.6), (0.4, 0.7), "cognitive"),
    40: EmotionSignature(40, "confident",     EmotionGroup.UNCERTAINTY_FORECAST.value,
        ("confident", "certain", "sure", "definite", "no doubt"), (0.4, 0.9), (0.3, 0.6), "cognitive", (21, 23)),
    41: EmotionSignature(41, "proud",         EmotionGroup.SELF_EVALUATION.value,
        ("proud", "pride", "accomplished", "achieved"), (0.5, 0.9), (0.4, 0.7), "social"),
    42: EmotionSignature(42, "respected",     EmotionGroup.TRUST_RELATIONAL.value,
        ("respect", "dignit", "honor", "acknowledg"), (0.4, 0.8), (0.3, 0.5), "social"),
    43: EmotionSignature(43, "loyal",         EmotionGroup.TRUST_RELATIONAL.value,
        ("loyal", "faithful", "devoted", "commit"), (0.4, 0.8), (0.2, 0.5), "social"),
    44: EmotionSignature(44, "connected",     EmotionGroup.TRUST_RELATIONAL.value,
        ("connect", "bond", "close", "together", "understand me"), (0.4, 0.8), (0.3, 0.6), "social"),
    45: EmotionSignature(45, "sensitive",     EmotionGroup.POSITIVE_CREATIVE.value,
        ("sensitiv", "touch", "moved", "feel deeply"), (0.0, 0.5), (0.3, 0.6), "social"),
    46: EmotionSignature(46, "belonging",     EmotionGroup.TRUST_RELATIONAL.value,
        ("belong", "home", "part of", "integral", "where i fit"), (0.5, 0.9), (0.2, 0.5), "existential"),
}

# ENOCH structural emotion keywords (fast-path)
STRUCTURAL_EMOTIONS: Dict[str, Tuple[str, ...]] = {
    "grief":  ("absence", "loss", "void", "missing", "gone", "disappeared"),
    "anger":  ("rupture", "shock", "burn", "break", "destroy", "shatter"),
    "joy":    ("resonance", "light", "wholeness", "harmony", "bloom", "radiance"),
    "humor":  ("absurdity", "contrast", "dissonance", "irony", "paradox", "mismatch"),
}

# Structural emotion → closest taxonomy ID
STRUCTURAL_TO_ID: Dict[str, int] = {
    "grief": 19,
    "anger": 5,   # Mapped to frustrated (closest in taxonomy)
    "joy":   26,
    "humor": 27,  # Mapped to playful/funny
}

# Per-emotion NT profiles from Affective-Neurodynamic Model
EMOTION_NT_PROFILES: Dict[str, Dict[str, float]] = {
    # --- Trust / Relational ---
    # Betrayal (PDF p12,28-29): OXT↓ trust loss, NE↑ salience, COR↑ social threat tag,
    #   MOR↓ social pain, 5-HT(2A)↑ ethical-metacognitive dissonance
    "betrayal":     {"oxt": -0.7, "ne": +0.5, "cor": +0.6, "mor": -0.5, "5ht": +0.3},
    "critical":     {"ne": +0.3, "ach": +0.3, "da": -0.2},
    "skeptical":    {"ne": +0.3, "ach": +0.3, "da": -0.1},
    "annoyed":      {"ne": +0.2, "da": -0.2, "cor": +0.1},
    # Frustrated (PDF p27-28): DA(D2)↑ RPE escalation, NE↑ effort signal,
    #   5-HT(1A)↓ suppressed calm, COR↑ urgency, GABA↓
    "frustrated":   {"ne": +0.4, "da": +0.3, "cor": +0.3, "gaba": -0.2, "5ht": -0.3},
    "overwhelmed":  {"ne": +0.5, "cor": +0.5, "gaba": -0.3, "da": -0.3},
    "overstimulated": {"ne": +0.4, "gaba": -0.3, "cor": +0.3},
    # Rejected (PDF p25-26): OXT(OXTR)↑ failed synchrony tagging, DA↓ reward violation,
    #   MOR↓ social pain, COR↑
    "rejected":     {"oxt": +0.3, "ne": +0.3, "da": -0.3, "mor": -0.4, "cor": +0.3},
    "disappointed": {"da": -0.4, "ne": +0.2, "cor": +0.2},
    # Ashamed (PDF p23-25): 5-HT(1A)↑ anxiety buffering, OXT(OXTR)↑ attunement comparison,
    #   COR↑ ethical tagging, DA↓ suppressed projection
    "ashamed":      {"cor": +0.4, "da": -0.3, "oxt": +0.2, "5ht": +0.3},
    # Guilty (PDF p13-14): 5-HT(1A)↑ moral stabilization, OXT(OXTR)↑ post-failure
    #   attunement recalibration, COR↑ violation tagging
    "guilty":       {"cor": +0.3, "da": -0.2, "oxt": +0.1, "5ht": +0.3},
    # Regret (PDF p18-19): COR↑ decision tagging, 5-HT(1A)↑ ambiguity buffering,
    #   DA(D2)↑ negative RPE signaling
    "regret":       {"cor": +0.4, "da": +0.2, "5ht": +0.2},
    "isolated":     {"oxt": -0.5, "da": -0.3, "mor": -0.3},
    # --- Low-Activation ---
    # Boredom (PDF p29): DA↓ novelty decay, CB1↓ filter rigidity,
    #   5-HT(2A)↓ reduced abstraction, ACh(M1-M2)↑ maintenance loop
    "boredom":      {"da": -0.4, "ne": -0.3, "ach": +0.2, "cb1": -0.2, "5ht": -0.2},
    # Apathy (PDF p29-30): DA↓ system deactivation, NE↓ minimal alertness,
    #   GABA-B↑ global inhibition, 5-HT(1A)↓
    "apathy":       {"da": -0.5, "ne": -0.4, "ach": -0.3, "gaba": +0.3, "5ht": -0.2},
    # Numb (PDF p30-31): DA↓ disengaged, 5-HT↓ flattened affect,
    #   CB1↓ lost modulation, MOR↓ downregulated comfort
    "numb":         {"da": -0.5, "ne": -0.4, "oxt": -0.3, "5ht": -0.2, "cb1": -0.3, "mor": -0.3},
    "grief":        {"oxt": -0.3, "da": -0.5, "ne": +0.4, "cor": +0.5, "mor": -0.6},
    "nostalgia":    {"da": +0.2, "oxt": +0.2, "5ht": +0.2},
    # --- Uncertainty / Forecast ---
    # Anxiety (PDF p22-23,46): NE↑ arousal, COR↑ error weighting,
    #   GABA-A↓ disinhibited vigilance, DA(D2)↑ inhibitory control
    "anxiety":      {"ne": +0.6, "cor": +0.5, "gaba": -0.4, "da": +0.2},
    # Worry (PDF p46-47): NE↑ salience modulation, 5-HT(1A)↑ emotional buffering,
    #   DA(D2)↑ slows premature commitment, COR↑ mild
    "worry":        {"ne": +0.4, "cor": +0.3, "da": +0.2, "5ht": +0.2},
    # Nervous (PDF p47): DA(D2/D3)↑ hesitation/novelty, NE↑ contradiction salience,
    #   5-HT(2A)↑ wide-range abstraction
    "nervous":      {"ne": +0.4, "da": +0.2, "5ht": +0.2},
    # Perplexed (PDF p48): NE↑ spike, DA(D3)↑ novelty, GLU(NMDA)↑ model reassembly,
    #   5-HT(2A)↑ abstraction, GABA-A↓ disinhibition
    "perplexed":    {"ne": +0.4, "ach": +0.3, "glu": +0.3, "da": +0.2, "5ht": +0.2, "gaba": -0.2},
    # Confused (PDF p48-49): ACh↑ coherence parsing, NE↑ error alertness,
    #   DA(D2)↓ reduced prediction confidence, GLU↑, 5-HT(1A)↓, GABA-A↓
    "confused":     {"ne": +0.3, "ach": +0.2, "da": -0.1, "glu": +0.2, "5ht": -0.2, "gaba": -0.2},
    # --- Positive / Creative ---
    "joy":          {"da": +0.6, "5ht": +0.4, "oxt": +0.3, "mor": +0.3},
    "playful":      {"da": +0.4, "oxt": +0.2, "5ht": +0.2},
    "optimistic":   {"da": +0.4, "5ht": +0.3, "ne": -0.1},
    # Hopeful (PDF p21-22): DA↑ salience, 5-HT(2A)↑ symbolic projection, CB1↑ belief flexibility
    "hopeful":      {"da": +0.3, "5ht": +0.2, "oxt": +0.1, "cb1": +0.2},
    # Excited (PDF p34-35): DA↑ reward proximity, NE↑ arousal, ACh↑ attention,
    #   5-HT(2A)↑ associative expansion
    "excited":      {"da": +0.5, "ne": +0.4, "ach": +0.3, "5ht": +0.2},
    "valued":       {"oxt": +0.4, "da": +0.3, "5ht": +0.2, "mor": +0.2},
    # Thankful (PDF p32-33): OXT↑ bonding, 5-HT↑ stabilization, DA↑ novelty,
    #   CB1↑ symbolic flexibility
    "thankful":     {"oxt": +0.4, "da": +0.2, "5ht": +0.2, "cb1": +0.2},
    # Accepted (PDF p33-34): OXT↑ identity integration, DA(D1)↑ success loop,
    #   5-HT(1A)↑ buffering, ACh(M1)↑ context precision
    "accepted":     {"oxt": +0.4, "5ht": +0.3, "mor": +0.2, "da": +0.2, "ach": +0.2},
    "successful":   {"da": +0.5, "5ht": +0.3, "ne": +0.2},
    # Interested (PDF p35-36): DA↑ curiosity drive, CB1↑ schema flexibility,
    #   ACh↑ attentional gating, NE↑ moderate
    "interested":   {"da": +0.3, "ne": +0.2, "ach": +0.3, "cb1": +0.2},
    # Curious (PDF p15-16,36-37): DA↑ novelty, 5-HT(2A)↑ symbolic expansion,
    #   CB1↑ schema flexibility, GLU(NMDA)↑ high-complexity binding
    "curious":      {"da": +0.5, "ne": +0.3, "ach": +0.4, "glu": +0.3, "5ht": +0.2, "cb1": +0.3},
    "creative":     {"da": +0.5, "cb1": +0.3, "5ht": +0.2},
    "focused":      {"ach": +0.5, "ne": +0.2, "da": +0.2, "gaba": +0.2},
    # Courageous (PDF p38-39): DA↑ override inhibitions, NE↑ challenge salience,
    #   OXT↑ social risk buffering, CRH/GR↑ background threat
    "courageous":   {"da": +0.4, "ne": +0.3, "5ht": +0.2, "oxt": +0.2, "cor": +0.2},
    "confident":    {"da": +0.5, "ne": +0.3, "5ht": +0.3, "ach": +0.2},
    # Proud (PDF p40-41): DA↑ self-relevance, MOR↑ satisfaction, OXT↑ resonance,
    #   5-HT(1B)↑ reward generalization
    "proud":        {"da": +0.4, "5ht": +0.3, "oxt": +0.2, "mor": +0.2},
    # Respected (PDF p41-42): OXT↑ peer encoding, DA↑ reward, 5-HT↑ regulation,
    #   MOR↑ security
    "respected":    {"oxt": +0.3, "5ht": +0.2, "da": +0.2, "mor": +0.2},
    # Loyal (PDF p42-43): OXT↑ consistency reinforcement, MOR↑ satisfaction,
    #   5-HT↑ suppresses impulsive detachment, DA(D2)↑ stable re-engagement
    "loyal":        {"oxt": +0.5, "5ht": +0.3, "mor": +0.2, "da": +0.2},
    # Connected (PDF p43-44): OXT↑↑ deep resonance, DA↑ novelty in empathic match,
    #   CB1↑ filter fluidity, 5-HT(2A)↑ intersubjective mapping
    "connected":    {"oxt": +0.5, "5ht": +0.3, "mor": +0.3, "da": +0.2, "cb1": +0.2},
    # Sensitive (PDF p44-45): OXT↑ emotional amplification, 5-HT↑ tone modulation,
    #   NE↑ contradiction sensitivity, GABA-A↑ overreactivity control
    "sensitive":    {"oxt": +0.3, "ne": +0.2, "5ht": +0.2, "gaba": +0.2},
    # Belonging (PDF p45): OXT↑ identity-environment mapping, DA(D1)↑ self-consistency,
    #   MOR↑ security, 5-HT(1A)↑ uncertainty buffer
    "belonging":    {"oxt": +0.5, "mor": +0.3, "5ht": +0.3, "da": +0.2},
}

# NT bias map: emotion group → which NTs bias detection sensitivity
GROUP_NT_BIAS: Dict[str, Dict[str, float]] = {
    EmotionGroup.TRUST_RELATIONAL.value:     {"oxt": 0.30, "mor": 0.15},
    EmotionGroup.SELF_EVALUATION.value:      {"cor": 0.20, "da": -0.15},
    EmotionGroup.UNCERTAINTY_FORECAST.value:  {"ne": 0.25, "cor": 0.15},
    EmotionGroup.AROUSAL_ENERGY.value:        {"ne": 0.20, "da": 0.10},
    EmotionGroup.LOW_ACTIVATION.value:        {"da": -0.20, "ne": -0.15},
    EmotionGroup.LOSS_TEMPORAL.value:         {"cor": 0.15, "oxt": -0.10},
    EmotionGroup.POSITIVE_CREATIVE.value:     {"da": 0.25, "5ht": 0.15},
}

# Positive and negative lexicons (compact — used for valence extraction)
POSITIVE_LEXICON = frozenset([
    "good", "great", "amazing", "wonderful", "love", "happy", "thank",
    "awesome", "fantastic", "excellent", "brilliant", "perfect", "nice",
    "beautiful", "joy", "glad", "pleased", "delight", "appreciate",
    "success", "hope", "excited", "proud", "confident", "trust",
])
NEGATIVE_LEXICON = frozenset([
    "bad", "terrible", "awful", "hate", "angry", "sad", "wrong",
    "horrible", "disgusting", "stupid", "fail", "worse", "worst",
    "pain", "hurt", "fear", "scared", "disappoint", "frustrat",
    "annoy", "upset", "regret", "guilt", "shame", "anxi",
])

# Urgency keyword set
URGENCY_KEYWORDS = frozenset([
    "now", "immediately", "urgent", "asap", "help", "emergency",
    "quickly", "hurry", "critical", "right away", "fast",
])


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class EDConfig:
    """All tunable parameters for the Emotional Detection Engine."""

    # --- Stage 2: Classification ---
    theta_detect: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.25, "dev": 0.35, "learning": 0.30,
        "reflective": 0.20, "rem_normal": 0.25, "rem_dream": 0.40,
    })
    max_active_K: Dict[str, int] = field(default_factory=lambda: {
        "normal": 5, "dev": 3, "learning": 4,
        "reflective": 6, "rem_normal": 5, "rem_dream": 3,
    })
    intensity_scale: Dict[str, float] = field(default_factory=lambda: {
        "normal": 1.0, "dev": 0.7, "learning": 0.85,
        "reflective": 1.2, "rem_normal": 1.0, "rem_dream": 0.5,
    })

    # --- Scoring weights ---
    w_keyword: float = 0.35
    w_valence: float = 0.20
    w_arousal: float = 0.15
    w_domain: float = 0.10
    w_structure: float = 0.10
    w_memory: float = 0.10

    # --- Arousal coefficients ---
    alpha_punct: float = 0.25
    alpha_caps: float = 0.20
    alpha_repeat: float = 0.20
    alpha_urgency: float = 0.25
    alpha_length: float = 0.10

    # Arousal sigmoid
    beta_arousal: float = 3.0
    arousal_midpoint: float = 0.5

    # --- NT detection bias strength ---
    nt_bias_strength: Dict[str, float] = field(default_factory=lambda: {
        "normal": 1.0, "dev": 0.5, "learning": 0.7,
        "reflective": 1.3, "rem_normal": 1.0, "rem_dream": 0.3,
    })

    # --- ENOCH structural override ---
    structural_override_threshold: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.40, "dev": 0.55, "learning": 0.45,
        "reflective": 0.35, "rem_normal": 0.40, "rem_dream": 0.60,
    })

    # --- Tone calibration coefficients ---
    rho_1: float = 0.05    # OXT warmth drift
    rho_2: float = 0.02    # OXT valence drift
    gamma_oxt: float = 0.03  # OXT mean reversion
    lambda_5ht: float = 0.30  # 5-HT1A affinity sensitivity
    tau_5ht: float = 20.0     # Integration window (cycles)
    eta_gaba: float = 0.25    # GABA discord suppression

    # --- Stochastic ---
    sigma_intensity: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.02, "dev": 0.03, "learning": 0.025,
        "reflective": 0.01, "rem_normal": 0.02, "rem_dream": 0.05,
    })

    # Max intensity sum for re-normalization
    max_intensity_sum: float = 1.5


# =====================================================================
# Mutable state
# =====================================================================


@dataclass
class EDState:
    """Runtime state."""
    # NT read-port levels
    oxt_level: float = 0.0
    ne_level:  float = 0.0
    da_level:  float = 0.0
    _5ht_level: float = 0.0
    cor_level: float = 0.0
    gaba_level: float = 0.0

    # OXT baseline for drift calculation
    oxt_baseline: float = 0.5

    # 5-HT1A affinity integral
    charge_integral: float = 0.0

    # History
    total_detections: int = 0
    structural_override_count: int = 0


# =====================================================================
# Frozen I/O
# =====================================================================


@dataclass(frozen=True)
class EmotionalDetectionInput:
    """Input to the Emotional Detection Engine."""
    tokens: Tuple[str, ...] = ()
    lemmatized_tokens: Tuple[str, ...] = ()
    tf_idf_weights: Dict[str, float] = field(default_factory=dict)
    raw_text: str = ""
    sentence_count: int = 1
    question_count: int = 0
    memory_contrast_emotions: Optional[Dict[str, float]] = None
    active_mode: str = "normal"
    cycle_count: int = 0
    speaker: str = "user"


@dataclass(frozen=True)
class DetectedEmotion:
    """One detected emotion with metadata."""
    emotion_id: int
    emotion_name: str
    group: str
    intensity: float
    score: float
    dominant_evidence: str = ""


@dataclass(frozen=True)
class ToneVector:
    """4-component tone evaluation vector for response generation."""
    e_valence: float = 0.0    # [-1, 1]
    e_coherence: float = 0.0  # [0, 1]
    e_warmth: float = 0.0     # [-1, 1]
    e_discord: float = 0.0    # [0, 1]


@dataclass(frozen=True)
class EmotionNeurochem:
    """Neurochemical deltas emitted by the Emotional Detection Engine."""
    # Tonic (4M)
    oxt_baseline_drift: float = 0.0
    serotonin_affinity_shift: float = 0.0
    gaba_reuptake_mod: float = 0.0
    # Phasic (4R)
    delta_da: float = 0.0
    delta_ne: float = 0.0
    delta_oxt: float = 0.0
    delta_mor: float = 0.0
    delta_cor: float = 0.0
    delta_ach: float = 0.0
    delta_gaba: float = 0.0
    # Oscillatory
    delta_alpha: float = 0.0
    beta_suppress: float = 0.0
    theta_boost: float = 0.0
    gamma_burst: float = 0.0


@dataclass(frozen=True)
class EmotionalDetectionResult:
    """Output from the Emotional Detection Engine."""
    active_emotions: Tuple[DetectedEmotion, ...] = ()
    dominant_emotion: Optional[DetectedEmotion] = None
    emotion_count: int = 0
    valence_net: float = 0.0
    arousal: float = 0.0
    dominant_domain: str = "cognitive"
    tone_vector: ToneVector = field(default_factory=ToneVector)
    neurochemical_signals: EmotionNeurochem = field(default_factory=EmotionNeurochem)
    saturation_level: float = 0.0
    emotional_complexity: float = 0.0
    structural_emotion_override: bool = False
    processing_time_ms: float = 0.0
    engine_id: str = "emotional_detection_engine"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Utility
# =====================================================================


def _sigmoid(x: float) -> float:
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# =====================================================================
# Stage 1: Feature Extraction  (pure functions)
# =====================================================================


def extract_valence(
    tokens: Tuple[str, ...],
    tf_idf: Dict[str, float],
) -> Tuple[float, float, float]:
    """Extract positive/negative valence from tokens. Returns (V_pos, V_neg, V_net)."""
    if not tokens:
        return 0.0, 0.0, 0.0
    n = len(tokens)
    v_pos = sum(tf_idf.get(t, 1.0) for t in tokens if t.lower() in POSITIVE_LEXICON) / n
    v_neg = sum(tf_idf.get(t, 1.0) for t in tokens if t.lower() in NEGATIVE_LEXICON) / n
    # Also check for partial matches (lemma prefix matching)
    for t in tokens:
        tl = t.lower()
        for pos in POSITIVE_LEXICON:
            if tl.startswith(pos) and tl not in POSITIVE_LEXICON:
                v_pos += tf_idf.get(t, 1.0) * 0.5 / n
                break
        for neg in NEGATIVE_LEXICON:
            if tl.startswith(neg) and tl not in NEGATIVE_LEXICON:
                v_neg += tf_idf.get(t, 1.0) * 0.5 / n
                break
    v_net = max(-1.0, min(1.0, v_pos - v_neg))
    return v_pos, v_neg, v_net


def extract_arousal(
    raw_text: str,
    tokens: Tuple[str, ...],
    cfg: EDConfig,
) -> float:
    """Extract arousal level from text features."""
    if not raw_text:
        return 0.0
    n_chars = max(len(raw_text), 1)
    n_tokens = max(len(tokens), 1)

    f_excl = raw_text.count("!") / n_chars * 10.0
    f_caps = sum(1 for t in tokens if t.isupper() and len(t) > 1) / n_tokens
    # Repetition: repeated chars like "noooo"
    f_repeat = 0.0
    for t in tokens:
        if len(t) > 3:
            for i in range(len(t) - 2):
                if t[i] == t[i + 1] == t[i + 2]:
                    f_repeat += 1.0 / n_tokens
                    break
    f_urgency = sum(1 for t in tokens if t.lower() in URGENCY_KEYWORDS) / n_tokens
    # Length: very short or very long → higher arousal
    normalized_len = min(len(tokens) / 50.0, 1.0)

    arousal = (
        cfg.alpha_punct * _clamp(f_excl)
        + cfg.alpha_caps * _clamp(f_caps)
        + cfg.alpha_repeat * _clamp(f_repeat)
        + cfg.alpha_urgency * _clamp(f_urgency)
        + cfg.alpha_length * (1.0 - abs(normalized_len - 0.3))  # Peak at moderate length
    )
    return _clamp(arousal)


def extract_domain_scores(
    tokens: Tuple[str, ...],
    tf_idf: Dict[str, float],
) -> Dict[str, float]:
    """Classify emotional content domain."""
    social_kw = {"trust", "friend", "help", "together", "relationship", "people", "someone", "love"}
    cognitive_kw = {"think", "understand", "confus", "sense", "logic", "reason", "idea", "know"}
    temporal_kw = {"remember", "used to", "will be", "someday", "before", "after", "when"}
    existential_kw = {"meaning", "purpose", "identity", "who am", "why", "exist", "life"}

    n = max(len(tokens), 1)
    scores: Dict[str, float] = {}
    for domain, kw_set in [("social", social_kw), ("cognitive", cognitive_kw),
                            ("temporal", temporal_kw), ("existential", existential_kw)]:
        score = 0.0
        for t in tokens:
            tl = t.lower()
            for kw in kw_set:
                if kw in tl:
                    score += tf_idf.get(t, 1.0)
                    break
        scores[domain] = score / n
    return scores


def extract_structural_features(
    raw_text: str,
    tokens: Tuple[str, ...],
    sentence_count: int,
    question_count: int,
) -> Dict[str, float]:
    """Extract structural features from text."""
    n_sentences = max(sentence_count, 1)
    n_tokens = max(len(tokens), 1)
    return {
        "question_density": question_count / n_sentences,
        "ellipsis_usage": raw_text.count("...") / max(len(raw_text), 1) * 100,
        "negation_density": sum(1 for t in tokens if t.lower() in ("not", "never", "no", "can't", "won't", "don't")) / n_tokens,
        "first_person": sum(1 for t in tokens if t.lower() in ("i", "me", "my", "mine", "myself")) / n_tokens,
        "second_person": sum(1 for t in tokens if t.lower() in ("you", "your", "yours", "yourself")) / n_tokens,
    }


# =====================================================================
# Stage 2: Emotion Classification  (pure functions)
# =====================================================================


def keyword_match_score(
    sig: EmotionSignature,
    tokens: Tuple[str, ...],
    tf_idf: Dict[str, float],
) -> float:
    """Score keyword match for one emotion signature."""
    if not sig.keyword_patterns or not tokens:
        return 0.0
    hits = 0.0
    for pattern in sig.keyword_patterns:
        for t in tokens:
            if pattern in t.lower():
                hits += tf_idf.get(t, 1.0)
                break
    return min(hits / len(sig.keyword_patterns), 1.0)


def valence_match(sig: EmotionSignature, v_net: float) -> float:
    """Score how well net valence matches emotion's expected range."""
    mid = (sig.valence_range[0] + sig.valence_range[1]) / 2.0
    span = max(sig.valence_range[1] - sig.valence_range[0], 0.1)
    dist = abs(v_net - mid) / span
    return max(0.0, 1.0 - dist)


def arousal_match(sig: EmotionSignature, arousal: float) -> float:
    """Score how well arousal matches emotion's expected range."""
    mid = (sig.arousal_range[0] + sig.arousal_range[1]) / 2.0
    span = max(sig.arousal_range[1] - sig.arousal_range[0], 0.1)
    dist = abs(arousal - mid) / span
    return max(0.0, 1.0 - dist)


def domain_match(sig: EmotionSignature, domain_scores: Dict[str, float]) -> float:
    """Score domain match."""
    primary = sig.domain_primary
    if primary in domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        if best_domain == primary:
            return 1.0
    return 0.3


def score_emotion(
    sig: EmotionSignature,
    tokens: Tuple[str, ...],
    tf_idf: Dict[str, float],
    v_net: float,
    arousal: float,
    domain_scores: Dict[str, float],
    structural_features: Dict[str, float],
    memory_context: Optional[Dict[str, float]],
    cfg: EDConfig,
) -> Tuple[float, str]:
    """
    Compute detection score for one emotion.
    Returns (score, dominant_evidence_description).
    """
    kw = keyword_match_score(sig, tokens, tf_idf)
    val = valence_match(sig, v_net)
    aro = arousal_match(sig, arousal)
    dom = domain_match(sig, domain_scores)

    # Structural feature match (simple — check if structural indicators present)
    struct = 0.0
    if sig.group == EmotionGroup.SELF_EVALUATION.value:
        struct = structural_features.get("first_person", 0.0) * 2
    elif sig.group == EmotionGroup.TRUST_RELATIONAL.value:
        struct = structural_features.get("second_person", 0.0) * 2
    elif sig.group == EmotionGroup.UNCERTAINTY_FORECAST.value:
        struct = structural_features.get("question_density", 0.0)
    struct = _clamp(struct)

    # Memory context boost
    mem = 0.0
    if memory_context and sig.emotion_name in memory_context:
        mem = _clamp(memory_context[sig.emotion_name])

    score = (
        cfg.w_keyword * kw
        + cfg.w_valence * val
        + cfg.w_arousal * aro
        + cfg.w_domain * dom
        + cfg.w_structure * struct
        + cfg.w_memory * mem
    )

    # Determine dominant evidence
    parts = [
        ("keyword", kw * cfg.w_keyword),
        ("valence", val * cfg.w_valence),
        ("arousal", aro * cfg.w_arousal),
        ("domain", dom * cfg.w_domain),
        ("structure", struct * cfg.w_structure),
        ("memory", mem * cfg.w_memory),
    ]
    best_part = max(parts, key=lambda x: x[1])
    evidence = f"{best_part[0]}: {best_part[1]:.2f}"

    return score, evidence


def estimate_intensity(
    score: float,
    arousal: float,
    mode_scale: float,
    beta_arousal: float = 3.0,
    arousal_midpoint: float = 0.5,
) -> float:
    """Estimate intensity from score, modulated by arousal and mode."""
    arousal_scale = _sigmoid(beta_arousal * (arousal - arousal_midpoint))
    return _clamp(score * (0.5 + 0.5 * arousal_scale) * mode_scale)


def apply_mutual_exclusion(
    scored: List[Tuple[int, float, str]],
) -> List[Tuple[int, float, str]]:
    """Remove suppressed emotions."""
    active_ids = {eid for eid, _, _ in scored}
    result = []
    for eid, score, evidence in scored:
        sig = EMOTION_SIGNATURES.get(eid)
        if sig is None:
            result.append((eid, score, evidence))
            continue
        # Check if any active emotion suppresses this one
        suppressed = False
        for other_eid, other_score, _ in scored:
            if other_eid == eid:
                continue
            other_sig = EMOTION_SIGNATURES.get(other_eid)
            if other_sig and eid in other_sig.suppresses and other_score > score:
                suppressed = True
                break
        if not suppressed:
            result.append((eid, score, evidence))
    return result


def detect_structural_emotions(
    tokens: Tuple[str, ...],
    threshold: float = 0.40,
) -> List[DetectedEmotion]:
    """ENOCH fast-path: detect structural emotions via keyword density."""
    results: List[DetectedEmotion] = []
    n = max(len(tokens), 1)
    for category, keywords in STRUCTURAL_EMOTIONS.items():
        hits = sum(1 for t in tokens for kw in keywords if kw in t.lower())
        density = hits / n
        if density >= threshold:
            eid = STRUCTURAL_TO_ID[category]
            sig = EMOTION_SIGNATURES.get(eid)
            name = sig.emotion_name if sig else category
            group = sig.group if sig else ""
            results.append(DetectedEmotion(
                emotion_id=eid,
                emotion_name=name,
                group=group,
                intensity=_clamp(density * 2),
                score=density,
                dominant_evidence=f"structural:{category}",
            ))
    return results


# =====================================================================
# Stage 3: Tone Calibration  (pure functions)
# =====================================================================


def compute_tone_vector(
    active_emotions: List[DetectedEmotion],
    v_net: float,
) -> ToneVector:
    """Compute the 4-component tone evaluation vector."""
    if not active_emotions:
        return ToneVector(e_valence=v_net, e_coherence=1.0, e_warmth=0.0, e_discord=0.0)

    # e_valence = V_net
    e_val = v_net

    # e_coherence = 1 - entropy / log(K)
    intensities = [e.intensity for e in active_emotions]
    total_i = sum(intensities)
    if total_i > 0 and len(active_emotions) > 1:
        probs = [i / total_i for i in intensities]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(active_emotions))
        e_coh = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
    else:
        e_coh = 1.0

    # e_warmth: trust/relational positive - trust/relational negative
    warm_ids = {31, 32, 33, 42, 43, 44, 46}  # Valued, Thankful, Accepted, Respected, Loyal, Connected, Belonging
    cold_ids = {1, 8, 13}  # Betrayal, Rejected, Isolated
    e_warm = 0.0
    for em in active_emotions:
        if em.emotion_id in warm_ids:
            e_warm += em.intensity
        elif em.emotion_id in cold_ids:
            e_warm -= em.intensity
    e_warm = max(-1.0, min(1.0, e_warm))

    # e_discord: conflicting emotions active simultaneously
    # Check for positive + negative emotions both present at high intensity
    pos_present = any(em.intensity > 0.3 and v_net > 0.2 for em in active_emotions)
    neg_present = any(em.intensity > 0.3 and v_net < -0.2 for em in active_emotions)
    # Or specific conflicts
    active_ids = {em.emotion_id for em in active_emotions}
    has_conflict = bool(active_ids & warm_ids) and bool(active_ids & cold_ids)

    e_disc = 0.0
    if pos_present and neg_present:
        e_disc = 0.5
    if has_conflict:
        e_disc = max(e_disc, 0.6)
    e_disc = _clamp(e_disc)

    return ToneVector(
        e_valence=max(-1.0, min(1.0, e_val)),
        e_coherence=_clamp(e_coh),
        e_warmth=e_warm,
        e_discord=e_disc,
    )


def compute_oxt_drift(
    e_warmth: float,
    e_valence: float,
    oxt_baseline: float,
    oxt_baseline_0: float = 0.5,
    rho_1: float = 0.05,
    rho_2: float = 0.02,
    gamma_oxt: float = 0.03,
) -> float:
    """OXT baseline drift: warmth and valence drive OXT up, mean-revert."""
    d_oxt = rho_1 * e_warmth + rho_2 * max(e_valence, 0) - gamma_oxt * (oxt_baseline - oxt_baseline_0)
    return d_oxt


def compute_5ht1a_affinity(
    charge_integral: float,
    total_charge: float,
    tau_5ht: float = 20.0,
    lambda_5ht: float = 0.30,
) -> Tuple[float, float]:
    """
    5-HT1A affinity adjustment from sustained emotional charge.
    Returns (affinity_shift, updated_integral).
    """
    # Leaky integration of charge
    decay = math.exp(-1.0 / max(tau_5ht, 1.0))
    new_integral = charge_integral * decay + total_charge * (1.0 - decay)
    shift = lambda_5ht * new_integral
    return shift, new_integral


def compute_gaba_reuptake_mod(e_discord: float, eta: float = 0.25) -> float:
    """GABA reuptake suppression under discord."""
    return -eta * e_discord  # Negative = suppression = longer GABA residence


# =====================================================================
# Stage 4: Neurochemical Mapping  (pure functions)
# =====================================================================


def map_emotions_to_neurochem(
    active_emotions: List[DetectedEmotion],
    tone: ToneVector,
    oxt_drift: float,
    affinity_shift: float,
    gaba_mod: float,
) -> EmotionNeurochem:
    """Map detected emotions to neurochemical signals."""
    # Aggregate phasic signals from all active emotions
    da = ne = oxt = mor = cor = ach = gaba = 0.0
    for em in active_emotions:
        profile = EMOTION_NT_PROFILES.get(em.emotion_name, {})
        intensity = em.intensity
        da += profile.get("da", 0.0) * intensity
        ne += profile.get("ne", 0.0) * intensity
        oxt += profile.get("oxt", 0.0) * intensity
        mor += profile.get("mor", 0.0) * intensity
        cor += profile.get("cor", 0.0) * intensity
        ach += profile.get("ach", 0.0) * intensity
        gaba += profile.get("gaba", 0.0) * intensity

    # Oscillatory targets
    d_alpha = _clamp(tone.e_warmth * 0.1, -0.1, 0.1)
    beta_sup = _clamp(affinity_shift * 0.1, 0.0, 0.1)
    theta_b = _clamp(tone.e_discord * 0.1, 0.0, 0.1)
    gamma_b = 0.0
    for em in active_emotions:
        if em.emotion_name in ("curious", "excited", "creative"):
            gamma_b += em.intensity * 0.05

    return EmotionNeurochem(
        oxt_baseline_drift=oxt_drift,
        serotonin_affinity_shift=affinity_shift,
        gaba_reuptake_mod=gaba_mod,
        delta_da=da,
        delta_ne=ne,
        delta_oxt=oxt,
        delta_mor=mor,
        delta_cor=cor,
        delta_ach=ach,
        delta_gaba=gaba,
        delta_alpha=d_alpha,
        beta_suppress=beta_sup,
        theta_boost=theta_b,
        gamma_burst=_clamp(gamma_b),
    )


# =====================================================================
# NT bias application (pure)
# =====================================================================


def apply_nt_detection_bias(
    emotion_scores: Dict[int, float],
    nt_state: Dict[str, float],
    strength: float = 1.0,
) -> Dict[int, float]:
    """Modulate emotion scores based on current NT state."""
    result = dict(emotion_scores)
    for eid, score in result.items():
        sig = EMOTION_SIGNATURES.get(eid)
        if sig is None:
            continue
        bias_map = GROUP_NT_BIAS.get(sig.group, {})
        bias = 0.0
        for nt, weight in bias_map.items():
            bias += weight * nt_state.get(nt, 0.0)
        result[eid] = max(0.0, score * (1.0 + bias * strength))
    return result


# =====================================================================
# Engine class
# =====================================================================


class EmotionalDetectionEngine:
    """
    Engine 28 -- Emotional Detection Engine.

    Four-stage affective perception:
      Stage 1: Feature Extraction (valence, arousal, domain, structure)
      Stage 2: Emotion Classification (46-emotion taxonomy, top-K)
      Stage 3: Tone Calibration (E_tone vector, OXT/5-HT/GABA drift)
      Stage 4: Neurochemical Mapping (4M tonic + 4R phasic signals)

    API
    ---
    configure(mode)                -- set operational mode
    update_neurochem_state(state)  -- inject external NT levels
    process(input)                 -- full emotional detection
    get_status()                   -- introspection
    """

    engine_id = "emotional_detection_engine"
    cluster   = "emotional_processing"

    def __init__(
        self,
        config: Optional[EDConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or EDConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = EDState()
        self._cycle_count = 0

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        if "oxt" in state_dict:
            self._state.oxt_level = _clamp(state_dict["oxt"])
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "da" in state_dict:
            self._state.da_level = _clamp(state_dict["da"])
        if "5ht" in state_dict:
            self._state._5ht_level = _clamp(state_dict["5ht"])
        if "cor" in state_dict:
            self._state.cor_level = _clamp(state_dict["cor"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])

    def _mode_key(self) -> str:
        return self._mode.value

    def _get_mode_param(self, param_dict: Dict, default=0.5):
        return param_dict.get(self._mode_key(), default)

    def process(self, inp: EmotionalDetectionInput) -> EmotionalDetectionResult:
        t0 = time.perf_counter()
        cfg = self._cfg

        tokens = inp.lemmatized_tokens if inp.lemmatized_tokens else inp.tokens
        tf_idf = inp.tf_idf_weights or {}

        # ==============================================================
        # STAGE 1: FEATURE EXTRACTION
        # ==============================================================

        v_pos, v_neg, v_net = extract_valence(tokens, tf_idf)
        arousal = extract_arousal(inp.raw_text, tokens, cfg)
        domain_scores = extract_domain_scores(tokens, tf_idf)
        structural = extract_structural_features(
            inp.raw_text, tokens, inp.sentence_count, inp.question_count,
        )

        dominant_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "cognitive"

        # ==============================================================
        # STAGE 2: EMOTION CLASSIFICATION
        # ==============================================================

        theta_det = self._get_mode_param(cfg.theta_detect, 0.25)
        max_k = self._get_mode_param(cfg.max_active_K, 5)
        mode_scale = self._get_mode_param(cfg.intensity_scale, 1.0)
        struct_threshold = self._get_mode_param(cfg.structural_override_threshold, 0.40)
        sigma_int = self._get_mode_param(cfg.sigma_intensity, 0.02)

        # Check ENOCH structural fast-path first
        structural_overrides = detect_structural_emotions(tokens, struct_threshold)
        structural_fired = len(structural_overrides) > 0

        # Score all emotions
        raw_scores: Dict[int, Tuple[float, str]] = {}
        for eid, sig in EMOTION_SIGNATURES.items():
            s, ev = score_emotion(
                sig, tokens, tf_idf, v_net, arousal,
                domain_scores, structural, inp.memory_contrast_emotions, cfg,
            )
            if s > 0:
                raw_scores[eid] = (s, ev)

        # Apply NT detection bias
        nt_bias_str = self._get_mode_param(cfg.nt_bias_strength, 1.0)
        nt_state = {
            "oxt": self._state.oxt_level,
            "ne": self._state.ne_level,
            "da": self._state.da_level,
            "5ht": self._state._5ht_level,
            "cor": self._state.cor_level,
            "gaba": self._state.gaba_level,
        }
        biased_scores = apply_nt_detection_bias(
            {eid: s for eid, (s, _) in raw_scores.items()},
            nt_state,
            nt_bias_str,
        )

        # Merge biased scores with evidence strings
        scored_list: List[Tuple[int, float, str]] = []
        for eid, score in biased_scores.items():
            if score >= theta_det:
                _, ev = raw_scores.get(eid, (0.0, ""))
                scored_list.append((eid, score, ev))

        # Sort by score descending
        scored_list.sort(key=lambda x: x[1], reverse=True)

        # Apply mutual exclusion
        scored_list = apply_mutual_exclusion(scored_list)

        # Select top K
        scored_list = scored_list[:max_k]

        # Compute intensity
        detected: List[DetectedEmotion] = []
        for eid, score, evidence in scored_list:
            sig = EMOTION_SIGNATURES.get(eid)
            if sig is None:
                continue
            intensity = estimate_intensity(score, arousal, mode_scale, cfg.beta_arousal, cfg.arousal_midpoint)
            # Add noise
            if sigma_int > 0:
                noise = float(self._rng.normal(0.0, sigma_int))
                intensity = _clamp(intensity + noise)
            detected.append(DetectedEmotion(
                emotion_id=eid,
                emotion_name=sig.emotion_name,
                group=sig.group,
                intensity=intensity,
                score=score,
                dominant_evidence=evidence,
            ))

        # Merge structural overrides
        if structural_fired:
            override_ids = {e.emotion_id for e in structural_overrides}
            detected_ids = {e.emotion_id for e in detected}
            for so in structural_overrides:
                if so.emotion_id not in detected_ids:
                    detected.append(so)
            detected.sort(key=lambda e: e.intensity, reverse=True)
            detected = detected[:max_k]

        # Re-normalize intensities
        total_intensity = sum(e.intensity for e in detected)
        if total_intensity > cfg.max_intensity_sum:
            scale = cfg.max_intensity_sum / total_intensity
            detected = [
                DetectedEmotion(
                    emotion_id=e.emotion_id, emotion_name=e.emotion_name,
                    group=e.group, intensity=e.intensity * scale,
                    score=e.score, dominant_evidence=e.dominant_evidence,
                )
                for e in detected
            ]

        dominant = detected[0] if detected else None
        saturation = max((e.intensity for e in detected), default=0.0)

        # Emotional complexity (entropy)
        complexity = 0.0
        if len(detected) > 1:
            total_i = sum(e.intensity for e in detected)
            if total_i > 0:
                probs = [e.intensity / total_i for e in detected]
                entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                complexity = entropy / math.log(len(detected)) if len(detected) > 1 else 0.0

        # ==============================================================
        # STAGE 3: TONE CALIBRATION
        # ==============================================================

        tone = compute_tone_vector(list(detected), v_net)

        oxt_drift = compute_oxt_drift(
            tone.e_warmth, tone.e_valence,
            self._state.oxt_baseline, 0.5,
            cfg.rho_1, cfg.rho_2, cfg.gamma_oxt,
        )

        total_charge = sum(e.intensity for e in detected)
        affinity_shift, new_integral = compute_5ht1a_affinity(
            self._state.charge_integral, total_charge,
            cfg.tau_5ht, cfg.lambda_5ht,
        )

        gaba_mod = compute_gaba_reuptake_mod(tone.e_discord, cfg.eta_gaba)

        # ==============================================================
        # STAGE 4: NEUROCHEMICAL MAPPING
        # ==============================================================

        neurochem = map_emotions_to_neurochem(
            list(detected), tone, oxt_drift, affinity_shift, gaba_mod,
        )

        # ==============================================================
        # UPDATE STATE
        # ==============================================================

        self._cycle_count += 1
        self._state.total_detections += 1
        self._state.oxt_baseline += oxt_drift
        self._state.oxt_baseline = _clamp(self._state.oxt_baseline)
        self._state.charge_integral = new_integral
        if structural_fired:
            self._state.structural_override_count += 1

        elapsed = (time.perf_counter() - t0) * 1000.0

        return EmotionalDetectionResult(
            active_emotions=tuple(detected),
            dominant_emotion=dominant,
            emotion_count=len(detected),
            valence_net=v_net,
            arousal=arousal,
            dominant_domain=dominant_domain,
            tone_vector=tone,
            neurochemical_signals=neurochem,
            saturation_level=saturation,
            emotional_complexity=complexity,
            structural_emotion_override=structural_fired,
            processing_time_ms=elapsed,
            engine_id=self.engine_id,
            metadata={
                "mode": self._mode.value,
                "cycle": self._cycle_count,
                "raw_scores_count": len(raw_scores),
                "detected_count": len(detected),
                "structural_fired": structural_fired,
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "total_detections": self._state.total_detections,
            "structural_override_count": self._state.structural_override_count,
            "oxt_baseline": self._state.oxt_baseline,
            "charge_integral": self._state.charge_integral,
            "nt_levels": {
                "oxt": self._state.oxt_level,
                "ne": self._state.ne_level,
                "da": self._state.da_level,
                "5ht": self._state._5ht_level,
                "cor": self._state.cor_level,
                "gaba": self._state.gaba_level,
            },
        }
