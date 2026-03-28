"""
ZA-DOS v0.6 — Subject Classifier (spec §2.3).

Classifies input text into one of 7 SubjectCategory values so the
engine_toolkit can apply subject-specific tier promotions/demotions.

Two entry points:
  classify_subject()          — uses TokenizerResult + ExpansionResult
  classify_subject_from_text() — simple keyword-based fallback
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from zados.core.types import SubjectCategory

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Keyword sets per subject category
# ------------------------------------------------------------------

SUBJECT_KEYWORD_MAPS: Dict[SubjectCategory, Set[str]] = {
    SubjectCategory.TECHNICAL: {
        "algorithm", "code", "software", "hardware", "programming",
        "debug", "compile", "api", "database", "server", "deploy",
        "function", "variable", "class", "method", "library",
        "framework", "bug", "syntax", "binary", "protocol",
        "architecture", "stack", "module", "interface", "endpoint",
    },
    SubjectCategory.SCIENTIFIC: {
        "hypothesis", "experiment", "theory", "physics", "chemistry",
        "biology", "research", "data", "analysis", "statistical",
        "equation", "formula", "observation", "empirical", "evidence",
        "quantum", "molecular", "cellular", "genome", "evolution",
        "neuroscience", "particle", "thermodynamic", "entropy",
    },
    SubjectCategory.PHILOSOPHICAL: {
        "meaning", "existence", "consciousness", "morality", "ethics",
        "truth", "reality", "freedom", "justice", "virtue",
        "ontology", "epistemology", "metaphysics", "dialectic",
        "phenomenology", "existential", "determinism", "dualism",
        "paradox", "absurd", "nihilism", "stoic", "utilitarian",
    },
    SubjectCategory.SOCIAL: {
        "relationship", "emotion", "feeling", "friend", "family",
        "conflict", "communication", "empathy", "trust", "love",
        "anger", "grief", "anxiety", "support", "connection",
        "lonely", "hurt", "upset", "jealous", "grateful",
        "community", "belonging", "rejection", "attachment",
    },
    SubjectCategory.CREATIVE: {
        "story", "poem", "music", "art", "design", "paint",
        "write", "novel", "character", "plot", "compose",
        "imagine", "creative", "inspiration", "aesthetic",
        "metaphor", "narrative", "sketch", "sculpt", "lyric",
        "fiction", "fantasy", "genre", "theme", "motif",
    },
    SubjectCategory.PRACTICAL: {
        "how", "step", "guide", "recipe", "fix", "repair",
        "build", "make", "install", "setup", "configure",
        "plan", "budget", "schedule", "organise", "organize",
        "travel", "cook", "clean", "move", "buy", "sell",
        "price", "cost", "deadline", "task", "checklist",
    },
}

# ------------------------------------------------------------------
# Aggregate feature heuristics (from TokenizerResult/ExpansionResult)
# ------------------------------------------------------------------

_FEATURE_CATEGORY_HINTS: Dict[str, SubjectCategory] = {
    "hedging_density":  SubjectCategory.PHILOSOPHICAL,
    "emotional_vocab":  SubjectCategory.SOCIAL,
    "technical_vocab":  SubjectCategory.TECHNICAL,
    "imperative_ratio": SubjectCategory.PRACTICAL,
    "figurative_lang":  SubjectCategory.CREATIVE,
}


def classify_subject(
    tokenizer_result: Any = None,
    expansion_result: Any = None,
) -> SubjectCategory:
    """Classify subject using rich pipeline data.

    Parameters
    ----------
    tokenizer_result : TokenizerResult, optional
        Output from E23-adjacent Tokenizer.  If available, tokens are
        extracted for keyword matching and AggregateFeatures are used
        for heuristic boosts.
    expansion_result : ExpansionResult, optional
        Output from SemanticExpander.  Provides synonym/hypernym tokens.

    Returns
    -------
    SubjectCategory
    """
    tokens: set[str] = set()

    # Gather tokens from tokenizer
    if tokenizer_result is not None:
        raw_tokens = getattr(tokenizer_result, "tokens", None)
        if raw_tokens:
            tokens.update(t.lower() for t in raw_tokens)

    # Gather expanded tokens
    if expansion_result is not None:
        expanded = getattr(expansion_result, "expanded_tokens", None)
        if expanded:
            tokens.update(t.lower() for t in expanded)

    if not tokens:
        return SubjectCategory.MIXED

    # Score each category by keyword overlap
    scores: Dict[SubjectCategory, float] = {}
    for cat, keywords in SUBJECT_KEYWORD_MAPS.items():
        overlap = tokens & keywords
        scores[cat] = len(overlap)

    # Apply aggregate-feature boosts
    if tokenizer_result is not None:
        features = getattr(tokenizer_result, "aggregate_features", None)
        if features and isinstance(features, dict):
            for feat_key, target_cat in _FEATURE_CATEGORY_HINTS.items():
                val = features.get(feat_key, 0.0)
                if val > 0.3:
                    scores[target_cat] = scores.get(target_cat, 0.0) + val * 2.0

    # Pick highest-scoring category (ties → MIXED)
    if not scores or max(scores.values()) == 0.0:
        return SubjectCategory.MIXED

    top_score = max(scores.values())
    top_cats = [cat for cat, s in scores.items() if s == top_score]
    if len(top_cats) == 1:
        return top_cats[0]
    return SubjectCategory.MIXED


def classify_subject_from_text(text: str) -> SubjectCategory:
    """Simple keyword-based fallback when no pipeline data is available.

    Parameters
    ----------
    text : str
        Raw user input text.

    Returns
    -------
    SubjectCategory
    """
    if not text:
        return SubjectCategory.MIXED

    lower = text.lower()
    words = set(lower.split())

    scores: Dict[SubjectCategory, float] = {}
    for cat, keywords in SUBJECT_KEYWORD_MAPS.items():
        overlap = words & keywords
        scores[cat] = len(overlap)

    if not scores or max(scores.values()) == 0.0:
        return SubjectCategory.MIXED

    top_score = max(scores.values())
    top_cats = [cat for cat, s in scores.items() if s == top_score]
    if len(top_cats) == 1:
        return top_cats[0]
    return SubjectCategory.MIXED
