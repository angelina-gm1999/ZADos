"""
ZA-DOS Core Pipeline — Phase 1: Perception Layer (spec Part IV).

Runs the five perception engines (E23, E8, E11, E18, E19) plus the
Tokenizer and SemanticExpander infrastructure to produce a
PerceptionSnapshot.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from zados.core.types import InputBundle, PerceptionSnapshot

log = logging.getLogger(__name__)


def run_perception(
    bundle: InputBundle,
    engines: Dict[int, Any],
    nt_state_lowercase: Dict[str, float],
    tokenizer: Any = None,
    semantic_expander: Any = None,
    stmm: Any = None,
) -> PerceptionSnapshot:
    """Run the five perception engines and return a PerceptionSnapshot.

    Parameters
    ----------
    bundle : InputBundle
        Incoming pipeline input.
    engines : dict
        engine_number → engine instance  (must contain 23, 8, 11, 18, 19).
    nt_state_lowercase : dict
        NT name (lowercase) → concentration.
    tokenizer : Tokenizer, optional
        If provided, tokenizes ``bundle.raw_text`` before E23.
    semantic_expander : SemanticExpander, optional
        If provided, expands tokenizer output before E23.
    stmm : STMMStore, optional
        If provided, used to derive context signals for E11 / E23.
    """
    snap = PerceptionSnapshot()

    # ------------------------------------------------------------------
    # Step (a) — Tokenize + expand (pipeline infrastructure)
    # ------------------------------------------------------------------
    tokenizer_result = None
    expansion_result = None

    if tokenizer is not None:
        try:
            tokenizer_result = tokenizer.process(bundle.raw_text)
        except Exception:
            log.warning("Tokenizer failed; proceeding with empty tokenizer_result.")

    if semantic_expander is not None and tokenizer_result is not None:
        try:
            expansion_result = semantic_expander.process(tokenizer_result)
        except Exception:
            log.warning("SemanticExpander failed; proceeding without expansion.")

    # Derive token list for engines that accept simple List[str]
    tokens: List[str] = []
    if tokenizer_result is not None:
        tokens = [t.surface for t in getattr(tokenizer_result, "tokens", [])]
    if not tokens:
        tokens = bundle.raw_text.split()

    # ------------------------------------------------------------------
    # Step (b) — E23: Intention Map
    # ------------------------------------------------------------------
    e23 = engines.get(23)
    if e23 is not None:
        try:
            e23.update_neurochem_state(nt_state_lowercase)
            e23_input = _build_e23_input(
                bundle, tokenizer_result, expansion_result, stmm,
            )
            e23_result = e23.process(e23_input)

            snap.intent_result = e23_result
            snap.intent_archetype = getattr(e23_result, "primary_archetype", "")
            snap.intent_vector = getattr(e23_result, "intent_labels", {})
            snap.intent_confidence = getattr(e23_result, "intent_confidence", 0.0)
            snap.engine_statuses["E23"] = _safe_status(e23)
        except Exception:
            log.exception("E23 (IntentionMap) failed.")

    # If E23 didn't produce an archetype, fall back to bundle then default
    if not snap.intent_archetype:
        snap.intent_archetype = bundle.intent_archetype or "REFLECTIVE"

    # ------------------------------------------------------------------
    # Step (c) — E8: Relevance Scoring
    # ------------------------------------------------------------------
    e8 = engines.get(8)
    if e8 is not None:
        try:
            e8.update_neurochem_state(nt_state_lowercase)
            e8_input = {
                "nt_state": nt_state_lowercase,
                "items": _tokens_to_items(tokens),
            }
            e8_result = e8.process(e8_input)
            snap.ranked_facets = e8_result.get("scored_items", [])
            snap.engine_statuses["E8"] = _safe_status(e8)
        except Exception:
            log.exception("E8 (RelevanceScoring) failed.")

    # ------------------------------------------------------------------
    # Step (d) — E11: Input Relevance Evaluation
    # ------------------------------------------------------------------
    e11 = engines.get(11)
    if e11 is not None:
        try:
            e11.update_neurochem_state(nt_state_lowercase)
            e11_input = _build_e11_input(bundle, tokens, stmm, nt_state_lowercase)
            e11_result = e11.process(e11_input)
            snap.filtered_facets = [_ire_output_to_dict(e11_result)]
            snap.engine_statuses["E11"] = _safe_status(e11)
        except Exception:
            log.exception("E11 (InputRelevance) failed.")

    # ------------------------------------------------------------------
    # Step (e) — E18: Data Analysis
    # ------------------------------------------------------------------
    e18 = engines.get(18)
    if e18 is not None:
        try:
            e18.update_neurochem_state(nt_state_lowercase)
            e18_input = _build_e18_input(bundle, tokens)
            e18_result = e18.process(e18_input)
            snap.entity_triples = _extract_triples(e18_result)
            snap.engine_statuses["E18"] = _safe_status(e18)
        except Exception:
            log.exception("E18 (DataAnalysis) failed.")

    # ------------------------------------------------------------------
    # Step (f) — E19: Pattern Identification
    # ------------------------------------------------------------------
    e19 = engines.get(19)
    if e19 is not None:
        try:
            e19.update_neurochem_state(nt_state_lowercase)
            e19_input = {
                "nt_state": nt_state_lowercase,
                "text": bundle.raw_text,
                "tokens": tokens,
            }
            e19_result = e19.process(e19_input)
            snap.pattern_list = e19_result.get("new_patterns", []) + e19_result.get("confirmed_patterns", [])
            snap.engine_statuses["E19"] = _safe_status(e19)
        except Exception:
            log.exception("E19 (PatternIdentification) failed.")

    return snap


# ------------------------------------------------------------------
# Input builders
# ------------------------------------------------------------------

def _build_e23_input(
    bundle: InputBundle,
    tokenizer_result: Any,
    expansion_result: Any,
    stmm: Any,
) -> Any:
    """Build IntentionMapInput for E23."""
    from zados.cognitive_engines.py_engines.intention_map_engine import IntentionMapInput

    kwargs: Dict[str, Any] = {}
    if tokenizer_result is not None:
        kwargs["tokenizer_result"] = tokenizer_result
    if expansion_result is not None:
        kwargs["expansion_result"] = expansion_result

    # Affective signals from bundle emotion profile
    ep = bundle.emotion_profile
    if ep:
        # Estimate valence from profile: positive emotions > 0, negative < 0
        pos = sum(v for k, v in ep.items() if k in _POS_EMOTIONS)
        neg = sum(v for k, v in ep.items() if k in _NEG_EMOTIONS)
        total = pos + neg
        kwargs["emotional_valence"] = (pos - neg) / max(total, 0.01)
        kwargs["emotional_intensity"] = min(total / 3.0, 1.0)

    # Context signals from STMM
    if stmm is not None:
        ia = stmm.intention_analysis
        if ia.primary_intention:
            prev_vec = getattr(ia, "_prev_intent_vector", None)
            if prev_vec:
                kwargs["historical_intent"] = prev_vec

    return IntentionMapInput(**kwargs)


def _build_e11_input(
    bundle: InputBundle,
    tokens: List[str],
    stmm: Any,
    nt_state: Dict[str, float],
) -> Any:
    """Build IREPhase1Input for E11."""
    from zados.cognitive_engines.py_engines.input_relevance_evaluation_engine import (
        IREPhase1Input,
    )

    kwargs: Dict[str, Any] = {
        "current_text": bundle.raw_text,
        "tokens": tokens,
        "nt_levels": nt_state,
    }

    if stmm is not None:
        buf = stmm.active_message_buffer
        kwargs["stmm_user_messages"] = [
            m.text for m in getattr(buf, "messages", [])
            if getattr(m, "speaker", None) and str(getattr(m, "speaker", "")).endswith("USER")
        ]
        kwargs["stmm_system_responses"] = [
            m.text for m in getattr(buf, "messages", [])
            if getattr(m, "speaker", None) and str(getattr(m, "speaker", "")).endswith("SYSTEM")
        ]

    return IREPhase1Input(**kwargs)


def _build_e18_input(bundle: InputBundle, tokens: List[str]) -> Any:
    """Build DataAnalysisInput for E18."""
    from zados.cognitive_engines.py_engines.data_analysis_engine import DataAnalysisInput

    return DataAnalysisInput(
        raw_text=bundle.raw_text,
        tokens=tokens,
    )


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def _extract_triples(e18_result: Any) -> List:
    """Pull entity-relation-entity triples from DataAnalysisResult."""
    triples = []
    for rel in getattr(e18_result, "relations", []):
        src = getattr(rel, "source", "")
        tgt = getattr(rel, "target", "")
        rtype = getattr(rel, "relation_type", "")
        triples.append((src, rtype, tgt))
    return triples


def _ire_output_to_dict(result: Any) -> Dict[str, Any]:
    """Convert IREPhase1Output to a plain dict."""
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    return {}


def _tokens_to_items(tokens: List[str]) -> List[Dict[str, Any]]:
    """Convert a token list into E8-compatible item dicts."""
    return [{"item_id": f"tok_{i}", "metadata": {"text": t}} for i, t in enumerate(tokens)]


def _safe_status(engine: Any) -> Dict[str, Any]:
    """Safely call engine.get_status()."""
    try:
        return engine.get_status()
    except Exception:
        return {}


# Rough positive/negative sets for quick valence estimation
_POS_EMOTIONS = frozenset({
    "joy", "excited", "hopeful", "proud", "grateful", "content",
    "amused", "inspired", "connected", "belonging", "accepted",
    "thankful", "respected", "courageous", "interested", "curious",
})
_NEG_EMOTIONS = frozenset({
    "sad", "angry", "fearful", "anxious", "frustrated", "ashamed",
    "guilty", "rejected", "betrayal", "lonely", "jealous", "envious",
    "disgusted", "contempt", "worried", "nervous", "numb", "apathy",
    "boredom", "regret", "hopeless",
})
