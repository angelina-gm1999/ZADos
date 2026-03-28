"""
Cross-contrast helpers — align engine flags against memory contrast matches.

Extracts salient flags from Phase 3 engine results (E1 contradictions,
E5 biases, E19 patterns, E4 fallacies, E24 heuristics) and notes where
they align, diverge from, or extend matched memory entries.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from zados.core.thinking_blocks.types import MemoryCrossNote

log = logging.getLogger(__name__)


def extract_engine_flags(engine_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Extract salient flags from Phase 3 engine results.

    Returns a flat dict keyed by short engine label, containing the
    most important diagnostic outputs for the thinking pass.
    """
    flags: Dict[str, Any] = {}

    # E1 — contradiction detection
    e1 = engine_results.get(1, {})
    if e1:
        contradictions = e1.get("contradictions", e1.get("detected_contradictions", []))
        if contradictions:
            flags["e1_contradictions"] = [
                _coerce_str(c) for c in (contradictions[:3] if isinstance(contradictions, list) else [str(contradictions)])
            ]

    # E5 — bias detection
    e5 = engine_results.get(5, {})
    if e5:
        biases = e5.get("biases", e5.get("detected_biases", []))
        if biases:
            flags["e5_biases"] = [
                _coerce_str(b) for b in (biases[:3] if isinstance(biases, list) else [str(biases)])
            ]

    # E4 — fallacy detection
    e4 = engine_results.get(4, {})
    if e4:
        fallacies = e4.get("fallacies", e4.get("detected_fallacies", []))
        if fallacies:
            flags["e4_fallacies"] = [
                _coerce_str(f) for f in (fallacies[:3] if isinstance(fallacies, list) else [str(fallacies)])
            ]

    # E19 — pattern analysis
    e19 = engine_results.get(19, {})
    if e19:
        patterns = e19.get("patterns", e19.get("detected_patterns", []))
        if patterns:
            flags["e19_patterns"] = [
                _coerce_str(p) for p in (patterns[:3] if isinstance(patterns, list) else [str(patterns)])
            ]

    # E24 — heuristic bias
    e24 = engine_results.get(24, {})
    if e24:
        heuristics = e24.get("heuristics", e24.get("detected_heuristics", []))
        if heuristics:
            flags["e24_heuristics"] = [
                _coerce_str(h) for h in (heuristics[:3] if isinstance(heuristics, list) else [str(heuristics)])
            ]

    # E7 — simulated opposition
    e7 = engine_results.get(7, {})
    if e7:
        opposition = e7.get("opposition_summary", e7.get("challenge", ""))
        if opposition:
            flags["e7_opposition"] = _coerce_str(opposition)

    # E14 — socratic reasoning
    e14 = engine_results.get(14, {})
    if e14:
        questions = e14.get("questions", e14.get("socratic_questions", []))
        if questions:
            flags["e14_socratic"] = [
                _coerce_str(q) for q in (questions[:2] if isinstance(questions, list) else [str(questions)])
            ]

    return flags


def build_cross_contrast_notes(
    engine_flags: Dict[str, Any],
    memory_matches: List[Dict[str, Any]],
) -> List[MemoryCrossNote]:
    """Build cross-contrast notes between engine flags and memory matches.

    For each engine flag type, checks memory matches for confirmation,
    divergence, or extension.

    Returns a list of MemoryCrossNote objects (max 8 total).
    """
    notes: List[MemoryCrossNote] = []

    _FLAG_TO_ENGINE: Dict[str, str] = {
        "e1_contradictions": "E1",
        "e5_biases": "E5",
        "e4_fallacies": "E4",
        "e19_patterns": "E19",
        "e24_heuristics": "E24",
        "e7_opposition": "E7",
        "e14_socratic": "E14",
    }

    for flag_key, engine_id in _FLAG_TO_ENGINE.items():
        if flag_key not in engine_flags:
            continue
        flag_value = engine_flags[flag_key]
        flag_items = flag_value if isinstance(flag_value, list) else [str(flag_value)]

        for flag_detail in flag_items[:2]:
            for mem in memory_matches[:4]:
                mem_summary = mem.get("summary", mem.get("content_summary", ""))
                mem_id = mem.get("packet_id", mem.get("entry_id", mem.get("id", "")))
                if not mem_summary:
                    continue
                relation = _infer_relation(flag_key, flag_detail, mem_summary)
                if relation:
                    notes.append(MemoryCrossNote(
                        engine_id=engine_id,
                        flag_type=flag_key,
                        flag_detail=str(flag_detail)[:200],
                        memory_match_id=mem_id,
                        memory_summary=mem_summary[:200],
                        relation=relation,
                    ))
                    if len(notes) >= 8:
                        return notes
    return notes


def _infer_relation(flag_key: str, flag_detail: str, mem_summary: str) -> str:
    """Infer alignment/divergence relation between flag and memory.

    Heuristic: look for keyword overlap.  Returns "" if no meaningful
    relation can be inferred.
    """
    flag_words = set(_tokenize(flag_detail))
    mem_words = set(_tokenize(mem_summary))
    overlap = len(flag_words & mem_words)

    if overlap == 0:
        return ""

    # Contradictions / fallacies / biases tend to diverge
    if flag_key in ("e1_contradictions", "e5_biases", "e4_fallacies"):
        return "diverges" if overlap >= 2 else "extends"
    # Patterns and heuristics confirm or extend
    if flag_key in ("e19_patterns", "e24_heuristics"):
        return "confirms" if overlap >= 3 else "extends"
    return "extends"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer, lowercased, stops removed."""
    _STOPS = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
               "to", "and", "or", "that", "this", "it", "be"}
    return [w for w in text.lower().split() if len(w) > 3 and w not in _STOPS]


def _coerce_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "__dict__"):
        return str({k: v for k, v in obj.__dict__.items() if not k.startswith("_")})
    return str(obj)
