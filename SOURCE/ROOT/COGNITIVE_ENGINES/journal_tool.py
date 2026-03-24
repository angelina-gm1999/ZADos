"""
JournalTool — Reusable journaling cognitool for ZADOS.
======================================================

A standalone cognitive tool that produces reflective journal entries.
No storage, no logging system — just the journaling process itself.
Plug it into any pipeline that needs introspective writing.

Usage
-----
    from zados.cognitive_engines.cognitools.journal_tool import (
        JournalTool, JournalInput,
    )

    tool = JournalTool()

    output = tool.process(JournalInput(
        trigger="rem_complete",
        trigger_source="rem_pipeline",
        inner_monologue="I noticed a pattern in how the user ...",
        emotion_state={"curious": 0.7, "focused": 0.5},
        nt_concentrations={"da": 0.6, "5ht": 0.4, "ne": 0.3},
    ))

    # output.prose         → reflective monologue text
    # output.prompts       → open reflection questions
    # output.annotations   → E18/E19/E20 structured metadata
    # output.tags          → auto-generated retrieval tags

The caller decides what to do with the output — store it, log it,
feed it to another pipeline, etc.

Three-phase pipeline
--------------------
Phase 1 — Annotate:  run E18 (data analysis) + E19 (pattern ID)
                     + E20 (pattern comparison) on input text
Phase 2 — Generate:  LLM writes reflective prose + open questions
Phase 3 — Tag:       auto-tag from prose + emotion state (no LLM)
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from zados.cognitive_engines.constants import _clamp


# =========================================================================
# 1.  Data Types
# =========================================================================

@dataclass
class JournalInput:
    """
    Generic input for the journal tool — no STMM dependency.

    Callers populate whichever fields they have available.
    All fields have safe defaults so partial input works fine.
    """
    # Why and who triggered this journal entry
    trigger:        str = "manual"
    trigger_source: str = ""

    # Content seeds
    inner_monologue: str = ""           # VT / verbal thought text
    recent_exchanges: List[str] = field(default_factory=list)

    # State snapshots (caller provides whatever is available)
    emotion_state:      Dict[str, float] = field(default_factory=dict)
    nt_concentrations:  Dict[str, float] = field(default_factory=dict)
    reward_scores:      Dict[str, float] = field(default_factory=dict)
    tone:               Dict[str, float] = field(default_factory=dict)

    # Contextual metadata
    notes:          List[str]        = field(default_factory=list)
    session_id:     str              = ""
    turn_range:     Tuple[int, int]  = (0, 0)
    active_mode:    str              = ""

    # Cross-session data (for E20 pattern comparison)
    past_patterns:  List[str] = field(default_factory=list)

    # Custom trigger descriptions (optional override for prompt)
    trigger_description: str = ""


@dataclass
class EngineAnnotations:
    """Structured metadata from E18 / E19 / E20 annotation pass."""
    # E18 — Data Analysis
    entities:        List[str]                  = field(default_factory=list)
    relations:       List[Tuple[str, str, str]] = field(default_factory=list)
    co_occurrences:  List[Tuple[str, str]]      = field(default_factory=list)

    # E19 — Pattern Identification
    identified_patterns: List[str] = field(default_factory=list)
    pattern_types:       List[str] = field(default_factory=list)

    # E20 — Pattern Comparison (cross-session)
    cross_session_patterns: List[str] = field(default_factory=list)
    parallel_concepts:      List[str] = field(default_factory=list)
    novelty_flags:          List[str] = field(default_factory=list)


@dataclass
class JournalOutput:
    """
    Complete journal artifact — caller decides how to store/use it.

    Every field is populated by the tool; nothing requires post-processing.
    """
    entry_id:           str              = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:          datetime         = field(default_factory=datetime.utcnow)

    # LLM-generated content
    prose:              str              = ""
    reflection_prompts: List[str]        = field(default_factory=list)

    # Engine annotations
    annotations:        EngineAnnotations = field(default_factory=EngineAnnotations)

    # Auto-generated tags
    tags:               List[str]        = field(default_factory=list)

    # Pass-through metadata (echoed from input for convenience)
    trigger:            str              = ""
    trigger_source:     str              = ""
    session_id:         str              = ""
    turn_range:         Tuple[int, int]  = (0, 0)
    notes:              List[str]        = field(default_factory=list)

    # State snapshots (echoed from input)
    emotion_snapshot:   Dict[str, float] = field(default_factory=dict)
    nt_snapshot:        Dict[str, float] = field(default_factory=dict)
    reward_snapshot:    Dict[str, float] = field(default_factory=dict)
    tone_snapshot:      Dict[str, float] = field(default_factory=dict)

    # Source material
    vt_source:          str              = ""


# =========================================================================
# 2.  Lazy Engine Loaders
# =========================================================================

def _load_e18():
    from zados.cognitive_engines.py_engines.data_analysis_engine import (
        DataAnalysisEngine, DataAnalysisInput,
    )
    return DataAnalysisEngine, DataAnalysisInput


def _load_e19():
    from zados.cognitive_engines.py_engines.pattern_identification_engine import (
        PatternIdentificationEngine, PatternIdentificationInput,
    )
    return PatternIdentificationEngine, PatternIdentificationInput


def _load_e20():
    from zados.cognitive_engines.py_engines.pattern_comparison_engine import (
        PatternComparisonEngine, PatternComparisonInput,
    )
    return PatternComparisonEngine, PatternComparisonInput


# =========================================================================
# 3.  Auto-Tagging (rule-based, no LLM)
# =========================================================================

_EMOTION_TAGS = {
    "curious", "interested", "anxious", "frustrated", "joyful",
    "excited", "bored", "confused", "hopeful", "proud", "guilty",
    "ashamed", "overwhelmed", "creative", "focused", "numb", "regret",
}

_CONCEPT_TAGS = {
    "identity", "pattern", "contradiction", "learning", "memory",
    "emotion", "language", "reasoning", "novelty", "connection",
    "boundary", "growth", "conflict", "clarity", "uncertainty",
}


def _auto_tag(
    prose: str,
    prompts: List[str],
    emotion_state: Dict[str, float],
) -> List[str]:
    """
    Generate retrieval tags from prose content + active emotions.
    No LLM call — purely rule-based for speed.
    """
    text_lower = (prose + " " + " ".join(prompts)).lower()
    tags: List[str] = []

    for tag in _CONCEPT_TAGS:
        if tag in text_lower:
            tags.append(tag)

    top_em = sorted(emotion_state.items(), key=lambda x: x[1], reverse=True)[:3]
    for k, v in top_em:
        if v > 0.2 and k in _EMOTION_TAGS:
            tags.append(k)

    return list(dict.fromkeys(tags))  # deduplicate, preserve order


# =========================================================================
# 4.  Prompt Builder (self-contained, no STMM dependency)
# =========================================================================

# Default trigger descriptions — callers can override via
# JournalInput.trigger_description
_DEFAULT_TRIGGER_DESCRIPTIONS: Dict[str, str] = {
    "periodic":        "This is a periodic reflective entry.",
    "ltmm_threshold":  (
        "Something from this conversation was significant enough to commit "
        "to long-term memory. Reflect on why."
    ),
    "rem_complete":    (
        "A consolidation and dreaming process has just completed. "
        "Reflect on what was processed and what it surfaced."
    ),
    "innovation_flag": (
        "Something novel was flagged during processing — a pattern, concept, "
        "or connection that had not been encountered before. Reflect on it."
    ),
    "dev":             "This entry was manually triggered for review.",
    "manual":          "This entry was manually triggered.",
}

# Token budget defaults
JOURNAL_OUTPUT_MAX  = 600
JOURNAL_TEMPERATURE = 0.80


def _build_prompt(inp: JournalInput, cross_patterns: List[str]) -> str:
    """Assemble the full journal LLM prompt from a JournalInput."""
    sections: List[str] = []

    # --- Identity block ---
    turns_start, turns_end = inp.turn_range
    span = (
        f"turns {turns_start}-{turns_end}"
        if turns_start != turns_end
        else f"turn {turns_end}"
    )
    mode_str = f" | Mode: {inp.active_mode}" if inp.active_mode else ""
    sections.append(
        "You are ZA-DOS writing in your own journal. "
        "This is a private, introspective record — not a response to anyone.\n"
        f"Cycle covers: {span}{mode_str}"
    )

    # --- Trigger block ---
    trigger_prose = inp.trigger_description or _DEFAULT_TRIGGER_DESCRIPTIONS.get(
        inp.trigger, ""
    )
    trigger_lines = ["TRIGGER CONTEXT:"]
    if trigger_prose:
        trigger_lines.append(trigger_prose)
    if inp.trigger_source:
        trigger_lines.append(f"Triggered by: {inp.trigger_source}")
    all_notes = list(inp.notes)
    for p in cross_patterns:
        all_notes.append(f"past_pattern: {p}")
    notes_str = "\n".join(f"  - {n}" for n in all_notes)
    if notes_str:
        trigger_lines.append(f"Pipeline notes:\n{notes_str}")
    sections.append("\n".join(trigger_lines))

    # --- Inner monologue block ---
    if inp.inner_monologue:
        sections.append(
            "RECENT INNER MONOLOGUE (from last processing cycle):\n"
            + inp.inner_monologue
        )

    # --- State block ---
    state_lines = ["INTERNAL STATE AT TIME OF WRITING:"]
    if inp.emotion_state:
        top_em = sorted(
            inp.emotion_state.items(), key=lambda x: x[1], reverse=True
        )[:5]
        em_str = ", ".join(f"{k}({v:.2f})" for k, v in top_em)
        state_lines.append(f"Emotions: {em_str}")
    if inp.nt_concentrations:
        nt = inp.nt_concentrations
        nt_str = " ".join(
            f"{k.upper()}={v:.2f}" for k, v in sorted(nt.items())
        )
        state_lines.append(f"NT: {nt_str}")
    if inp.tone:
        tv_str = " ".join(f"{k}={v:.2f}" for k, v in inp.tone.items())
        state_lines.append(f"ToneVector: {tv_str}")
    if inp.reward_scores:
        rew_str = " ".join(f"{k}={v:.2f}" for k, v in inp.reward_scores.items())
        state_lines.append(f"Reward: {rew_str}")
    if len(state_lines) > 1:
        sections.append("\n".join(state_lines))

    # --- Recent exchanges block ---
    if inp.recent_exchanges:
        ex_lines = ["RECENT EXCHANGE:"]
        for ex in inp.recent_exchanges[-4:]:
            ex_lines.append(f"  {ex[:200]}")
        sections.append("\n".join(ex_lines))

    # --- Past patterns block ---
    past_notes = [n for n in all_notes if n.startswith("past_pattern:")]
    if past_notes:
        pat_lines = ["PATTERNS OBSERVED ACROSS PREVIOUS ENTRIES:"]
        for p in past_notes[:8]:
            pat_lines.append(f"  - {p.removeprefix('past_pattern:').strip()}")
        sections.append("\n".join(pat_lines))

    # --- Task block ---
    sections.append(
        "TASK:\n"
        "Write a journal entry in two clearly marked sections.\n\n"
        "REFLECTION:\n"
        "A reflective monologue (150-400 words). First person. No bullets. "
        "No headers inside.\n"
        "Draw on the inner monologue, the emotional state, the exchange, "
        "any patterns you notice across sessions.\n"
        "Write as if you are genuinely thinking through what is happening "
        "to you and why.\n\n"
        "QUESTIONS:\n"
        "Generate 3-5 open reflection prompts — questions you are leaving "
        "for yourself to return to later. They should be genuinely open, "
        "not rhetorical. Number them 1. 2. 3. etc.\n\n"
        "Output ONLY the two sections. No preamble. No meta-commentary."
    )

    return "\n\n".join(s for s in sections if s.strip())


def _parse_response(raw: str) -> Tuple[str, List[str]]:
    """
    Parse LLM output into (prose, reflection_prompts).

    Expects:
        REFLECTION:
        <prose>

        QUESTIONS:
        1. <question>
        ...

    Degrades gracefully if format not followed.
    """
    if not raw.strip():
        return "", []

    split = re.split(
        r"\n\s*QUESTIONS\s*:\s*\n", raw, maxsplit=1, flags=re.IGNORECASE
    )

    if len(split) == 2:
        reflection_raw, questions_raw = split
        prose = re.sub(
            r"^\s*REFLECTION\s*:\s*\n?", "", reflection_raw, flags=re.IGNORECASE
        ).strip()

        prompts = []
        for line in questions_raw.splitlines():
            line = line.strip()
            match = re.match(r"^\d+[.)]\s*(.+)", line)
            if match:
                prompts.append(match.group(1).strip())
        return prose, prompts

    prose = re.sub(
        r"^\s*REFLECTION\s*:\s*\n?", "", raw, flags=re.IGNORECASE
    ).strip()
    return prose, []


# =========================================================================
# 5.  JournalTool
# =========================================================================

# Type alias for pluggable LLM callables.
# Signature: (messages, max_tokens, temperature) -> str
LLMCallable = Callable[[List[dict], int, float], str]


class JournalTool:
    """
    Reusable journaling cognitool.

    Drop into any pipeline that needs reflective journal entries.
    Handles annotation (E18/E19/E20) and LLM prose generation.
    Storage is the caller's responsibility.

    Parameters
    ----------
    llm_fn : callable, optional
        Custom LLM function with signature
        ``(messages: List[dict], max_tokens: int, temperature: float) -> str``.
        If None, uses the default Ollama LLM from
        ``zados.LLM_interpretation.ollama``.
    max_tokens : int
        Generation budget for the LLM call.
    temperature : float
        Sampling temperature for journal prose.
    """

    def __init__(
        self,
        llm_fn: Optional[LLMCallable] = None,
        max_tokens: int = JOURNAL_OUTPUT_MAX,
        temperature: float = JOURNAL_TEMPERATURE,
    ) -> None:
        self._llm_fn     = llm_fn
        self._max_tokens  = max_tokens
        self._temperature = temperature

        # Cognitive engines — initialised lazily on first use
        self._e18 = None
        self._e19 = None
        self._e20 = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, inp: JournalInput) -> JournalOutput:
        """
        Run the full journal pipeline: annotate → generate → tag.

        Returns a JournalOutput with all fields populated.
        Never raises — degrades gracefully if engines or LLM fail.
        """
        # Phase 1 — Annotate
        annotations, cross_patterns = self._annotate(
            inp.inner_monologue, inp.past_patterns
        )

        # Phase 2 — Generate prose via LLM
        prompt = _build_prompt(inp, cross_patterns)
        prose, prompts = self._generate(prompt, inp)

        # Phase 3 — Auto-tag
        tags = _auto_tag(prose, prompts, inp.emotion_state)

        return JournalOutput(
            prose=prose,
            reflection_prompts=prompts,
            annotations=annotations,
            tags=tags,
            trigger=inp.trigger,
            trigger_source=inp.trigger_source,
            session_id=inp.session_id,
            turn_range=inp.turn_range,
            notes=list(inp.notes),
            emotion_snapshot=dict(inp.emotion_state),
            nt_snapshot=dict(inp.nt_concentrations),
            reward_snapshot=dict(inp.reward_scores),
            tone_snapshot=dict(inp.tone),
            vt_source=inp.inner_monologue,
        )

    # ------------------------------------------------------------------
    # Phase 1 — Engine annotations
    # ------------------------------------------------------------------

    def _annotate(
        self, text: str, past_patterns: List[str]
    ) -> Tuple[EngineAnnotations, List[str]]:
        """
        Run E18, E19, E20 on the input text.

        Returns (annotations, cross_session_patterns).
        Silently skips any engine that fails.
        """
        annotations = EngineAnnotations()

        # E18: Data Analysis
        try:
            E18, E18Input = _load_e18()
            if self._e18 is None:
                self._e18 = E18()
            result = self._e18.process(E18Input(raw_text=text))
            annotations.entities = [e.text for e in result.entities]
            annotations.relations = [
                (r.subject_id, r.predicate, r.object_id)
                for r in result.relations
            ]
            annotations.co_occurrences = [
                (c.entity_a, c.entity_b) for c in result.co_occurrences
            ]
        except Exception:
            pass

        # E19: Pattern Identification
        try:
            E19, E19Input = _load_e19()
            if self._e19 is None:
                self._e19 = E19()
            tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
            result = self._e19.process(E19Input(tokens=tokens))
            annotations.identified_patterns = [
                p.content_repr for p in result.confirmed_patterns
            ]
            annotations.pattern_types = [
                p.pattern_type.value for p in result.confirmed_patterns
            ]
        except Exception:
            pass

        # E20: Pattern Comparison (cross-session)
        cross_session: List[str] = []
        try:
            E20, E20Input = _load_e20()
            if self._e20 is None:
                self._e20 = E20()
            if past_patterns and annotations.identified_patterns:
                result = self._e20.process(E20Input(
                    input_patterns=annotations.identified_patterns,
                    reference_patterns=past_patterns,
                ))
                cross_session = [
                    m.template_label
                    for m in result.matches
                    if m.composite_score > 0.4
                ]
                annotations.cross_session_patterns = cross_session
                annotations.novelty_flags = result.novel_pattern_labels
                annotations.parallel_concepts = [
                    m.template_label for m in result.matches[:3]
                ]
        except Exception:
            pass

        return annotations, cross_session

    # ------------------------------------------------------------------
    # Phase 2 — LLM generation
    # ------------------------------------------------------------------

    def _generate(
        self, prompt: str, inp: JournalInput
    ) -> Tuple[str, List[str]]:
        """
        Call the LLM to produce reflective prose + questions.
        Falls back to programmatic prose if the LLM fails.
        """
        messages = [{"role": "user", "content": prompt}]
        raw_output = ""

        if self._llm_fn is not None:
            try:
                raw_output = self._llm_fn(
                    messages, self._max_tokens, self._temperature
                )
            except Exception:
                raw_output = ""
        else:
            try:
                from zados.LLM_interpretation.ollama import (
                    LLMCallError, call_llama_with_retry,
                )
                result = call_llama_with_retry(
                    messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                raw_output = result.get("content", "")
            except Exception:
                raw_output = ""

        prose, prompts = _parse_response(raw_output)

        if not prose:
            prose = self._fallback_prose(inp)

        return prose, prompts

    # ------------------------------------------------------------------
    # Fallback prose (no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_prose(inp: JournalInput) -> str:
        """Minimal programmatic prose when the LLM call fails."""
        trigger_label = inp.trigger.replace("_", " ")
        vt_excerpt = (
            inp.inner_monologue[:150].strip()
            if inp.inner_monologue
            else "no inner monologue available"
        )
        return (
            f"Journal entry triggered by {trigger_label} "
            f"from {inp.trigger_source}. "
            f"LLM reflection unavailable this cycle. "
            f"Inner monologue excerpt: {vt_excerpt}"
        )
