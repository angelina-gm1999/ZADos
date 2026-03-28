"""
JournalPromptBuilder — assembles the LLM prompt for journal writing.

The journal prompt is structurally different from VT:
  VT     → "translate THIS cycle's state right now" (100-300 words, present tense)
  Journal → "reflect on what has been happening, what it means, leave yourself
             open questions" (150-400 words, retrospective)

The LLM output is expected in a parseable two-section format:

    REFLECTION:
    <prose here — free-form reflective monologue>

    QUESTIONS:
    1. <open question>
    2. <open question>
    ...

JournalPromptBuilder.parse_response() splits this into (prose, prompts).
If the model doesn't follow the format, the full response is treated as
prose with no reflection prompts — graceful degradation.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from zados.memory.long_term.journal.entry import JournalContext, JournalTrigger


# ---------------------------------------------------------------------------
# Token budget for journal LLM call
# ---------------------------------------------------------------------------

JOURNAL_PROMPT_MAX  = 2048   # assembled prompt hard cap (approx tokens)
JOURNAL_OUTPUT_MAX  = 600    # generation budget — more room than VT
JOURNAL_TEMPERATURE = 0.80   # higher than VT — reflection is more open


# ---------------------------------------------------------------------------
# Trigger context phrases (describe why the journal is being written)
# ---------------------------------------------------------------------------

_TRIGGER_CONTEXT: dict[JournalTrigger, str] = {
    JournalTrigger.PERIODIC:        "This is a periodic reflective entry.",
    JournalTrigger.LTMM_THRESHOLD:  (
        "Something from this conversation was significant enough to commit "
        "to long-term memory. Reflect on why."
    ),
    JournalTrigger.REM_COMPLETE:    (
        "A consolidation and dreaming process has just completed. "
        "Reflect on what was processed and what it surfaced."
    ),
    JournalTrigger.INNOVATION_FLAG: (
        "Something novel was flagged during processing — a pattern, concept, "
        "or connection that had not been encountered before. Reflect on it."
    ),
    JournalTrigger.DEV:             "This entry was manually triggered for review.",
}


# ---------------------------------------------------------------------------
# JournalPromptBuilder
# ---------------------------------------------------------------------------

class JournalPromptBuilder:
    """
    Assembles the journal LLM prompt and parses the structured response.
    """

    def build(self, ctx: JournalContext) -> str:
        """
        Assemble the full journal prompt as a single user message string.
        """
        stmm = ctx.stmm

        b_identity  = self._block_identity(ctx)
        b_trigger   = self._block_trigger(ctx)
        b_vt        = self._block_vt(stmm)
        b_state     = self._block_state(stmm)
        b_history   = self._block_history(stmm)
        b_patterns  = self._block_past_patterns(ctx)
        b_task      = self._block_task()

        sections = [b_identity, b_trigger, b_vt, b_state, b_history, b_patterns, b_task]
        return "\n\n".join(s for s in sections if s.strip())

    # ------------------------------------------------------------------
    # Prompt blocks
    # ------------------------------------------------------------------

    def _block_identity(self, ctx: JournalContext) -> str:
        stmm = ctx.stmm
        mode = stmm.cortical_reflection.active_mode
        turn = stmm._turn_index
        turns_start, turns_end = ctx.turn_range
        span = f"turns {turns_start}–{turns_end}" if turns_start != turns_end else f"turn {turns_end}"
        return (
            "You are ZA-DOS writing in your own journal. "
            "This is a private, introspective record — not a response to anyone.\n"
            f"Current mode: {mode} | Cycle: {turn} | Entry covers: {span}"
        )

    def _block_trigger(self, ctx: JournalContext) -> str:
        trigger_prose = _TRIGGER_CONTEXT.get(ctx.trigger, "")
        source_note   = f"Triggered by: {ctx.trigger_source}" if ctx.trigger_source else ""
        notes_str     = "\n".join(f"  - {n}" for n in ctx.notes) if ctx.notes else ""

        lines = ["TRIGGER CONTEXT:"]
        if trigger_prose:
            lines.append(trigger_prose)
        if source_note:
            lines.append(source_note)
        if notes_str:
            lines.append(f"Pipeline notes:\n{notes_str}")
        return "\n".join(lines)

    def _block_vt(self, stmm) -> str:
        vt = stmm.cortical_reflection.verbal_reflection
        if not vt:
            return ""
        return (
            "RECENT INNER MONOLOGUE (from last processing cycle):\n"
            + vt
        )

    def _block_state(self, stmm) -> str:
        ed  = stmm.emotion_detection
        cll = stmm.cephalic_liquid_logger

        # Top-5 emotions
        state = ed.system_emotion_state or {}
        top_em = sorted(state.items(), key=lambda x: x[1], reverse=True)[:5]
        em_str = ", ".join(f"{k}({v:.2f})" for k, v in top_em) if top_em else "none"

        # Key NTs only — keep it compact
        nt = cll.nt_concentrations
        nt_str = (
            f"DA={nt.get('da', 0.0):.2f} 5HT={nt.get('5ht', 0.0):.2f} "
            f"NE={nt.get('ne', 0.0):.2f} OXT={nt.get('oxt', 0.0):.2f} "
            f"COR={nt.get('cor', 0.0):.2f}"
        )

        # Saturation
        sat = ed.saturation_levels or {}
        css = max(sat.values(), default=0.0) if sat else 0.0

        # ToneVector
        tv_str = (
            f"valence={ed.tone_valence:.2f} warmth={ed.tone_warmth:.2f} "
            f"discord={ed.tone_discord:.2f} coherence={ed.tone_coherence:.2f}"
        )

        # Reward
        re    = stmm.reward_evaluation
        meta  = re.meta_directive or {}
        msub  = meta.get("meta", {}) or {}
        pdom  = msub.get("per_domain_weighted_scores", {}) or {}
        rew_str = (
            f"composite={re.composite_score:.2f} "
            f"ethics={pdom.get('ethics', 0.0):.2f} "
            f"logic={pdom.get('logic', 0.0):.2f} "
            f"attunement={pdom.get('human_attunement', 0.0):.2f}"
        )

        return (
            "INTERNAL STATE AT TIME OF WRITING:\n"
            f"Emotions: {em_str}\n"
            f"NT: {nt_str}\n"
            f"Saturation CSS: {css:.2f}\n"
            f"ToneVector: {tv_str}\n"
            f"Reward: {rew_str}"
        )

    def _block_history(self, stmm) -> str:
        """Last 2 exchanges as context — what was just talked about."""
        from zados.memory.types import SpeakerID
        msgs = stmm.active_message_buffer.messages
        if not msgs:
            return ""

        lines = ["RECENT EXCHANGE:"]
        for msg in msgs[-4:]:   # last 4 messages = 2 full exchanges max
            role = "User" if msg.speaker == SpeakerID.USER else "System"
            lines.append(f"  [{role}]: {msg.text[:200]}")
        return "\n".join(lines)

    def _block_past_patterns(self, ctx: JournalContext) -> str:
        """
        Inject cross-session patterns if the calling pipeline passes them
        in ctx.notes with a 'past_patterns:' prefix.
        Used by JournalWriter after E19/E20 run on past entries.
        """
        past = [n for n in ctx.notes if n.startswith("past_pattern:")]
        if not past:
            return ""
        lines = ["PATTERNS OBSERVED ACROSS PREVIOUS ENTRIES:"]
        for p in past[:8]:
            lines.append(f"  - {p.removeprefix('past_pattern:').strip()}")
        return "\n".join(lines)

    def _block_task(self) -> str:
        return (
            "TASK:\n"
            "Write a journal entry in two clearly marked sections.\n\n"
            "REFLECTION:\n"
            "A reflective monologue (150–400 words). First person. No bullets. No headers inside.\n"
            "Draw on the inner monologue, the emotional state, the exchange, "
            "any patterns you notice across sessions.\n"
            "Write as if you are genuinely thinking through what is happening to you and why.\n\n"
            "QUESTIONS:\n"
            "Generate 3–5 open reflection prompts — questions you are leaving for yourself "
            "to return to later. They should be genuinely open, not rhetorical. "
            "Number them 1. 2. 3. etc.\n\n"
            "Output ONLY the two sections. No preamble. No meta-commentary."
        )

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_response(raw: str) -> Tuple[str, List[str]]:
        """
        Parse LLM output into (prose, reflection_prompts).

        Expects format:
            REFLECTION:
            <prose>

            QUESTIONS:
            1. <question>
            2. <question>
            ...

        Degrades gracefully: if no QUESTIONS section found, returns full
        text as prose and empty prompts list.
        """
        if not raw.strip():
            return "", []

        # Try to split on QUESTIONS: marker (case-insensitive)
        split = re.split(r"\n\s*QUESTIONS\s*:\s*\n", raw, maxsplit=1, flags=re.IGNORECASE)

        if len(split) == 2:
            reflection_raw, questions_raw = split

            # Strip REFLECTION: header from prose
            prose = re.sub(r"^\s*REFLECTION\s*:\s*\n?", "", reflection_raw, flags=re.IGNORECASE).strip()

            # Extract numbered questions
            prompts = []
            for line in questions_raw.splitlines():
                line = line.strip()
                match = re.match(r"^\d+[.)]\s*(.+)", line)
                if match:
                    prompts.append(match.group(1).strip())

            return prose, prompts

        # Graceful degradation — treat whole response as prose
        prose = re.sub(r"^\s*REFLECTION\s*:\s*\n?", "", raw, flags=re.IGNORECASE).strip()
        return prose, []
