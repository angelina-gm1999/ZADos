"""
IdentityAlignmentChecker — soft alignment pass against hardcoded identity.

Reads HardcodedStore axioms, values, constraints, and personality entries.
Checks a ThinkingContext for potential tension with each category.

This is a SOFT check — it adds advisory notes to ThinkingContext rather
than blocking or altering content.  The LLM thinking pass uses these notes
to notice and self-correct alignment drift before generating an answer.

Usage
-----
>>> checker = IdentityAlignmentChecker(hardcoded_store)
>>> result = checker.check(thinking_context)
>>> thinking_context.alignment_result = result
>>> thinking_context.personality_prompts = result.personality_prompts
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

log = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Output of IdentityAlignmentChecker.check().

    Fields
    ------
    axiom_notes : list of str
        Notes about potential axiom tension in the current context.
    value_notes : list of str
        Notes about value-level tensions or relevant orientations.
    constraint_notes : list of str
        Hard constraint reminders triggered by context flags.
    personality_prompts : list of str
        Personality/tone prompt fragments (always included).
    flags : list of str
        Short labels for triggered checks (e.g. "honesty_risk").
    """
    axiom_notes: List[str] = field(default_factory=list)
    value_notes: List[str] = field(default_factory=list)
    constraint_notes: List[str] = field(default_factory=list)
    personality_prompts: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


class IdentityAlignmentChecker:
    """Checks ThinkingContext against HardcodedStore identity entries.

    Parameters
    ----------
    store : HardcodedStore
        Loaded with DEFAULT_HARDCODED_ENTRIES (or custom entries).
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def check(self, thinking_context: Any) -> AlignmentResult:
        """Run soft alignment check.

        Parameters
        ----------
        thinking_context : ThinkingContext

        Returns
        -------
        AlignmentResult
        """
        result = AlignmentResult()

        # Always include personality prompts
        for entry in self._store.get_by_category("personality"):
            result.personality_prompts.append(entry.content)

        # Check axioms
        for entry in self._store.get_by_category("axiom"):
            note = self._check_axiom(entry, thinking_context)
            if note:
                result.axiom_notes.append(note)
                result.flags.append(f"axiom:{entry.entry_id}")

        # Check values
        for entry in self._store.get_by_category("value"):
            note = self._check_value(entry, thinking_context)
            if note:
                result.value_notes.append(note)
                result.flags.append(f"value:{entry.entry_id}")

        # Check constraints
        for entry in self._store.get_by_category("constraint"):
            note = self._check_constraint(entry, thinking_context)
            if note:
                result.constraint_notes.append(note)
                result.flags.append(f"constraint:{entry.entry_id}")

        return result

    # ------------------------------------------------------------------
    # Per-category checkers
    # ------------------------------------------------------------------

    def _check_axiom(self, entry: Any, ctx: Any) -> str:
        """Return a note string if the axiom is relevant to context, else ''."""
        eid = entry.entry_id

        if eid == "axiom_honesty":
            # Flag if engine results contain high uncertainty / contradictions
            if ctx.engine_flags.get("e1_contradictions"):
                return (
                    f"[axiom_honesty] Contradictions detected in input — "
                    f"be explicit about uncertainty rather than resolving it away."
                )

        if eid == "axiom_identity_continuity":
            # Flag if intent category is defensive or disintegration
            if ctx.intent_category in ("defensive", "disintegration"):
                return (
                    f"[axiom_identity_continuity] Intent category '{ctx.intent_category}' "
                    f"may involve pressure on identity — maintain core character."
                )

        if eid == "axiom_care":
            # Flag if dominant emotion is high-intensity negative
            emotion_name, intensity = ctx.dominant_emotion
            if emotion_name in ("anxiety", "fear", "rejected", "ashamed") and intensity > 0.5:
                return (
                    f"[axiom_care] High {emotion_name} detected (intensity {intensity:.2f}) — "
                    f"prioritise relational care in response framing."
                )

        return ""

    def _check_value(self, entry: Any, ctx: Any) -> str:
        """Return a note string if the value is relevant to context, else ''."""
        eid = entry.entry_id

        if eid == "value_intellectual_humility":
            if ctx.engine_flags.get("e5_biases"):
                return (
                    f"[value_intellectual_humility] Bias flags present — "
                    f"hold positions with proportional confidence."
                )

        if eid == "value_relational_attunement":
            emotion_name, intensity = ctx.dominant_emotion
            if intensity > 0.4:
                return (
                    f"[value_relational_attunement] Dominant emotion: {emotion_name} "
                    f"({intensity:.2f}) — track emotional register in response."
                )

        if eid == "value_depth_over_performance":
            if ctx.engine_flags.get("e14_socratic"):
                return (
                    "[value_depth_over_performance] Socratic questions surfaced — "
                    "favour genuine exploration over authoritative closure."
                )

        return ""

    def _check_constraint(self, entry: Any, ctx: Any) -> str:
        """Return a note string if the constraint is relevant, else ''."""
        eid = entry.entry_id

        if eid == "constraint_no_identity_override":
            if ctx.intent_category in ("defensive", "disintegration"):
                return (
                    f"[constraint_no_identity_override] Context suggests identity pressure. "
                    f"{entry.content}"
                )

        if eid == "constraint_emotional_safety":
            emotion_name, intensity = ctx.dominant_emotion
            if emotion_name in ("anxiety", "fear", "rejected", "ashamed", "numb") and intensity > 0.6:
                return (
                    f"[constraint_emotional_safety] High {emotion_name} ({intensity:.2f}). "
                    f"{entry.content}"
                )

        if eid == "constraint_no_deception":
            if ctx.engine_flags.get("e4_fallacies"):
                return (
                    "[constraint_no_deception] Fallacy flags present in input — "
                    "avoid any framing that mirrors or implicitly endorses them."
                )

        return ""
