"""
JournalWriter — the journal plugin (LTMM adapter).

Thin adapter that bridges the STMM-based LTMM journal pipeline to
the reusable :class:`JournalTool` cognitool.

Pipeline usage
--------------
    journal = JournalWriter(store)

    journal.write(JournalContext(
        trigger=JournalTrigger.REM_COMPLETE,
        trigger_source="rem_pipeline",
        stmm=stmm,
        turn_range=(first_turn, stmm._turn_index),
        session_id="session_42",
    ))

Internally, JournalWriter:
  1. Converts STMM + JournalContext → JournalInput  (generic)
  2. Calls JournalTool.process(input) → JournalOutput (generic)
  3. Converts JournalOutput → JournalEntry            (LTMM-specific)
  4. Persists via JournalStore
"""
from __future__ import annotations

from typing import Optional

from zados.cognitive_engines.cognitools.journal_tool import (
    JournalInput,
    JournalTool,
)
from zados.memory.long_term.journal.entry import (
    EngineAnnotations,
    JournalContext,
    JournalEntry,
    ReviewStatus,
)
from zados.memory.long_term.journal.store import JournalStore


class JournalWriter:
    """
    Plugin object.  Instantiate once per session; reuse across pipelines.

    Parameters
    ----------
    store : JournalStore
        The shared journal store.  Must be the same instance across the
        session so cross-entry pattern comparison works correctly.
    """

    def __init__(self, store: JournalStore) -> None:
        self._store = store
        self._tool  = JournalTool()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write(self, ctx: JournalContext) -> Optional[JournalEntry]:
        """
        Full journal write: convert → process → persist.

        Returns the completed JournalEntry.
        """
        # ---- Convert STMM context → generic JournalInput ----
        inp = self._context_to_input(ctx)

        # ---- Run the journaling tool ----
        output = self._tool.process(inp)

        # ---- Convert JournalOutput → JournalEntry ----
        annotations = EngineAnnotations(
            entities=output.annotations.entities,
            relations=output.annotations.relations,
            co_occurrences=output.annotations.co_occurrences,
            identified_patterns=output.annotations.identified_patterns,
            pattern_types=output.annotations.pattern_types,
            cross_session_patterns=output.annotations.cross_session_patterns,
            parallel_concepts=output.annotations.parallel_concepts,
            novelty_flags=output.annotations.novelty_flags,
        )

        entry = JournalEntry(
            entry_id=output.entry_id,
            timestamp=output.timestamp,
            session_id=ctx.session_id,
            turn_range=ctx.turn_range,
            trigger=ctx.trigger,
            trigger_source=ctx.trigger_source,
            prose=output.prose,
            reflection_prompts=output.reflection_prompts,
            vt_source=output.vt_source,
            annotations=annotations,
            emotion_snapshot=output.emotion_snapshot,
            nt_snapshot=output.nt_snapshot,
            reward_snapshot=output.reward_snapshot,
            tone_snapshot=output.tone_snapshot,
            tags=output.tags,
            pipeline_notes=list(ctx.notes),
        )

        # ---- Link to recent related entries ----
        self._link_related(entry)

        # ---- Persist ----
        self._store.write(entry)
        return entry

    # ------------------------------------------------------------------
    # STMM → JournalInput conversion
    # ------------------------------------------------------------------

    def _context_to_input(self, ctx: JournalContext) -> JournalInput:
        """Extract all relevant data from STMM into a generic JournalInput."""
        stmm = ctx.stmm

        # Snapshots
        ed   = stmm.emotion_detection
        cll  = stmm.cephalic_liquid_logger
        re_  = stmm.reward_evaluation
        meta = re_.meta_directive or {}
        msub = meta.get("meta", {}) or {}

        emotion_state = dict(ed.system_emotion_state or {})
        nt_concentrations = dict(cll.nt_concentrations or {})
        reward_scores = dict(msub.get("per_domain_weighted_scores", {}) or {})
        tone = {
            "valence":   ed.tone_valence,
            "warmth":    ed.tone_warmth,
            "discord":   ed.tone_discord,
            "coherence": ed.tone_coherence,
        }

        # Inner monologue
        vt_text = stmm.cortical_reflection.verbal_reflection or ""

        # Recent exchanges
        recent_exchanges = []
        try:
            from zados.memory.types import SpeakerID
            msgs = stmm.active_message_buffer.messages
            for msg in msgs[-4:]:
                role = "User" if msg.speaker == SpeakerID.USER else "System"
                recent_exchanges.append(f"[{role}]: {msg.text[:200]}")
        except Exception:
            pass

        # Cross-session patterns from store
        past_patterns = self._store.get_all_patterns()

        # Active mode
        active_mode = stmm.cortical_reflection.active_mode or ""

        return JournalInput(
            trigger=ctx.trigger.value,
            trigger_source=ctx.trigger_source,
            inner_monologue=vt_text,
            recent_exchanges=recent_exchanges,
            emotion_state=emotion_state,
            nt_concentrations=nt_concentrations,
            reward_scores=reward_scores,
            tone=tone,
            notes=list(ctx.notes),
            session_id=ctx.session_id,
            turn_range=ctx.turn_range,
            active_mode=active_mode,
            past_patterns=past_patterns,
        )

    # ------------------------------------------------------------------
    # Entry linking
    # ------------------------------------------------------------------

    def _link_related(self, entry: JournalEntry) -> None:
        """
        Search past entries for semantic relatives and link bidirectionally.
        Only considers the 20 most recent entries to keep it cheap.
        """
        if len(self._store) == 0:
            return

        query = entry.prose[:300]
        candidates = self._store.search(query, limit=3)
        for score, past_entry in candidates:
            if score > 0.35 and past_entry.entry_id != entry.entry_id:
                self._store.link_entries(entry.entry_id, past_entry.entry_id)
