"""
ThinkingBlockBuilder — assembles ThinkingContext from pipeline state.

Called after Phase 3 completes (and after Phase 2 NT modulation), before
Phase 4 (VT / LLM pass 1).

Gathers:
  - Phase 3 engine result flags (compressed)
  - Memory contrast matches from STMM (both flat + scoped passes)
  - Cross-contrast notes (engine flags × memory matches)
  - Last 2 MTMM conversation turns
  - Held thinking blocks pulled from LTMM thoughts/held_blocks
  - Mission briefing from bundle / session
  - Current reward profile, intent category, dominant emotion, NT snapshot
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class ThinkingBlockBuilder:
    """Assembles a ThinkingContext from PipelineState + STMM + session."""

    def build(
        self,
        state: Any,           # PipelineState
        stmm: Any,            # STMMStore
        session: Any,         # SessionState
        ltmm: Any = None,     # MemoryLayer (for held blocks query)
    ) -> Any:
        """Build and return a ThinkingContext.

        Parameters
        ----------
        state : PipelineState
        stmm : STMMStore
        session : SessionState
        ltmm : memory layer, optional
            Used to query unreviewed held thinking blocks.

        Returns
        -------
        ThinkingContext
        """
        from zados.core.thinking_blocks.types import (
            ThinkingContext,
            ConversationTurn,
            HeldBlock,
        )
        from zados.core.thinking_blocks.cross_contrast import (
            extract_engine_flags,
            build_cross_contrast_notes,
        )

        ctx = ThinkingContext()

        # Mission briefing
        bundle = state.bundle
        ctx.mission_briefing = str(
            getattr(bundle, "mission_briefing", None)
            or getattr(session, "mission_briefing", None)
            or ""
        )

        # Reward profile + intent
        ctx.reward_profile_name = (
            state.modulation.reward_profile_name
            if state.modulation
            else getattr(session, "reward_profile_name", "regular_input")
        )

        if state.perception and state.perception.intent_result:
            ir = state.perception.intent_result
            if hasattr(ir, "intent_category"):
                cat = ir.intent_category
                ctx.intent_category = cat.value.lower() if hasattr(cat, "value") else str(cat).lower()
            elif hasattr(ir, "dominant_intent"):
                ctx.intent_category = str(ir.dominant_intent).lower()

        # NT snapshot
        ctx.nt_snapshot = (
            state.modulation.nt_snapshot if state.modulation else {}
        )

        # Dominant emotion (from extractor result or E28)
        if state.modulation and getattr(state.modulation, "extractor_result", None):
            ctx.dominant_emotion = state.modulation.extractor_result.dominant_emotion
        elif state.dispatch and state.dispatch.e28_result:
            e28 = state.dispatch.e28_result
            if hasattr(e28, "detected_emotions") and e28.detected_emotions:
                de = max(
                    e28.detected_emotions,
                    key=lambda x: getattr(x, "intensity", 0.0),
                )
                ctx.dominant_emotion = (
                    getattr(de, "emotion_name", getattr(de, "name", "none")),
                    float(getattr(de, "intensity", 0.0)),
                )

        # Engine flags
        engine_results: Dict[int, Any] = {}
        if state.dispatch:
            engine_results = state.dispatch.engine_results
        ctx.engine_flags = extract_engine_flags(engine_results)

        # Memory matches from STMM
        ctx.memory_matches = _collect_memory_matches(stmm)

        # Cross-contrast notes
        ctx.cross_contrast_notes = build_cross_contrast_notes(
            ctx.engine_flags,
            ctx.memory_matches,
        )

        # Last 2 MTMM conversation turns
        ctx.recent_turns = _collect_recent_turns(stmm, n=2)

        # Held thinking blocks
        ctx.held_blocks = _collect_held_blocks(ltmm, stmm)

        return ctx


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _collect_memory_matches(stmm: Any) -> List[Dict[str, Any]]:
    """Extract matched memory entries from stmm.memory_contrast."""
    try:
        mc = stmm.memory_contrast
        matches = []
        for m in getattr(mc, "matched_entries", []):
            matches.append({
                "entry_id": getattr(m, "entry_id", ""),
                "source_tier": getattr(m, "source_tier", ""),
                "similarity": getattr(m, "similarity", 0.0),
                "content_summary": getattr(m, "content_summary", ""),
            })
        return matches
    except Exception:
        return []


def _collect_recent_turns(stmm: Any, n: int = 2) -> List[Any]:
    """Collect last n conversation turns from STMM."""
    from zados.core.thinking_blocks.types import ConversationTurn
    turns = []
    try:
        tracker = stmm.brain_process_tracker
        # Try conversation_history first
        history = getattr(tracker, "conversation_history", [])
        if not history:
            # Fall back to user_messages + assistant_messages interleaved
            user_msgs = getattr(stmm, "user_messages", [])
            asst_msgs = getattr(stmm, "assistant_messages", [])
            for i, (u, a) in enumerate(zip(reversed(user_msgs[-n:]), reversed(asst_msgs[-n:]))):
                turns.append(ConversationTurn(role="user", text=str(u), turn_index=i))
                turns.append(ConversationTurn(role="assistant", text=str(a), turn_index=i))
            return turns[:n * 2]
        for i, entry in enumerate(history[-n:]):
            role = entry.get("role", "user") if isinstance(entry, dict) else getattr(entry, "role", "user")
            text = entry.get("text", entry.get("content", "")) if isinstance(entry, dict) else getattr(entry, "text", "")
            turns.append(ConversationTurn(role=role, text=str(text), turn_index=i))
    except Exception:
        pass
    return turns


def _collect_held_blocks(ltmm: Any, stmm: Any) -> List[Any]:
    """Query unreviewed held thinking blocks from LTMM thoughts/held_blocks."""
    from zados.core.thinking_blocks.types import HeldBlock
    blocks = []

    if ltmm is None:
        return blocks

    try:
        # Try namespace store for thoughts/held_blocks
        store = None
        if hasattr(ltmm, "namespace_store"):
            store = ltmm.namespace_store
        elif hasattr(ltmm, "get_namespace_store"):
            store = ltmm.get_namespace_store("thoughts/held_blocks")

        if store is None:
            return blocks

        # Search for unreviewed blocks
        results = []
        if hasattr(store, "search_folder"):
            results = store.search_folder(
                "thoughts/held_blocks",
                tags_required=frozenset(),
                tags_excluded=frozenset({"reviewed"}),
                max_results=5,
            )
        elif hasattr(store, "search"):
            results = store.search(
                query="",
                folder="thoughts/held_blocks",
                max_results=5,
                exclude_tags=["reviewed"],
            )

        for entry in results:
            content = getattr(entry, "content", "") or str(entry.get("content", "")) if isinstance(entry, dict) else ""
            tags = getattr(entry, "tags", []) if not isinstance(entry, dict) else entry.get("tags", [])
            trigger = ""
            metadata = getattr(entry, "metadata", {}) if not isinstance(entry, dict) else entry.get("metadata", {})
            if isinstance(metadata, dict):
                trigger = metadata.get("trigger_summary", metadata.get("trigger", ""))

            eid = getattr(entry, "entry_id", "") if not isinstance(entry, dict) else entry.get("entry_id", "")
            blocks.append(HeldBlock(
                block_id=str(eid),
                content=str(content)[:500],
                trigger_summary=str(trigger)[:200],
                tags=list(tags) if tags else [],
            ))

    except Exception:
        log.debug("Failed to collect held thinking blocks from LTMM.")

    return blocks
