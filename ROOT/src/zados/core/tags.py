"""
ZADOS Tag Taxonomy — centralized tag namespace and builder functions.

All MemoryPacket.flags, InputBundle.context_flags, JournalEntry.tags, and
TimeContextSnapshot.flags should use tags built by the helpers in this module
so the entire architecture shares a consistent, searchable label space.

Namespace overview
------------------
  pipeline:*   Origin pipeline (regular_input, rem, dream, homework, etc.)
  mode:*        Operational mode token (normal, learning, rem, dream, ...)
  intent:*      Primary user intention (question, assertion, command, ...)
  signal:*      Learning / emotional signal (frustration, curiosity, ...)
  reward:*      Reward domain strength label (logic_high, ethics_low, ...)
  mem:*         Memory salience marker (high_significance, ltmm_promoted, ...)
  flag:*        Operational event flag (contradiction, paradox, innovation, ...)
  content:*     Content type classifier (academic, creative, ethical, ...)
  origin:*      Provenance tag for unsolved concepts (academic, identity, ...)

Pre-existing namespaces (defined elsewhere, listed here for completeness)
  time:*         morning / afternoon / evening / night      (time_context.py)
  day:*          monday – sunday                            (time_context.py)
  circadian:*    waking / active / wind_down / sleep        (time_context.py)
  elapsed:*      <seconds>s                                 (time_context.py)
  dream_signal:* curiosity / confusion / wonder / perplexed (dream pipeline)
  emphasis:*     <engine_name>                              (homework pipeline)
  axiom:*        <entry_id>                                 (identity alignment)
  value:*        <entry_id>                                 (identity alignment)
  constraint:*   <entry_id>                                 (identity alignment)

Usage
-----
    from zados.core.tags import T

    T.pipeline("rem")          # → "pipeline:rem"
    T.signal("curiosity")      # → "signal:curiosity"
    T.flag("contradiction")    # → "flag:contradiction"
    T.reward("logic", "high")  # → "reward:logic_high"
    T.mem("ltmm_promoted")     # → "mem:ltmm_promoted"
"""
from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Valid values per namespace (for documentation + validation helpers)
# ---------------------------------------------------------------------------

PIPELINE_NAMES = frozenset({
    "regular_input", "self_reflective",
    "learning_m1", "learning_m2", "learning_m3", "learning_m4", "learning_m5",
    "homework", "reflective",
    "rem", "dream", "triage",
    "autonomous",
})

MODE_NAMES = frozenset({
    "normal", "learning", "autonomous",
    "homework", "reflective",
    "rem", "dream", "triage",
})

INTENT_NAMES = frozenset({
    "question", "assertion", "command",
    "reflection", "exploration", "clarification", "social",
    "correction", "request", "unknown",
})

SIGNAL_NAMES = frozenset({
    # Learning-oriented
    "frustration", "curiosity", "confusion",
    "boredom", "anxiety", "overwhelmed",
    # Dream-phase
    "wonder", "perplexed",
    # Positive learning
    "engagement", "insight",
})

REWARD_DOMAINS = frozenset({"logic", "ethics", "innovation", "attunement"})
REWARD_LEVELS  = frozenset({"high", "mid", "low"})

MEM_LABELS = frozenset({
    "high_significance", "low_significance",
    "ltmm_promoted", "identity_relevant",
    "dream_candidate",
})

FLAG_NAMES = frozenset({
    "contradiction", "paradox", "innovation",
    "unsolved", "identity_violation", "alignment_fail",
    "llm_bypass", "soothing", "urgency_high", "urgency_elevated",
})

CONTENT_TYPES = frozenset({
    "academic", "creative", "ethical", "technical",
    "social", "metacognitive", "reflective",
})

ORIGIN_SOURCES = frozenset({
    "academic",     # academic buffer / homework-origin unsolved concepts
    "identity",     # identity-derived questions (axiom/value/constraint challenges)
    "dialectic",    # contradiction or paradox from dialectic processing
    "general",      # general unsolved question (no special origin)
})


# ---------------------------------------------------------------------------
# Tag builder — functional interface (call as T.pipeline("rem") etc.)
# ---------------------------------------------------------------------------

class _TagNamespace:
    """Provides T.namespace(value) → 'namespace:value' tag helpers."""

    # --- Core namespaces ---

    @staticmethod
    def pipeline(name: str) -> str:
        """pipeline:<name>  — origin pipeline identifier."""
        return f"pipeline:{name}"

    @staticmethod
    def mode(name: str) -> str:
        """mode:<name>  — operational mode token."""
        return f"mode:{name}"

    @staticmethod
    def intent(name: str) -> str:
        """intent:<name>  — primary user intention."""
        return f"intent:{name}"

    @staticmethod
    def signal(name: str) -> str:
        """signal:<name>  — learning / emotional signal."""
        return f"signal:{name}"

    @staticmethod
    def reward(domain: str, level: str) -> str:
        """reward:<domain>_<level>  — reward domain strength label."""
        return f"reward:{domain}_{level}"

    @staticmethod
    def mem(label: str) -> str:
        """mem:<label>  — memory salience / lifecycle marker."""
        return f"mem:{label}"

    @staticmethod
    def flag(name: str) -> str:
        """flag:<name>  — operational event flag."""
        return f"flag:{name}"

    @staticmethod
    def content(type_: str) -> str:
        """content:<type>  — content domain classifier."""
        return f"content:{type_}"

    @staticmethod
    def origin(source: str) -> str:
        """origin:<source>  — provenance tag for unsolved concepts.

        Used by sleep pipelines to differentiate processing strategy:
        - ``origin:academic``  → REM raises logic weight; Dream deprioritises
        - ``origin:identity``  → REM raises ethics+attunement; Dream deprioritises
        - ``origin:dialectic`` → REM raises logic+ethics; Dream standard priority
        - ``origin:general``   → neutral; standard priority in both pipelines
        """
        return f"origin:{source}"

    # --- Convenience builders for compound tags ---

    @staticmethod
    def reward_from_score(domain: str, score: float) -> str:
        """Build a reward domain tag from a 0-1 score.

        score >= 0.65 → high, 0.35–0.65 → mid, < 0.35 → low
        """
        if score >= 0.65:
            level = "high"
        elif score >= 0.35:
            level = "mid"
        else:
            level = "low"
        return f"reward:{domain}_{level}"

    @staticmethod
    def signals_from_list(signals: List[str]) -> List[str]:
        """Return a list of signal:* tags from a list of signal names."""
        return [f"signal:{s}" for s in signals]

    @staticmethod
    def pipeline_tags_for_sleep(pipeline: str, signals: List[str]) -> List[str]:
        """Convenience: return the standard tag set for a sleep pipeline run."""
        return [
            f"pipeline:{pipeline}",
            f"mode:{pipeline}",
        ] + [f"signal:{s}" for s in signals]


# Module-level singleton — import as `from zados.core.tags import T`
T = _TagNamespace()
