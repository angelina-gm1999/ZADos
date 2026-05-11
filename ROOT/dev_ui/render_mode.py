"""Renderers for the `mode` command group."""
from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Categorised view of MODE_TO_PROFILE — keeps `mode list` readable.
_MODE_GROUPS: List[tuple[str, list[tuple[str, str, str]]]] = [
    ("Conversational", [
        ("Regular",       "regular_input",        "default conversational mode"),
        ("Normal",        "regular_input",        "alias for Regular"),
        ("RegularInput",  "regular_input",        "explicit mode token"),
    ]),
    ("Neurosymbolic auto-selected", [
        ("EmpathicAttunement",     "reflective",          "empathy-led response"),
        ("ComfortAmplifier",       "reflective",          "soothing / supportive"),
        ("CuriosityDrive",         "exploratory_sandbox", "wide exploration"),
        ("CreativeDivergence",     "creative_sandbox",    "high-novelty creative"),
        ("ConceptualSynthesis",    "creative_sandbox",    "abstract synthesis"),
        ("AnalyticalFilter",       "analysis",            "structured analysis"),
        ("HypercriticalLogicScan", "analysis",            "strict logic audit"),
        ("HyperRationalEngine",    "analysis",            "rationalist mode"),
        ("LogicMode",              "analysis",            "general logic"),
        ("ConvergentRefiner",      "analysis",            "convergent refinement"),
        ("LiteralSkeptic",         "analysis",            "literal/skeptic stance"),
        ("PrecisionRuleFidelity",  "analysis",            "rule-driven precision"),
        ("Containment",            "ethics_training",     "safety / containment"),
        ("RecoveryReset",          "ethics_training",     "recovery / de-escalation"),
    ]),
    ("Learning (Matrioshka M1–M5)", [
        ("M1",  "receptive_learning",     "teach me something"),
        ("M2",  "critical_review",        "review / quiz me"),
        ("M3",  "dialectic_exploration",  "explore / socratic"),
        ("M4",  "curiosity_driven",       "questions I have"),
        ("M5",  "autonomous_study",       "independent study"),
    ]),
    ("Sleep", [
        ("SleepMode_Triage", "sleep_triage", "lightweight consolidation"),
        ("SleepMode_REM",    "sleep_deep",   "REM consolidation"),
        ("SleepMode_Dream",  "sleep_dream",  "dream / creative recombination"),
    ]),
    ("Meta-learning / commanded", [
        ("MetaLearning_Homework",   "homework_processing",  "structured 6-phase processing"),
        ("MetaLearning_Reflective", "reflective_synthesis", "post-session synthesis"),
        ("SelfReflectiveQuery",     "self_reflective",      "introspection on unsolved"),
    ]),
]


def render_mode_list() -> Any:
    parts: list[Any] = []
    for group_name, entries in _MODE_GROUPS:
        tbl = Table(title=group_name, title_style="bold", padding=(0, 1))
        tbl.add_column("mode", min_width=22)
        tbl.add_column("reward profile", min_width=22)
        tbl.add_column("description", overflow="fold")
        for name, profile, desc in entries:
            tbl.add_row(name, profile, desc)
        parts.append(tbl)
    parts.append(Text(
        "\nUse `mode set <name>` to switch.  Learning modes accept M1..M5 directly.\n"
        "Use `mode briefing \"<text>\"` to set the session-level context prompt.",
        style="dim",
    ))
    return Group(*parts)


def render_mode_show(session: Any, state: Any) -> Any:
    """Current mode + briefing + cluster weights for this turn."""
    if session is None:
        return Text("(no session)", style="dim")

    rows = [
        ("session_id",      str(getattr(session, "session_id", "?"))),
        ("session_mode",    str(getattr(session, "session_mode", "?"))),
        ("initial_mode",    str(getattr(session, "initial_mode", "?"))),
        ("active_learning", str(getattr(session, "active_learning_mode", None) or "-")),
        ("reward profile",  str(getattr(session, "reward_profile_name", "?"))),
        ("branch",          str(getattr(session, "branch", "?"))),
    ]
    briefing = getattr(session, "mission_briefing", None)
    rows.append(("mission briefing", _briefing_text(briefing)))

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column(overflow="fold")
    for k, v in rows:
        tbl.add_row(f"{k}:", v)

    parts: list[Any] = [Panel(tbl, title="current mode", title_align="left",
                              border_style="cyan", padding=(0, 1))]

    # Cluster weights from the most recent turn, if available.
    weights: Dict[str, float] = {}
    if state is not None:
        mod = getattr(state, "modulation", None)
        weights = getattr(mod, "engine_weights", None) or {}
    if weights:
        wtbl = Table(title="cluster weights (last turn)", title_style="bold", padding=(0, 1))
        wtbl.add_column("cluster")
        wtbl.add_column("weight", justify="right")
        for k, v in sorted(weights.items(),
                           key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0):
            if isinstance(v, (int, float)):
                wtbl.add_row(k, f"{v:+.3f}")
        parts.append(wtbl)

    return Group(*parts)


def _briefing_text(b: Any) -> str:
    if b is None:
        return "(none)"
    if isinstance(b, str):
        return b[:240]
    # Could be a MemoryPacket or similar
    for attr in ("verbal_summary", "content", "text", "summary"):
        v = getattr(b, attr, None)
        if v:
            return str(v)[:240]
    return f"<{type(b).__name__}>"
