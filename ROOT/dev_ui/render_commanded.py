"""Renderers for commanded pipelines: REM, Dream, Homework, Reflective.

Each pipeline returns a plain ``dict`` summary from the classifier — no
PipelineState, so these views are pure summary statistics rather than full
per-phase inspection panels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render_rem_result(result: Dict[str, Any]) -> Any:
    """REM pipeline summary — retroactive learning + consolidation stats."""
    if not isinstance(result, dict):
        return Text(f"(unexpected REM result type: {type(result).__name__})", style="red")

    status = result.get("status", "?")
    stats = [
        ("session_id",            str(result.get("session_id", "-"))),
        ("processing_time_s",     f"{float(result.get('processing_time_s', 0) or 0):.3f}"),
        ("packets_scanned",       _count(result, "packets_scanned")),
        ("packets_consolidated",  _count(result, "packets_consolidated")),
    ]

    parts: List[Any] = [
        Panel(_status_line(status, "REM consolidation"),
              border_style=_status_color(status), padding=(0, 1)),
        _kv_table(stats),
    ]

    # Retroactive learning signals
    signals = result.get("dominant_signals") or []
    if signals:
        sigs = Table(title="dominant emotional signals (retroactive learning)",
                     title_style="bold", padding=(0, 1))
        sigs.add_column("signal")
        sigs.add_column("strength", justify="right")
        for s in signals:
            if isinstance(s, dict):
                sigs.add_row(str(s.get("name", "?")), f"{float(s.get('strength', 0)):.3f}")
            elif isinstance(s, (list, tuple)) and len(s) >= 2:
                sigs.add_row(str(s[0]), f"{float(s[1]):.3f}")
            else:
                sigs.add_row(str(s), "-")
        parts.append(sigs)

    parts.extend(_domain_adjustments_table(result.get("domain_weight_adjustments")))
    return Group(*parts)


def render_dream_result(result: Dict[str, Any]) -> Any:
    """Dream pipeline summary — recombination + emotional drivers."""
    if not isinstance(result, dict):
        return Text(f"(unexpected dream result: {type(result).__name__})", style="red")

    status = result.get("status", "?")
    stats = [
        ("session_id",          str(result.get("session_id", "-"))),
        ("processing_time_s",   f"{float(result.get('processing_time_s', 0) or 0):.3f}"),
        ("candidates_found",    _count(result, "candidates_found")),
        ("candidates_processed", _count(result, "candidates_processed")),
        ("novel_connections",   _count(result, "novel_connections")),
    ]

    parts: List[Any] = [
        Panel(_status_line(status, "Dream / creative recombination"),
              border_style="purple", padding=(0, 1)),
        _kv_table(stats),
    ]

    drivers = result.get("dominant_signals") or []
    if drivers:
        dtbl = Table(title="emotional driver profile", title_style="bold", padding=(0, 1))
        dtbl.add_column("signal")
        dtbl.add_column("strength", justify="right")
        for s in drivers:
            if isinstance(s, dict):
                dtbl.add_row(str(s.get("name", "?")), f"{float(s.get('strength', 0)):.3f}")
            elif isinstance(s, (list, tuple)) and len(s) >= 2:
                dtbl.add_row(str(s[0]), f"{float(s[1]):.3f}")
            else:
                dtbl.add_row(str(s), "-")
        parts.append(dtbl)

    parts.extend(_domain_adjustments_table(result.get("domain_weight_adjustments")))
    return Group(*parts)


def render_homework_result(result: Dict[str, Any]) -> Any:
    """6-phase homework run summary — `HomeworkRunSummary` stats."""
    if not isinstance(result, dict):
        return Text(f"(unexpected homework result: {type(result).__name__})", style="red")

    status = result.get("status", "?")

    # 6-phase tickbox stepper.
    phases = [
        ("Phase 0",  "Input assembly & triage",
         f"batches={result.get('batches_processed', 0)}, logs={result.get('logs_processed', 0)}"),
        ("Phase 1",  "Analysis",
         f"meta_patterns={len(result.get('meta_patterns', []))}"),
        ("Phase 2",  "Processing",
         f"contradictions resolved={result.get('contradictions_resolved', 0)}, "
         f"unresolved={result.get('contradictions_unresolved', 0)}"),
        ("Phase 3",  "Question resolution",
         f"resolved={result.get('questions_resolved', 0)}, "
         f"new={result.get('questions_new', 0)}, "
         f"dream candidates={result.get('dream_candidates_flagged', 0)}"),
        ("Phase 4",  "Synthesis",
         f"validated={result.get('lessons_validated', 0)}, "
         f"pending={result.get('lessons_pending', 0)}, "
         f"core updates={result.get('core_memory_updates_applied', 0)}"),
        ("Phase 5",  "Output & handoff",
         f"fallacy/bias flags={result.get('fallacy_bias_flags', 0)}"),
    ]
    stepper = Table(show_header=False, padding=(0, 1), box=None)
    stepper.add_column(style="green", min_width=8)   # ✓ Phase n
    stepper.add_column(style="bold", min_width=24)
    stepper.add_column(overflow="fold")
    for tag, name, detail in phases:
        stepper.add_row(f"✓ {tag}", name, detail)

    # Top-level summary table.
    summary = _kv_table([
        ("session_id",        str(result.get("session_id", "-"))),
        ("processing_time_s", f"{float(result.get('processing_time_s', 0) or 0):.3f}"),
    ])

    parts: List[Any] = [
        Panel(_status_line(status, "Homework — 6-phase structured processing"),
              border_style="yellow", padding=(0, 1)),
        summary,
        Panel(stepper, title="phases", title_align="left",
              border_style="grey50", padding=(0, 1)),
    ]

    # Processing emphasis (subject → deficit_domain).
    emphasis = result.get("processing_emphasis") or {}
    if isinstance(emphasis, dict) and emphasis:
        etbl = Table(title="processing emphasis", title_style="bold", padding=(0, 1))
        etbl.add_column("subject")
        etbl.add_column("deficit domain")
        for sub, dom in emphasis.items():
            etbl.add_row(str(sub), str(dom))
        parts.append(etbl)

    return Group(*parts)


def render_reflective_result(result: Dict[str, Any]) -> Any:
    """Reflective pipeline — E31 meta-learning + E32 identity coherence."""
    if not isinstance(result, dict):
        return Text(f"(unexpected reflective result: {type(result).__name__})", style="red")

    status = result.get("status", "?")
    coh_status = result.get("identity_coherence_status", "?")
    coh_score = float(result.get("coherence_score", 0) or 0)
    coh_color = "green" if coh_score >= 0.7 else ("yellow" if coh_score >= 0.4 else "red")

    # E31 (meta-learning) stats
    e31_rows = [
        ("learning_patterns",     _count(result, "learning_patterns")),
        ("recurring_failures",    _count(result, "recurring_failures")),
        ("mode_effectiveness",    f"{len(result.get('mode_effectiveness') or {})} entries"),
        ("subject_proficiencies", f"{len(result.get('subject_proficiencies') or {})} entries"),
        ("style_preferences",     str(len(result.get("style_preferences") or []))),
        ("learning_recommendations", str(len(result.get("learning_recommendations") or []))),
        ("learning_logs_analysed", _count(result, "learning_logs_analysed")),
    ]

    # E32 (identity coherence) stats
    e32_rows = [
        ("identity_coherence_status", Text(str(coh_status), style=coh_color)),
        ("coherence_score",           Text(f"{coh_score:.3f}", style=coh_color)),
        ("core_contradictions",       _count(result, "core_contradictions")),
        ("fragile_conclusions",       _count(result, "fragile_conclusions")),
        ("alignment_issues",          _count(result, "alignment_issues")),
        ("identity_themes",           str(len(result.get("identity_themes") or []))),
    ]

    # Mutations applied
    mut_rows = [
        ("conclusions_reinforced",            _count(result, "conclusions_reinforced")),
        ("conclusions_created",               _count(result, "conclusions_created")),
        ("conclusions_recommended_for_update", _count(result, "conclusions_recommended_for_update")),
        ("journal_entries_created",           _count(result, "journal_entries_created")),
        ("pending_updates_analysed",          _count(result, "pending_updates_analysed")),
        ("cross_references",                  _count(result, "cross_references")),
    ]

    # Input summary
    input_rows = [
        ("fallacy_flags_processed", _count(result, "fallacy_flags_processed")),
        ("bias_flags_processed",    _count(result, "bias_flags_processed")),
        ("meta_patterns_processed", _count(result, "meta_patterns_processed")),
    ]

    parts: List[Any] = [
        Panel(_status_line(status, "Reflective — meta-learning + identity coherence"),
              border_style="cyan", padding=(0, 1)),
        _kv_table([
            ("session_id",        str(result.get("session_id", "-"))),
            ("processing_time_s", f"{float(result.get('processing_time_s', 0) or 0):.3f}"),
        ]),
        Panel(_kv_table(e31_rows), title="E31 — meta-learning analysis",
              title_align="left", border_style="grey50", padding=(0, 1)),
        Panel(_kv_table(e32_rows), title="E32 — identity coherence",
              title_align="left", border_style="grey50", padding=(0, 1)),
        Panel(_kv_table(mut_rows), title="identity store mutations",
              title_align="left", border_style="grey50", padding=(0, 1)),
        Panel(_kv_table(input_rows), title="input summary",
              title_align="left", border_style="grey50", padding=(0, 1)),
    ]
    return Group(*parts)


def render_sleep_status(session: Any) -> Any:
    """Current sleep state — session_mode + active learning + branch."""
    if session is None:
        return Text("(no session)", style="dim")

    mode = getattr(session, "session_mode", "regular")
    in_sleep = mode == "sleep"
    rows = [
        ("session_mode", Text(mode, style="purple" if in_sleep else "dim")),
        ("active_learning_mode", str(getattr(session, "active_learning_mode", None) or "-")),
        ("initial_mode", str(getattr(session, "initial_mode", "-"))),
        ("reward_profile", str(getattr(session, "reward_profile_name", "-"))),
        ("branch", str(getattr(session, "branch", "?"))),
    ]
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right")
    grid.add_column()
    for k, v in rows:
        grid.add_row(f"{k}:", v if isinstance(v, Text) else Text(str(v)))

    body = Panel(
        grid,
        title=f"sleep status: {'IN SLEEP' if in_sleep else 'awake'}",
        title_align="left",
        border_style="purple" if in_sleep else "grey50", padding=(0, 1),
    )
    return body


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kv_table(rows: List[Tuple[str, Any]]) -> Any:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right")
    grid.add_column()
    for k, v in rows:
        grid.add_row(f"{k}:", v if isinstance(v, Text) else Text(str(v)))
    return grid


def _count(d: Dict[str, Any], key: str) -> str:
    v = d.get(key, 0)
    if isinstance(v, (list, tuple, dict, set)):
        return str(len(v))
    return str(v)


def _status_line(status: str, title: str) -> Text:
    color = _status_color(status)
    return Text(f"{title}  —  status: {status}", style=color)


def _status_color(status: str) -> str:
    s = (status or "").lower()
    if s == "completed":
        return "green"
    if s in ("partial", "deferred"):
        return "yellow"
    if s in ("error", "failed"):
        return "red"
    return "grey50"


def _domain_adjustments_table(adjustments: Any) -> List[Any]:
    if not isinstance(adjustments, dict) or not adjustments:
        return []
    tbl = Table(title="domain weight adjustments", title_style="bold", padding=(0, 1))
    tbl.add_column("domain")
    tbl.add_column("delta", justify="right")
    for k, v in adjustments.items():
        if isinstance(v, (int, float)):
            style = "green" if v > 0 else ("red" if v < 0 else "dim")
            tbl.add_row(str(k), Text(f"{v:+.3f}", style=style))
        else:
            tbl.add_row(str(k), str(v))
    return [tbl]
