"""Pure render helpers — input: state objects, output: rich Renderables.

Kept side-effect-free so they can be unit-tested without a live console.
"""
from __future__ import annotations

from typing import Any, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------

def render_status_line(dev: Any) -> Text:
    """One-line summary printed above every prompt."""
    s = dev.session
    if s is None:
        return Text("[zados] (no session)  verbosity={}".format(dev.verbosity), style="dim")

    sess_id = (getattr(s, "session_id", "") or "")[:8]
    branch = getattr(s, "branch", "?")
    turn = len(dev.history)
    mode = getattr(s, "initial_mode", "?")
    if getattr(s, "active_learning_mode", None):
        mode = f"{mode}/{s.active_learning_mode}"
    profile = getattr(s, "reward_profile_name", "?")

    line = (
        f"[zados] sess={sess_id} branch={branch} turn={turn} "
        f"mode={mode} profile={profile} verbosity={dev.verbosity}"
    )
    return Text(line, style="dim")


# ---------------------------------------------------------------------------
# Turn result rendering
# ---------------------------------------------------------------------------

def render_answer_panel(result: Any) -> Panel:
    """Wrap the final answer in a panel titled 'AI'."""
    answer = _extract_answer(result)
    return Panel(answer or "(no answer)", title="AI", title_align="left",
                 border_style="cyan", padding=(0, 1))


def render_error_panel(message: str) -> Panel:
    return Panel(Text(message, style="red"), title="error", title_align="left",
                 border_style="red", padding=(0, 1))


def render_auto_block(result: Any, verbosity: str) -> Optional[Any]:
    """Auto-display block printed after each turn, scoped to verbosity level.

    quiet  -> nothing (caller renders final_answer only)
    normal -> dominant emotion + directive + selected mode
    nerd   -> normal block + reward summary + engines run/skipped + NT delta
    """
    if verbosity == "quiet":
        return None

    state = _state_of(result)
    if state is None:
        return None

    rows: list[tuple[str, str]] = []

    # mode / directive / dominant emotion (normal+nerd)
    mode_token = _safe_attr(state, "modulation", "mode_token")
    if mode_token:
        rows.append(("mode", str(mode_token)))

    profile = _safe_attr(state, "modulation", "reward_profile_name")
    if profile:
        rows.append(("profile", str(profile)))

    directive = getattr(result, "directive", None) or _safe_attr(state, "answer", "directive")
    if directive:
        rows.append(("directive", str(directive)))

    intent = _safe_attr(state, "perception", "intent_archetype")
    if intent:
        rows.append(("intent", str(intent)))

    dom_emotion = _dominant_emotion(state)
    if dom_emotion:
        rows.append(("dominant emotion", dom_emotion))

    if verbosity == "nerd":
        engines_run, engines_skipped = _engine_counts(state)
        rows.append(("engines", f"{engines_run} run, {engines_skipped} skipped"))

        reward_summary = _reward_summary(state)
        if reward_summary:
            rows.append(("reward", reward_summary))

        nt_delta = _nt_top_movers(state)
        if nt_delta:
            rows.append(("NT top", nt_delta))

    if not rows:
        return None

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column()
    for k, v in rows:
        tbl.add_row(f"{k}:", v)
    return Panel(tbl, title="turn detail", title_align="left",
                 border_style="grey50", padding=(0, 1))


def render_turn_block(result: Any, verbosity: str) -> Any:
    """Final answer panel + (optional) auto-detail block, grouped."""
    answer = render_answer_panel(result)
    auto = render_auto_block(result, verbosity)
    return Group(answer, auto) if auto else answer


# ---------------------------------------------------------------------------
# History table
# ---------------------------------------------------------------------------

def render_history_table(history: list, n: int = 10) -> Table:
    rows = history[-n:]
    start_idx = len(history) - len(rows)
    tbl = Table(title=f"last {len(rows)} turn(s)", title_style="bold",
                show_lines=False, padding=(0, 1))
    tbl.add_column("#", style="dim", justify="right")
    tbl.add_column("mode")
    tbl.add_column("intent")
    tbl.add_column("directive")
    tbl.add_column("answer", overflow="fold", max_width=60)
    for i, r in enumerate(rows):
        idx = start_idx + i
        state = _state_of(r)
        mode = _safe_attr(state, "modulation", "mode_token") or "-"
        intent = _safe_attr(state, "perception", "intent_archetype") or "-"
        directive = getattr(r, "directive", None) or _safe_attr(state, "answer", "directive") or "-"
        answer = _truncate(_extract_answer(r) or "", 80)
        tbl.add_row(str(idx), str(mode), str(intent), str(directive), answer)
    return tbl


def render_turn_detail(result: Any) -> Any:
    """Full per-turn detail (chat show <idx>)."""
    state = _state_of(result)
    rows: list[tuple[str, str]] = []
    rows.append(("intent",      str(_safe_attr(state, "perception", "intent_archetype") or "-")))
    rows.append(("directive",   str(getattr(result, "directive", None) or "-")))
    rows.append(("mode",        str(_safe_attr(state, "modulation", "mode_token") or "-")))
    rows.append(("profile",     str(_safe_attr(state, "modulation", "reward_profile_name") or "-")))
    rows.append(("emotion",     _dominant_emotion(state) or "-"))
    thinking = _safe_attr(state, "thinking", "thinking_trace") or ""
    rows.append(("thinking",    _truncate(thinking, 200) or "-"))

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column()
    for k, v in rows:
        tbl.add_row(f"{k}:", v)

    answer = render_answer_panel(result)
    return Group(answer, Panel(tbl, title="detail", title_align="left",
                                border_style="grey50", padding=(0, 1)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def unwrap_pipeline_result(result: Any) -> Any:
    """Return the underlying PipelineResult regardless of wrapping.

    Handles:
      - PipelineResult (returned by regular + commanded direct calls)
      - LearningModeResult.pipeline_result
      - SelfRefResult.pipeline_result
      - dict results from sleep/homework — returns None (no PipelineState)
    """
    if result is None:
        return None
    if hasattr(result, "state") and getattr(result, "state", None) is not None:
        return result
    inner = getattr(result, "pipeline_result", None)
    if inner is not None:
        return inner
    return None


def _extract_answer(result: Any) -> str:
    """Best-effort extraction of the user-facing text from any result type."""
    if result is None:
        return ""
    # Direct PipelineResult.final_answer
    direct = getattr(result, "final_answer", None)
    if direct:
        return direct
    # LearningModeResult / SelfRefResult have a pipeline_result inside
    inner = getattr(result, "pipeline_result", None)
    if inner is not None and getattr(inner, "final_answer", None):
        return inner.final_answer
    # SelfRefResult.synthesis
    synth = getattr(result, "synthesis", None)
    if synth:
        return synth
    # dict (sleep/homework/etc.)
    if isinstance(result, dict):
        for k in ("final_answer", "summary", "message", "status"):
            v = result.get(k)
            if v:
                return str(v)
    return ""


def _state_of(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "state") and getattr(result, "state", None) is not None:
        return result.state
    inner = getattr(result, "pipeline_result", None)
    if inner is not None:
        return getattr(inner, "state", None)
    return None


def _safe_attr(obj: Any, *path: str) -> Any:
    cur = obj
    for p in path:
        if cur is None:
            return None
        cur = getattr(cur, p, None)
    return cur


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _dominant_emotion(state: Any) -> Optional[str]:
    """Pull top-1 emotion from E28 result if available."""
    e28 = _safe_attr(state, "dispatch", "e28_result")
    if e28 is None:
        return None
    # E28 may expose .top_emotion, .dominant_emotion, .emotions dict, etc.
    for attr in ("top_emotion", "dominant_emotion", "primary_emotion"):
        v = getattr(e28, attr, None)
        if v:
            return str(v)
    emotions = getattr(e28, "emotions", None)
    if isinstance(emotions, dict) and emotions:
        top = max(emotions.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)
        return f"{top[0]} ({top[1]:.2f})" if isinstance(top[1], (int, float)) else str(top[0])
    return None


def _engine_counts(state: Any) -> tuple[int, int]:
    dispatch = getattr(state, "dispatch", None)
    if dispatch is None:
        return (0, 0)
    run = getattr(dispatch, "engines_run", None) or getattr(dispatch, "engine_results", None) or {}
    skipped = getattr(dispatch, "engines_skipped", None) or []
    nrun = len(run) if hasattr(run, "__len__") else 0
    nskip = len(skipped) if hasattr(skipped, "__len__") else 0
    return (nrun, nskip)


def _reward_summary(state: Any) -> Optional[str]:
    reward = getattr(state, "reward", None)
    if reward is None:
        return None
    # Try common shapes
    meta = getattr(reward, "meta_directive", None)
    if meta is not None:
        directive = getattr(meta, "directive", None) or (
            meta.get("directive") if isinstance(meta, dict) else None
        )
        if directive:
            return f"meta_directive={directive}"
    phase5 = getattr(reward, "phase5_result", None) or reward
    return _truncate(repr(phase5), 80)


def _nt_top_movers(state: Any) -> Optional[str]:
    snap = _safe_attr(state, "modulation", "nt_snapshot")
    if not isinstance(snap, dict) or not snap:
        return None
    # Just show the 3 highest values (no delta tracking yet — that's nerd-mode v2).
    top3 = sorted(snap.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0,
                  reverse=True)[:3]
    return " ".join(f"{k}={v:.2f}" for k, v in top3 if isinstance(v, (int, float)))
