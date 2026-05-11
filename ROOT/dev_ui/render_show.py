"""Renderers for the `show` command group.

Each function takes raw state objects and returns a rich Renderable.  No
console writes happen here — the shell's `do_show` is responsible for
printing.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Group
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Constants — canonical orderings & label sets
# ---------------------------------------------------------------------------

# Canonical 12-NT order for display.  Matches spec §3.2.
NT_ORDER: Tuple[str, ...] = (
    "glu", "gaba", "da", "5ht", "ne", "ach",
    "oxt", "mor", "cb1", "crh", "cortisol", "histamine",
)

# Oscillation bands shown as a strip.
OSC_BANDS: Tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma", "sigma")
OSC_CROSS: Tuple[str, ...] = ("theta_gamma", "alpha_beta", "delta_sigma")

METRIC_FIELDS: Tuple[str, ...] = (
    "motivation", "empathy", "cognitive_rigidity", "fatigue",
    "precision", "openness", "anxiety", "social_engagement",
    "dream_permissiveness", "consolidation_depth", "narrative_plasticity",
)

_SLEEP_METRICS = {"dream_permissiveness", "consolidation_depth", "narrative_plasticity"}

_BLOCKS = " ▁▂▃▄▅▆▇█"   # 9 levels (index 0 = empty, 8 = full)

# Engine grid: 6 cols × 6 rows = 36 slots for IDs 1..32 + 4 empty.
_ENGINE_SHORT_NAMES: Dict[int, str] = {
    1: "Contradict",   2: "Paradox",     3: "SOAR",         4: "Fallacy",
    5: "Bias",         6: "LogicTrap",   7: "SimOpp",       8: "Relevance",
    9: "AtomSpace",   10: "PLN",        11: "InputRel",    12: "LogicBrain",
    13: "SimBrain",   14: "Socratic",   15: "Decision",    16: "ECAN",
    17: "RewardLrn",  18: "DataAnal",   19: "PatternID",   20: "PatternCmp",
    21: "Strategic",  22: "CtxLearn",   23: "Intent",      24: "Heuristic",
    25: "Recursive",  26: "Uncertain",  27: "Homeostat",   28: "Emotion",
    29: "MemCompr",   30: "Retroactv",  31: "ReflLearn",   32: "ReflIdent",
}


# ---------------------------------------------------------------------------
# show neurochem
# ---------------------------------------------------------------------------

def render_neurochem(state: Any, full: bool = False) -> Any:
    """12-NT heatmap + osc bands + 11 metrics."""
    nt_snap: Dict[str, float] = _safe(state, "modulation", "nt_snapshot") or {}
    osc_snap: Dict[str, float] = _safe(state, "modulation", "osc_snapshot") or {}
    metrics: Dict[str, float] = _safe(state, "modulation", "metrics_dict") or {}
    mode_token = _safe(state, "modulation", "mode_token") or "-"

    nt_panel = Panel(
        _render_nt_strip(nt_snap, full=full),
        title=f"neurotransmitters  (mode={mode_token})",
        title_align="left", border_style="cyan", padding=(0, 1),
    )
    osc_panel = Panel(
        _render_osc_strip(osc_snap),
        title="oscillations",
        title_align="left", border_style="magenta", padding=(0, 1),
    )
    metrics_panel = Panel(
        _render_metrics_table(metrics, mode_token),
        title="neurochemical metrics",
        title_align="left", border_style="green", padding=(0, 1),
    )
    return Group(nt_panel, osc_panel, metrics_panel)


def _render_nt_strip(snap: Dict[str, float], full: bool) -> Any:
    grid = Table.grid(padding=(0, 1))
    for _ in NT_ORDER:
        grid.add_column(justify="center", min_width=6)

    bars = [_block_bar(snap.get(k, 0.0)) for k in NT_ORDER]
    labels = [Text(k, style="dim") for k in NT_ORDER]

    grid.add_row(*[Text(b, style=_bar_style(snap.get(k, 0.0))) for k, b in zip(NT_ORDER, bars)])
    grid.add_row(*labels)
    if full:
        grid.add_row(*[Text(f"{snap.get(k, 0.0):.3f}", style="dim") for k in NT_ORDER])
    return grid


def _render_osc_strip(snap: Dict[str, float]) -> Any:
    grid = Table.grid(padding=(0, 1))
    bands = OSC_BANDS + OSC_CROSS
    for _ in bands:
        grid.add_column(justify="center", min_width=8)
    grid.add_row(*[Text(_block_bar(snap.get(b, 0.0)), style=_bar_style(snap.get(b, 0.0))) for b in bands])
    grid.add_row(*[Text(b, style="dim") for b in bands])
    grid.add_row(*[Text(f"{snap.get(b, 0.0):.2f}", style="dim") for b in bands])
    return grid


def _render_metrics_table(metrics: Dict[str, float], mode: str) -> Any:
    sleep_active = "sleep" in mode.lower() or "rem" in mode.lower() or "dream" in mode.lower()
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(justify="right")
    tbl.add_column()
    tbl.add_column(justify="right")
    tbl.add_column()

    pairs = list(METRIC_FIELDS)
    rows = [pairs[i:i + 2] for i in range(0, len(pairs), 2)]
    for row in rows:
        cells: List[Any] = []
        for name in row:
            v = float(metrics.get(name, 0.0) or 0.0)
            greyed = (name in _SLEEP_METRICS) and not sleep_active
            label_style = "dim" if greyed else None
            value_style = "dim" if greyed else _bar_style(v)
            cells.append(Text(name, style=label_style))
            cells.append(Text(f"{_block_bar(v)} {v:.2f}", style=value_style))
        if len(cells) < 4:
            cells.extend([Text(""), Text("")])
        tbl.add_row(*cells)
    return tbl


# ---------------------------------------------------------------------------
# show reward
# ---------------------------------------------------------------------------

def render_reward(state: Any, session: Any) -> Any:
    """Mode header + reward profile + tonic/phasic pathway + NT signals."""
    mode_token = _safe(state, "modulation", "mode_token") or "-"
    profile_name = _safe(state, "modulation", "reward_profile_name") or "-"

    header = Panel(
        Text(f"mode: {mode_token}    profile: {profile_name}", style="bold"),
        border_style=_mode_color(mode_token),
        padding=(0, 1),
    )

    # Static vs. learned domain weights -------------------------------------
    try:
        from zados.reward.profile import PROFILE_REGISTRY
        static = PROFILE_REGISTRY.get(profile_name)
        static_weights: Dict[str, float] = dict(static.domain_weights) if static else {}
    except Exception:
        static_weights = {}

    learned: Dict[str, float] = getattr(session, "learned_domain_weights", {}) or {}
    # learned keys come with "_weight" suffix
    learned_norm = {k.replace("_weight", ""): v for k, v in learned.items()}

    domains = ["logic", "ethics", "innovation", "human_attunement"]
    weights_tbl = Table(title="domain weights", title_style="bold", show_lines=False,
                        padding=(0, 1))
    weights_tbl.add_column("domain")
    weights_tbl.add_column("static",  justify="right")
    weights_tbl.add_column("learned", justify="right")
    weights_tbl.add_column("delta",   justify="right")
    for d in domains:
        s_w = static_weights.get(d, 0.0)
        # learned key is "attunement" not "human_attunement"
        l_key = "attunement" if d == "human_attunement" else d
        l_w = learned_norm.get(l_key, s_w)
        delta = l_w - s_w
        delta_style = "green" if delta > 0.001 else ("red" if delta < -0.001 else "dim")
        weights_tbl.add_row(
            d,
            f"{s_w:.2f}",
            f"{l_w:.2f}",
            Text(f"{delta:+.2f}", style=delta_style),
        )

    # Phase 5 pathway summary -----------------------------------------------
    p5 = _safe(state, "reward", "phase5_result")
    pathway_rows: List[Tuple[str, str]] = []

    meta = getattr(p5, "meta_directive", None) if p5 else None
    if meta is not None:
        if isinstance(meta, dict):
            directive = (
                "abstain" if meta.get("abstain")
                else ("suppress" if meta.get("suppress") else "allow")
            )
            reason = meta.get("reason", "")
        else:
            directive = getattr(meta, "directive", "?")
            reason = getattr(meta, "reason", "")
        pathway_rows.append(("meta_directive", _colored_directive(directive)))
        if reason:
            pathway_rows.append(("reason", str(reason)))

    urgency = getattr(p5, "urgency_risk", None) if p5 else None
    if urgency is not None:
        pathway_rows.append(("urgency_risk", _scalar_with_bar(urgency)))

    selected_mode = getattr(p5, "selected_mode", None) if p5 else None
    if selected_mode:
        pathway_rows.append(("selected_mode", str(selected_mode)))

    tonic_applied = _safe(state, "reward", "tonic_applied")
    phasic_applied = _safe(state, "reward", "phasic_applied")
    pathway_rows.append(("tonic applied",  Text(str(bool(tonic_applied)),
                         style="green" if tonic_applied else "dim")))
    pathway_rows.append(("phasic applied", Text(str(bool(phasic_applied)),
                         style="green" if phasic_applied else "dim")))

    pathway_grid = Table.grid(padding=(0, 1))
    pathway_grid.add_column(style="dim", justify="right")
    pathway_grid.add_column()
    for k, v in pathway_rows:
        pathway_grid.add_row(f"{k}:", v if isinstance(v, Text) else Text(str(v)))
    pathway_panel = Panel(pathway_grid, title="phase 5 pathway", title_align="left",
                          border_style="grey50", padding=(0, 1))

    # Domain scores ---------------------------------------------------------
    domain_results = getattr(p5, "domain_results", None) if p5 else None
    domain_panel: Any = None
    if isinstance(domain_results, dict) and domain_results:
        dtbl = Table.grid(padding=(0, 1))
        dtbl.add_column(style="dim", justify="right")
        dtbl.add_column()
        for dname, dres in domain_results.items():
            score = _extract_score(dres)
            dtbl.add_row(f"{dname}:", _scalar_with_bar(score))
        domain_panel = Panel(dtbl, title="domain scores", title_align="left",
                             border_style="grey50", padding=(0, 1))

    # NT signals applied ----------------------------------------------------
    nt_signals = getattr(p5, "nt_signals", None) if p5 else None
    nt_panel: Any = None
    if isinstance(nt_signals, dict) and nt_signals:
        sig_tbl = Table.grid(padding=(0, 2))
        sig_tbl.add_column(style="dim", justify="right")
        sig_tbl.add_column()
        for k, v in nt_signals.items():
            if isinstance(v, (int, float)):
                style = "green" if v > 0 else ("red" if v < 0 else "dim")
                sig_tbl.add_row(f"{k}:", Text(f"{v:+.3f}", style=style))
            else:
                sig_tbl.add_row(f"{k}:", Text(str(v)[:60]))
        nt_panel = Panel(sig_tbl, title="NT signals (applied next turn)",
                         title_align="left", border_style="grey50", padding=(0, 1))

    parts = [header, weights_tbl, pathway_panel]
    if domain_panel is not None:
        parts.append(domain_panel)
    if nt_panel is not None:
        parts.append(nt_panel)
    return Group(*parts)


# ---------------------------------------------------------------------------
# show engines
# ---------------------------------------------------------------------------

def render_engines(state: Any, registered_ids: Iterable[int]) -> Any:
    """6×6 engine grid + cluster weights."""
    dispatch = getattr(state, "dispatch", None)
    run: List[int] = list(getattr(dispatch, "engines_run", []) or [])
    skipped: List[int] = list(getattr(dispatch, "engines_skipped", []) or [])
    registered = set(registered_ids)

    grid = Table(title="engines (green=ran, yellow=skipped, dim=unregistered)",
                 title_style="bold", show_header=False, padding=(0, 1))
    for _ in range(6):
        grid.add_column(justify="left", min_width=14)

    cells: List[Text] = []
    for eid in range(1, 33):
        name = _ENGINE_SHORT_NAMES.get(eid, f"E{eid}")
        label = f"E{eid:2d} {name}"
        if eid in run:
            cells.append(Text(label, style="bold green"))
        elif eid in skipped:
            cells.append(Text(label, style="yellow"))
        elif eid in registered:
            cells.append(Text(label, style="dim"))
        else:
            cells.append(Text(label, style="red dim"))

    while len(cells) % 6 != 0:
        cells.append(Text(""))
    for i in range(0, len(cells), 6):
        grid.add_row(*cells[i:i + 6])

    summary = Text(
        f"\n{len(run)} ran  /  {len(skipped)} skipped  /  "
        f"{len(registered) - len(run) - len(skipped)} idle  /  "
        f"{32 - len(registered)} unregistered",
        style="dim",
    )

    # Engine weights — these are cluster-level in this codebase, not per-engine.
    weights = _safe(state, "modulation", "engine_weights") or {}
    weights_panel: Any = None
    if isinstance(weights, dict) and weights:
        wtbl = Table.grid(padding=(0, 1))
        wtbl.add_column(style="dim", justify="right")
        wtbl.add_column()
        for k, v in sorted(weights.items(), key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0):
            if isinstance(v, (int, float)):
                wtbl.add_row(f"{k}:", _scalar_with_bar(v))
            else:
                wtbl.add_row(f"{k}:", Text(str(v)[:60]))
        weights_panel = Panel(wtbl, title="cluster weights",
                              title_align="left", border_style="grey50", padding=(0, 1))

    parts: List[Any] = [grid, summary]
    if weights_panel is not None:
        parts.append(weights_panel)
    return Group(*parts)


# ---------------------------------------------------------------------------
# show engine <id>
# ---------------------------------------------------------------------------

def render_engine_inspector(state: Any, engine_id: int) -> Any:
    """Structured view for known engines; JSON fallback otherwise."""
    dispatch = getattr(state, "dispatch", None)
    results = getattr(dispatch, "engine_results", None) or {}
    if engine_id not in results:
        skipped = list(getattr(dispatch, "engines_skipped", []) or [])
        if engine_id in skipped:
            return Panel(Text(f"E{engine_id} was skipped this turn.", style="yellow"),
                         title=f"E{engine_id}", title_align="left",
                         border_style="yellow", padding=(0, 1))
        return Panel(Text(f"E{engine_id} did not run this turn.", style="dim"),
                     title=f"E{engine_id}", title_align="left",
                     border_style="grey50", padding=(0, 1))

    result = results[engine_id]
    name = _ENGINE_SHORT_NAMES.get(engine_id, f"E{engine_id}")
    title = f"E{engine_id}  {name}"

    # Specific renderers for the engines spec called out.
    if engine_id == 8 and isinstance(result, dict):
        return Panel(_render_e8(result), title=title, title_align="left",
                     border_style="cyan", padding=(0, 1))
    if engine_id == 18 and isinstance(result, dict):
        return Panel(_render_e18(result), title=title, title_align="left",
                     border_style="cyan", padding=(0, 1))
    if engine_id == 19 and isinstance(result, dict):
        return Panel(_render_e19(result), title=title, title_align="left",
                     border_style="cyan", padding=(0, 1))
    if engine_id == 23 and isinstance(result, dict):
        return Panel(_render_e23(result, state), title=title, title_align="left",
                     border_style="cyan", padding=(0, 1))
    if engine_id == 28:
        e28 = getattr(dispatch, "e28_result", None) or result
        return Panel(_render_e28(e28), title=title, title_align="left",
                     border_style="cyan", padding=(0, 1))

    # JSON fallback
    return Panel(_json_dump(result), title=title, title_align="left",
                 border_style="cyan", padding=(0, 1))


def _render_e8(r: Dict[str, Any]) -> Any:
    facets = r.get("ranked_facets") or r.get("facets") or []
    if not facets:
        return Text("(no facets)", style="dim")
    tbl = Table(title="ranked facets", title_style="bold", padding=(0, 1))
    tbl.add_column("facet")
    tbl.add_column("score", justify="right")
    for f in facets[:20]:
        if isinstance(f, dict):
            tbl.add_row(str(f.get("name", f.get("facet", "?"))),
                        f"{float(f.get('score', 0)):.3f}")
        else:
            tbl.add_row(str(f), "-")
    return tbl


def _render_e18(r: Dict[str, Any]) -> Any:
    triples = r.get("entity_triples") or r.get("triples") or []
    if not triples:
        return Text("(no triples)", style="dim")
    tbl = Table(title="entity triples", title_style="bold", padding=(0, 1))
    tbl.add_column("subject")
    tbl.add_column("relation")
    tbl.add_column("object")
    for t in triples[:30]:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            tbl.add_row(str(t[0]), str(t[1]), str(t[2]))
        else:
            tbl.add_row(str(t), "-", "-")
    return tbl


def _render_e19(r: Dict[str, Any]) -> Any:
    patterns = r.get("patterns") or r.get("pattern_list") or []
    if not patterns:
        return Text("(no patterns)", style="dim")
    tbl = Table(title="patterns", title_style="bold", padding=(0, 1))
    tbl.add_column("id / name")
    tbl.add_column("kind")
    tbl.add_column("confidence", justify="right")
    for p in patterns[:30]:
        if isinstance(p, dict):
            tbl.add_row(
                str(p.get("id") or p.get("name", "?")),
                str(p.get("kind", p.get("type", "-"))),
                f"{float(p.get('confidence', 0)):.2f}",
            )
        else:
            tbl.add_row(str(p), "-", "-")
    return tbl


def _render_e23(r: Dict[str, Any], state: Any) -> Any:
    archetype = (_safe(state, "perception", "intent_archetype")
                 or r.get("intent_archetype") or "?")
    confidence = (_safe(state, "perception", "intent_confidence")
                  or r.get("confidence") or 0.0)
    coeffs = r.get("coefficients") or r.get("intent_vector") or \
             _safe(state, "perception", "intent_vector") or {}

    head = Text(f"archetype: {archetype}   confidence: {float(confidence):.2f}",
                style="bold")
    if not coeffs:
        return Group(head, Text("(no coefficients)", style="dim"))

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column()
    for k, v in sorted(coeffs.items(), key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0):
        if isinstance(v, (int, float)):
            tbl.add_row(f"{k}:", _scalar_with_bar(v))
    return Group(head, tbl)


def _render_e28(e28: Any) -> Any:
    if e28 is None:
        return Text("(E28 result unavailable)", style="dim")
    emotions: Optional[Dict[str, float]] = None
    for attr in ("emotions", "emotion_intensities", "intensities"):
        v = getattr(e28, attr, None)
        if isinstance(v, dict):
            emotions = v
            break
    if emotions is None and isinstance(e28, dict):
        emotions = e28.get("emotions") or e28.get("intensities")
    if not emotions:
        return Text("(no emotion data)", style="dim")

    tbl = Table(title="emotion intensities", title_style="bold", padding=(0, 1))
    tbl.add_column("emotion")
    tbl.add_column("intensity")
    sorted_em = sorted(emotions.items(), key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0)
    for name, v in sorted_em[:28]:
        if isinstance(v, (int, float)):
            tbl.add_row(name, _scalar_with_bar(v))
    return tbl


# ---------------------------------------------------------------------------
# show thinking
# ---------------------------------------------------------------------------

def render_thinking(state: Any) -> Any:
    th = getattr(state, "thinking", None)
    trace = getattr(th, "thinking_trace", None) or ""
    skipped = getattr(th, "skipped", False)
    reason = getattr(th, "skip_reason", "") or ""

    body: Any
    if not trace.strip():
        body = Text("(no thinking trace)", style="dim italic")
    else:
        body = Text(trace, style="italic")

    title = "thinking"
    if skipped:
        title = f"thinking — SKIPPED ({reason})" if reason else "thinking — SKIPPED"

    return Panel(body, title=title, title_align="left",
                 border_style="grey50", padding=(0, 1))


# ---------------------------------------------------------------------------
# show classification
# ---------------------------------------------------------------------------

def render_classification(classification: Any) -> Any:
    if classification is None:
        return Text("(no classification — supply text or send a turn first)", style="dim")
    rows = [
        ("input_type",   getattr(classification.input_type, "value", str(classification.input_type))),
        ("sub_type",     getattr(classification.sub_type, "value", str(classification.sub_type))),
        ("variant",      getattr(classification.variant, "value", str(classification.variant))
                         if classification.variant is not None else "-"),
        ("route_target", classification.route_target),
        ("confidence",   f"{classification.confidence:.2f}"),
    ]
    lm = getattr(classification, "learning_mode_number", 0)
    if lm:
        rows.append(("learning_mode", f"M{lm}"))
    ri = getattr(classification, "raw_input", None)
    if ri is not None and getattr(ri, "text", ""):
        rows.append(("raw_text", ri.text[:120]))

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column()
    for k, v in rows:
        tbl.add_row(f"{k}:", str(v))
    return Panel(tbl, title="classification", title_align="left",
                 border_style="cyan", padding=(0, 1))


# ---------------------------------------------------------------------------
# show perception
# ---------------------------------------------------------------------------

def render_perception(state: Any) -> Any:
    p = getattr(state, "perception", None)
    if p is None:
        return Text("(no perception snapshot)", style="dim")

    rows = [
        ("intent_archetype",  p.intent_archetype or "-"),
        ("intent_confidence", f"{p.intent_confidence:.2f}"),
    ]
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column()
    for k, v in rows:
        tbl.add_row(f"{k}:", str(v))

    parts: List[Any] = [Panel(tbl, title="perception", title_align="left",
                              border_style="cyan", padding=(0, 1))]

    if p.intent_vector:
        ivt = Table(title="intent vector", title_style="bold", padding=(0, 1))
        ivt.add_column("axis")
        ivt.add_column("weight", justify="right")
        for k, v in sorted(p.intent_vector.items(), key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0)[:15]:
            if isinstance(v, (int, float)):
                ivt.add_row(k, _scalar_with_bar(v))
        parts.append(ivt)

    if p.ranked_facets:
        ft = Table(title=f"ranked facets ({len(p.ranked_facets)})", title_style="bold", padding=(0, 1))
        ft.add_column("facet")
        ft.add_column("score", justify="right")
        for f in p.ranked_facets[:10]:
            if isinstance(f, dict):
                ft.add_row(str(f.get("name", f.get("facet", "?"))),
                           f"{float(f.get('score', 0)):.3f}")
        parts.append(ft)

    if p.entity_triples:
        et = Table(title=f"entity triples ({len(p.entity_triples)})", title_style="bold", padding=(0, 1))
        et.add_column("subject"); et.add_column("relation"); et.add_column("object")
        for t in p.entity_triples[:10]:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                et.add_row(str(t[0]), str(t[1]), str(t[2]))
        parts.append(et)

    if p.pattern_list:
        pt = Table(title=f"patterns ({len(p.pattern_list)})", title_style="bold", padding=(0, 1))
        pt.add_column("name"); pt.add_column("kind"); pt.add_column("conf", justify="right")
        for pat in p.pattern_list[:10]:
            if isinstance(pat, dict):
                pt.add_row(
                    str(pat.get("name", pat.get("id", "?"))),
                    str(pat.get("kind", pat.get("type", "-"))),
                    f"{float(pat.get('confidence', 0)):.2f}",
                )
        parts.append(pt)

    return Group(*parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(obj: Any, *path: str) -> Any:
    cur = obj
    for p in path:
        if cur is None:
            return None
        cur = getattr(cur, p, None)
    return cur


def _block_bar(v: float, width: int = 4) -> str:
    """Render a fractional value [0,1] as a row of unicode block characters."""
    v = max(0.0, min(1.0, float(v or 0.0)))
    total = v * width * 8
    chars: List[str] = []
    for _ in range(width):
        if total >= 8:
            chars.append("█")
            total -= 8
        elif total > 0:
            idx = int(round(total))
            chars.append(_BLOCKS[idx])
            total = 0
        else:
            chars.append(" ")
    return "".join(chars)


def _bar_style(v: float) -> str:
    v = float(v or 0.0)
    if v >= 0.7:
        return "bold red"
    if v >= 0.4:
        return "yellow"
    if v > 0.0:
        return "green"
    return "dim"


def _scalar_with_bar(v: float) -> Text:
    if not isinstance(v, (int, float)):
        return Text(str(v))
    return Text(f"{_block_bar(float(v))}  {float(v):+.3f}", style=_bar_style(abs(float(v))))


def _extract_score(domain_result: Any) -> float:
    if domain_result is None:
        return 0.0
    if isinstance(domain_result, dict):
        for k in ("score", "weighted_score", "total"):
            v = domain_result.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    for k in ("score", "weighted_score", "total"):
        v = getattr(domain_result, k, None)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _mode_color(mode: str) -> str:
    m = (mode or "").lower()
    if "learning" in m or m.startswith("m"):
        return "blue"
    if "sleep" in m or "rem" in m:
        return "purple"
    if "dream" in m:
        return "magenta"
    if "homework" in m:
        return "yellow"
    if "reflect" in m:
        return "cyan"
    return "grey50"


def _colored_directive(d: str) -> Text:
    d = (d or "").lower()
    if d == "allow":
        return Text("ALLOW", style="bold green")
    if d == "suppress":
        return Text("SUPPRESS", style="bold red")
    if d == "abstain":
        return Text("ABSTAIN", style="bold yellow")
    return Text(d.upper() or "?", style="bold")


def _json_dump(obj: Any) -> Any:
    """Best-effort JSON dump, with truncation for huge strings."""
    import json

    def _default(o: Any) -> Any:
        if hasattr(o, "__dict__"):
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        return str(o)

    def _truncate_strings(o: Any, max_len: int = 500) -> Any:
        if isinstance(o, str) and len(o) > max_len:
            return o[: max_len - 1] + "…"
        if isinstance(o, dict):
            return {k: _truncate_strings(v, max_len) for k, v in o.items()}
        if isinstance(o, list):
            return [_truncate_strings(x, max_len) for x in o]
        return o

    try:
        payload = json.loads(json.dumps(obj, default=_default))
        payload = _truncate_strings(payload)
        return JSON.from_data(payload)
    except Exception as exc:  # noqa: BLE001
        return Text(f"(could not render JSON: {exc})\n\n{repr(obj)[:1000]}", style="dim")
