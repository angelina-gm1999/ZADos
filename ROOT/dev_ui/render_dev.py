"""Renderers for the `dev` and `nt` command groups."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Group
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ===========================================================================
# REWARD
# ===========================================================================

def render_reward_profiles() -> Any:
    """List every profile in PROFILE_REGISTRY with its 4 domain weights."""
    try:
        from zados.reward.profile import PROFILE_REGISTRY
    except Exception as exc:  # noqa: BLE001
        return Text(f"(cannot import PROFILE_REGISTRY: {exc})", style="red")

    tbl = Table(title="reward profiles", title_style="bold", padding=(0, 1))
    tbl.add_column("name")
    tbl.add_column("logic", justify="right")
    tbl.add_column("ethics", justify="right")
    tbl.add_column("innovation", justify="right")
    tbl.add_column("attunement", justify="right")
    tbl.add_column("supr.", justify="right", style="dim")
    tbl.add_column("abst.", justify="right", style="dim")

    for name in sorted(PROFILE_REGISTRY.keys()):
        p = PROFILE_REGISTRY[name]
        dw = p.domain_weights or {}
        tbl.add_row(
            name,
            f"{dw.get('logic', 0):.2f}",
            f"{dw.get('ethics', 0):.2f}",
            f"{dw.get('innovation', 0):.2f}",
            f"{dw.get('human_attunement', 0):.2f}",
            f"{p.suppression_bias:.2f}",
            f"{p.abstention_bias:.2f}",
        )
    return tbl


def render_reward_profile_detail(name: str) -> Any:
    try:
        from zados.reward.profile import PROFILE_REGISTRY
    except Exception as exc:  # noqa: BLE001
        return Text(f"(cannot import PROFILE_REGISTRY: {exc})", style="red")

    p = PROFILE_REGISTRY.get(name)
    if p is None:
        valid = ", ".join(sorted(PROFILE_REGISTRY.keys()))
        return Text(f"unknown profile: {name!r}\nvalid: {valid}", style="red")

    rows: List[Tuple[str, str]] = [
        ("name", str(p.name)),
        ("suppression_bias", f"{p.suppression_bias:.3f}"),
        ("abstention_bias", f"{p.abstention_bias:.3f}"),
    ]
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right")
    grid.add_column()
    for k, v in rows:
        grid.add_row(f"{k}:", v)

    dw_tbl = Table(title="domain weights", title_style="bold", padding=(0, 1))
    dw_tbl.add_column("domain")
    dw_tbl.add_column("weight", justify="right")
    for d, w in (p.domain_weights or {}).items():
        dw_tbl.add_row(d, f"{w:.3f}")

    parts: List[Any] = [
        Panel(grid, title=f"profile: {name}", title_align="left",
              border_style="cyan", padding=(0, 1)),
        dw_tbl,
    ]
    if p.threshold_tolerances:
        thr_tbl = Table(title="threshold tolerances", title_style="bold", padding=(0, 1))
        thr_tbl.add_column("threshold")
        thr_tbl.add_column("value", justify="right")
        for k, v in p.threshold_tolerances.items():
            thr_tbl.add_row(k, f"{v:.3f}")
        parts.append(thr_tbl)
    return Group(*parts)


def render_reward_map() -> Any:
    try:
        from zados.core.mode_profiles import MODE_TO_PROFILE
    except Exception as exc:  # noqa: BLE001
        return Text(f"(cannot import MODE_TO_PROFILE: {exc})", style="red")

    tbl = Table(title="MODE_TO_PROFILE", title_style="bold", padding=(0, 1))
    tbl.add_column("mode token")
    tbl.add_column("reward profile")
    for mode, profile in MODE_TO_PROFILE.items():
        tbl.add_row(mode, profile)
    return tbl


def render_reward_learned(session: Any) -> Any:
    if session is None:
        return Text("(no session)", style="dim")
    learned = getattr(session, "learned_domain_weights", {}) or {}
    if not learned:
        return Text("(no learned weights yet — E17 has not adjusted any domain)", style="dim")

    tbl = Table(title="learned_domain_weights", title_style="bold", padding=(0, 1))
    tbl.add_column("key")
    tbl.add_column("value", justify="right")
    for k, v in sorted(learned.items()):
        if isinstance(v, (int, float)):
            tbl.add_row(k, f"{v:+.3f}")
        else:
            tbl.add_row(k, str(v))
    return tbl


# ===========================================================================
# NEUROCHEM
# ===========================================================================

# canonical order shared with render_show
_NT_ORDER_UPPER = ("GLU", "GABA", "DA", "5HT", "NE", "ACh", "OXT", "MOR",
                   "CB1", "CRH", "cortisol", "histamine")


def render_nt_state(neurochem: Any, full: bool = False) -> Any:
    """Tabular view of current NT concentrations + tonic/phasic/F."""
    if neurochem is None:
        return Text("(no neurochem engine)", style="dim")
    reg = neurochem.registry

    tbl = Table(title="neurotransmitter state", title_style="bold", padding=(0, 1))
    tbl.add_column("NT")
    tbl.add_column("C", justify="right")
    if full:
        tbl.add_column("C_tonic", justify="right")
        tbl.add_column("C_phasic", justify="right")
        tbl.add_column("F (fatigue)", justify="right")

    for name in _NT_ORDER_UPPER:
        try:
            nt = reg.get_neurotransmitter(name)
        except Exception:  # noqa: BLE001
            continue
        if nt is None:
            continue
        c = float(getattr(nt, "C", 0.0) or 0.0)
        row = [name, f"{c:.3f}"]
        if full:
            row += [
                f"{float(getattr(nt, 'C_tonic', 0.0) or 0.0):.3f}",
                f"{float(getattr(nt, 'C_phasic', 0.0) or 0.0):.3f}",
                f"{float(getattr(nt, 'F', 0.0) or 0.0):.3f}",
            ]
        tbl.add_row(*row)

    osc = reg.get_oscillations()
    parts: List[Any] = [tbl]
    if osc is not None:
        otbl = Table(title="oscillations (live)", title_style="bold", padding=(0, 1))
        otbl.add_column("band")
        otbl.add_column("amplitude", justify="right")
        for band in ("delta", "theta", "alpha", "beta", "gamma", "sigma"):
            v = float(getattr(osc, band, 0.0) or 0.0)
            otbl.add_row(band, f"{v:.3f}")
        parts.append(otbl)

    metrics_dict = {}
    try:
        v = neurochem.get_neurosymbolic_readout()
        if hasattr(v, "as_dict"):
            metrics_dict = v.as_dict()
        elif isinstance(v, dict):
            metrics_dict = v
    except Exception:  # noqa: BLE001
        pass
    if metrics_dict:
        mtbl = Table(title="neurosymbolic readout", title_style="bold", padding=(0, 1))
        mtbl.add_column("metric")
        mtbl.add_column("value", justify="right")
        for k, v in sorted(metrics_dict.items()):
            if isinstance(v, (int, float)):
                mtbl.add_row(k, f"{v:.3f}")
        parts.append(mtbl)

    return Group(*parts)


def render_nt_metrics_only(neurochem: Any) -> Any:
    metrics_dict = {}
    try:
        v = neurochem.get_neurosymbolic_readout()
        if hasattr(v, "as_dict"):
            metrics_dict = v.as_dict()
        elif isinstance(v, dict):
            metrics_dict = v
    except Exception as exc:  # noqa: BLE001
        return Text(f"(readout failed: {exc})", style="red")

    if not metrics_dict:
        return Text("(no metrics)", style="dim")
    tbl = Table(title="NeurochemicalMetrics", title_style="bold", padding=(0, 1))
    tbl.add_column("metric")
    tbl.add_column("value", justify="right")
    for k, v in sorted(metrics_dict.items()):
        if isinstance(v, (int, float)):
            tbl.add_row(k, f"{v:.3f}")
    return tbl


def apply_nt_set(neurochem: Any, name: str, value: float) -> str:
    """Set a single NT concentration. Returns status string."""
    if neurochem is None:
        return "no neurochem engine"
    canonical = _canon_nt_name(name)
    try:
        nt = neurochem.registry.get_neurotransmitter(canonical)
    except Exception as exc:  # noqa: BLE001
        return f"failed to fetch NT {canonical!r}: {exc}"
    if nt is None:
        valid = neurochem.registry.neurotransmitter_names()
        return f"unknown NT {name!r} (canonical: {canonical!r}). Valid: {', '.join(valid)}"
    v = max(0.0, min(1.0, float(value)))
    if hasattr(nt, "set_concentration"):
        nt.set_concentration(v)
    else:
        nt.C = v
    return f"{canonical}.C = {v:.3f}"


def apply_nt_reset(neurochem: Any) -> str:
    """Reset all NTs to their baseline (C_baseline if available, else 0.5)."""
    if neurochem is None:
        return "no neurochem engine"
    reg = neurochem.registry
    n = 0
    for name in reg.neurotransmitter_names():
        try:
            nt = reg.get_neurotransmitter(name)
            cfg = reg.get_config(name) if hasattr(reg, "get_config") else None
            baseline = getattr(cfg, "C_baseline", None) if cfg is not None else None
            v = float(baseline) if isinstance(baseline, (int, float)) else 0.5
            if hasattr(nt, "set_concentration"):
                nt.set_concentration(v)
            else:
                nt.C = v
            n += 1
        except Exception:  # noqa: BLE001
            continue
    return f"reset {n} NT(s) to baseline"


def apply_reward_override(session: Any, weights: Dict[str, float]) -> str:
    """Write all four domain weights into session.learned_domain_weights.

    Keys accepted: logic, ethics, innovation, attunement.
    Stored with `_weight` suffix to match E17 / Phase 5 conventions.
    """
    if session is None:
        return "no session"
    out = dict(getattr(session, "learned_domain_weights", {}) or {})
    for k in ("logic", "ethics", "innovation", "attunement"):
        if k in weights:
            out[f"{k}_weight"] = float(max(0.0, min(1.0, weights[k])))
    session.learned_domain_weights = out
    rendered = ", ".join(f"{k}={v:.2f}" for k, v in sorted(out.items()))
    return f"learned_domain_weights = {{{rendered}}}"


def apply_reward_reset(session: Any) -> str:
    if session is None:
        return "no session"
    session.learned_domain_weights = {}
    return "learned_domain_weights cleared — static profile in effect"


# ===========================================================================
# PIPELINE
# ===========================================================================

def render_pipeline_last(result: Any, full: bool = False) -> Any:
    """JSON-tree dump of the last PipelineResult."""
    if result is None:
        return Text("(no turn yet)", style="dim")
    return Panel(
        _json_dump_state(result, full=full),
        title="last pipeline result",
        title_align="left", border_style="cyan", padding=(0, 1),
    )


def render_pipeline_dispatch(result: Any) -> Any:
    """Engine dispatch log: per-engine timing + summary keys."""
    state = getattr(result, "state", None) if result is not None else None
    if state is None:
        return Text("(no pipeline state)", style="dim")
    dispatch = getattr(state, "dispatch", None)
    if dispatch is None:
        return Text("(no dispatch result)", style="dim")

    results = getattr(dispatch, "engine_results", {}) or {}
    run = list(getattr(dispatch, "engines_run", []) or [])
    skipped = list(getattr(dispatch, "engines_skipped", []) or [])

    tbl = Table(title="engine dispatch (this turn)", title_style="bold", padding=(0, 1))
    tbl.add_column("E#", justify="right")
    tbl.add_column("status")
    tbl.add_column("ms", justify="right")
    tbl.add_column("keys", overflow="fold", max_width=70)

    for eid in sorted(set(run) | set(skipped)):
        if eid in skipped:
            tbl.add_row(f"E{eid}", Text("SKIP", style="yellow"), "-", "-")
            continue
        res = results.get(eid, {})
        ms = "-"
        if isinstance(res, dict) and "processing_time_ms" in res:
            ms = f"{float(res['processing_time_ms']):.3f}"
        keys: List[str] = []
        if isinstance(res, dict):
            keys = [k for k in res.keys() if not k.startswith("_")]
        tbl.add_row(f"E{eid}", Text("RUN", style="green"), ms, ", ".join(keys[:10]))

    total_ms = sum(
        float(r.get("processing_time_ms", 0.0))
        for r in results.values() if isinstance(r, dict)
    )
    summary = Text(
        f"\n{len(run)} engines ran  /  {len(skipped)} skipped  /  "
        f"total dispatch ≈ {total_ms:.3f} ms",
        style="dim",
    )
    return Group(tbl, summary)


def render_pipeline_errors(errors: List[dict]) -> Any:
    if not errors:
        return Text("(no errors captured in this session)", style="dim green")
    tbl = Table(title=f"runtime errors ({len(errors)})", title_style="bold red",
                padding=(0, 1))
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("time", style="dim")
    tbl.add_column("context")
    tbl.add_column("type", style="red")
    tbl.add_column("message", overflow="fold", max_width=70)
    for i, e in enumerate(errors):
        t = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        tbl.add_row(str(i), t, e["context"], e["type"], e["message"][:200])
    return tbl


def render_pipeline_error_detail(errors: List[dict], idx: int) -> Any:
    if not (0 <= idx < len(errors)):
        return Text(f"error {idx} out of range (have {len(errors)})", style="red")
    e = errors[idx]
    rows = [
        ("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["timestamp"]))),
        ("context", e["context"]),
        ("turn", str(e["turn"])),
        ("type", e["type"]),
        ("message", e["message"]),
    ]
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right")
    grid.add_column(overflow="fold")
    for k, v in rows:
        grid.add_row(f"{k}:", v)
    return Group(
        Panel(grid, title=f"error #{idx}", title_align="left",
              border_style="red", padding=(0, 1)),
        Panel(Text(e["traceback"], style="dim red"), title="traceback",
              title_align="left", border_style="grey50", padding=(0, 1)),
    )


# ===========================================================================
# Internal helpers
# ===========================================================================

_NT_NAME_ALIASES: Dict[str, str] = {
    "da":    "DA",  "dopamine":   "DA",
    "5ht":   "5HT", "5-ht":       "5HT", "serotonin":  "5HT",
    "ne":    "NE",  "norepinephrine": "NE",
    "ach":   "ACh", "acetylcholine":  "ACh",
    "oxt":   "OXT", "oxytocin":   "OXT",
    "mor":   "MOR", "mu-opioid":  "MOR", "opioid": "MOR",
    "cb1":   "CB1", "endocannabinoid": "CB1",
    "crh":   "CRH",
    "cor":   "cortisol", "cortisol":  "cortisol",
    "gaba":  "GABA",
    "glu":   "GLU", "glutamate":  "GLU",
    "his":   "histamine", "histamine":  "histamine",
}


def _canon_nt_name(name: str) -> str:
    return _NT_NAME_ALIASES.get(name.strip().lower(), name)


def _json_dump_state(result: Any, full: bool = False) -> Any:
    def _default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if hasattr(o, "as_dict"):
            try:
                return o.as_dict()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(o, "__dict__"):
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        return str(o)

    def _truncate(o: Any, max_len: int) -> Any:
        if isinstance(o, str) and len(o) > max_len:
            return o[: max_len - 1] + "…"
        if isinstance(o, dict):
            return {k: _truncate(v, max_len) for k, v in o.items()}
        if isinstance(o, list):
            return [_truncate(x, max_len) for x in o]
        return o

    try:
        raw = json.loads(json.dumps(result, default=_default))
        if not full:
            raw = _truncate(raw, 500)
        return JSON.from_data(raw)
    except Exception as exc:  # noqa: BLE001
        return Text(f"(JSON dump failed: {exc})\n\n{repr(result)[:2000]}", style="dim")
