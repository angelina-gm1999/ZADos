"""Renderers for the `mem` command group.

Covers STMM (active turn state), MTMM (mid-term packet log), and LTMM
(namespaced long-term stores).

LTMM uses dotted paths so the command surface stays uniform:
  mem ltmm identity.core list
  mem ltmm knowledge.lessons show <id>
  mem ltmm thoughts.unsolved_buffer list
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from rich.console import Group
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# STMM
# ---------------------------------------------------------------------------

def render_stmm_current(stmm: Any) -> Any:
    """Labelled snapshot of the active STMM cycle."""
    if stmm is None:
        return Text("(no STMM)", style="dim")

    snap_fn = getattr(stmm, "snapshot", None)
    snap: Dict[str, Any] = {}
    if callable(snap_fn):
        try:
            snap = snap_fn() or {}
            if not isinstance(snap, dict):
                snap = {"snapshot": snap}
        except Exception as exc:  # noqa: BLE001
            snap = {"_snapshot_error": f"{type(exc).__name__}: {exc}"}

    rows: List[Tuple[str, str]] = []
    rows.append(("turn_index", str(getattr(stmm, "turn_index", "?"))))

    # Active message buffer — try to summarize.
    amb = getattr(stmm, "active_message_buffer", None)
    if amb is not None:
        msgs = _first_attr(amb, ("messages", "buffer", "entries"))
        if isinstance(msgs, list):
            rows.append(("messages", f"{len(msgs)} buffered"))
        else:
            rows.append(("active_buffer", type(amb).__name__))

    # Each sub-result block — keep one-liner unless data is small.
    for attr in ("emotion_detection", "fractal_decomposition",
                 "intention_analysis", "memory_contrast", "reward_evaluation"):
        v = getattr(stmm, attr, None)
        if v is None:
            continue
        rows.append((attr, _one_line_summary(v)))

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column(overflow="fold")
    for k, v in rows:
        tbl.add_row(f"{k}:", v)

    parts: List[Any] = [
        Panel(tbl, title="STMM (current cycle)", title_align="left",
              border_style="cyan", padding=(0, 1)),
    ]
    if snap:
        parts.append(Panel(_json_dump(snap), title="snapshot()", title_align="left",
                           border_style="grey50", padding=(0, 1)))
    return Group(*parts)


def render_stmm_tracker(stmm: Any) -> Any:
    """Brain process tracker — phase-completion flags."""
    tracker = getattr(stmm, "brain_process_tracker", None)
    if tracker is None:
        return Text("(no brain_process_tracker)", style="dim")
    flags: Dict[str, Any] = {}
    for attr in ("flags", "state", "stages"):
        v = getattr(tracker, attr, None)
        if isinstance(v, dict):
            flags = v
            break
    if not flags:
        # Fall back to __dict__
        flags = {k: v for k, v in getattr(tracker, "__dict__", {}).items()
                 if not k.startswith("_")}
    if not flags:
        return Text("(tracker has no readable flags)", style="dim")

    tbl = Table(title="brain process tracker", title_style="bold", padding=(0, 1))
    tbl.add_column("stage")
    tbl.add_column("value")
    for k, v in flags.items():
        if isinstance(v, bool):
            tbl.add_row(str(k), Text("✓" if v else "·", style="green" if v else "dim"))
        else:
            tbl.add_row(str(k), str(v))
    return tbl


# ---------------------------------------------------------------------------
# MTMM
# ---------------------------------------------------------------------------

def render_mtmm_packets(mtmm: Any, n: int = 10) -> Any:
    packets = _safe_call(mtmm, "get_all_packets") or []
    if not packets:
        return Text("(no MTMM packets)", style="dim")

    tail = packets[-n:]
    tbl = Table(title=f"last {len(tail)} MTMM packet(s) of {len(packets)} total",
                title_style="bold", padding=(0, 1))
    tbl.add_column("#", style="dim", justify="right")
    tbl.add_column("id")
    tbl.add_column("turn")
    tbl.add_column("user msg", max_width=40, overflow="fold")
    tbl.add_column("response", max_width=40, overflow="fold")
    tbl.add_column("flags")

    start = len(packets) - len(tail)
    for i, p in enumerate(tail):
        idx = start + i
        pid = _safe_get(p, "packet_id", "id") or "-"
        turn = _safe_get(p, "turn_index", "turn", "session_turn") or "-"
        user = _truncate(_safe_get(p, "user_message", "user_msg", "input_text") or "", 60)
        resp = _truncate(_safe_get(p, "response", "answer", "system_response", "final_answer") or "", 60)
        flags = _packet_flags(p)
        tbl.add_row(str(idx), str(pid)[:12], str(turn), user, resp, flags)
    return tbl


def render_mtmm_packet(mtmm: Any, packet_id: str) -> Any:
    """Detail view for one packet — uses get_by_id if available, else linear scan."""
    pkt = _safe_call(mtmm, "get_by_id", packet_id)
    if pkt is None:
        # Try linear scan
        all_p = _safe_call(mtmm, "get_all_packets") or []
        for p in all_p:
            pid = _safe_get(p, "packet_id", "id")
            if str(pid) == packet_id or str(pid).startswith(packet_id):
                pkt = p
                break
    if pkt is None:
        return Text(f"(no packet found for id {packet_id!r})", style="dim")
    return Panel(_object_panel(pkt), title=f"packet {packet_id}",
                 title_align="left", border_style="cyan", padding=(0, 1))


def render_mtmm_trends(mtmm: Any) -> Any:
    trends = getattr(mtmm, "trends", None)
    if trends is None:
        return Text("(no trends)", style="dim")
    return Panel(_object_panel(trends), title="session trends",
                 title_align="left", border_style="green", padding=(0, 1))


# ---------------------------------------------------------------------------
# LTMM — namespace overview + generic store inspector
# ---------------------------------------------------------------------------

# Path map: dotted name -> (memory.<namespace>.<store>)
_LTMM_NAMESPACES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("identity", ("core", "conclusions", "hardcoded", "journal", "pending")),
    ("knowledge", ("lessons", "library", "notebook", "knowledge_maps",
                   "academic_buffer", "academic_questions", "cognitools_data")),
    ("thoughts", ("general_questions", "held_blocks",
                  "overview_logs", "unsolved_buffer")),
)


def render_ltmm_overview(memory: Any) -> Any:
    """Per-namespace summary of each LTMM store with entry count."""
    parts: List[Any] = []
    for ns_name, store_names in _LTMM_NAMESPACES:
        ns = getattr(memory, ns_name, None)
        if ns is None:
            continue
        tbl = Table(title=f"{ns_name}", title_style="bold", padding=(0, 1))
        tbl.add_column("store")
        tbl.add_column("type")
        tbl.add_column("count", justify="right")
        tbl.add_column("hint", style="dim")
        for sname in store_names:
            store = getattr(ns, sname, None)
            if store is None:
                tbl.add_row(sname, "-", "-", "(not present)")
                continue
            count = _count_entries(store)
            cmd_hint = f"mem ltmm {ns_name}.{sname} list"
            tbl.add_row(sname, type(store).__name__, str(count), cmd_hint)
        parts.append(tbl)

    if not parts:
        return Text("(no LTMM namespaces)", style="dim")

    # Header explaining the dotted-path syntax.
    header = Panel(
        Text(
            "LTMM stores — use `mem ltmm <namespace>.<store> list [N]` to browse\n"
            "or `mem ltmm <namespace>.<store> show <id>` for a single entry.",
            style="dim",
        ),
        border_style="grey50", padding=(0, 1),
    )
    return Group(header, *parts)


def resolve_ltmm_store(memory: Any, dotted: str) -> Tuple[Optional[Any], str]:
    """Resolve `<namespace>.<store>` → store instance.  Returns (store, error)."""
    if "." not in dotted:
        return None, f"path must be of form `<namespace>.<store>` (got {dotted!r})"
    ns_name, store_name = dotted.split(".", 1)
    ns = getattr(memory, ns_name, None)
    if ns is None:
        valid = ", ".join(n for n, _ in _LTMM_NAMESPACES)
        return None, f"unknown namespace {ns_name!r}.  Valid: {valid}"
    store = getattr(ns, store_name, None)
    if store is None:
        valid = ", ".join(s for _, ss in _LTMM_NAMESPACES if _ == ns_name for s in ss)
        return None, f"unknown store {store_name!r} in namespace {ns_name!r}"
    return store, ""


def render_ltmm_store_list(store: Any, dotted: str, n: int = 20) -> Any:
    """List entries in a store using whatever accessor it exposes."""
    entries = _list_entries(store)
    if entries is None:
        return Text(f"(store {dotted} exposes no list accessor)", style="dim")
    if not entries:
        return Text(f"({dotted}: empty)", style="dim")

    tail = entries[-n:]
    start = len(entries) - len(tail)
    tbl = Table(title=f"{dotted} — last {len(tail)} of {len(entries)}",
                title_style="bold", padding=(0, 1))
    tbl.add_column("#", style="dim", justify="right")
    tbl.add_column("id")
    tbl.add_column("preview", overflow="fold", max_width=80)

    for i, e in enumerate(tail):
        idx = start + i
        eid = _entry_id(e)
        preview = _entry_preview(e)
        tbl.add_row(str(idx), str(eid)[:14], preview)
    return tbl


def render_ltmm_store_show(store: Any, dotted: str, entry_id: str) -> Any:
    """Show a single entry — get_by_id first, else linear scan by id-prefix match."""
    entry = _safe_call(store, "get_by_id", entry_id)
    if entry is None:
        # Linear scan by prefix
        entries = _list_entries(store) or []
        for e in entries:
            eid = str(_entry_id(e))
            if eid == entry_id or eid.startswith(entry_id):
                entry = e
                break
    if entry is None:
        return Text(f"(no entry in {dotted} with id matching {entry_id!r})", style="dim")
    return Panel(_object_panel(entry), title=f"{dotted}  /  {entry_id}",
                 title_align="left", border_style="cyan", padding=(0, 1))


# ---------------------------------------------------------------------------
# Specialized cross-cutting logs (memory.manager.logs)
# ---------------------------------------------------------------------------

_LOG_NAMES: Tuple[str, ...] = (
    "contradiction", "paradox", "self_reflect", "identity",
    "learning", "dream", "sandbox", "unsolved",
)


def render_logs_overview(memory: Any) -> Any:
    """One-line summary per specialized log."""
    logs_ns = _safe_attr(memory, "manager", "logs")
    if logs_ns is None:
        return Text("(no manager.logs namespace)", style="dim")

    tbl = Table(title="specialized logs", title_style="bold", padding=(0, 1))
    tbl.add_column("log")
    tbl.add_column("type")
    tbl.add_column("count", justify="right")
    tbl.add_column("hint", style="dim")

    for name in _LOG_NAMES:
        log = getattr(logs_ns, name, None)
        if log is None:
            tbl.add_row(name, "-", "-", "(not present)")
            continue
        entries = _list_log_entries(log)
        count = str(len(entries)) if entries is not None else "?"
        tbl.add_row(name, type(log).__name__, count, f"mem logs {name}")
    return tbl


def render_log_entries(memory: Any, log_name: str, n: int = 20) -> Any:
    logs_ns = _safe_attr(memory, "manager", "logs")
    if logs_ns is None:
        return Text("(no manager.logs namespace)", style="dim")
    log = getattr(logs_ns, log_name, None)
    if log is None:
        valid = ", ".join(_LOG_NAMES)
        return Text(f"unknown log {log_name!r}.  valid: {valid}", style="red")

    entries = _list_log_entries(log)
    if entries is None:
        return Text(f"({log_name}: no readable list accessor)", style="dim")
    if not entries:
        return Text(f"({log_name}: empty)", style="dim")

    tail = entries[-n:]
    start = len(entries) - len(tail)
    tbl = Table(title=f"{log_name} — {len(tail)} of {len(entries)}",
                title_style="bold", padding=(0, 1))
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("id")
    tbl.add_column("preview", overflow="fold", max_width=80)
    for i, e in enumerate(tail):
        tbl.add_row(str(start + i), str(_entry_id(e))[:14], _entry_preview(e))
    return tbl


def _list_log_entries(log: Any) -> Optional[List[Any]]:
    # Try common accessors in order.
    for m in ("get_all", "get_unresolved", "get_pending", "get_validated",
              "get_all_active"):
        fn = getattr(log, m, None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    return list(v.values())
                if isinstance(v, tuple):
                    return list(v)
            except Exception:  # noqa: BLE001
                continue
    return None


def _safe_attr(obj: Any, *path: str) -> Any:
    cur = obj
    for p in path:
        if cur is None:
            return None
        cur = getattr(cur, p, None)
    return cur


# ---------------------------------------------------------------------------
# Helpers shared across renderers
# ---------------------------------------------------------------------------

def _object_panel(obj: Any) -> Any:
    """Two-column key/value table over a dataclass/object's public attrs."""
    if is_dataclass(obj):
        items = [(f.name, getattr(obj, f.name, None)) for f in fields(obj)]
    elif isinstance(obj, dict):
        items = list(obj.items())
    else:
        items = [(k, v) for k, v in vars(obj).items() if not k.startswith("_")] \
                if hasattr(obj, "__dict__") else []

    if not items:
        return _json_dump(obj)

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", justify="right")
    tbl.add_column(overflow="fold")
    for k, v in items:
        tbl.add_row(f"{k}:", _format_value(v))
    return tbl


def _format_value(v: Any) -> Any:
    if v is None:
        return Text("None", style="dim")
    if isinstance(v, bool):
        return Text(str(v), style="green" if v else "dim")
    if isinstance(v, (int, float)):
        return Text(str(v))
    if isinstance(v, str):
        return _truncate(v, 200)
    if isinstance(v, (list, tuple)):
        if not v:
            return Text("[]", style="dim")
        return Text(f"[{len(v)} item(s)]: " + _truncate(", ".join(str(x) for x in v[:5]), 120))
    if isinstance(v, dict):
        if not v:
            return Text("{}", style="dim")
        keys = ", ".join(str(k) for k in list(v.keys())[:6])
        return Text(f"{{{len(v)} keys}}: {keys}{'…' if len(v) > 6 else ''}")
    if is_dataclass(v):
        return Text(f"<{type(v).__name__}>")
    return Text(repr(v)[:200])


def _one_line_summary(obj: Any) -> str:
    if obj is None:
        return "-"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}({len(obj)})"
    if isinstance(obj, dict):
        return f"dict({len(obj)} keys)"
    # Look for a 'results' or similar list.
    for attr in ("results", "entries", "items", "data"):
        v = getattr(obj, attr, None)
        if isinstance(v, (list, tuple)):
            return f"{type(obj).__name__} ({len(v)} entries)"
    return type(obj).__name__


def _packet_flags(p: Any) -> str:
    flags: List[str] = []
    for f in ("has_contradiction", "has_paradox", "flagged"):
        if _safe_get(p, f):
            flags.append(f.replace("has_", "").upper()[:3])
    sig = _safe_get(p, "significance", "salience")
    if isinstance(sig, (int, float)) and sig >= 0.7:
        flags.append("HI")
    return " ".join(flags) if flags else "-"


def _count_entries(store: Any) -> int:
    for m in ("get_all", "get_all_active"):
        fn = getattr(store, m, None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, (list, tuple, dict)):
                    return len(v)
            except Exception:  # noqa: BLE001
                pass
    return 0


def _list_entries(store: Any) -> Optional[List[Any]]:
    for m in ("get_all", "get_all_active"):
        fn = getattr(store, m, None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    return list(v.values())
                if isinstance(v, tuple):
                    return list(v)
            except Exception:  # noqa: BLE001
                continue
    return None


def _entry_id(e: Any) -> str:
    for k in ("id", "entry_id", "packet_id", "question_id", "lesson_id",
              "memory_id", "block_id", "log_id", "key"):
        v = _safe_get(e, k)
        if v:
            return str(v)
    return f"<{type(e).__name__}>"


def _entry_preview(e: Any) -> str:
    for k in ("title", "summary", "question_text", "content", "text",
              "verbal_summary", "value", "name"):
        v = _safe_get(e, k)
        if v:
            return _truncate(str(v), 100)
    if isinstance(e, dict):
        return _truncate(str(e), 100)
    return _truncate(repr(e), 100)


def _safe_get(obj: Any, *keys: str) -> Any:
    for k in keys:
        if isinstance(obj, dict):
            if k in obj and obj[k] not in ("", None):
                return obj[k]
        else:
            v = getattr(obj, k, None)
            if v not in ("", None):
                return v
    return None


def _safe_call(obj: Any, method: str, *args: Any) -> Any:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return None


def _first_attr(obj: Any, names: Iterable[str]) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _json_dump(obj: Any) -> Any:
    import json

    def _default(o: Any) -> Any:
        if hasattr(o, "__dict__"):
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        return str(o)

    def _truncate_strings(o: Any, max_len: int = 400) -> Any:
        if isinstance(o, str) and len(o) > max_len:
            return o[: max_len - 1] + "…"
        if isinstance(o, dict):
            return {k: _truncate_strings(v, max_len) for k, v in o.items()}
        if isinstance(o, list):
            return [_truncate_strings(x, max_len) for x in o]
        return o

    try:
        return JSON.from_data(_truncate_strings(json.loads(json.dumps(obj, default=_default))))
    except Exception:  # noqa: BLE001
        return Text(repr(obj)[:1000], style="dim")
