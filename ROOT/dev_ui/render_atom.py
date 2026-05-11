"""Renderers + actions for the `atom` and `map` command groups.

Operates against the AtomSpace engine (E9).  Map management piggybacks on
``memory.knowledge.knowledge_maps`` for persistence — each saved map is one
entry in KnowledgeMapStore containing E9's exported dict.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Group
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ===========================================================================
# ATOM commands
# ===========================================================================

def render_atom_list(
    atomspace: Any,
    type_filter: Optional[str] = None,
    name_filter: Optional[str] = None,
    limit: int = 30,
) -> Any:
    """List atoms in the active AtomSpace."""
    if atomspace is None:
        return Text("(AtomSpace (E9) is not registered)", style="red")

    atoms = _all_atoms(atomspace)
    if type_filter:
        tf = type_filter.lower()
        atoms = [a for a in atoms if _atom_type(a).lower() == tf]
    if name_filter:
        nf = name_filter.lower()
        atoms = [a for a in atoms if nf in (_atom_name(a) or "").lower()]

    total = len(atoms)
    if not atoms:
        msg = f"(no atoms match filters)" if (type_filter or name_filter) \
              else "(AtomSpace is empty)"
        return Text(msg, style="dim")

    tail = atoms[-limit:]
    tbl = Table(
        title=f"atoms — showing {len(tail)} of {total}"
              + (f"  type={type_filter}" if type_filter else "")
              + (f"  name~{name_filter}" if name_filter else ""),
        title_style="bold", padding=(0, 1),
    )
    tbl.add_column("id",   max_width=12, overflow="ellipsis")
    tbl.add_column("type", max_width=18)
    tbl.add_column("name", overflow="fold", max_width=40)
    tbl.add_column("TV (s, c)", justify="right")
    tbl.add_column("AV (sti, lti)", justify="right")

    for a in tail:
        s, c = _truth(a)
        sti, lti = _attention(a)
        tbl.add_row(
            str(_atom_id(a)),
            _atom_type(a),
            _atom_name(a) or "-",
            f"{s:.2f}, {c:.2f}",
            f"{sti:.2f}, {lti:.2f}",
        )
    return tbl


def render_atom_show(atomspace: Any, atom_id: str) -> Any:
    """Full atom detail with incoming + outgoing links."""
    if atomspace is None:
        return Text("(AtomSpace (E9) is not registered)", style="red")

    atom = _get_atom(atomspace, atom_id)
    if atom is None:
        return Text(f"(no atom found for id {atom_id!r})", style="dim")

    s, c = _truth(atom)
    sti, lti = _attention(atom)
    rows: List[Tuple[str, str]] = [
        ("id",   str(_atom_id(atom))),
        ("type", _atom_type(atom)),
        ("name", _atom_name(atom) or "-"),
        ("TV (strength)", f"{s:.3f}"),
        ("TV (confidence)", f"{c:.3f}"),
        ("AV (STI)", f"{sti:.3f}"),
        ("AV (LTI)", f"{lti:.3f}"),
    ]
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right")
    grid.add_column(overflow="fold")
    for k, v in rows:
        grid.add_row(f"{k}:", v)

    parts: List[Any] = [Panel(grid, title=f"atom {_atom_id(atom)}", title_align="left",
                              border_style="cyan", padding=(0, 1))]

    # Outgoing links
    out = _safe_call(atomspace, "get_links_from", _atom_id(atom)) or []
    if out:
        ot = Table(title=f"outgoing links ({len(out)})", title_style="bold", padding=(0, 1))
        ot.add_column("id", max_width=12, overflow="ellipsis")
        ot.add_column("type")
        ot.add_column("target ids")
        for link in out[:30]:
            targets = _outgoing_ids(link, exclude_self=_atom_id(atom))
            ot.add_row(str(_atom_id(link)), _atom_type(link),
                       ", ".join(str(t) for t in targets[:6]) +
                       ("…" if len(targets) > 6 else ""))
        parts.append(ot)

    # Incoming links
    inc = _safe_call(atomspace, "get_links_to", _atom_id(atom)) or \
          _safe_call(atomspace, "get_incoming", _atom_id(atom)) or []
    if inc:
        it = Table(title=f"incoming links ({len(inc)})", title_style="bold", padding=(0, 1))
        it.add_column("id", max_width=12, overflow="ellipsis")
        it.add_column("type")
        it.add_column("source ids")
        for link in inc[:30]:
            sources = _outgoing_ids(link, exclude_self=_atom_id(atom))
            it.add_row(str(_atom_id(link)), _atom_type(link),
                       ", ".join(str(t) for t in sources[:6]) +
                       ("…" if len(sources) > 6 else ""))
        parts.append(it)

    return Group(*parts)


def render_atom_search(atomspace: Any, query: str, limit: int = 30) -> Any:
    """Substring match against atom names."""
    if atomspace is None:
        return Text("(AtomSpace (E9) is not registered)", style="red")
    q = query.lower()
    matches = [a for a in _all_atoms(atomspace)
               if q in (_atom_name(a) or "").lower()]
    if not matches:
        return Text(f"(no atoms with name containing {query!r})", style="dim")
    tail = matches[-limit:]
    tbl = Table(title=f"search '{query}' — {len(matches)} match(es), showing {len(tail)}",
                title_style="bold", padding=(0, 1))
    tbl.add_column("id", max_width=12, overflow="ellipsis")
    tbl.add_column("type")
    tbl.add_column("name", overflow="fold")
    for a in tail:
        tbl.add_row(str(_atom_id(a)), _atom_type(a), _atom_name(a) or "-")
    return tbl


def apply_atom_add_node(atomspace: Any, atom_type: str, name: str,
                        strength: float = 1.0, confidence: float = 0.9) -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    fn = getattr(atomspace, "add_node", None)
    if not callable(fn):
        return "AtomSpace has no add_node() method"
    try:
        atom = fn(atom_type, name)
        if atom is None:
            return f"add_node returned None for ({atom_type}, {name})"
        aid = _atom_id(atom)
        # Best-effort TV update
        upd = getattr(atomspace, "update_truth_value", None)
        if callable(upd):
            try:
                upd(aid, strength=strength, confidence=confidence)
            except Exception:  # noqa: BLE001
                pass
        return f"added atom {aid} ({atom_type}: {name})"
    except Exception as exc:  # noqa: BLE001
        return f"add_node failed: {type(exc).__name__}: {exc}"


def apply_atom_add_link(atomspace: Any, link_type: str, outgoing_ids: List[str],
                        strength: float = 1.0, confidence: float = 0.9) -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    fn = getattr(atomspace, "add_link", None)
    if not callable(fn):
        return "AtomSpace has no add_link() method"
    # Resolve atom ids to atom objects (some impls take objects, some ids)
    atoms = []
    for aid in outgoing_ids:
        a = _get_atom(atomspace, aid)
        if a is None:
            return f"unknown atom id: {aid}"
        atoms.append(a)
    # Try both calling conventions
    try:
        link = fn(link_type, atoms)
    except Exception:  # noqa: BLE001
        try:
            link = fn(link_type, [_atom_id(a) for a in atoms])
        except Exception as exc:  # noqa: BLE001
            return f"add_link failed: {type(exc).__name__}: {exc}"
    if link is None:
        return f"add_link returned None for ({link_type}, {outgoing_ids})"
    lid = _atom_id(link)
    upd = getattr(atomspace, "update_truth_value", None)
    if callable(upd):
        try:
            upd(lid, strength=strength, confidence=confidence)
        except Exception:  # noqa: BLE001
            pass
    return f"added link {lid} ({link_type}: {' → '.join(outgoing_ids)})"


def apply_atom_set(atomspace: Any, atom_id: str,
                   strength: Optional[float] = None,
                   confidence: Optional[float] = None,
                   sti: Optional[float] = None,
                   lti: Optional[float] = None) -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    if _get_atom(atomspace, atom_id) is None:
        return f"unknown atom id: {atom_id}"
    msgs: List[str] = []
    if strength is not None or confidence is not None:
        fn = getattr(atomspace, "update_truth_value", None)
        if callable(fn):
            try:
                fn(atom_id, strength=strength, confidence=confidence)
                msgs.append(f"TV: s={strength}, c={confidence}")
            except Exception as exc:  # noqa: BLE001
                msgs.append(f"TV update failed: {exc}")
    if sti is not None or lti is not None:
        fn = getattr(atomspace, "update_attention_value", None)
        if callable(fn):
            try:
                fn(atom_id, sti=sti, lti=lti)
                msgs.append(f"AV: sti={sti}, lti={lti}")
            except Exception as exc:  # noqa: BLE001
                msgs.append(f"AV update failed: {exc}")
    return f"atom {atom_id} updated" + (" — " + "; ".join(msgs) if msgs else "")


def apply_atom_delete(atomspace: Any, atom_id_or_name: str) -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    # Resolve name/prefix → real id so remove_atom doesn't silently no-op.
    atom = _get_atom(atomspace, atom_id_or_name)
    if atom is None:
        # Try name lookup
        by_name = _safe_call(atomspace, "get_by_name", atom_id_or_name)
        if isinstance(by_name, list) and by_name:
            atom = by_name[0]
        elif by_name is not None and not isinstance(by_name, list):
            atom = by_name
    if atom is None:
        return f"no atom found matching {atom_id_or_name!r}"

    real_id = _atom_id(atom)
    fn = getattr(atomspace, "remove_atom", None)
    if not callable(fn):
        return "AtomSpace has no remove_atom() method"
    try:
        result = fn(real_id)
    except Exception as exc:  # noqa: BLE001
        return f"remove_atom failed: {type(exc).__name__}: {exc}"
    # Confirm it really went away.
    if _get_atom(atomspace, str(real_id)) is None:
        return f"atom {real_id} removed"
    return f"remove_atom({real_id}) returned {result!r} but atom still present"


def render_atom_status(atomspace: Any) -> Any:
    if atomspace is None:
        return Text("(AtomSpace (E9) is not registered)", style="red")
    status = _safe_call(atomspace, "get_status") or {}
    total = status.get("total_atoms") if isinstance(status, dict) else None
    if total is None:
        total = len(_all_atoms(atomspace))

    nt_levels = status.get("nt_levels") if isinstance(status, dict) else None

    rows: List[Tuple[str, str]] = [
        ("engine_id",     str(status.get("engine_id", "atomspace_engine"))),
        ("cluster",       str(status.get("cluster", "-"))),
        ("total_atoms",   str(total)),
        ("next_timetag",  str(status.get("next_timetag", "-"))),
        ("tick_counter",  str(status.get("tick_counter", "-"))),
        ("mode",          str(status.get("mode", "-"))),
    ]

    # Type histogram from live atoms.
    type_counts: Dict[str, int] = {}
    for a in _all_atoms(atomspace):
        t = _atom_type(a)
        type_counts[t] = type_counts.get(t, 0) + 1

    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right")
    grid.add_column()
    for k, v in rows:
        grid.add_row(f"{k}:", v)

    parts: List[Any] = [
        Panel(grid, title="AtomSpace (E9)", title_align="left",
              border_style="cyan", padding=(0, 1)),
    ]
    if type_counts:
        tt = Table(title=f"type histogram ({len(type_counts)} types)",
                   title_style="bold", padding=(0, 1))
        tt.add_column("type")
        tt.add_column("count", justify="right")
        for t, n in sorted(type_counts.items(), key=lambda kv: -kv[1]):
            tt.add_row(t, str(n))
        parts.append(tt)
    if isinstance(nt_levels, dict) and nt_levels:
        ntt = Table.grid(padding=(0, 1))
        ntt.add_column(style="dim", justify="right")
        ntt.add_column()
        for k, v in nt_levels.items():
            if isinstance(v, (int, float)):
                ntt.add_row(f"{k}:", f"{v:.3f}")
        parts.append(Panel(ntt, title="NT modulation state",
                           title_align="left", border_style="grey50",
                           padding=(0, 1)))
    return Group(*parts)


# ===========================================================================
# MAP commands — AtomSpace snapshots persisted as JSON files on disk
# ===========================================================================
#
# We use plain JSON files rather than KnowledgeMapStore because the latter
# is a concept-graph store (KnowledgeMap dataclass with nodes/links/lessons),
# not a generic AtomSpace-export container.  KnowledgeMapStore stays
# browsable via `mem ltmm knowledge.knowledge_maps list`.

# Default location: ROOT/dev_ui_maps/
def _maps_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, "dev_ui_maps")


def render_map_list(memory: Any) -> Any:
    d = _maps_dir()
    if not os.path.isdir(d):
        return Text(f"(no saved maps — directory not created yet: {d})", style="dim")
    files = [f for f in sorted(os.listdir(d)) if f.endswith(".json")]
    if not files:
        return Text(f"(no saved maps in {d})", style="dim")
    tbl = Table(title=f"saved AtomSpace snapshots ({len(files)}) in {d}",
                title_style="bold", padding=(0, 1))
    tbl.add_column("name")
    tbl.add_column("size",        justify="right")
    tbl.add_column("atoms",       justify="right")
    tbl.add_column("saved at",    style="dim")
    for fname in files:
        full = os.path.join(d, fname)
        try:
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
            atoms_n = len(data.get("atoms", [])) if isinstance(data, dict) else "-"
        except Exception:  # noqa: BLE001
            atoms_n = "?"
        st = os.stat(full)
        size_kb = f"{st.st_size / 1024:.1f}K"
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        tbl.add_row(fname[:-5], size_kb, str(atoms_n), ts)
    return tbl


def apply_map_save(memory: Any, atomspace: Any, name: str,
                   description: str = "") -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    if not name or "/" in name or "\\" in name:
        return f"invalid map name: {name!r}"
    fn = getattr(atomspace, "export_to_dict", None)
    if not callable(fn):
        return "AtomSpace has no export_to_dict() method"
    try:
        export = fn()
    except Exception as exc:  # noqa: BLE001
        return f"export_to_dict failed: {type(exc).__name__}: {exc}"

    d = _maps_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    payload = {
        "_dev_ui_map": True,
        "name": name,
        "description": description,
        "saved_at": time.time(),
        "atom_count": len(export.get("atoms", [])),
        **export,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        return f"map save failed: {type(exc).__name__}: {exc}"
    return f"map saved: {path} ({payload['atom_count']} atoms)"


def apply_map_load(memory: Any, atomspace: Any, name: str) -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    d = _maps_dir()
    path = os.path.join(d, f"{name}.json")
    if not os.path.isfile(path):
        return f"no saved map: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return f"failed to read {path}: {type(exc).__name__}: {exc}"
    fn = getattr(atomspace, "import_from_dict", None)
    if not callable(fn):
        return "AtomSpace has no import_from_dict() method"
    try:
        fn(data)
    except Exception as exc:  # noqa: BLE001
        return f"import_from_dict failed: {type(exc).__name__}: {exc}"
    return f"map loaded from {path} ({len(data.get('atoms', []))} atoms in payload)"


def apply_map_export_file(atomspace: Any, path: str) -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    try:
        export = atomspace.export_to_dict()
    except Exception as exc:  # noqa: BLE001
        return f"export_to_dict failed: {type(exc).__name__}: {exc}"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        return f"failed to write {path}: {type(exc).__name__}: {exc}"
    return f"exported {len(export.get('atoms', []))} atoms → {path}"


def apply_map_import_file(atomspace: Any, path: str) -> str:
    if atomspace is None:
        return "AtomSpace (E9) is not registered"
    if not os.path.isfile(path):
        return f"file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return f"failed to read {path}: {type(exc).__name__}: {exc}"
    try:
        atomspace.import_from_dict(data)
    except Exception as exc:  # noqa: BLE001
        return f"import_from_dict failed: {type(exc).__name__}: {exc}"
    return f"imported {len(data.get('atoms', []))} atoms from {path}"


# ===========================================================================
# Helpers
# ===========================================================================

def _all_atoms(atomspace: Any) -> List[Any]:
    for m in ("get_all_atoms", "list_atoms"):
        fn = getattr(atomspace, m, None)
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
    return []


def _get_atom(atomspace: Any, atom_id: str) -> Any:
    for m in ("get_atom", "get_by_id"):
        fn = getattr(atomspace, m, None)
        if callable(fn):
            try:
                v = fn(atom_id)
                if v is not None:
                    return v
            except Exception:  # noqa: BLE001
                continue
    # Linear scan for prefix match (atom ids can be timetag ints)
    for a in _all_atoms(atomspace):
        aid = str(_atom_id(a))
        if aid == atom_id or aid.startswith(atom_id):
            return a
    return None


def _atom_id(atom: Any) -> Any:
    for k in ("id", "atom_id", "timetag", "tag", "uuid"):
        v = _attr(atom, k)
        if v is not None:
            return v
    return id(atom)


def _atom_type(atom: Any) -> str:
    for k in ("atom_type", "type", "node_type", "link_type"):
        v = _attr(atom, k)
        if v is not None:
            return str(getattr(v, "name", v))
    return type(atom).__name__


def _atom_name(atom: Any) -> Optional[str]:
    for k in ("name", "value", "label"):
        v = _attr(atom, k)
        if v is not None and not isinstance(v, (list, tuple, dict)):
            return str(v)
    return None


def _truth(atom: Any) -> Tuple[float, float]:
    tv = _attr(atom, "truth_value", "tv")
    if tv is not None:
        s = _attr(tv, "strength", "s")
        c = _attr(tv, "confidence", "c")
        return (float(s or 0.0), float(c or 0.0))
    return (float(_attr(atom, "strength", "s") or 0.0),
            float(_attr(atom, "confidence", "c") or 0.0))


def _attention(atom: Any) -> Tuple[float, float]:
    av = _attr(atom, "attention_value", "av")
    if av is not None:
        return (float(_attr(av, "sti", "STI") or 0.0),
                float(_attr(av, "lti", "LTI") or 0.0))
    return (float(_attr(atom, "sti", "STI") or 0.0),
            float(_attr(atom, "lti", "LTI") or 0.0))


def _outgoing_ids(link: Any, exclude_self: Any = None) -> List[Any]:
    outs = _attr(link, "outgoing", "outgoing_set", "targets")
    if not isinstance(outs, (list, tuple)):
        return []
    result = []
    for o in outs:
        oid = _atom_id(o) if not isinstance(o, (str, int)) else o
        if oid == exclude_self:
            continue
        result.append(oid)
    return result


def _attr(obj: Any, *keys: str) -> Any:
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


def _get_map_store(memory: Any) -> Any:
    try:
        return memory.knowledge.knowledge_maps
    except AttributeError:
        return None


def _find_map_entry(store: Any, name_or_id: str) -> Any:
    entries = _safe_call(store, "get_all") or []
    for e in entries:
        if str(_entry_id(e)) == name_or_id:
            return e
        if str(_attr(e, "name", "title", "label") or "") == name_or_id:
            return e
    # Prefix match
    for e in entries:
        eid = str(_entry_id(e))
        if eid.startswith(name_or_id):
            return e
    return None


def _entry_id(e: Any) -> Any:
    return _attr(e, "id", "entry_id", "map_id", "key") or f"<{type(e).__name__}>"
