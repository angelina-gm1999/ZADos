"""
ZADOS Bridge Server — JSON Serializers.

safe_json(obj) converts any ZADOS dataclass / result object into a
plain dict/list/primitive that FastAPI can return as JSON.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any


def safe_json(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert obj to a JSON-safe structure.

    Handles:
      - None, bool, int, float, str  → as-is
      - enum.Enum                    → .value
      - dataclasses                  → dict of fields (recursive)
      - dict                         → recurse on values
      - list / tuple / set           → list (recursive)
      - objects with .as_dict()      → call it, then recurse
      - anything else                → str(obj)

    _depth caps recursion at 12 to prevent runaway on circular refs.
    """
    if _depth > 12:
        return str(obj)

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, enum.Enum):
        return obj.value

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: safe_json(getattr(obj, f.name), _depth + 1)
            for f in dataclasses.fields(obj)
        }

    if isinstance(obj, dict):
        return {str(k): safe_json(v, _depth + 1) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [safe_json(item, _depth + 1) for item in obj]

    if hasattr(obj, "as_dict"):
        try:
            return safe_json(obj.as_dict(), _depth + 1)
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return safe_json(obj.__dict__, _depth + 1)
        except Exception:
            pass

    return str(obj)


def session_snapshot(session: Any) -> dict:
    """Return the subset of SessionState the Godot client needs every turn."""
    if session is None:
        return {}
    return {
        "session_id":          getattr(session, "session_id", ""),
        "branch":              getattr(session, "branch", "C"),
        "turn_count":          getattr(session, "turn_count", 0),
        "active_mode":         getattr(session, "initial_mode", "Normal"),
        "session_mode":        getattr(session, "session_mode", "regular"),
        "active_learning_mode": getattr(session, "active_learning_mode", None),
        "reward_profile_name": getattr(session, "reward_profile_name", "regular_input"),
    }


def pipeline_state_snapshot(state: Any) -> dict:
    """Serialize the PipelineState phases into a compact dict for the client."""
    if state is None:
        return {}

    def _mod(m: Any) -> dict:
        if m is None:
            return {}
        return {
            "mode_token":          getattr(m, "mode_token", ""),
            "reward_profile_name": getattr(m, "reward_profile_name", ""),
            "engine_weights":      safe_json(getattr(m, "engine_weights", {})),
            "metrics_dict":        safe_json(getattr(m, "metrics_dict", {})),
            "nt_snapshot":         safe_json(getattr(m, "nt_snapshot", {})),
            "osc_snapshot":        safe_json(getattr(m, "osc_snapshot", {})),
        }

    def _dispatch(d: Any) -> dict:
        if d is None:
            return {}
        return {
            "engines_run":     safe_json(getattr(d, "engines_run", [])),
            "engines_skipped": safe_json(getattr(d, "engines_skipped", [])),
            "engine_results":  safe_json(getattr(d, "engine_results", {})),
            "e28_result":      safe_json(getattr(d, "e28_result", None)),
        }

    return {
        "turn_index": getattr(state, "turn_index", 0),
        "perception": safe_json(getattr(state, "perception", None)),
        "modulation": _mod(getattr(state, "modulation", None)),
        "dispatch":   _dispatch(getattr(state, "dispatch", None)),
        "thinking":   safe_json(getattr(state, "thinking", None)),
        "reward":     safe_json(getattr(state, "reward", None)),
        "answer":     safe_json(getattr(state, "answer", None)),
    }


def process_response(pipeline_result: Any, session: Any) -> dict:
    """Build the full response dict for POST /process."""
    final_answer = getattr(pipeline_result, "final_answer", "") or ""
    directive    = getattr(pipeline_result, "directive", "allow") or "allow"
    state        = getattr(pipeline_result, "state", None)

    # LearningModeResult wraps PipelineResult in .pipeline_result
    if not hasattr(pipeline_result, "final_answer"):
        inner = getattr(pipeline_result, "pipeline_result", None)
        if inner is not None:
            final_answer = getattr(inner, "final_answer", "") or ""
            directive    = getattr(inner, "directive", "allow") or "allow"
            state        = getattr(inner, "state", None)

    return {
        "final_answer": final_answer,
        "directive":    directive,
        "session":      session_snapshot(session),
        "state":        pipeline_state_snapshot(state),
    }
