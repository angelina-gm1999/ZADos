"""
Unified mastergrid parser for neurosymbolic encoding (Appendix K.7).

Parses and encodes all 6 canonical forms:
1. Base triplet:      NT->R:MOD[,MOD...]
2. Gated triplet:     GATE{NT->R:MOD[,MOD...]}
3. Phasic/tonic:      NT.P->R:MOD or NT.T->R:MOD (also NT•/NT~)
4. Plasticity ops:    INT(R), UPR(R), SWITCH(Ra->Rb)
5. State expression:  STATE(Name)=w1*Var+w2*Var-w3*Var
6. Conditional trigger: IF(condition)=>action1;action2;ACTIVATE(Mode)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Parsed data structures (frozen for immutability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedTriplet:
    """Parsed base or gated triplet (K.7 Forms 1-3)."""
    nt: str
    receptor: str
    modifiers: Tuple[str, ...]
    gate: Optional[str] = None
    signal_mode: Optional[str] = None  # "phasic" / "tonic" / None


@dataclass(frozen=True)
class ParsedOperator:
    """Parsed plasticity operator (K.7 Form 4)."""
    operator: str       # "INT", "UPR", "SWITCH", "DSN", "REC"
    target: str
    target_b: Optional[str] = None  # for SWITCH


@dataclass(frozen=True)
class ParsedStateExpr:
    """Parsed composite state expression (K.7 Form 5)."""
    name: str
    terms: Tuple[Tuple[float, str], ...]  # (weight, variable_key)


@dataclass(frozen=True)
class ParsedTrigger:
    """Parsed conditional trigger (K.7 Form 6)."""
    condition: str
    actions: Tuple  # of ParsedTriplet | ParsedOperator | str
    activate_mode: Optional[str] = None


ParsedEntry = Union[ParsedTriplet, ParsedOperator, ParsedStateExpr, ParsedTrigger]

# Gate tokens recognized as oscillatory gates (K.7.2D)
_GATE_TOKENS = {
    "DELTA", "THETA", "ALPHA", "BETA", "GAMMA",
    "THETA_GAMMA", "ALPHA_BETA",
}

# Operator prefixes (K.4)
_OPERATOR_PREFIXES = ("INT(", "UPR(", "SWITCH(", "DSN(", "REC(")


# ---------------------------------------------------------------------------
# Entry classification (K.7.5 step 2)
# ---------------------------------------------------------------------------

def classify_entry(entry: str) -> str:
    """
    Classify a single mastergrid entry by prefix.

    Returns one of: 'trigger', 'state', 'operator', 'gated_triplet',
    'activate', 'triplet'.
    """
    s = entry.strip()
    if s.startswith("IF("):
        return "trigger"
    if s.startswith("STATE("):
        return "state"
    if any(s.startswith(op) for op in _OPERATOR_PREFIXES):
        return "operator"
    if s.startswith("ACTIVATE("):
        return "activate"
    # Check for gated triplet: GATE{...}
    brace = s.find("{")
    if brace > 0 and s.endswith("}"):
        prefix = s[:brace]
        if prefix in _GATE_TOKENS:
            return "gated_triplet"
    return "triplet"


# ---------------------------------------------------------------------------
# Triplet parsing (K.7.4 Forms 1-3)
# ---------------------------------------------------------------------------

def _strip_signal_mode(nt: str) -> Tuple[str, Optional[str]]:
    """Extract phasic/tonic marker from NT token."""
    # Unicode markers
    if nt.endswith("\u2022"):  # •
        return nt[:-1], "phasic"
    if nt.endswith("~"):
        return nt[:-1], "tonic"
    # ASCII markers
    if nt.endswith(".P"):
        return nt[:-2], "phasic"
    if nt.endswith(".T"):
        return nt[:-2], "tonic"
    return nt, None


def parse_triplet_entry(entry: str) -> ParsedTriplet:
    """
    Parse a single base or gated triplet string.

    Handles Forms 1-3:
    - DA->D1:UP_ACT
    - GAMMA{Glu->NMDA:UP_AFF}
    - DA.P->D1:UP_ACT
    """
    s = entry.strip()
    gate = None

    # Check for gate wrapper
    brace = s.find("{")
    if brace > 0 and s.endswith("}"):
        prefix = s[:brace]
        if prefix in _GATE_TOKENS:
            gate = prefix
            s = s[brace + 1:-1]

    # Split on -> (ASCII arrow)
    if "->" not in s:
        # Try Unicode arrow →
        if "\u2192" not in s:
            raise ValueError(f"Invalid triplet format (no arrow): {entry}")
        parts = s.split("\u2192", 1)
    else:
        parts = s.split("->", 1)

    if len(parts) != 2:
        raise ValueError(f"Invalid triplet format: {entry}")

    nt_raw = parts[0].strip()
    nt, signal_mode = _strip_signal_mode(nt_raw)

    # Split receptor:modifiers
    rm = parts[1].split(":", 1)
    if len(rm) != 2:
        raise ValueError(f"Invalid triplet format (no colon): {entry}")

    receptor = rm[0].strip()
    mod_str = rm[1].strip()
    modifiers = tuple(m.strip() for m in mod_str.split(",") if m.strip())

    return ParsedTriplet(
        nt=nt,
        receptor=receptor,
        modifiers=modifiers,
        gate=gate,
        signal_mode=signal_mode,
    )


# ---------------------------------------------------------------------------
# Operator parsing (K.7.4 Form 4)
# ---------------------------------------------------------------------------

def parse_operator_entry(entry: str) -> ParsedOperator:
    """
    Parse INT(R), UPR(R), DSN(R), REC(R), or SWITCH(Ra->Rb).
    """
    s = entry.strip()

    # SWITCH(Ra->Rb)
    m = re.match(r"SWITCH\((.+?)\s*->\s*(.+?)\)", s)
    if m:
        return ParsedOperator("SWITCH", m.group(1).strip(), m.group(2).strip())

    # INT(R), UPR(R), DSN(R), REC(R)
    m = re.match(r"(INT|UPR|DSN|REC)\((.+?)\)", s)
    if m:
        return ParsedOperator(m.group(1), m.group(2).strip())

    raise ValueError(f"Invalid operator format: {entry}")


# ---------------------------------------------------------------------------
# State expression parsing (K.7.4 Form 5)
# ---------------------------------------------------------------------------

def parse_state_entry(entry: str) -> ParsedStateExpr:
    """
    Parse STATE(Name)=w1*Var+w2*Var-w3*Var.
    """
    s = entry.strip()

    m = re.match(r"STATE\((.+?)\)\s*=\s*(.+)", s)
    if not m:
        raise ValueError(f"Invalid state expression: {entry}")

    name = m.group(1).strip()
    expr = m.group(2).strip()

    terms = []
    # Split into signed tokens: handle +/- as delimiters while keeping sign
    # Insert a '+' at the beginning if expression starts with a variable
    tokens = re.split(r"(?=[+-])", expr)

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Parse weight*variable
        parts = tok.split("*", 1)
        if len(parts) == 2:
            weight = float(parts[0].strip())
            var = parts[1].strip()
        else:
            # Bare variable with implicit weight 1.0
            weight = 1.0
            var = parts[0].strip()
        terms.append((weight, var))

    return ParsedStateExpr(name=name, terms=tuple(terms))


# ---------------------------------------------------------------------------
# Trigger parsing (K.7.4 Form 6)
# ---------------------------------------------------------------------------

def parse_trigger_entry(entry: str) -> ParsedTrigger:
    """
    Parse IF(condition)=>action1;action2;ACTIVATE(Mode).
    """
    s = entry.strip()

    # Split on =>
    if "=>" not in s:
        raise ValueError(f"Invalid trigger format (no =>): {entry}")

    cond_part, action_part = s.split("=>", 1)

    # Extract condition from IF(...)
    m = re.match(r"IF\((.+)\)", cond_part.strip())
    if not m:
        raise ValueError(f"Invalid trigger condition: {cond_part}")
    condition = m.group(1).strip()

    # Split actions by ;
    raw_actions = [a.strip() for a in action_part.split(";") if a.strip()]

    actions = []
    activate_mode = None

    for action_str in raw_actions:
        # Check for ACTIVATE(Mode)
        am = re.match(r"ACTIVATE\((.+?)\)", action_str)
        if am:
            activate_mode = am.group(1).strip()
            continue
        # Check for operator
        kind = classify_entry(action_str)
        if kind == "operator":
            actions.append(parse_operator_entry(action_str))
        elif kind in ("triplet", "gated_triplet"):
            actions.append(parse_triplet_entry(action_str))
        else:
            # Keep as raw string for unknown action types
            actions.append(action_str)

    return ParsedTrigger(
        condition=condition,
        actions=tuple(actions),
        activate_mode=activate_mode,
    )


# ---------------------------------------------------------------------------
# Top-level parser (K.7.5)
# ---------------------------------------------------------------------------

def parse_mastergrid(text: str) -> List[ParsedEntry]:
    """
    Parse a | delimited mastergrid string into typed entries.

    Parameters
    ----------
    text : str
        Mastergrid string with | delimited entries.

    Returns
    -------
    list
        List of ParsedTriplet, ParsedOperator, ParsedStateExpr, or ParsedTrigger.
    """
    if not text or not text.strip():
        return []

    raw_entries = [e.strip() for e in text.split("|") if e.strip()]
    results = []

    for raw in raw_entries:
        kind = classify_entry(raw)
        if kind == "trigger":
            results.append(parse_trigger_entry(raw))
        elif kind == "state":
            results.append(parse_state_entry(raw))
        elif kind == "operator":
            results.append(parse_operator_entry(raw))
        elif kind == "gated_triplet":
            results.append(parse_triplet_entry(raw))
        elif kind == "activate":
            # Standalone ACTIVATE — store as raw string
            m = re.match(r"ACTIVATE\((.+?)\)", raw)
            if m:
                results.append(raw)
            else:
                raise ValueError(f"Invalid ACTIVATE: {raw}")
        else:
            results.append(parse_triplet_entry(raw))

    return results


# ---------------------------------------------------------------------------
# Encoding (roundtrip support)
# ---------------------------------------------------------------------------

def encode_triplet(t: ParsedTriplet) -> str:
    """Encode a ParsedTriplet back to mastergrid string."""
    nt = t.nt
    if t.signal_mode == "phasic":
        nt = f"{nt}.P"
    elif t.signal_mode == "tonic":
        nt = f"{nt}.T"

    mods = ",".join(t.modifiers)
    base = f"{nt}->{t.receptor}:{mods}"

    if t.gate:
        return f"{t.gate}{{{base}}}"
    return base


def encode_operator(op: ParsedOperator) -> str:
    """Encode a ParsedOperator back to mastergrid string."""
    if op.operator == "SWITCH":
        return f"SWITCH({op.target}->{op.target_b})"
    return f"{op.operator}({op.target})"


def encode_state_expr(se: ParsedStateExpr) -> str:
    """Encode a ParsedStateExpr back to mastergrid string."""
    parts = []
    for i, (w, var) in enumerate(se.terms):
        if i == 0:
            parts.append(f"{w:g}*{var}")
        elif w >= 0:
            parts.append(f"+{w:g}*{var}")
        else:
            parts.append(f"{w:g}*{var}")
    return f"STATE({se.name})={''.join(parts)}"


def encode_trigger(trig: ParsedTrigger) -> str:
    """Encode a ParsedTrigger back to mastergrid string."""
    parts = []
    for a in trig.actions:
        if isinstance(a, ParsedTriplet):
            parts.append(encode_triplet(a))
        elif isinstance(a, ParsedOperator):
            parts.append(encode_operator(a))
        elif isinstance(a, str):
            parts.append(a)
    if trig.activate_mode:
        parts.append(f"ACTIVATE({trig.activate_mode})")
    actions_str = ";".join(parts)
    return f"IF({trig.condition})=>{actions_str}"


def encode_entry(entry: ParsedEntry) -> str:
    """Encode any parsed entry back to mastergrid string."""
    if isinstance(entry, ParsedTriplet):
        return encode_triplet(entry)
    elif isinstance(entry, ParsedOperator):
        return encode_operator(entry)
    elif isinstance(entry, ParsedStateExpr):
        return encode_state_expr(entry)
    elif isinstance(entry, ParsedTrigger):
        return encode_trigger(entry)
    elif isinstance(entry, str):
        return entry
    raise TypeError(f"Unknown entry type: {type(entry)}")


def encode_mastergrid(entries: List[ParsedEntry]) -> str:
    """Encode entries back to | delimited mastergrid string."""
    return " | ".join(encode_entry(e) for e in entries)
