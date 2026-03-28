"""
Conditional trigger evaluation for neurosymbolic encoding (Appendix K.6).

IF(condition) => action1; action2; ACTIVATE(Mode)

Provides:
- A safe condition evaluator (tokenizer + recursive descent parser)
- Variable namespace builder from system state
- Trigger evaluation returning TriggerResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriggerDefinition:
    """A conditional trigger rule (K.6.1)."""
    condition_str: str
    actions: Tuple[str, ...]
    activate_mode: Optional[str] = None
    persistence_window: Optional[float] = None


@dataclass(frozen=True)
class TriggerResult:
    """Result of evaluating a trigger (K.6.4)."""
    fired: bool
    mode: Optional[str] = None
    actions: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Safe condition evaluator — tokenizer + recursive descent parser
#
# Supports: >, >=, <, <=, ==, !=, AND, OR, NOT, parentheses,
#           numeric literals, variable names.
# ---------------------------------------------------------------------------

# Token types
_TOK_NUM = "NUM"
_TOK_VAR = "VAR"
_TOK_OP = "OP"       # comparison operators
_TOK_AND = "AND"
_TOK_OR = "OR"
_TOK_NOT = "NOT"
_TOK_LPAREN = "("
_TOK_RPAREN = ")"
_TOK_EOF = "EOF"


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    """Tokenize a condition expression."""
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue

        # Parentheses
        if c == "(":
            tokens.append((_TOK_LPAREN, "("))
            i += 1
            continue
        if c == ")":
            tokens.append((_TOK_RPAREN, ")"))
            i += 1
            continue

        # Comparison operators (>=, <=, ==, !=, >, <)
        if c in ">" and i + 1 < len(expr) and expr[i + 1] == "=":
            tokens.append((_TOK_OP, ">="))
            i += 2
            continue
        if c == "<" and i + 1 < len(expr) and expr[i + 1] == "=":
            tokens.append((_TOK_OP, "<="))
            i += 2
            continue
        if c == "=" and i + 1 < len(expr) and expr[i + 1] == "=":
            tokens.append((_TOK_OP, "=="))
            i += 2
            continue
        if c == "!" and i + 1 < len(expr) and expr[i + 1] == "=":
            tokens.append((_TOK_OP, "!="))
            i += 2
            continue
        if c == ">":
            tokens.append((_TOK_OP, ">"))
            i += 1
            continue
        if c == "<":
            tokens.append((_TOK_OP, "<"))
            i += 1
            continue

        # Number (including negative and decimal)
        if c.isdigit() or (c == "." and i + 1 < len(expr) and expr[i + 1].isdigit()):
            j = i
            if c == "-":
                j += 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            tokens.append((_TOK_NUM, expr[i:j]))
            i = j
            continue

        # Identifiers and keywords
        if c.isalpha() or c == "_":
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] in "_-"):
                j += 1
            word = expr[i:j]
            if word == "AND":
                tokens.append((_TOK_AND, "AND"))
            elif word == "OR":
                tokens.append((_TOK_OR, "OR"))
            elif word == "NOT":
                tokens.append((_TOK_NOT, "NOT"))
            else:
                tokens.append((_TOK_VAR, word))
            i = j
            continue

        raise ValueError(f"Unexpected character '{c}' at position {i} in: {expr}")

    tokens.append((_TOK_EOF, ""))
    return tokens


class _Parser:
    """Recursive descent parser for condition expressions."""

    def __init__(self, tokens: List[Tuple[str, str]], variables: Dict[str, float]):
        self.tokens = tokens
        self.variables = variables
        self.pos = 0

    def _peek(self) -> Tuple[str, str]:
        return self.tokens[self.pos]

    def _advance(self) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tok_type: str) -> Tuple[str, str]:
        tok = self._advance()
        if tok[0] != tok_type:
            raise ValueError(f"Expected {tok_type}, got {tok}")
        return tok

    def parse(self) -> bool:
        result = self._parse_or()
        if self._peek()[0] != _TOK_EOF:
            raise ValueError(f"Unexpected token after expression: {self._peek()}")
        return result

    def _parse_or(self) -> bool:
        left = self._parse_and()
        while self._peek()[0] == _TOK_OR:
            self._advance()
            right = self._parse_and()
            left = left or right
        return left

    def _parse_and(self) -> bool:
        left = self._parse_not()
        while self._peek()[0] == _TOK_AND:
            self._advance()
            right = self._parse_not()
            left = left and right
        return left

    def _parse_not(self) -> bool:
        if self._peek()[0] == _TOK_NOT:
            self._advance()
            return not self._parse_not()
        return self._parse_comparison()

    def _parse_comparison(self) -> bool:
        # Could be: value OP value  or  (or_expr)  or  just a value (truthy check)
        if self._peek()[0] == _TOK_LPAREN:
            self._advance()
            result = self._parse_or()
            self._expect(_TOK_RPAREN)
            return result

        left = self._parse_value()

        if self._peek()[0] == _TOK_OP:
            op = self._advance()[1]
            right = self._parse_value()
            return self._apply_op(op, left, right)

        # Truthy check: non-zero → True
        return left != 0.0

    def _parse_value(self) -> float:
        tok_type, tok_val = self._peek()
        if tok_type == _TOK_NUM:
            self._advance()
            return float(tok_val)
        if tok_type == _TOK_VAR:
            self._advance()
            return self.variables.get(tok_val, 0.0)
        if tok_type == _TOK_LPAREN:
            # Parenthesized value — parse as sub-expression, return float
            self._advance()
            result = self._parse_or()
            self._expect(_TOK_RPAREN)
            return 1.0 if result else 0.0
        raise ValueError(f"Expected value, got {self._peek()}")

    @staticmethod
    def _apply_op(op: str, left: float, right: float) -> bool:
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        raise ValueError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_condition(condition_str: str, variables: Dict[str, float]) -> bool:
    """
    Evaluate a condition expression against a variable namespace.

    Parameters
    ----------
    condition_str : str
        Condition like "beta>0.6 AND S_DA-D2>0.7".
    variables : dict
        Variable name → float value mapping.

    Returns
    -------
    bool
        Whether the condition is satisfied.
    """
    tokens = _tokenize(condition_str)
    parser = _Parser(tokens, variables)
    return parser.parse()


def build_variable_namespace(
    concentrations: Optional[Dict[str, float]] = None,
    saturations: Optional[Dict[str, float]] = None,
    oscillations: Optional[Dict[str, float]] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build a flat variable namespace for condition evaluation (K.6.2).

    Adds prefixed and unprefixed names:
    - Concentrations: C_DA → value, also DA if no collision
    - Saturations: S_DA_D1 → value, also S_DA-D1 (hyphen variant)
    - Oscillations: phi_theta → value, also theta, phi_theta_gamma → value
    - Metrics: direct names (Fatigue, LogicMode, etc.)

    Parameters
    ----------
    concentrations, saturations, oscillations, metrics : dict or None

    Returns
    -------
    dict
        Flat variable namespace.
    """
    ns: Dict[str, float] = {}

    if concentrations:
        for k, v in concentrations.items():
            ns[f"C_{k}"] = v
            ns[k] = v  # short name

    if saturations:
        for k, v in saturations.items():
            ns[f"S_{k}"] = v
            # Also add hyphen variant: S_DA_D1 → S_DA-D1
            ns[f"S_{k.replace('_', '-', 1)}"] = v

    if oscillations:
        for k, v in oscillations.items():
            ns[f"phi_{k}"] = v
            ns[k] = v  # short name

    if metrics:
        for k, v in metrics.items():
            ns[k] = v

    return ns


def evaluate_trigger(
    trigger: TriggerDefinition,
    variables: Dict[str, float],
) -> TriggerResult:
    """
    Evaluate a single trigger against current variables.

    Parameters
    ----------
    trigger : TriggerDefinition
    variables : dict

    Returns
    -------
    TriggerResult
    """
    fired = evaluate_condition(trigger.condition_str, variables)
    if fired:
        return TriggerResult(
            fired=True,
            mode=trigger.activate_mode,
            actions=trigger.actions,
        )
    return TriggerResult(fired=False)


def evaluate_all_triggers(
    triggers: List[TriggerDefinition],
    variables: Dict[str, float],
) -> List[TriggerResult]:
    """
    Evaluate all triggers and return results.

    Parameters
    ----------
    triggers : list of TriggerDefinition
    variables : dict

    Returns
    -------
    list of TriggerResult
    """
    return [evaluate_trigger(t, variables) for t in triggers]
