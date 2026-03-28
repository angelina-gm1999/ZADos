"""
Engine 3 -- SOAR Production Rule Engine  (``soar_production_engine``)
=====================================================================
Executive controller that implements a SOAR-style decision cycle with
neurochemically-modulated production rules, preference-based operator
selection, impasse-driven delegation, and chunking (learning).

Five-phase decision cycle:
  * **Phase 1 — Input**: Populate working memory from engine outputs,
    NT state, reward scores, and memory contrast.
  * **Phase 2 — Elaboration**: Fire state-elaboration productions in
    parallel until quiescence.  ACh scales matching sensitivity.
  * **Phase 3 — Proposal**: Propose operators with acceptable
    preferences; compare operators with better/worse/indifferent prefs.
    DA biases exploratory, 5-HT biases conservative operators.
  * **Phase 4 — Decision**: SOAR preference resolution with NT-weighted
    dominance.  Impasses delegate to existing engines:
      TIE         → E13 (Simulation Brain)
      CONFLICT    → E1  (Contradiction) + E14 (Socratic)
      NO_CHANGE   → E7  (Simulated Opposition)
      STATE_NO_CHANGE → E26 (Uncertainty)
  * **Phase 5 — Application**: Fire application productions, populate
    output-link, emit neurochemical signals, create chunks on impasse
    resolution.

Neurochemical coupling:
  DA   — exploration bias, reward prediction on selection
  5-HT — conservation bias, consolidation on chunking
  NE   — urgency bias, conflict signal on impasse
  ACh  — matching depth, attention load
  COR  — risk suppression, escalation stress
  GABA — inhibition, preference suppression
  OXT  — social operator bias
  CB1  — creativity / unconventional operator bias (REM_DREAM)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Enums
# =====================================================================


class Support(str, Enum):
    """WME persistence semantics."""
    I_SUPPORTED = "i-supported"     # Architecture input
    O_SUPPORTED = "o-supported"     # Operator result
    JUSTIFIED = "justified"         # Elaboration result


class ProductionType(str, Enum):
    """Production rule classification."""
    ELABORATION = "elaboration"
    PROPOSAL = "proposal"
    COMPARISON = "comparison"
    APPLICATION = "application"


class ConditionTest(str, Enum):
    """Comparison operations for production conditions."""
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EXISTS = "exists"


class PrefType(str, Enum):
    """SOAR preference types."""
    REQUIRE = "require"
    ACCEPTABLE = "acceptable"
    BEST = "best"
    BETTER = "better"
    WORSE = "worse"
    INDIFFERENT = "indifferent"
    PROHIBIT = "prohibit"
    REJECT = "reject"


class ImpasseType(str, Enum):
    """Impasse classification."""
    TIE = "tie"
    CONFLICT = "conflict"
    NO_CHANGE = "no_change"
    STATE_NO_CHANGE = "state_no_change"
    NONE = "none"


# =====================================================================
# Frozen Data Structures
# =====================================================================


@dataclass(frozen=True)
class WME:
    """Working Memory Element — (identifier, attribute, value) triple."""
    identifier: str
    attribute: str
    value: Any
    timetag: int
    support: Support
    source_operator: Optional[str] = None


@dataclass(frozen=True)
class Condition:
    """A single condition in a production rule."""
    identifier_var: str             # Variable or literal identifier
    attribute: str                  # Attribute to test
    value_test: ConditionTest       # Comparison
    value: Any = None               # Test value (or variable "<var>")
    binding_var: Optional[str] = None  # Variable to bind matched value


@dataclass(frozen=True)
class Action:
    """A single action in a production rule."""
    action_type: str                # "add" | "remove" | "propose" | "prefer"
    identifier_var: str             # Variable or literal identifier
    attribute: Optional[str] = None
    value: Any = None
    support: Support = Support.JUSTIFIED
    operator_name: Optional[str] = None
    pref_type: Optional[str] = None
    reference_var: Optional[str] = None
    strength: float = 1.0


@dataclass(frozen=True)
class Production:
    """A production rule: conditions → actions."""
    name: str
    conditions: Tuple[Condition, ...]
    actions: Tuple[Action, ...]
    prod_type: ProductionType
    activation: float = 1.0
    learned: bool = False
    chunk_source: Optional[str] = None


@dataclass(frozen=True)
class Operator:
    """A proposed operator (action to take)."""
    op_id: str
    name: str
    parameters: Dict[str, Any]
    source_production: str


@dataclass(frozen=True)
class Preference:
    """A value judgment about an operator."""
    operator_id: str
    pref_type: PrefType
    reference_id: Optional[str] = None
    strength: float = 1.0
    source_production: str = ""


@dataclass(frozen=True)
class Impasse:
    """Describes a decision-cycle impasse."""
    impasse_type: ImpasseType
    tied_operators: Tuple[str, ...] = ()
    conflicting_prefs: Tuple[Preference, ...] = ()
    stuck_operator: Optional[str] = None
    cycle_count: int = 0
    delegation_target: Optional[str] = None


# =====================================================================
# Configuration
# =====================================================================


_MODES = ("normal", "dev", "learning", "reflective", "rem_normal", "rem_dream")


@dataclass(frozen=True)
class SOARConfig:
    """All tunable parameters for the SOAR Production Engine."""

    # --- Elaboration Phase ---
    max_elaboration_rounds: Dict[str, int] = field(default_factory=lambda: {
        "normal": 10, "dev": 15, "learning": 12,
        "reflective": 20, "rem_normal": 10, "rem_dream": 8,
    })
    matching_threshold: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.70, "dev": 0.50, "learning": 0.60,
        "reflective": 0.80, "rem_normal": 0.70, "rem_dream": 0.40,
    })

    # --- Decision Phase ---
    max_substates: Dict[str, int] = field(default_factory=lambda: {
        "normal": 3, "dev": 5, "learning": 4,
        "reflective": 6, "rem_normal": 3, "rem_dream": 2,
    })
    preference_epsilon: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.01, "dev": 0.05, "learning": 0.02,
        "reflective": 0.005, "rem_normal": 0.01, "rem_dream": 0.08,
    })

    # --- Chunking ---
    chunk_confidence_min: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.60, "dev": 0.40, "learning": 0.50,
        "reflective": 0.70, "rem_normal": 0.60, "rem_dream": 0.30,
    })
    max_chunk_conditions: int = 8
    chunk_decay_rate: float = 0.01

    # --- NT Modulation Weights ---
    da_exploration_weight: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.30, "dev": 0.50, "learning": 0.40,
        "reflective": 0.20, "rem_normal": 0.30, "rem_dream": 0.60,
    })
    sht_conservation_weight: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.30, "dev": 0.20, "learning": 0.30,
        "reflective": 0.40, "rem_normal": 0.30, "rem_dream": 0.10,
    })
    ne_urgency_weight: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.30, "dev": 0.30, "learning": 0.20,
        "reflective": 0.20, "rem_normal": 0.30, "rem_dream": 0.10,
    })
    cor_risk_suppression: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.30, "dev": 0.20, "learning": 0.30,
        "reflective": 0.40, "rem_normal": 0.30, "rem_dream": 0.10,
    })
    ach_matching_depth: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.30, "dev": 0.40, "learning": 0.30,
        "reflective": 0.50, "rem_normal": 0.30, "rem_dream": 0.20,
    })
    gaba_inhibition_weight: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.20, "dev": 0.10, "learning": 0.20,
        "reflective": 0.30, "rem_normal": 0.20, "rem_dream": 0.10,
    })
    oxt_social_weight: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.20, "dev": 0.20, "learning": 0.20,
        "reflective": 0.30, "rem_normal": 0.20, "rem_dream": 0.10,
    })
    cb1_creativity_weight: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.20, "dev": 0.30, "learning": 0.20,
        "reflective": 0.10, "rem_normal": 0.20, "rem_dream": 0.50,
    })

    # --- Neurochem Output Scaling ---
    beta_da_reward: float = 0.15
    beta_ne_conflict: float = 0.12
    beta_5ht_consolidation: float = 0.10
    beta_cor_escalation: float = 0.08
    beta_ach_attention: float = 0.10
    beta_gaba_inhibition: float = 0.08
    psi_beta: float = 0.12
    psi_theta_gamma: float = 0.10
    psi_alpha_suppress: float = 0.08

    # --- WM Limits ---
    max_wm_size: int = 500
    max_productions_per_cycle: int = 50


# =====================================================================
# Mutable State
# =====================================================================


@dataclass
class SOARState:
    """Mutable state for the SOAR decision cycle."""

    # --- Working Memory ---
    wm: Dict[int, WME] = field(default_factory=dict)
    index_by_id: Dict[str, Set[int]] = field(default_factory=dict)
    index_by_attr: Dict[str, Set[int]] = field(default_factory=dict)
    index_by_id_attr: Dict[Tuple[str, str], Set[int]] = field(default_factory=dict)
    next_timetag: int = 0
    next_identifier: int = 0
    next_op_id: int = 0

    # --- Production Memory ---
    productions: Dict[str, Production] = field(default_factory=dict)
    fired_this_cycle: Set[str] = field(default_factory=set)

    # --- Operator State ---
    proposed_operators: Dict[str, Operator] = field(default_factory=dict)
    preferences: List[Preference] = field(default_factory=list)
    selected_operator: Optional[str] = None
    previous_operator: Optional[str] = None

    # --- Substate Stack ---
    substate_stack: List[Dict] = field(default_factory=list)
    current_impasse: Optional[Impasse] = None
    impasse_cycle_count: int = 0

    # --- Neurochemical State ---
    da_level: float = 0.5
    ne_level: float = 0.5
    _5ht_level: float = 0.5
    ach_level: float = 0.5
    cor_level: float = 0.3
    gaba_level: float = 0.5
    oxt_level: float = 0.5
    cb1_level: float = 0.3

    # --- Counters ---
    cycle_count: int = 0
    total_productions_fired: int = 0
    total_chunks_learned: int = 0
    total_impasses: int = 0

    # --- Input/Output link identifiers ---
    input_link_ids: Set[int] = field(default_factory=set)
    output_link_id: str = "output-link"


# =====================================================================
# Frozen I/O Dataclasses
# =====================================================================


@dataclass(frozen=True)
class SOARInput:
    """Input to the SOAR decision cycle."""
    engine_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nt_state: Dict[str, float] = field(default_factory=dict)
    reward_scores: Dict[str, float] = field(default_factory=dict)
    memory_contrast: Optional[Dict[str, Any]] = None
    active_mode: str = "NORMAL"
    cycle_id: int = 0


@dataclass(frozen=True)
class SOARNeurochem:
    """Neurochemical signals emitted by the SOAR engine."""
    delta_da: float = 0.0
    delta_ne: float = 0.0
    delta_5ht: float = 0.0
    delta_cor: float = 0.0
    delta_ach: float = 0.0
    delta_gaba: float = 0.0
    beta_boost: float = 0.0
    theta_gamma_boost: float = 0.0
    alpha_suppress: float = 0.0


@dataclass(frozen=True)
class SOARResult:
    """Output of the SOAR decision cycle."""
    selected_operator: Optional[Operator]
    action_commands: Tuple[Dict[str, Any], ...]
    impasse: Impasse
    delegation_requests: Tuple[Dict[str, Any], ...]
    chunks_learned: Tuple[Production, ...]
    neurochemical_signals: SOARNeurochem
    cycle_count: int = 0
    elaboration_rounds: int = 0
    productions_fired: int = 0
    wm_size: int = 0
    engine_id: str = "soar_production_engine"
    processing_time_ms: float = 0.0


# =====================================================================
# Pure Helper Functions
# =====================================================================



def _mode_key(mode: str) -> str:
    """Normalize operational mode to config key."""
    m = mode.lower().replace(" ", "_")
    if m in _MODES:
        return m
    return "normal"


def _cfg(param: Dict[str, Any], mode: str) -> Any:
    """Extract mode-dependent config value."""
    return param.get(_mode_key(mode), param.get("normal"))


def _new_identifier(state: SOARState) -> str:
    """Generate a new unique identifier."""
    state.next_identifier += 1
    return f"I{state.next_identifier}"


def create_wme(
    identifier: str,
    attribute: str,
    value: Any,
    timetag: int,
    support: Support,
    source_operator: Optional[str] = None,
) -> WME:
    """Construct a WME."""
    return WME(
        identifier=identifier,
        attribute=attribute,
        value=value,
        timetag=timetag,
        support=support,
        source_operator=source_operator,
    )


def add_wme_to_state(state: SOARState, wme: WME) -> None:
    """Insert WME into working memory and update all three indexes."""
    state.wm[wme.timetag] = wme
    # Index by identifier
    state.index_by_id.setdefault(wme.identifier, set()).add(wme.timetag)
    # Index by attribute
    state.index_by_attr.setdefault(wme.attribute, set()).add(wme.timetag)
    # Index by (identifier, attribute)
    key = (wme.identifier, wme.attribute)
    state.index_by_id_attr.setdefault(key, set()).add(wme.timetag)


def remove_wme_from_state(state: SOARState, timetag: int) -> Optional[WME]:
    """Remove WME from working memory and clean up indexes."""
    wme = state.wm.pop(timetag, None)
    if wme is None:
        return None
    # Clean index_by_id
    id_set = state.index_by_id.get(wme.identifier)
    if id_set is not None:
        id_set.discard(timetag)
        if not id_set:
            del state.index_by_id[wme.identifier]
    # Clean index_by_attr
    attr_set = state.index_by_attr.get(wme.attribute)
    if attr_set is not None:
        attr_set.discard(timetag)
        if not attr_set:
            del state.index_by_attr[wme.attribute]
    # Clean index_by_id_attr
    key = (wme.identifier, wme.attribute)
    ia_set = state.index_by_id_attr.get(key)
    if ia_set is not None:
        ia_set.discard(timetag)
        if not ia_set:
            del state.index_by_id_attr[key]
    return wme


def _add_wme_quick(state: SOARState, identifier: str, attribute: str,
                   value: Any, support: Support,
                   source_operator: Optional[str] = None) -> WME:
    """Helper: create + add a WME in one call."""
    tt = state.next_timetag
    state.next_timetag += 1
    wme = create_wme(identifier, attribute, value, tt, support, source_operator)
    add_wme_to_state(state, wme)
    return wme


def _test_value(test: ConditionTest, wm_val: Any, cond_val: Any) -> bool:
    """Evaluate a single condition test."""
    if test == ConditionTest.EXISTS:
        return True
    if test == ConditionTest.EQ:
        return wm_val == cond_val
    if test == ConditionTest.NEQ:
        return wm_val != cond_val
    try:
        wf = float(wm_val)
        cf = float(cond_val)
    except (TypeError, ValueError):
        return False
    if test == ConditionTest.GT:
        return wf > cf
    if test == ConditionTest.GTE:
        return wf >= cf
    if test == ConditionTest.LT:
        return wf < cf
    if test == ConditionTest.LTE:
        return wf <= cf
    return False


def _resolve_var(val: Any, bindings: Dict[str, Any]) -> Any:
    """If val is a variable string like '<x>', resolve from bindings."""
    if isinstance(val, str) and val.startswith("<") and val.endswith(">"):
        return bindings.get(val, val)
    return val


def match_condition(
    cond: Condition,
    state: SOARState,
    bindings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Test a single condition against working memory.
    Returns list of possible binding extensions (one per matching WME).
    """
    results: List[Dict[str, Any]] = []
    resolved_id = _resolve_var(cond.identifier_var, bindings)

    # Determine candidate WMEs
    if isinstance(resolved_id, str) and not (resolved_id.startswith("<") and resolved_id.endswith(">")):
        # Literal identifier — use index_by_id_attr or index_by_id
        candidates_tt = state.index_by_id_attr.get((resolved_id, cond.attribute), set())
    else:
        # Variable identifier — use index_by_attr
        candidates_tt = state.index_by_attr.get(cond.attribute, set())

    resolved_val = _resolve_var(cond.value, bindings)

    for tt in candidates_tt:
        wme = state.wm.get(tt)
        if wme is None:
            continue
        # Check attribute match
        if wme.attribute != cond.attribute:
            continue
        # Check identifier binding
        if isinstance(resolved_id, str) and resolved_id.startswith("<") and resolved_id.endswith(">"):
            # Variable — need to bind or match
            if resolved_id in bindings:
                if bindings[resolved_id] != wme.identifier:
                    continue
            # else: will bind below
        else:
            if wme.identifier != resolved_id:
                continue

        # Check value test
        if not _test_value(cond.value_test, wme.value, resolved_val):
            continue

        # Build new bindings
        new_bindings = dict(bindings)
        if isinstance(cond.identifier_var, str) and cond.identifier_var.startswith("<"):
            new_bindings[cond.identifier_var] = wme.identifier
        if cond.binding_var is not None:
            new_bindings[cond.binding_var] = wme.value
        results.append(new_bindings)

    return results


def match_production(
    prod: Production,
    state: SOARState,
    threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    Match all conditions of a production against WM.
    Returns the first valid binding set, or None if no match.
    Activation must be >= threshold.
    """
    if prod.activation < threshold:
        return None

    binding_sets: List[Dict[str, Any]] = [{}]

    for cond in prod.conditions:
        new_binding_sets: List[Dict[str, Any]] = []
        for bindings in binding_sets:
            matches = match_condition(cond, state, bindings)
            new_binding_sets.extend(matches)
        if not new_binding_sets:
            return None
        binding_sets = new_binding_sets
        # Limit combinatorial explosion
        if len(binding_sets) > 100:
            binding_sets = binding_sets[:100]

    return binding_sets[0] if binding_sets else None


def find_matching_productions(
    state: SOARState,
    threshold: float,
    type_filter: Optional[ProductionType] = None,
) -> List[Tuple[Production, Dict[str, Any]]]:
    """Scan production memory for matching productions."""
    results: List[Tuple[Production, Dict[str, Any]]] = []
    for prod in state.productions.values():
        if type_filter is not None and prod.prod_type != type_filter:
            continue
        bindings = match_production(prod, state, threshold)
        if bindings is not None:
            results.append((prod, bindings))
    return results


def fire_production(
    prod: Production,
    bindings: Dict[str, Any],
    state: SOARState,
    selected_op: Optional[str] = None,
) -> Tuple[List[WME], List[Operator], List[Preference]]:
    """
    Execute a production's actions with bound variables.
    Returns (new_wmes, new_operators, new_preferences).
    """
    new_wmes: List[WME] = []
    new_operators: List[Operator] = []
    new_preferences: List[Preference] = []

    for action in prod.actions:
        resolved_id = _resolve_var(action.identifier_var, bindings)
        resolved_val = _resolve_var(action.value, bindings)

        if action.action_type == "add":
            if isinstance(resolved_id, str) and resolved_id.startswith("<new"):
                resolved_id = _new_identifier(state)
                bindings[action.identifier_var] = resolved_id
            src_op = selected_op if action.support == Support.O_SUPPORTED else None
            wme = _add_wme_quick(
                state, resolved_id, action.attribute or "",
                resolved_val, action.support, src_op,
            )
            new_wmes.append(wme)

        elif action.action_type == "remove":
            key = (str(resolved_id), action.attribute or "")
            tts = list(state.index_by_id_attr.get(key, set()))
            for tt in tts:
                remove_wme_from_state(state, tt)

        elif action.action_type == "propose":
            state.next_op_id += 1
            op_id = f"op_{state.next_op_id}"
            params = {}
            # Collect operator params from action attributes encoded in value
            if isinstance(resolved_val, dict):
                params = dict(resolved_val)
            elif action.attribute:
                params[action.attribute] = resolved_val
            op = Operator(
                op_id=op_id,
                name=action.operator_name or str(resolved_val),
                parameters=params,
                source_production=prod.name,
            )
            new_operators.append(op)
            # Auto-add acceptable preference
            pref = Preference(
                operator_id=op_id,
                pref_type=PrefType.ACCEPTABLE,
                strength=action.strength,
                source_production=prod.name,
            )
            new_preferences.append(pref)

        elif action.action_type == "prefer":
            ref_id = None
            if action.reference_var:
                ref_id = _resolve_var(action.reference_var, bindings)
                if isinstance(ref_id, str) and ref_id.startswith("<"):
                    ref_id = None
            pref = Preference(
                operator_id=str(resolved_id),
                pref_type=PrefType(action.pref_type) if action.pref_type else PrefType.ACCEPTABLE,
                reference_id=ref_id if isinstance(ref_id, str) else None,
                strength=action.strength,
                source_production=prod.name,
            )
            new_preferences.append(pref)

    return new_wmes, new_operators, new_preferences


def elaborate_until_quiescence(
    state: SOARState,
    max_rounds: int,
    threshold: float,
) -> int:
    """
    Run elaboration match-fire cycles until quiescence.
    Returns number of rounds executed.
    """
    state.fired_this_cycle.clear()
    rounds = 0
    for _ in range(max_rounds):
        matched = find_matching_productions(
            state, threshold, type_filter=ProductionType.ELABORATION,
        )
        # Filter out already-fired productions (prevent infinite loops)
        new_matches = [
            (p, b) for p, b in matched
            if p.name not in state.fired_this_cycle
        ]
        if not new_matches:
            break
        for prod, bindings in new_matches:
            fire_production(prod, bindings, state)
            state.fired_this_cycle.add(prod.name)
            state.total_productions_fired += 1
        rounds += 1
    return rounds


def propose_operators(
    state: SOARState,
    threshold: float,
) -> None:
    """Run proposal productions to populate proposed_operators and preferences."""
    state.proposed_operators.clear()
    state.preferences.clear()

    matched = find_matching_productions(
        state, threshold, type_filter=ProductionType.PROPOSAL,
    )
    for prod, bindings in matched:
        _, new_ops, new_prefs = fire_production(prod, bindings, state)
        for op in new_ops:
            state.proposed_operators[op.op_id] = op
        state.preferences.extend(new_prefs)
        state.total_productions_fired += 1


def collect_comparison_preferences(
    state: SOARState,
    threshold: float,
) -> None:
    """Run comparison productions to add preferences between operators."""
    matched = find_matching_productions(
        state, threshold, type_filter=ProductionType.COMPARISON,
    )
    for prod, bindings in matched:
        _, _, new_prefs = fire_production(prod, bindings, state)
        state.preferences.extend(new_prefs)
        state.total_productions_fired += 1


def apply_nt_preference_bias(
    state: SOARState,
    config: SOARConfig,
    mode: str,
) -> List[Preference]:
    """Apply neurochemical modulation to preference strengths."""
    w_da = _cfg(config.da_exploration_weight, mode)
    w_5ht = _cfg(config.sht_conservation_weight, mode)
    w_ne = _cfg(config.ne_urgency_weight, mode)
    w_cor = _cfg(config.cor_risk_suppression, mode)
    w_gaba = _cfg(config.gaba_inhibition_weight, mode)
    w_oxt = _cfg(config.oxt_social_weight, mode)
    w_cb1 = _cfg(config.cb1_creativity_weight, mode)

    modulated: List[Preference] = []
    for pref in state.preferences:
        op = state.proposed_operators.get(pref.operator_id)
        if op is None:
            modulated.append(pref)
            continue

        params = op.parameters
        s = pref.strength

        # DA → exploration bias
        if params.get("novelty", 0.0) > 0.5:
            s *= (1.0 + w_da * (state.da_level - 0.5))
        # 5-HT → conservation bias
        if params.get("familiarity", 0.0) > 0.5:
            s *= (1.0 + w_5ht * (state._5ht_level - 0.5))
        # NE → urgency bias
        if params.get("urgency", 0.0) > 0.5:
            s *= (1.0 + w_ne * (state.ne_level - 0.5))
        # Cortisol → risk suppression
        if params.get("risk", 0.0) > 0.5:
            s *= (1.0 - w_cor * state.cor_level)
        # GABA → inhibition
        if pref.pref_type == PrefType.ACCEPTABLE:
            confidence = params.get("confidence", 0.5)
            s *= (1.0 - w_gaba * state.gaba_level * (1.0 - confidence))
        # OXT → social bias
        if params.get("social", 0.0) > 0.5:
            s *= (1.0 + w_oxt * (state.oxt_level - 0.5))
        # CB1 → creativity bias
        if params.get("unconventional", 0.0) > 0.5:
            s *= (1.0 + w_cb1 * (state.cb1_level - 0.5))

        s = _clamp(s, 0.0, 2.0)
        modulated.append(Preference(
            operator_id=pref.operator_id,
            pref_type=pref.pref_type,
            reference_id=pref.reference_id,
            strength=s,
            source_production=pref.source_production,
        ))
    return modulated


def resolve_preferences(
    preferences: List[Preference],
    operators: Dict[str, Operator],
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[Optional[str], ImpasseType, Tuple[str, ...], Tuple[Preference, ...]]:
    """
    SOAR preference semantics.
    Returns (winner_id, impasse_type, tied_ops, conflicting_prefs).
    """
    if not operators:
        return None, ImpasseType.STATE_NO_CHANGE, (), ()

    remaining = set(operators.keys())

    # Step 1: Filter by require/prohibit/reject
    require_set: Set[str] = set()
    prohibit_set: Set[str] = set()
    for p in preferences:
        if p.pref_type == PrefType.REQUIRE:
            require_set.add(p.operator_id)
        elif p.pref_type in (PrefType.PROHIBIT, PrefType.REJECT):
            prohibit_set.add(p.operator_id)

    remaining -= prohibit_set
    if require_set:
        remaining &= require_set
    if not remaining:
        conflict_prefs = tuple(p for p in preferences
                               if p.pref_type in (PrefType.PROHIBIT, PrefType.REJECT, PrefType.REQUIRE))
        return None, ImpasseType.CONFLICT, (), conflict_prefs

    # Step 2: Compute dominance scores
    dominance: Dict[str, float] = {op_id: 0.0 for op_id in remaining}
    for p in preferences:
        if p.operator_id not in remaining:
            continue
        if p.pref_type == PrefType.BEST:
            dominance[p.operator_id] += p.strength * 2.0
        elif p.pref_type == PrefType.BETTER:
            dominance[p.operator_id] += p.strength
            if p.reference_id and p.reference_id in dominance:
                dominance[p.reference_id] -= p.strength
        elif p.pref_type == PrefType.WORSE:
            dominance[p.operator_id] -= p.strength
            if p.reference_id and p.reference_id in dominance:
                dominance[p.reference_id] += p.strength

    # Step 3: Select candidates (not dominated)
    if not dominance:
        return None, ImpasseType.STATE_NO_CHANGE, (), ()

    max_score = max(dominance.values())
    candidates = [op_id for op_id, score in dominance.items()
                  if score >= max_score - epsilon]

    if len(candidates) == 1:
        return candidates[0], ImpasseType.NONE, (), ()

    # Step 4: Check indifference among tied candidates
    indifferent_pairs: Set[Tuple[str, str]] = set()
    for p in preferences:
        if p.pref_type == PrefType.INDIFFERENT and p.reference_id:
            pair = tuple(sorted((p.operator_id, p.reference_id)))
            indifferent_pairs.add(pair)

    all_indifferent = True
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            pair = tuple(sorted((a, b)))
            if pair not in indifferent_pairs:
                all_indifferent = False
                break
        if not all_indifferent:
            break

    if all_indifferent:
        # Random tie-break among indifferent operators
        winner = candidates[rng.integers(len(candidates))]
        return winner, ImpasseType.NONE, (), ()

    # TIE impasse
    return None, ImpasseType.TIE, tuple(candidates), ()


def detect_impasse_delegation(impasse_type: ImpasseType) -> Optional[str]:
    """Map impasse type to delegation target engine."""
    mapping = {
        ImpasseType.TIE: "simulation_brain_engine",
        ImpasseType.CONFLICT: "contradiction_detection_engine",
        ImpasseType.NO_CHANGE: "simulated_opposition_engine",
        ImpasseType.STATE_NO_CHANGE: "uncertainty_pattern_engine",
    }
    return mapping.get(impasse_type)


def apply_operator_productions(
    state: SOARState,
    selected_op_id: str,
    threshold: float,
) -> Tuple[List[WME], bool]:
    """
    Fire application productions for the selected operator.
    Returns (new_wmes, any_fired).
    """
    all_new_wmes: List[WME] = []
    matched = find_matching_productions(
        state, threshold, type_filter=ProductionType.APPLICATION,
    )
    any_fired = False
    for prod, bindings in matched:
        new_wmes, _, _ = fire_production(prod, bindings, state, selected_op=selected_op_id)
        all_new_wmes.extend(new_wmes)
        state.total_productions_fired += 1
        any_fired = True
    return all_new_wmes, any_fired


def populate_input_link(state: SOARState, inp: SOARInput) -> None:
    """Convert SOARInput into WMEs on the input-link."""
    # Remove old input-link WMEs
    old_tts = list(state.input_link_ids)
    for tt in old_tts:
        remove_wme_from_state(state, tt)
    state.input_link_ids.clear()

    # Engine outputs
    for engine_id, output_dict in inp.engine_outputs.items():
        ident = _new_identifier(state)
        w1 = _add_wme_quick(state, ident, "^type", "engine_output", Support.I_SUPPORTED)
        state.input_link_ids.add(w1.timetag)
        w2 = _add_wme_quick(state, ident, "^engine-id", engine_id, Support.I_SUPPORTED)
        state.input_link_ids.add(w2.timetag)
        for key, value in output_dict.items():
            w = _add_wme_quick(state, ident, f"^{key}", value, Support.I_SUPPORTED)
            state.input_link_ids.add(w.timetag)

    # NT state
    for nt_key, level in inp.nt_state.items():
        ident = _new_identifier(state)
        w1 = _add_wme_quick(state, ident, "^type", "nt_level", Support.I_SUPPORTED)
        state.input_link_ids.add(w1.timetag)
        w2 = _add_wme_quick(state, ident, "^nt", nt_key, Support.I_SUPPORTED)
        state.input_link_ids.add(w2.timetag)
        w3 = _add_wme_quick(state, ident, "^level", level, Support.I_SUPPORTED)
        state.input_link_ids.add(w3.timetag)

    # Reward scores
    for domain, score in inp.reward_scores.items():
        ident = _new_identifier(state)
        w1 = _add_wme_quick(state, ident, "^type", "reward", Support.I_SUPPORTED)
        state.input_link_ids.add(w1.timetag)
        w2 = _add_wme_quick(state, ident, "^domain", domain, Support.I_SUPPORTED)
        state.input_link_ids.add(w2.timetag)
        w3 = _add_wme_quick(state, ident, "^score", score, Support.I_SUPPORTED)
        state.input_link_ids.add(w3.timetag)

    # Memory contrast (if present)
    if inp.memory_contrast:
        ident = _new_identifier(state)
        w1 = _add_wme_quick(state, ident, "^type", "memory_contrast", Support.I_SUPPORTED)
        state.input_link_ids.add(w1.timetag)
        for key, value in inp.memory_contrast.items():
            w = _add_wme_quick(state, ident, f"^{key}", value, Support.I_SUPPORTED)
            state.input_link_ids.add(w.timetag)


def extract_output_link(state: SOARState) -> Tuple[Dict[str, Any], ...]:
    """Extract output-link WMEs as action command dicts."""
    commands: List[Dict[str, Any]] = []
    tts = state.index_by_id.get(state.output_link_id, set())
    for tt in tts:
        wme = state.wm.get(tt)
        if wme is not None:
            commands.append({
                "attribute": wme.attribute,
                "value": wme.value,
            })
    return tuple(commands)


def decay_o_supported(state: SOARState, deselected_op: Optional[str]) -> int:
    """Remove o-supported WMEs from a deselected operator. Returns count removed."""
    if deselected_op is None:
        return 0
    to_remove: List[int] = []
    for tt, wme in state.wm.items():
        if wme.support == Support.O_SUPPORTED and wme.source_operator == deselected_op:
            to_remove.append(tt)
    for tt in to_remove:
        remove_wme_from_state(state, tt)
    return len(to_remove)


def decay_chunk_activations(state: SOARState, decay_rate: float, threshold: float) -> int:
    """Decay chunk activation levels. Returns number of chunks that became dormant."""
    dormant = 0
    updated: Dict[str, Production] = {}
    for name, prod in state.productions.items():
        if prod.learned:
            new_act = max(0.0, prod.activation - decay_rate)
            if new_act < threshold * 0.5 and prod.activation >= threshold * 0.5:
                dormant += 1
            updated[name] = Production(
                name=prod.name,
                conditions=prod.conditions,
                actions=prod.actions,
                prod_type=prod.prod_type,
                activation=new_act,
                learned=prod.learned,
                chunk_source=prod.chunk_source,
            )
        else:
            updated[name] = prod
    state.productions = updated
    return dormant


def create_chunk(
    impasse: Impasse,
    pre_conditions: Tuple[Condition, ...],
    resolution_actions: Tuple[Action, ...],
    confidence: float,
    config: SOARConfig,
    mode: str,
    chunk_counter: int,
) -> Optional[Production]:
    """Create a learned production from a resolved impasse."""
    min_conf = _cfg(config.chunk_confidence_min, mode)
    if confidence < min_conf:
        return None

    # Limit conditions
    conditions = pre_conditions[:config.max_chunk_conditions]

    # Determine chunk type from impasse
    type_map = {
        ImpasseType.TIE: ProductionType.COMPARISON,
        ImpasseType.CONFLICT: ProductionType.ELABORATION,
        ImpasseType.NO_CHANGE: ProductionType.APPLICATION,
        ImpasseType.STATE_NO_CHANGE: ProductionType.PROPOSAL,
    }
    prod_type = type_map.get(impasse.impasse_type, ProductionType.ELABORATION)

    return Production(
        name=f"chunk_{chunk_counter}",
        conditions=conditions,
        actions=resolution_actions,
        prod_type=prod_type,
        activation=confidence,
        learned=True,
        chunk_source=f"impasse_{impasse.impasse_type.value}_{chunk_counter}",
    )


def compute_soar_neurochem(
    state: SOARState,
    impasse: Impasse,
    chunks_learned: int,
    productions_fired_this_cycle: int,
    config: SOARConfig,
) -> SOARNeurochem:
    """Calculate neurochemical signals for this cycle."""
    wm_load = min(1.0, len(state.wm) / max(1, config.max_wm_size))
    fire_load = min(1.0, productions_fired_this_cycle / max(1, config.max_productions_per_cycle))

    delta_da = 0.0
    delta_ne = 0.0
    delta_5ht = 0.0
    delta_cor = 0.0
    delta_ach = 0.0
    delta_gaba = 0.0
    beta_boost = 0.0
    theta_gamma_boost = 0.0
    alpha_suppress = 0.0

    # Successful selection
    if state.selected_operator is not None and impasse.impasse_type == ImpasseType.NONE:
        op = state.proposed_operators.get(state.selected_operator)
        confidence = op.parameters.get("confidence", 0.5) if op else 0.5
        delta_da = config.beta_da_reward * confidence
        beta_boost = config.psi_beta * fire_load

    # Impasse detected
    if impasse.impasse_type not in (ImpasseType.NONE,):
        severity = 0.5
        if impasse.impasse_type == ImpasseType.CONFLICT:
            severity = 0.8
        elif impasse.impasse_type == ImpasseType.STATE_NO_CHANGE:
            severity = 0.7
        elif impasse.impasse_type == ImpasseType.TIE:
            severity = 0.4
        delta_ne = config.beta_ne_conflict * severity
        depth = len(state.substate_stack)
        delta_cor = config.beta_cor_escalation * (depth / max(1, 3))

    # Chunks learned
    if chunks_learned > 0:
        delta_5ht = config.beta_5ht_consolidation * min(1.0, chunks_learned * 0.5)
        theta_gamma_boost = config.psi_theta_gamma * min(1.0, chunks_learned * 0.5)

    # Attention load
    delta_ach = config.beta_ach_attention * wm_load

    # Preference suppression
    n_total = len(state.preferences)
    n_suppressed = sum(1 for p in state.preferences
                       if p.pref_type in (PrefType.PROHIBIT, PrefType.REJECT))
    if n_total > 0:
        delta_gaba = config.beta_gaba_inhibition * (n_suppressed / n_total)

    # WM overload → alpha suppress
    alpha_suppress = config.psi_alpha_suppress * wm_load

    return SOARNeurochem(
        delta_da=_clamp(delta_da, -1.0, 1.0),
        delta_ne=_clamp(delta_ne, -1.0, 1.0),
        delta_5ht=_clamp(delta_5ht, -1.0, 1.0),
        delta_cor=_clamp(delta_cor, -1.0, 1.0),
        delta_ach=_clamp(delta_ach, -1.0, 1.0),
        delta_gaba=_clamp(delta_gaba, -1.0, 1.0),
        beta_boost=_clamp(beta_boost, 0.0, 1.0),
        theta_gamma_boost=_clamp(theta_gamma_boost, 0.0, 1.0),
        alpha_suppress=_clamp(alpha_suppress, 0.0, 1.0),
    )


def _build_delegation_request(
    impasse: Impasse,
    state: SOARState,
) -> Optional[Dict[str, Any]]:
    """Build a delegation request dict for an impasse."""
    target = detect_impasse_delegation(impasse.impasse_type)
    if target is None:
        return None

    context: Dict[str, Any] = {"impasse_type": impasse.impasse_type.value}
    if impasse.tied_operators:
        context["tied_operators"] = list(impasse.tied_operators)
        context["operator_details"] = {
            op_id: {"name": op.name, "params": op.parameters}
            for op_id, op in state.proposed_operators.items()
            if op_id in impasse.tied_operators
        }
    if impasse.stuck_operator:
        op = state.proposed_operators.get(impasse.stuck_operator)
        context["stuck_operator"] = {
            "id": impasse.stuck_operator,
            "name": op.name if op else "unknown",
            "params": op.parameters if op else {},
        }

    return {
        "type": "impasse_delegation",
        "source_engine": "soar_production_engine",
        "target_engine": target,
        "impasse_type": impasse.impasse_type.value,
        "context": context,
        "urgency": state.ne_level,
    }


# =====================================================================
# Engine Class
# =====================================================================


class SOARProductionEngine:
    """
    Engine 3 — SOAR Production Rule Engine.

    Implements a five-phase decision cycle:
    Input → Elaboration → Proposal → Decision → Application.

    Uses neurochemically-modulated preference resolution with impasse
    delegation to existing ZADOS engines and chunking for learning.
    """

    engine_id = "soar_production_engine"
    cluster = "executive_control"

    def __init__(
        self,
        config: Optional[SOARConfig] = None,
        rng_seed: int = 42,
    ) -> None:
        self._config = config or SOARConfig()
        self._rng = np.random.default_rng(rng_seed)
        self._state = SOARState()
        self._mode = "normal"

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def configure(self, mode: str) -> None:
        """Set the operational mode."""
        if isinstance(mode, OperationalMode):
            mode = mode.value
        self._mode = _mode_key(mode)

    # -----------------------------------------------------------------
    # Neurochemical Interface
    # -----------------------------------------------------------------

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Update NT concentrations from pipeline — Pattern A dict."""
        _cl = _clamp
        if "da" in state_dict:
            self._state.da_level = _cl(state_dict["da"])
        if "ne" in state_dict:
            self._state.ne_level = _cl(state_dict["ne"])
        if "5ht" in state_dict:
            self._state._5ht_level = _cl(state_dict["5ht"])
        if "ach" in state_dict:
            self._state.ach_level = _cl(state_dict["ach"])
        if "cor" in state_dict:
            self._state.cor_level = _cl(state_dict["cor"])
        if "gaba" in state_dict:
            self._state.gaba_level = _cl(state_dict["gaba"])
        if "oxt" in state_dict:
            self._state.oxt_level = _cl(state_dict["oxt"])
        if "cb1" in state_dict:
            self._state.cb1_level = _cl(state_dict["cb1"])

    # -----------------------------------------------------------------
    # Production Management
    # -----------------------------------------------------------------

    def add_production(self, production: Production) -> None:
        """Add a production rule to production memory."""
        self._state.productions[production.name] = production

    def remove_production(self, name: str) -> Optional[Production]:
        """Remove a production rule by name."""
        return self._state.productions.pop(name, None)

    def get_production(self, name: str) -> Optional[Production]:
        """Get a production by name."""
        return self._state.productions.get(name)

    def list_productions(self) -> List[str]:
        """List all production names."""
        return list(self._state.productions.keys())

    # -----------------------------------------------------------------
    # Main Process
    # -----------------------------------------------------------------

    def process(self, inp: SOARInput) -> SOARResult:
        """
        Execute one full SOAR decision cycle.

        Five phases: Input → Elaboration → Proposal → Decision → Application.
        """
        t0 = time.perf_counter()
        state = self._state
        config = self._config
        mode = self._mode
        fired_before = state.total_productions_fired

        # ---- Phase 1: INPUT ----
        # Decay o-supported WMEs from previous operator
        if state.previous_operator is not None:
            decay_o_supported(state, state.previous_operator)
        # Populate input-link
        populate_input_link(state, inp)
        state.cycle_count += 1

        # ---- Phase 2: ELABORATION ----
        max_elab = _cfg(config.max_elaboration_rounds, mode)
        threshold = _cfg(config.matching_threshold, mode)
        # ACh modulation of threshold
        ach_depth = _cfg(config.ach_matching_depth, mode)
        effective_threshold = threshold * (1.0 + ach_depth * (state.ach_level - 0.5))
        effective_threshold = _clamp(effective_threshold, 0.1, 1.0)

        elab_rounds = elaborate_until_quiescence(
            state, max_elab, effective_threshold,
        )

        # ---- Phase 3: PROPOSAL ----
        propose_operators(state, effective_threshold)
        collect_comparison_preferences(state, effective_threshold)

        # Apply NT preference bias
        state.preferences = apply_nt_preference_bias(state, config, mode)

        # ---- Phase 4: DECISION ----
        eps = _cfg(config.preference_epsilon, mode)
        winner_id, impasse_type, tied_ops, conflict_prefs = resolve_preferences(
            state.preferences, state.proposed_operators, eps, self._rng,
        )

        # Build impasse
        impasse = Impasse(
            impasse_type=impasse_type,
            tied_operators=tied_ops,
            conflicting_prefs=conflict_prefs,
            delegation_target=detect_impasse_delegation(impasse_type),
        )

        # Track impasse
        delegation_requests: List[Dict[str, Any]] = []
        if impasse_type != ImpasseType.NONE:
            state.total_impasses += 1
            state.current_impasse = impasse
            # Build delegation request
            deleg = _build_delegation_request(impasse, state)
            if deleg is not None:
                delegation_requests.append(deleg)

        # Select operator
        state.previous_operator = state.selected_operator
        state.selected_operator = winner_id

        # ---- Phase 5: APPLICATION ----
        app_wmes: List[WME] = []
        if winner_id is not None:
            app_wmes_list, any_fired = apply_operator_productions(
                state, winner_id, effective_threshold,
            )
            app_wmes = app_wmes_list
            # Check for NO_CHANGE impasse
            if not any_fired:
                impasse = Impasse(
                    impasse_type=ImpasseType.NO_CHANGE,
                    stuck_operator=winner_id,
                    delegation_target=detect_impasse_delegation(ImpasseType.NO_CHANGE),
                )
                state.total_impasses += 1
                deleg = _build_delegation_request(impasse, state)
                if deleg is not None:
                    delegation_requests.append(deleg)

        # Extract output-link
        action_commands = extract_output_link(state)

        # Chunking (no actual substate resolution in this cycle — chunks
        # come when delegation results arrive in future cycles)
        chunks_learned_list: List[Production] = []

        # Decay chunk activations
        decay_chunk_activations(state, config.chunk_decay_rate, threshold)

        # Compute neurochemical signals
        fired_this_cycle = state.total_productions_fired - fired_before
        neurochem = compute_soar_neurochem(
            state, impasse, len(chunks_learned_list),
            fired_this_cycle, config,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return SOARResult(
            selected_operator=state.proposed_operators.get(winner_id) if winner_id else None,
            action_commands=action_commands,
            impasse=impasse,
            delegation_requests=tuple(delegation_requests),
            chunks_learned=tuple(chunks_learned_list),
            neurochemical_signals=neurochem,
            cycle_count=state.cycle_count,
            elaboration_rounds=elab_rounds,
            productions_fired=fired_this_cycle,
            wm_size=len(state.wm),
            engine_id=self.engine_id,
            processing_time_ms=elapsed_ms,
        )

    # -----------------------------------------------------------------
    # Chunk Learning (called when delegation result arrives)
    # -----------------------------------------------------------------

    def learn_from_resolution(
        self,
        impasse: Impasse,
        pre_conditions: Tuple[Condition, ...],
        resolution_actions: Tuple[Action, ...],
        confidence: float,
    ) -> Optional[Production]:
        """
        Create a chunk from an impasse resolution.
        Called when a delegation result resolves a previous impasse.
        """
        chunk = create_chunk(
            impasse, pre_conditions, resolution_actions,
            confidence, self._config, self._mode,
            self._state.total_chunks_learned,
        )
        if chunk is not None:
            self._state.productions[chunk.name] = chunk
            self._state.total_chunks_learned += 1
        return chunk

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current engine status."""
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode,
            "cycle_count": self._state.cycle_count,
            "wm_size": len(self._state.wm),
            "production_count": len(self._state.productions),
            "chunks_learned": self._state.total_chunks_learned,
            "total_impasses": self._state.total_impasses,
            "total_productions_fired": self._state.total_productions_fired,
            "selected_operator": self._state.selected_operator,
            "da_level": self._state.da_level,
            "ne_level": self._state.ne_level,
            "_5ht_level": self._state._5ht_level,
            "ach_level": self._state.ach_level,
            "cor_level": self._state.cor_level,
            "gaba_level": self._state.gaba_level,
            "oxt_level": self._state.oxt_level,
            "cb1_level": self._state.cb1_level,
        }

    def introspect(self) -> Dict[str, Any]:
        """Detailed introspection for debugging."""
        status = self.get_status()
        status["proposed_operators"] = {
            op_id: {"name": op.name, "params": op.parameters}
            for op_id, op in self._state.proposed_operators.items()
        }
        status["preferences"] = [
            {"op": p.operator_id, "type": p.pref_type.value,
             "ref": p.reference_id, "strength": p.strength}
            for p in self._state.preferences
        ]
        status["current_impasse"] = (
            self._state.current_impasse.impasse_type.value
            if self._state.current_impasse else None
        )
        status["learned_productions"] = [
            name for name, prod in self._state.productions.items()
            if prod.learned
        ]
        return status
