"""
Engine 21 -- Strategic Decision Engine  (``strategic_decision_engine``)
======================================================================
Multi-step goal planning engine that maintains a hierarchical goal tree,
selects sub-goals via utility-feasibility scoring, tracks commitment
progress, detects stagnation, and triggers replanning when the current
strategy is failing.

Differs from Engine 15 (Decision Making Engine) in scope: E15 is a
single-cycle decision router, while E21 handles **multi-cycle**
strategies with persistent commitment tracking and plan revision.

Five-phase lifecycle per cycle:
  * **Phase 1 -- Context Intake**: receive new goals/context, ingest
    upstream detection flags, update NT read-ports.
  * **Phase 2 -- Commitment Evaluation**: check active commitments for
    stagnation, progress, and feasibility decay.
  * **Phase 3 -- Sub-goal Selection**: pick best actionable sub-goal
    from the goal tree via ``utility x feasibility`` ranking, modulated
    by NT levels and mode.
  * **Phase 4 -- Strategy Scoring**: evaluate candidate strategies on
    expected utility, risk, and resource cost; compute composite score.
  * **Phase 5 -- Revision & Emit**: detect stagnation / infeasibility,
    trigger replanning if needed, emit decision + neurochemical signals.

Neurochemical coupling (Pattern A — ``Dict[str, float]``):
  DA   -- exploration bonus: broadens strategy search, raises exploration
  5-HT -- conservation: raises revision threshold, favours current plan
  NE   -- urgency: lowers stagnation threshold, faster replanning
  ACh  -- evaluation depth: more sub-goals considered per cycle
  COR  -- risk aversion: penalises high-risk strategies
  GABA -- suppression: prunes low-utility goals from selection

Output write-port signals:
  beta_boost  -- active planning engagement
  theta_boost -- deep strategic deliberation
  da_delta    -- on novel strategy adoption
  ne_delta    -- on revision / stagnation detection
  5ht_delta   -- on plan consolidation (commitment progress)
  cor_delta   -- on high-risk strategy exposure

Key formulas:
  composite(s)  = w_u * E[U] - w_r * Risk + w_f * Feasibility - w_c * Cost
  stagnation(g) = stagnation_count(g) >= theta_stag * (1 - ne) * (1 + 5ht)
  exploration   = da * exploration_bonus * (1 + cb1_in_dream)
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Enums
# =====================================================================


class GoalStatus(str, Enum):
    """Lifecycle status of a goal in the goal tree."""
    PENDING    = "pending"       # Created but not yet actionable
    ACTIVE     = "active"        # Currently being pursued
    COMPLETED  = "completed"     # Successfully achieved
    ABANDONED  = "abandoned"     # Dropped (infeasible or superseded)
    STAGNANT   = "stagnant"      # Progress halted — revision candidate


class RevisionReason(str, Enum):
    """Why a plan revision was triggered."""
    STAGNATION          = "stagnation"            # No progress for N ticks
    FEASIBILITY_DECAY   = "feasibility_decay"     # Feasibility dropped below threshold
    UTILITY_DROP        = "utility_drop"          # Expected utility fell
    RISK_SPIKE          = "risk_spike"             # Risk exceeded tolerance
    EXTERNAL_OVERRIDE   = "external_override"     # Upstream signal forced replanning
    GOAL_COMPLETED      = "goal_completed"        # Current goal finished, need new one
    URGENCY_OVERRIDE    = "urgency_override"      # NE-driven fast replanning


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class SDConfig:
    """All tunable parameters for the Strategic Decision Engine."""

    # --- Goal tree ---
    max_goals: int = 64
    max_children_per_goal: int = 8
    max_depth: int = 6

    # --- Strategy pool ---
    max_strategies_per_goal: int = 12
    max_active_strategies: int = 5

    # --- Stagnation ---
    stagnation_threshold: Dict[str, int] = field(default_factory=lambda: {
        "normal": 5, "dev": 7, "learning": 6,
        "reflective": 8, "rem_normal": 5, "rem_dream": 3,
    })

    # --- Revision ---
    revision_feasibility_floor: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.20, "dev": 0.15, "learning": 0.18,
        "reflective": 0.25, "rem_normal": 0.20, "rem_dream": 0.10,
    })
    revision_utility_floor: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.15, "dev": 0.10, "learning": 0.12,
        "reflective": 0.20, "rem_normal": 0.15, "rem_dream": 0.05,
    })
    revision_risk_ceiling: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.80, "dev": 0.85, "learning": 0.80,
        "reflective": 0.75, "rem_normal": 0.80, "rem_dream": 0.95,
    })

    # --- Composite scoring weights ---
    w_utility: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.40, "dev": 0.45, "learning": 0.35,
        "reflective": 0.35, "rem_normal": 0.40, "rem_dream": 0.25,
    })
    w_feasibility: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.25, "dev": 0.25, "learning": 0.30,
        "reflective": 0.25, "rem_normal": 0.25, "rem_dream": 0.15,
    })
    w_risk: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.20, "dev": 0.15, "learning": 0.20,
        "reflective": 0.25, "rem_normal": 0.20, "rem_dream": 0.10,
    })
    w_cost: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.15, "dev": 0.15, "learning": 0.15,
        "reflective": 0.15, "rem_normal": 0.15, "rem_dream": 0.10,
    })

    # --- Exploration ---
    exploration_bonus: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.10, "dev": 0.08, "learning": 0.15,
        "reflective": 0.05, "rem_normal": 0.10, "rem_dream": 0.30,
    })

    # --- Sub-goal selection breadth (base count before ACh modulation) ---
    selection_breadth: Dict[str, int] = field(default_factory=lambda: {
        "normal": 5, "dev": 4, "learning": 6,
        "reflective": 4, "rem_normal": 5, "rem_dream": 8,
    })

    # --- NT coupling weights (read-port modulation) ---
    mu_da:   float = 0.25   # Exploration boost
    mu_5ht:  float = 0.20   # Conservation / revision resistance
    mu_ne:   float = 0.30   # Urgency / stagnation sensitivity
    mu_ach:  float = 0.20   # Evaluation depth
    mu_cor:  float = 0.25   # Risk aversion
    mu_gaba: float = 0.15   # Low-utility suppression
    mu_cb1:  float = 0.20   # Creative strategy discovery (REM_DREAM)

    # --- Write-port coefficients ---
    beta_da_novel:       float = 0.10  # DA on novel strategy adoption
    beta_5ht_progress:   float = 0.08  # 5-HT on commitment progress
    beta_ne_revision:    float = 0.12  # NE on stagnation/revision
    beta_cor_risk:       float = 0.10  # COR on high-risk exposure
    beta_ach_depth:      float = 0.06  # ACh on deep evaluation
    beta_gaba_prune:     float = 0.04  # GABA on goal pruning

    psi_beta:  float = 0.05  # Beta boost (active planning)
    psi_theta: float = 0.04  # Theta boost (deep deliberation)

    # --- Goal utility decay (per cycle, for non-progressing goals) ---
    utility_decay_rate: float = 0.02

    # --- Progress momentum (EMA smoothing) ---
    progress_ema_alpha: float = 0.3


# =====================================================================
# Mutable internal records  (not exported — engine-private)
# =====================================================================


@dataclass
class _GoalRecord:
    """Internal mutable goal state."""
    goal_id: str
    description: str
    parent_id: Optional[str]
    priority: float          # [0, 1]
    feasibility: float       # [0, 1]
    utility: float           # [0, 1]
    status: GoalStatus
    created_tick: int
    last_progress_tick: int
    progress: float          # [0, 1]  cumulative
    progress_velocity: float # EMA of progress deltas
    stagnation_count: int
    children_ids: List[str]
    metadata: Dict[str, Any]


@dataclass
class _StrategyRecord:
    """Internal mutable strategy state."""
    strategy_id: str
    goal_id: str
    expected_utility: float  # [0, 1]
    risk: float              # [0, 1]
    resource_cost: float     # [0, 1]
    composite: float         # computed
    adopted: bool
    created_tick: int
    metadata: Dict[str, Any]


@dataclass
class _CommitmentRecord:
    """Tracks an active commitment to a (goal, strategy) pair."""
    goal_id: str
    strategy_id: str
    start_tick: int
    progress: float          # [0, 1]
    stagnation_count: int
    last_progress_delta: float
    progress_velocity: float # EMA of progress deltas


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class _SDState:
    """Runtime state for NT read-ports and counters."""
    da_level:   float = 0.0
    _5ht_level: float = 0.0
    ne_level:   float = 0.0
    ach_level:  float = 0.0
    cor_level:  float = 0.0
    gaba_level: float = 0.0
    cb1_level:  float = 0.0

    total_goals_created:    int = 0
    total_strategies_scored: int = 0
    total_revisions:        int = 0
    total_goals_completed:  int = 0
    total_goals_abandoned:  int = 0


# =====================================================================
# Frozen output dataclasses
# =====================================================================


@dataclass(frozen=True)
class Goal:
    """Exported view of a goal node."""
    goal_id: str = ""
    description: str = ""
    parent_id: Optional[str] = None
    priority: float = 0.0
    feasibility: float = 0.0
    utility: float = 0.0
    status: GoalStatus = GoalStatus.PENDING
    created_tick: int = 0
    progress: float = 0.0
    children_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyScore:
    """Evaluated strategy with composite score."""
    strategy_id: str = ""
    goal_id: str = ""
    expected_utility: float = 0.0
    risk: float = 0.0
    resource_cost: float = 0.0
    composite: float = 0.0


@dataclass(frozen=True)
class Commitment:
    """Exported view of an active commitment."""
    goal_id: str = ""
    strategy_id: str = ""
    start_tick: int = 0
    progress: float = 0.0
    stagnation_count: int = 0


@dataclass(frozen=True)
class RevisionEvent:
    """Record of a plan revision."""
    goal_id: str = ""
    old_strategy_id: str = ""
    new_strategy_id: str = ""
    reason: RevisionReason = RevisionReason.STAGNATION
    tick: int = 0
    detail: str = ""


@dataclass(frozen=True)
class StrategicDecisionNeurochem:
    """Neurochemical deltas emitted by the Strategic Decision Engine."""
    da_delta:   float = 0.0
    _5ht_delta: float = 0.0
    ne_delta:   float = 0.0
    ach_delta:  float = 0.0
    cor_delta:  float = 0.0
    gaba_delta: float = 0.0
    beta_boost:  float = 0.0
    theta_boost: float = 0.0


@dataclass(frozen=True)
class StrategicDecisionInput:
    """Input to the Strategic Decision Engine's ``process()`` method."""
    # New goals to register
    new_goals: Tuple[Dict[str, Any], ...] = ()
    # New strategies to register
    new_strategies: Tuple[Dict[str, Any], ...] = ()
    # Progress updates: {goal_id: progress_delta}
    progress_updates: Dict[str, float] = field(default_factory=dict)
    # Goal completions: set of goal_ids to mark completed
    completed_goals: Tuple[str, ...] = ()
    # Goal abandonments: set of goal_ids to abandon
    abandoned_goals: Tuple[str, ...] = ()
    # External revision request
    force_revision_goals: Tuple[str, ...] = ()
    # Context
    system_entropy: float = 0.5
    contradiction_count: int = 0
    active_mode: str = "normal"
    cycle_count: int = 0
    # Emotion intensities (for future integration)
    emotion_intensities: Optional[Dict[str, float]] = None
    # Reward alignment scores (for strategy utility anchoring)
    reward_scores: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class StrategicDecisionResult:
    """Output from the Strategic Decision Engine."""
    # Active goals (all non-terminal goals)
    active_goals: Tuple[Goal, ...] = ()
    # Selected sub-goal for this cycle
    selected_goal: Optional[Goal] = None
    # Selected strategy for this cycle
    selected_strategy: Optional[StrategyScore] = None
    # All scored strategies this cycle
    scored_strategies: Tuple[StrategyScore, ...] = ()
    # Active commitments
    commitments: Tuple[Commitment, ...] = ()
    # Revisions triggered this cycle
    revisions: Tuple[RevisionEvent, ...] = ()
    # Stagnant goals detected
    stagnant_goals: Tuple[str, ...] = ()
    # Goals pruned (GABA suppression)
    pruned_goals: Tuple[str, ...] = ()
    # Neurochemical signals
    neurochem_signals: StrategicDecisionNeurochem = field(
        default_factory=StrategicDecisionNeurochem,
    )
    # Metadata
    processing_time_ms: float = 0.0
    engine_id: str = "strategic_decision_engine"
    cycle_count: int = 0
    mode: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Pure functions
# =====================================================================


def compute_strategy_composite(
    expected_utility: float,
    risk: float,
    resource_cost: float,
    feasibility: float,
    w_utility: float,
    w_risk: float,
    w_feasibility: float,
    w_cost: float,
    exploration_bonus: float = 0.0,
) -> float:
    """
    Compute composite strategy score.

    composite = w_u * E[U] * feas - w_r * Risk - w_c * Cost + exploration
    """
    score = (
        w_utility * expected_utility * feasibility
        - w_risk * risk
        - w_cost * resource_cost
        + w_feasibility * feasibility
        + exploration_bonus
    )
    return score


def compute_goal_selection_score(
    priority: float,
    utility: float,
    feasibility: float,
    progress: float,
    stagnation_count: int,
    da_level: float,
    cor_level: float,
    gaba_level: float,
) -> float:
    """
    Score a goal for sub-goal selection.

    Higher priority, utility, and feasibility increase selection score.
    Stagnation reduces score.  DA broadens (adds exploration noise
    to non-top goals).  COR penalises low-feasibility goals.
    GABA suppresses low-utility goals.
    """
    # Base score
    base = priority * 0.3 + utility * 0.4 + feasibility * 0.3

    # Progress factor: partially complete goals get a continuation bonus
    continuation = 0.1 * progress

    # Stagnation penalty
    stag_penalty = 0.05 * stagnation_count

    # COR risk aversion: penalise goals with low feasibility
    cor_penalty = cor_level * 0.15 * max(0.0, 0.5 - feasibility)

    # GABA suppression of low-utility goals
    gaba_suppress = gaba_level * 0.10 * max(0.0, 0.3 - utility)

    score = base + continuation - stag_penalty - cor_penalty - gaba_suppress
    return score


def check_stagnation(
    stagnation_count: int,
    threshold_base: int,
    ne_level: float,
    _5ht_level: float,
    mu_ne: float,
    mu_5ht: float,
) -> bool:
    """
    Determine if a goal/commitment is stagnant.

    Effective threshold = base * (1 - mu_ne * NE) * (1 + mu_5ht * 5-HT)
    NE lowers threshold (faster stagnation detection).
    5-HT raises threshold (more patience).
    """
    effective = threshold_base * (1.0 - mu_ne * ne_level) * (1.0 + mu_5ht * _5ht_level)
    effective = max(1.0, effective)
    return stagnation_count >= effective


def check_feasibility_revision(
    feasibility: float,
    floor: float,
    cor_level: float,
    mu_cor: float,
) -> bool:
    """
    Check if feasibility has decayed below the revision floor.

    COR raises the floor (more conservative about infeasible goals).
    """
    effective_floor = floor * (1.0 + mu_cor * cor_level)
    effective_floor = min(effective_floor, 0.95)
    return feasibility < effective_floor


def check_risk_revision(
    risk: float,
    ceiling: float,
    cor_level: float,
    mu_cor: float,
) -> bool:
    """
    Check if risk exceeds the revision ceiling.

    COR lowers the ceiling (less risk tolerance).
    """
    effective_ceiling = ceiling * (1.0 - mu_cor * cor_level * 0.5)
    effective_ceiling = max(0.10, effective_ceiling)
    return risk > effective_ceiling


def compute_neurochem_signals(
    novel_strategy_adopted: bool,
    progress_made: float,
    revisions_triggered: int,
    max_risk_exposure: float,
    evaluation_depth: int,
    goals_pruned: int,
    stagnant_count: int,
    cfg: SDConfig,
) -> StrategicDecisionNeurochem:
    """Compute write-port neurochemical deltas from cycle results."""
    da_delta = 0.0
    _5ht_delta = 0.0
    ne_delta = 0.0
    cor_delta = 0.0
    ach_delta = 0.0
    gaba_delta = 0.0
    beta_boost = 0.0
    theta_boost = 0.0

    # Novel strategy adoption → DA
    if novel_strategy_adopted:
        da_delta += cfg.beta_da_novel

    # Progress → 5-HT (consolidation signal)
    if progress_made > 0.0:
        _5ht_delta += cfg.beta_5ht_progress * _clamp(progress_made)

    # Revision / stagnation → NE
    if revisions_triggered > 0:
        ne_delta += cfg.beta_ne_revision * min(revisions_triggered, 3)
    if stagnant_count > 0:
        ne_delta += cfg.beta_ne_revision * 0.5 * min(stagnant_count, 3)

    # High-risk exposure → COR
    if max_risk_exposure > 0.5:
        cor_delta += cfg.beta_cor_risk * (max_risk_exposure - 0.5) * 2.0

    # Deep evaluation → ACh
    if evaluation_depth > 3:
        ach_delta += cfg.beta_ach_depth * min((evaluation_depth - 3) / 5.0, 1.0)

    # Goal pruning → GABA
    if goals_pruned > 0:
        gaba_delta += cfg.beta_gaba_prune * min(goals_pruned, 5)

    # Oscillatory: active planning → beta, deep deliberation → theta
    beta_boost = cfg.psi_beta
    if evaluation_depth > 5:
        theta_boost = cfg.psi_theta * min(evaluation_depth / 10.0, 1.0)

    return StrategicDecisionNeurochem(
        da_delta=da_delta,
        _5ht_delta=_5ht_delta,
        ne_delta=ne_delta,
        ach_delta=ach_delta,
        cor_delta=cor_delta,
        gaba_delta=gaba_delta,
        beta_boost=beta_boost,
        theta_boost=theta_boost,
    )


def export_goal(rec: _GoalRecord) -> Goal:
    """Convert internal _GoalRecord to frozen Goal for export."""
    return Goal(
        goal_id=rec.goal_id,
        description=rec.description,
        parent_id=rec.parent_id,
        priority=rec.priority,
        feasibility=rec.feasibility,
        utility=rec.utility,
        status=rec.status,
        created_tick=rec.created_tick,
        progress=rec.progress,
        children_ids=tuple(rec.children_ids),
        metadata=dict(rec.metadata),
    )


def export_commitment(rec: _CommitmentRecord) -> Commitment:
    """Convert internal _CommitmentRecord to frozen Commitment."""
    return Commitment(
        goal_id=rec.goal_id,
        strategy_id=rec.strategy_id,
        start_tick=rec.start_tick,
        progress=rec.progress,
        stagnation_count=rec.stagnation_count,
    )


def export_strategy(rec: _StrategyRecord) -> StrategyScore:
    """Convert internal _StrategyRecord to frozen StrategyScore."""
    return StrategyScore(
        strategy_id=rec.strategy_id,
        goal_id=rec.goal_id,
        expected_utility=rec.expected_utility,
        risk=rec.risk,
        resource_cost=rec.resource_cost,
        composite=rec.composite,
    )


# =====================================================================
# Engine class
# =====================================================================


class StrategicDecisionEngine:
    """
    Engine 21 -- Strategic Decision Engine.

    Multi-step goal planning with commitment tracking and plan revision.

    Lifecycle per cycle:
      Phase 1: Context Intake (register goals/strategies, apply progress)
      Phase 2: Commitment Evaluation (stagnation, feasibility decay)
      Phase 3: Sub-goal Selection (utility x feasibility ranking)
      Phase 4: Strategy Scoring (composite scores with NT modulation)
      Phase 5: Revision & Emit (revise, emit decision + neurochem)

    API
    ---
    configure(mode)                -- set operational mode
    update_neurochem_state(state)  -- inject external NT levels (Pattern A)
    add_goal(...)                  -- register a goal in the tree
    remove_goal(goal_id)           -- remove a goal from the tree
    add_strategy(...)              -- register a strategy for a goal
    update_progress(goal_id, delta) -- advance progress on a goal
    complete_goal(goal_id)         -- mark a goal as completed
    abandon_goal(goal_id)          -- mark a goal as abandoned
    select_next_action()           -- pick best (goal, strategy) pair
    process(input_data)            -- full cycle: intake+eval+select+score+emit
    get_status()                   -- introspection
    """

    engine_id = "strategic_decision_engine"
    cluster   = "reasoning"

    def __init__(
        self,
        config: Optional[SDConfig] = None,
    ) -> None:
        self._cfg = config or SDConfig()
        self._mode = OperationalMode.NORMAL
        self._state = _SDState()
        self._cycle_count = 0

        # Core data stores
        self._goals: Dict[str, _GoalRecord] = {}
        self._strategies: Dict[str, _StrategyRecord] = {}
        self._commitments: Dict[str, _CommitmentRecord] = {}  # keyed by goal_id

        # Root goals (no parent)
        self._root_ids: List[str] = []

    # -----------------------------------------------------------------
    # Mode & NT injection
    # -----------------------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        """Set operational mode."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Pattern A — inject external NT levels as Dict[str, float]."""
        if "da" in state_dict:
            self._state.da_level = _clamp(state_dict["da"])
        if "5ht" in state_dict:
            self._state._5ht_level = _clamp(state_dict["5ht"])
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "ach" in state_dict:
            self._state.ach_level = _clamp(state_dict["ach"])
        if "cor" in state_dict:
            self._state.cor_level = _clamp(state_dict["cor"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])
        if "cb1" in state_dict:
            self._state.cb1_level = _clamp(state_dict["cb1"])

    def _mode_key(self) -> str:
        return self._mode.value

    def _get_mode_param(self, param_dict: Dict, default=0.5):
        return param_dict.get(self._mode_key(), default)

    # -----------------------------------------------------------------
    # Goal management
    # -----------------------------------------------------------------

    def add_goal(
        self,
        goal_id: str,
        description: str,
        parent_id: Optional[str] = None,
        priority: float = 0.5,
        feasibility: float = 0.8,
        utility: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Goal:
        """
        Register a new goal in the goal tree.

        Parameters
        ----------
        goal_id : str
            Unique identifier.
        description : str
            Human-readable description.
        parent_id : str, optional
            Parent goal id (None for root goals).
        priority : float
            Importance weight in [0, 1].
        feasibility : float
            Estimated achievability in [0, 1].
        utility : float
            Expected value of achieving this goal in [0, 1].
        metadata : dict, optional
            Arbitrary key-value context.

        Returns
        -------
        Goal
            Frozen exported view of the registered goal.
        """
        if len(self._goals) >= self._cfg.max_goals:
            # Evict lowest-utility pending goal if at capacity
            self._evict_lowest_goal()

        if parent_id and parent_id not in self._goals:
            parent_id = None  # Graceful fallback to root

        # Check parent child limit
        if parent_id and parent_id in self._goals:
            parent_rec = self._goals[parent_id]
            if len(parent_rec.children_ids) >= self._cfg.max_children_per_goal:
                # Don't add — return existing if re-registering
                if goal_id in self._goals:
                    return export_goal(self._goals[goal_id])
                # Evict lowest-utility child
                self._evict_child(parent_id)

        rec = _GoalRecord(
            goal_id=goal_id,
            description=description,
            parent_id=parent_id,
            priority=_clamp(priority),
            feasibility=_clamp(feasibility),
            utility=_clamp(utility),
            status=GoalStatus.PENDING,
            created_tick=self._cycle_count,
            last_progress_tick=self._cycle_count,
            progress=0.0,
            progress_velocity=0.0,
            stagnation_count=0,
            children_ids=[],
            metadata=metadata or {},
        )

        self._goals[goal_id] = rec
        self._state.total_goals_created += 1

        if parent_id and parent_id in self._goals:
            self._goals[parent_id].children_ids.append(goal_id)
        else:
            if goal_id not in self._root_ids:
                self._root_ids.append(goal_id)

        return export_goal(rec)

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal and its descendants from the tree."""
        if goal_id not in self._goals:
            return False

        # Collect descendants (BFS)
        to_remove: List[str] = []
        queue = [goal_id]
        while queue:
            gid = queue.pop(0)
            if gid in self._goals:
                to_remove.append(gid)
                queue.extend(self._goals[gid].children_ids)

        for gid in to_remove:
            rec = self._goals.pop(gid, None)
            if rec and rec.parent_id and rec.parent_id in self._goals:
                parent = self._goals[rec.parent_id]
                if gid in parent.children_ids:
                    parent.children_ids.remove(gid)
            if gid in self._root_ids:
                self._root_ids.remove(gid)
            # Remove associated commitments
            self._commitments.pop(gid, None)
            # Remove associated strategies
            strat_ids = [
                s.strategy_id for s in self._strategies.values()
                if s.goal_id == gid
            ]
            for sid in strat_ids:
                del self._strategies[sid]

        return True

    def _evict_lowest_goal(self) -> None:
        """Evict the lowest-utility PENDING goal to make room."""
        pending = [
            g for g in self._goals.values()
            if g.status == GoalStatus.PENDING
        ]
        if not pending:
            # Evict lowest-utility non-active goal
            pending = [
                g for g in self._goals.values()
                if g.status not in (GoalStatus.ACTIVE, GoalStatus.COMPLETED)
            ]
        if pending:
            worst = min(pending, key=lambda g: g.utility)
            self.remove_goal(worst.goal_id)

    def _evict_child(self, parent_id: str) -> None:
        """Evict the lowest-utility child of a parent goal."""
        parent = self._goals[parent_id]
        if not parent.children_ids:
            return
        children = [
            self._goals[cid]
            for cid in parent.children_ids
            if cid in self._goals
        ]
        if children:
            worst = min(children, key=lambda g: g.utility)
            self.remove_goal(worst.goal_id)

    # -----------------------------------------------------------------
    # Strategy management
    # -----------------------------------------------------------------

    def add_strategy(
        self,
        strategy_id: str,
        goal_id: str,
        expected_utility: float = 0.5,
        risk: float = 0.3,
        resource_cost: float = 0.2,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StrategyScore:
        """
        Register a candidate strategy for a goal.

        Returns
        -------
        StrategyScore
            Frozen scored strategy.
        """
        if goal_id not in self._goals:
            raise ValueError(f"Goal '{goal_id}' not found")

        # Enforce per-goal strategy limit
        existing = [
            s for s in self._strategies.values()
            if s.goal_id == goal_id
        ]
        if len(existing) >= self._cfg.max_strategies_per_goal:
            # Replace lowest-composite strategy
            worst = min(existing, key=lambda s: s.composite)
            del self._strategies[worst.strategy_id]

        # Compute composite with current NT state
        goal_rec = self._goals[goal_id]
        mk = self._mode_key()
        composite = compute_strategy_composite(
            expected_utility=_clamp(expected_utility),
            risk=_clamp(risk),
            resource_cost=_clamp(resource_cost),
            feasibility=goal_rec.feasibility,
            w_utility=self._get_mode_param(self._cfg.w_utility, 0.40),
            w_risk=self._get_mode_param(self._cfg.w_risk, 0.20),
            w_feasibility=self._get_mode_param(self._cfg.w_feasibility, 0.25),
            w_cost=self._get_mode_param(self._cfg.w_cost, 0.15),
            exploration_bonus=0.0,
        )

        rec = _StrategyRecord(
            strategy_id=strategy_id,
            goal_id=goal_id,
            expected_utility=_clamp(expected_utility),
            risk=_clamp(risk),
            resource_cost=_clamp(resource_cost),
            composite=composite,
            adopted=False,
            created_tick=self._cycle_count,
            metadata=metadata or {},
        )
        self._strategies[strategy_id] = rec
        self._state.total_strategies_scored += 1

        return export_strategy(rec)

    # -----------------------------------------------------------------
    # Progress tracking
    # -----------------------------------------------------------------

    def update_progress(self, goal_id: str, progress_delta: float) -> float:
        """
        Advance progress on a goal.

        Returns new cumulative progress.
        """
        if goal_id not in self._goals:
            return 0.0

        rec = self._goals[goal_id]
        delta = max(0.0, progress_delta)
        rec.progress = _clamp(rec.progress + delta)
        rec.last_progress_tick = self._cycle_count

        # Update EMA velocity
        alpha = self._cfg.progress_ema_alpha
        rec.progress_velocity = alpha * delta + (1.0 - alpha) * rec.progress_velocity

        # Reset stagnation if real progress made
        if delta > 0.001:
            rec.stagnation_count = 0

        # Activate if still pending
        if rec.status == GoalStatus.PENDING and delta > 0.0:
            rec.status = GoalStatus.ACTIVE

        # Update commitment if exists
        if goal_id in self._commitments:
            com = self._commitments[goal_id]
            com.progress = rec.progress
            com.last_progress_delta = delta
            com.progress_velocity = rec.progress_velocity
            if delta > 0.001:
                com.stagnation_count = 0

        return rec.progress

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed."""
        if goal_id not in self._goals:
            return False
        rec = self._goals[goal_id]
        rec.status = GoalStatus.COMPLETED
        rec.progress = 1.0
        self._state.total_goals_completed += 1
        # Remove commitment
        self._commitments.pop(goal_id, None)
        return True

    def abandon_goal(self, goal_id: str) -> bool:
        """Mark a goal as abandoned."""
        if goal_id not in self._goals:
            return False
        rec = self._goals[goal_id]
        rec.status = GoalStatus.ABANDONED
        self._state.total_goals_abandoned += 1
        # Remove commitment
        self._commitments.pop(goal_id, None)
        return True

    # -----------------------------------------------------------------
    # Sub-goal selection
    # -----------------------------------------------------------------

    def _get_actionable_goals(self) -> List[_GoalRecord]:
        """Return goals that are eligible for selection (PENDING or ACTIVE)."""
        return [
            g for g in self._goals.values()
            if g.status in (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.STAGNANT)
        ]

    def select_next_action(self) -> Tuple[Optional[Goal], Optional[StrategyScore]]:
        """
        Pick the best (goal, strategy) pair for this cycle.

        Returns
        -------
        (Goal or None, StrategyScore or None)
        """
        actionable = self._get_actionable_goals()
        if not actionable:
            return None, None

        # Score each goal
        scored: List[Tuple[_GoalRecord, float]] = []
        for g in actionable:
            s = compute_goal_selection_score(
                priority=g.priority,
                utility=g.utility,
                feasibility=g.feasibility,
                progress=g.progress,
                stagnation_count=g.stagnation_count,
                da_level=self._state.da_level,
                cor_level=self._state.cor_level,
                gaba_level=self._state.gaba_level,
            )
            # DA exploration bonus: add noise proportional to DA
            exploration = (
                self._state.da_level
                * self._get_mode_param(self._cfg.exploration_bonus, 0.10)
            )
            # In REM_DREAM, CB1 amplifies exploration
            if self._mode == OperationalMode.REM_DREAM:
                exploration *= (1.0 + self._cfg.mu_cb1 * self._state.cb1_level)
            s += exploration * (0.5 - abs(0.5 - (hash(g.goal_id) % 100) / 100.0))
            scored.append((g, s))

        # Sort descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # ACh modulates how many we consider (breadth)
        breadth = self._get_mode_param(self._cfg.selection_breadth, 5)
        ach_bonus = int(self._cfg.mu_ach * self._state.ach_level * 4)
        breadth = min(breadth + ach_bonus, len(scored))

        # Pick the best from the top-N
        best_goal_rec = scored[0][0]

        # Find best strategy for the selected goal
        goal_strategies = [
            s for s in self._strategies.values()
            if s.goal_id == best_goal_rec.goal_id
        ]

        best_strategy = None
        if goal_strategies:
            # Re-score strategies with current NT state
            for s in goal_strategies:
                exploration_bonus_val = (
                    self._state.da_level
                    * self._get_mode_param(self._cfg.exploration_bonus, 0.10)
                    * (0.5 if not s.adopted else 0.0)
                )
                s.composite = compute_strategy_composite(
                    expected_utility=s.expected_utility,
                    risk=s.risk,
                    resource_cost=s.resource_cost,
                    feasibility=best_goal_rec.feasibility,
                    w_utility=self._get_mode_param(self._cfg.w_utility, 0.40),
                    w_risk=self._get_mode_param(self._cfg.w_risk, 0.20),
                    w_feasibility=self._get_mode_param(self._cfg.w_feasibility, 0.25),
                    w_cost=self._get_mode_param(self._cfg.w_cost, 0.15),
                    exploration_bonus=exploration_bonus_val,
                )

            best_strat_rec = max(goal_strategies, key=lambda s: s.composite)
            best_strategy = export_strategy(best_strat_rec)

        return export_goal(best_goal_rec), best_strategy

    # -----------------------------------------------------------------
    # Commitment management
    # -----------------------------------------------------------------

    def _create_or_update_commitment(
        self,
        goal_id: str,
        strategy_id: str,
    ) -> _CommitmentRecord:
        """Create or update a commitment record."""
        if goal_id in self._commitments:
            com = self._commitments[goal_id]
            com.strategy_id = strategy_id
            return com

        com = _CommitmentRecord(
            goal_id=goal_id,
            strategy_id=strategy_id,
            start_tick=self._cycle_count,
            progress=self._goals[goal_id].progress if goal_id in self._goals else 0.0,
            stagnation_count=0,
            last_progress_delta=0.0,
            progress_velocity=0.0,
        )
        self._commitments[goal_id] = com
        return com

    # -----------------------------------------------------------------
    # Phase 2: Commitment Evaluation
    # -----------------------------------------------------------------

    def _evaluate_commitments(self) -> Tuple[List[str], List[RevisionEvent]]:
        """
        Evaluate all active commitments for stagnation and feasibility.

        Returns (stagnant_goal_ids, revisions_triggered).
        """
        stagnant: List[str] = []
        revisions: List[RevisionEvent] = []
        mk = self._mode_key()
        threshold_base = self._get_mode_param(self._cfg.stagnation_threshold, 5)

        for goal_id, com in list(self._commitments.items()):
            if goal_id not in self._goals:
                # Orphan commitment — remove
                del self._commitments[goal_id]
                continue

            goal_rec = self._goals[goal_id]

            # Skip terminal goals
            if goal_rec.status in (GoalStatus.COMPLETED, GoalStatus.ABANDONED):
                del self._commitments[goal_id]
                continue

            # Increment stagnation if no progress this cycle
            if com.last_progress_delta < 0.001:
                com.stagnation_count += 1
                goal_rec.stagnation_count += 1
            else:
                com.stagnation_count = 0
                goal_rec.stagnation_count = 0

            # Check stagnation
            is_stagnant = check_stagnation(
                com.stagnation_count,
                threshold_base,
                self._state.ne_level,
                self._state._5ht_level,
                self._cfg.mu_ne,
                self._cfg.mu_5ht,
            )

            if is_stagnant:
                stagnant.append(goal_id)
                goal_rec.status = GoalStatus.STAGNANT

                # Attempt strategy revision
                rev = self._try_revision(
                    goal_id, com.strategy_id, RevisionReason.STAGNATION,
                )
                if rev:
                    revisions.append(rev)

            # Check feasibility decay
            feas_floor = self._get_mode_param(
                self._cfg.revision_feasibility_floor, 0.20,
            )
            if check_feasibility_revision(
                goal_rec.feasibility, feas_floor,
                self._state.cor_level, self._cfg.mu_cor,
            ):
                rev = self._try_revision(
                    goal_id, com.strategy_id, RevisionReason.FEASIBILITY_DECAY,
                )
                if rev:
                    revisions.append(rev)

            # Check risk spike on current strategy
            if com.strategy_id in self._strategies:
                strat = self._strategies[com.strategy_id]
                risk_ceil = self._get_mode_param(
                    self._cfg.revision_risk_ceiling, 0.80,
                )
                if check_risk_revision(
                    strat.risk, risk_ceil,
                    self._state.cor_level, self._cfg.mu_cor,
                ):
                    rev = self._try_revision(
                        goal_id, com.strategy_id, RevisionReason.RISK_SPIKE,
                    )
                    if rev:
                        revisions.append(rev)

            # Reset progress delta for next cycle
            com.last_progress_delta = 0.0

        return stagnant, revisions

    def _try_revision(
        self,
        goal_id: str,
        old_strategy_id: str,
        reason: RevisionReason,
    ) -> Optional[RevisionEvent]:
        """
        Attempt to revise strategy for a goal.

        Picks the best alternative strategy (excluding current).
        Returns RevisionEvent if a new strategy is found, else None.
        """
        alternatives = [
            s for s in self._strategies.values()
            if s.goal_id == goal_id and s.strategy_id != old_strategy_id
        ]

        if not alternatives:
            return None

        goal_rec = self._goals[goal_id]

        # Re-score alternatives
        for s in alternatives:
            s.composite = compute_strategy_composite(
                expected_utility=s.expected_utility,
                risk=s.risk,
                resource_cost=s.resource_cost,
                feasibility=goal_rec.feasibility,
                w_utility=self._get_mode_param(self._cfg.w_utility, 0.40),
                w_risk=self._get_mode_param(self._cfg.w_risk, 0.20),
                w_feasibility=self._get_mode_param(self._cfg.w_feasibility, 0.25),
                w_cost=self._get_mode_param(self._cfg.w_cost, 0.15),
                exploration_bonus=self._state.da_level * self._get_mode_param(
                    self._cfg.exploration_bonus, 0.10,
                ),
            )

        best_alt = max(alternatives, key=lambda s: s.composite)

        # Only revise if alternative is actually better than current
        current = self._strategies.get(old_strategy_id)
        if current and best_alt.composite <= current.composite:
            return None

        # Apply revision
        best_alt.adopted = True
        if old_strategy_id in self._strategies:
            self._strategies[old_strategy_id].adopted = False

        # Update commitment
        if goal_id in self._commitments:
            self._commitments[goal_id].strategy_id = best_alt.strategy_id
            self._commitments[goal_id].stagnation_count = 0

        # Re-activate stagnant goal
        if goal_rec.status == GoalStatus.STAGNANT:
            goal_rec.status = GoalStatus.ACTIVE
            goal_rec.stagnation_count = 0

        self._state.total_revisions += 1

        return RevisionEvent(
            goal_id=goal_id,
            old_strategy_id=old_strategy_id,
            new_strategy_id=best_alt.strategy_id,
            reason=reason,
            tick=self._cycle_count,
            detail=f"Revised due to {reason.value}; "
                   f"old composite={current.composite:.3f}, "
                   f"new composite={best_alt.composite:.3f}"
                   if current else f"Revised due to {reason.value}",
        )

    # -----------------------------------------------------------------
    # GABA-driven goal pruning
    # -----------------------------------------------------------------

    def _prune_low_utility_goals(self) -> List[str]:
        """
        GABA suppression: abandon very low-utility goals.

        Only triggers when GABA is elevated.
        """
        pruned: List[str] = []
        if self._state.gaba_level < 0.2:
            return pruned

        suppress_threshold = 0.15 + 0.10 * (1.0 - self._state.gaba_level)

        for g in list(self._goals.values()):
            if g.status in (GoalStatus.COMPLETED, GoalStatus.ABANDONED):
                continue
            if g.utility < suppress_threshold and g.progress < 0.1:
                g.status = GoalStatus.ABANDONED
                self._state.total_goals_abandoned += 1
                self._commitments.pop(g.goal_id, None)
                pruned.append(g.goal_id)

        return pruned

    # -----------------------------------------------------------------
    # Utility decay for non-progressing goals
    # -----------------------------------------------------------------

    def _apply_utility_decay(self) -> None:
        """Decay utility of non-progressing goals over time."""
        for g in self._goals.values():
            if g.status in (GoalStatus.COMPLETED, GoalStatus.ABANDONED):
                continue
            ticks_since = self._cycle_count - g.last_progress_tick
            if ticks_since > 0 and g.progress_velocity < 0.01:
                decay = self._cfg.utility_decay_rate * ticks_since
                g.utility = _clamp(g.utility - decay)

                # Also decay feasibility slightly
                g.feasibility = _clamp(g.feasibility - decay * 0.5)

    # -----------------------------------------------------------------
    # Batch strategy re-scoring
    # -----------------------------------------------------------------

    def _rescore_all_strategies(self) -> List[StrategyScore]:
        """Re-score all strategies with current NT state."""
        results: List[StrategyScore] = []

        for s in self._strategies.values():
            if s.goal_id not in self._goals:
                continue

            goal_rec = self._goals[s.goal_id]
            exploration_bonus_val = (
                self._state.da_level
                * self._get_mode_param(self._cfg.exploration_bonus, 0.10)
                * (0.5 if not s.adopted else 0.0)
            )

            s.composite = compute_strategy_composite(
                expected_utility=s.expected_utility,
                risk=s.risk,
                resource_cost=s.resource_cost,
                feasibility=goal_rec.feasibility,
                w_utility=self._get_mode_param(self._cfg.w_utility, 0.40),
                w_risk=self._get_mode_param(self._cfg.w_risk, 0.20),
                w_feasibility=self._get_mode_param(self._cfg.w_feasibility, 0.25),
                w_cost=self._get_mode_param(self._cfg.w_cost, 0.15),
                exploration_bonus=exploration_bonus_val,
            )
            results.append(export_strategy(s))

        results.sort(key=lambda s: s.composite, reverse=True)
        return results

    # -----------------------------------------------------------------
    # Main process()
    # -----------------------------------------------------------------

    def process(self, input_data: StrategicDecisionInput) -> StrategicDecisionResult:
        """
        Execute one full strategic decision cycle.

        Parameters
        ----------
        input_data : StrategicDecisionInput
            Goals, strategies, progress updates, context.

        Returns
        -------
        StrategicDecisionResult
            Decision output with selected goal, strategy, revisions,
            neurochemical signals.
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        # ==============================================================
        # PHASE 1: CONTEXT INTAKE
        # ==============================================================

        # Register new goals
        for gd in input_data.new_goals:
            self.add_goal(
                goal_id=gd.get("goal_id", f"goal_{uuid.uuid4().hex[:8]}"),
                description=gd.get("description", ""),
                parent_id=gd.get("parent_id"),
                priority=gd.get("priority", 0.5),
                feasibility=gd.get("feasibility", 0.8),
                utility=gd.get("utility", 0.5),
                metadata=gd.get("metadata"),
            )

        # Register new strategies
        for sd in input_data.new_strategies:
            goal_id = sd.get("goal_id", "")
            if goal_id in self._goals:
                self.add_strategy(
                    strategy_id=sd.get("strategy_id", f"strat_{uuid.uuid4().hex[:8]}"),
                    goal_id=goal_id,
                    expected_utility=sd.get("expected_utility", 0.5),
                    risk=sd.get("risk", 0.3),
                    resource_cost=sd.get("resource_cost", 0.2),
                    metadata=sd.get("metadata"),
                )

        # Apply progress updates
        total_progress = 0.0
        for gid, delta in input_data.progress_updates.items():
            p = self.update_progress(gid, delta)
            total_progress += delta

        # Mark completed goals
        for gid in input_data.completed_goals:
            self.complete_goal(gid)

        # Mark abandoned goals
        for gid in input_data.abandoned_goals:
            self.abandon_goal(gid)

        # Force revisions
        forced_revisions: List[RevisionEvent] = []
        for gid in input_data.force_revision_goals:
            if gid in self._commitments:
                rev = self._try_revision(
                    gid, self._commitments[gid].strategy_id,
                    RevisionReason.EXTERNAL_OVERRIDE,
                )
                if rev:
                    forced_revisions.append(rev)

        # ==============================================================
        # PHASE 2: COMMITMENT EVALUATION
        # ==============================================================

        stagnant_goals, commitment_revisions = self._evaluate_commitments()
        all_revisions = forced_revisions + commitment_revisions

        # ==============================================================
        # PHASE 3: SUB-GOAL SELECTION
        # ==============================================================

        # Apply utility decay before selection
        self._apply_utility_decay()

        # GABA pruning
        pruned_goals = self._prune_low_utility_goals()

        # Select best (goal, strategy)
        selected_goal, selected_strategy = self.select_next_action()

        # ==============================================================
        # PHASE 4: STRATEGY SCORING
        # ==============================================================

        scored_strategies = self._rescore_all_strategies()

        # If we selected a goal and strategy, create/update commitment
        novel_strategy_adopted = False
        if selected_goal and selected_strategy:
            # Check if this is a novel strategy
            if selected_strategy.strategy_id in self._strategies:
                strat_rec = self._strategies[selected_strategy.strategy_id]
                if not strat_rec.adopted:
                    novel_strategy_adopted = True
                    strat_rec.adopted = True

            # Activate the goal
            if selected_goal.goal_id in self._goals:
                goal_rec = self._goals[selected_goal.goal_id]
                if goal_rec.status in (GoalStatus.PENDING, GoalStatus.STAGNANT):
                    goal_rec.status = GoalStatus.ACTIVE

            self._create_or_update_commitment(
                selected_goal.goal_id,
                selected_strategy.strategy_id,
            )

        # ==============================================================
        # PHASE 5: REVISION & EMIT
        # ==============================================================

        # Compute max risk exposure
        max_risk = 0.0
        if scored_strategies:
            adopted_strats = [
                s for s in self._strategies.values() if s.adopted
            ]
            if adopted_strats:
                max_risk = max(s.risk for s in adopted_strats)

        # Evaluation depth = number of goals considered
        eval_depth = len(self._get_actionable_goals())

        # Compute neurochem signals
        neurochem = compute_neurochem_signals(
            novel_strategy_adopted=novel_strategy_adopted,
            progress_made=total_progress,
            revisions_triggered=len(all_revisions),
            max_risk_exposure=max_risk,
            evaluation_depth=eval_depth,
            goals_pruned=len(pruned_goals),
            stagnant_count=len(stagnant_goals),
            cfg=self._cfg,
        )

        # Build active goals list
        active_goals = tuple(
            export_goal(g)
            for g in self._goals.values()
            if g.status in (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.STAGNANT)
        )

        # Build commitments list
        commitments = tuple(
            export_commitment(c)
            for c in self._commitments.values()
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return StrategicDecisionResult(
            active_goals=active_goals,
            selected_goal=selected_goal,
            selected_strategy=selected_strategy,
            scored_strategies=tuple(scored_strategies),
            commitments=commitments,
            revisions=tuple(all_revisions),
            stagnant_goals=tuple(stagnant_goals),
            pruned_goals=tuple(pruned_goals),
            neurochem_signals=neurochem,
            processing_time_ms=elapsed,
            engine_id=self.engine_id,
            cycle_count=self._cycle_count,
            mode=self._mode.value,
            metadata={
                "total_goals": len(self._goals),
                "total_strategies": len(self._strategies),
                "total_commitments": len(self._commitments),
                "total_revisions_lifetime": self._state.total_revisions,
                "total_goals_completed": self._state.total_goals_completed,
                "total_goals_abandoned": self._state.total_goals_abandoned,
                "stagnant_detected": len(stagnant_goals),
                "pruned_this_cycle": len(pruned_goals),
                "novel_strategy_adopted": novel_strategy_adopted,
                "evaluation_depth": eval_depth,
                "nt_levels": {
                    "da": self._state.da_level,
                    "5ht": self._state._5ht_level,
                    "ne": self._state.ne_level,
                    "ach": self._state.ach_level,
                    "cor": self._state.cor_level,
                    "gaba": self._state.gaba_level,
                    "cb1": self._state.cb1_level,
                },
            },
        )

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine introspection dict."""
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "total_goals": len(self._goals),
            "total_strategies": len(self._strategies),
            "total_commitments": len(self._commitments),
            "total_goals_created": self._state.total_goals_created,
            "total_goals_completed": self._state.total_goals_completed,
            "total_goals_abandoned": self._state.total_goals_abandoned,
            "total_strategies_scored": self._state.total_strategies_scored,
            "total_revisions": self._state.total_revisions,
            "goal_ids": list(self._goals.keys()),
            "active_goal_ids": [
                g.goal_id for g in self._goals.values()
                if g.status == GoalStatus.ACTIVE
            ],
            "stagnant_goal_ids": [
                g.goal_id for g in self._goals.values()
                if g.status == GoalStatus.STAGNANT
            ],
            "commitment_goal_ids": list(self._commitments.keys()),
            "nt_levels": {
                "da": self._state.da_level,
                "5ht": self._state._5ht_level,
                "ne": self._state.ne_level,
                "ach": self._state.ach_level,
                "cor": self._state.cor_level,
                "gaba": self._state.gaba_level,
                "cb1": self._state.cb1_level,
            },
        }
