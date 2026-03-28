"""
Engine 13 -- Simulation Brain Engine  (``simulation_brain_engine``)
====================================================================
The system's imagination: runs mental simulations of possible outcomes,
scenarios, and trajectories, producing probabilistic forecasts for
decision-making, behavioral prediction, and response planning.

Four-phase pipeline:
  * **Phase 1 — Scenario Seeding**: extract seeds from intent, semantic
    alternatives, memory, contradiction-driven what-ifs.
  * **Phase 2 — Branching Expansion**: entropy-modulated scenario tree
    with temperature-controlled softmax branching + pruning.
  * **Phase 3 — Evaluation**: score leaves on consistency, reward
    alignment, plausibility, utility.
  * **Phase 4 — Synthesis**: collapse tree → expected outcome distribution,
    risk profile, recommended action, uncertainty export.

Key formulas from stochastic notes:
  κ_uncertainty(t) = ω₁·e_disint + ω₂·e_ambig + ω₃·Var[ΔC_i]
  δ_uncertainty(t) = leaky integral of κ with τ_forecast
  T(t) = T₀·(1 + α·δ_uncertainty)     (forecast temperature)
  D_rec(t) = ⌊δ₀ + δ₁·Φ_θγ − δ₂·Vol_Sym⌋  (recursion depth)

Self-tuning recursion-reward-rhythm loop:
  Φ_θγ↑ ⇒ Vol↓ ⇒ D_rec↑ ⇒ R_Sym↑ ⇒ CB1/5HT2A↑ ⇒ Φ_θγ↑

Neurochemical coupling:
  DA  — optimism bias, novelty seeking
  NE  — threat awareness, lower pruning
  ACh — sharper branch discrimination
  5-HT— more coherent scenarios
  CB1 — creative/divergent branching
  COR — catastrophizing bias
  GABA— reduced simulation depth
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class SBConfig:
    """All tunable parameters for the Simulation Brain Engine."""

    # --- Temperature & uncertainty ---
    T_0: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.80, "dev": 0.60, "learning": 0.70,
        "reflective": 0.90, "rem_normal": 0.80, "rem_dream": 2.00,
    })
    alpha_temp: Dict[str, float] = field(default_factory=lambda: {
        "normal": 1.50, "dev": 1.00, "learning": 1.20,
        "reflective": 1.80, "rem_normal": 1.50, "rem_dream": 2.50,
    })
    tau_forecast: Dict[str, float] = field(default_factory=lambda: {
        "normal": 5.0, "dev": 8.0, "learning": 6.0,
        "reflective": 8.0, "rem_normal": 5.0, "rem_dream": 2.0,
    })
    omega_1: float = 0.35  # Disintegration weight
    omega_2: float = 0.35  # Ambiguity weight
    omega_3: float = 0.30  # NT variance weight

    # --- Recursion depth ---
    delta_0: Dict[str, int] = field(default_factory=lambda: {
        "normal": 3, "dev": 4, "learning": 3,
        "reflective": 3, "rem_normal": 3, "rem_dream": 2,
    })
    delta_1: Dict[str, int] = field(default_factory=lambda: {
        "normal": 4, "dev": 5, "learning": 4,
        "reflective": 3, "rem_normal": 4, "rem_dream": 6,
    })
    delta_2: Dict[str, int] = field(default_factory=lambda: {
        "normal": 3, "dev": 2, "learning": 3,
        "reflective": 4, "rem_normal": 3, "rem_dream": 1,
    })
    D_min: Dict[str, int] = field(default_factory=lambda: {
        "normal": 2, "dev": 3, "learning": 2,
        "reflective": 2, "rem_normal": 2, "rem_dream": 1,
    })
    D_max: Dict[str, int] = field(default_factory=lambda: {
        "normal": 10, "dev": 12, "learning": 10,
        "reflective": 8, "rem_normal": 10, "rem_dream": 15,
    })

    # --- Branching ---
    B_base: Dict[str, int] = field(default_factory=lambda: {
        "normal": 3, "dev": 2, "learning": 3,
        "reflective": 3, "rem_normal": 3, "rem_dream": 5,
    })
    B_max: Dict[str, int] = field(default_factory=lambda: {
        "normal": 8, "dev": 6, "learning": 7,
        "reflective": 6, "rem_normal": 8, "rem_dream": 12,
    })
    B_temp: float = 3.0    # Temperature bonus branches
    B_depth: float = 2.0   # Depth penalty branches
    B_min: int = 2

    # --- Seeding ---
    N_seeds: Dict[str, int] = field(default_factory=lambda: {
        "normal": 3, "dev": 2, "learning": 3,
        "reflective": 4, "rem_normal": 3, "rem_dream": 5,
    })
    w_seed_prob: float = 0.40
    w_seed_novelty: float = 0.35
    w_seed_risk: float = 0.25

    # --- Pruning ---
    theta_prune: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.05, "dev": 0.08, "learning": 0.06,
        "reflective": 0.04, "rem_normal": 0.05, "rem_dream": 0.02,
    })
    max_nodes: Dict[str, int] = field(default_factory=lambda: {
        "normal": 200, "dev": 150, "learning": 180,
        "reflective": 150, "rem_normal": 200, "rem_dream": 500,
    })

    # --- Branch quality scoring ---
    w_consistency: float = 0.30
    w_coherence: float = 0.25
    w_reward: float = 0.20
    w_novelty: float = 0.15
    w_memory: float = 0.10

    # --- Leaf evaluation ---
    alpha_plaus: float = 0.35
    alpha_util: float = 0.40
    alpha_consist: float = 0.25

    # --- Forecast thresholds ---
    theta_positive: float = 0.65
    theta_negative: float = 0.35
    theta_disaster: float = 0.15

    # --- Neurochemical coupling ---
    mu_cb1: float = 0.30
    mu_5ht: float = 0.20
    mu_ne: float = 0.25
    mu_da: float = 0.20
    mu_ach: float = 0.15

    # Write-port coefficients
    beta_da_positive: float = 0.10
    beta_da_negative: float = 0.08
    beta_ne_threat: float = 0.12
    beta_cor_stress: float = 0.10
    beta_ach_depth: float = 0.08
    beta_cb1_creative: float = 0.08
    beta_gaba_release: float = 0.04
    beta_5ht_coherence: float = 0.06

    psi_theta_gamma: float = 0.06
    psi_beta: float = 0.04
    psi_gamma: float = 0.05

    # --- Emotion modulation ---
    kappa_curious: float = 0.25
    kappa_creative: float = 0.30
    kappa_confident: float = 0.15
    kappa_anxious: float = 0.10
    zeta_curious: float = 1.5
    zeta_anxious: float = 1.0
    zeta_confident: float = 1.0

    # --- Stochastic ---
    sigma_quality: float = 0.02
    sigma_prune: float = 0.005


# =====================================================================
# Mutable state
# =====================================================================


@dataclass
class SBState:
    """Runtime state."""
    # NT read-port levels
    da_level:  float = 0.0
    ne_level:  float = 0.0
    ach_level: float = 0.0
    _5ht_level: float = 0.0
    cb1_level: float = 0.0
    cor_level: float = 0.0
    gaba_level: float = 0.0

    # Uncertainty integrator (leaky)
    delta_uncertainty: float = 0.0

    # History
    total_simulations: int = 0
    total_nodes_expanded: int = 0
    total_branches_pruned: int = 0


# =====================================================================
# Frozen I/O
# =====================================================================


@dataclass(frozen=True)
class ScenarioSeed:
    """A seed for scenario tree expansion."""
    seed_id: str = ""
    source: str = ""           # "intention" | "interpretation" | "memory" | "contradiction"
    premise: str = ""
    probability: float = 0.5
    novelty: float = 0.0
    risk_relevance: float = 0.0


@dataclass(frozen=True)
class ScenarioNode:
    """One node in the scenario tree."""
    node_id: str = ""
    parent_id: Optional[str] = None
    depth: int = 0
    premise: str = ""
    action: str = ""
    outcome: str = ""
    probability: float = 0.0         # Conditional on parent
    cumulative_probability: float = 0.0
    consistency_score: float = 1.0
    reward_alignment: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    children_ids: Tuple[str, ...] = ()
    is_leaf: bool = True


@dataclass(frozen=True)
class ScenarioOutcome:
    """A terminal scenario with evaluation scores."""
    outcome_id: str = ""
    description: str = ""
    probability: float = 0.0
    utility: float = 0.0
    reward_alignment: Dict[str, float] = field(default_factory=dict)
    chain: Tuple[str, ...] = ()      # Node IDs from root to leaf


@dataclass(frozen=True)
class RiskProfile:
    """Risk analysis of the simulation's outcome space."""
    best_case_utility: float = 0.0
    worst_case_utility: float = 0.0
    modal_probability: float = 0.0
    expected_utility: float = 0.0
    outcome_variance: float = 0.0
    tail_risk: float = 0.0            # P(utility < theta_disaster)


@dataclass(frozen=True)
class SimulationNeurochem:
    """Neurochemical deltas emitted by the Simulation Brain Engine."""
    delta_da: float = 0.0
    delta_ne: float = 0.0
    delta_cor: float = 0.0
    delta_ach: float = 0.0
    delta_cb1: float = 0.0
    delta_5ht: float = 0.0
    delta_gaba: float = 0.0
    theta_gamma_boost: float = 0.0
    beta_boost: float = 0.0
    gamma_burst: float = 0.0


@dataclass(frozen=True)
class SimulationBrainInput:
    """Input to the Simulation Brain Engine."""
    # Intent context
    intent_descriptions: Tuple[str, ...] = ()
    intent_confidences: Tuple[float, ...] = ()
    # Semantic alternatives
    alternative_interpretations: Tuple[str, ...] = ()
    alternative_plausibilities: Tuple[float, ...] = ()
    # Memory-primed scenarios
    memory_scenarios: Tuple[str, ...] = ()
    memory_relevance_scores: Tuple[float, ...] = ()
    # Contradiction-driven
    contradiction_statements: Tuple[Tuple[str, str], ...] = ()  # (stmt_a, stmt_b) pairs
    # Uncertainty/oscillation
    system_entropy: float = 0.5
    theta_gamma_coupling: float = 0.5
    symbolic_reward_volatility: float = 0.3
    e_disintegration: float = 0.0
    e_ambiguity: float = 0.0
    nt_variance: float = 0.0
    # Emotion
    emotion_intensities: Optional[Dict[str, float]] = None
    # Reward
    reward_scores: Optional[Dict[str, float]] = None
    # Detection flags (counts for consistency checking)
    contradiction_count: int = 0
    fallacy_count: int = 0
    # Context
    active_mode: str = "normal"
    cycle_count: int = 0


@dataclass(frozen=True)
class SimulationBrainResult:
    """Output from the Simulation Brain Engine."""
    # Tree summary
    total_nodes: int = 0
    max_depth_reached: int = 0
    branches_pruned: int = 0
    # Outcomes
    outcomes: Tuple[ScenarioOutcome, ...] = ()
    modal_outcome: Optional[ScenarioOutcome] = None
    expected_utility: float = 0.0
    # Risk
    risk_profile: RiskProfile = field(default_factory=RiskProfile)
    # Forecast metrics
    forecast_temperature: float = 0.0
    recursion_depth: int = 0
    outcome_entropy: float = 0.0
    forecast_reliability: float = 0.0
    # Recommendation
    recommended_action: str = ""
    action_confidence: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    # Uncertainty export (for Engine 26)
    simulation_uncertainty: Dict[str, float] = field(default_factory=dict)
    # Neurochem
    neurochemical_signals: SimulationNeurochem = field(default_factory=SimulationNeurochem)
    # Metadata
    processing_time_ms: float = 0.0
    engine_id: str = "simulation_brain_engine"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Utility
# =====================================================================


def _softmax(scores: List[float], temperature: float) -> List[float]:
    """Temperature-scaled softmax."""
    if not scores:
        return []
    t = max(temperature, 1e-6)
    scaled = [s / t for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    if total < 1e-12:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


# =====================================================================
# Phase 1: Scenario Seeding  (pure)
# =====================================================================


def generate_seeds(
    inp: SimulationBrainInput,
) -> List[ScenarioSeed]:
    """Generate scenario seeds from all available sources."""
    seeds: List[ScenarioSeed] = []
    idx = 0

    # Source 1: Intent seeds
    for i, desc in enumerate(inp.intent_descriptions):
        conf = inp.intent_confidences[i] if i < len(inp.intent_confidences) else 0.5
        seeds.append(ScenarioSeed(
            seed_id=f"intent_{idx}", source="intention",
            premise=desc, probability=conf, novelty=0.3, risk_relevance=0.2,
        ))
        idx += 1

    # Source 2: Interpretation seeds
    for i, alt in enumerate(inp.alternative_interpretations):
        plaus = inp.alternative_plausibilities[i] if i < len(inp.alternative_plausibilities) else 0.3
        seeds.append(ScenarioSeed(
            seed_id=f"interp_{idx}", source="interpretation",
            premise=alt, probability=plaus, novelty=0.6, risk_relevance=0.3,
        ))
        idx += 1

    # Source 3: Memory-primed seeds
    for i, mem in enumerate(inp.memory_scenarios):
        rel = inp.memory_relevance_scores[i] if i < len(inp.memory_relevance_scores) else 0.3
        seeds.append(ScenarioSeed(
            seed_id=f"memory_{idx}", source="memory",
            premise=mem, probability=rel, novelty=0.2, risk_relevance=0.4,
        ))
        idx += 1

    # Source 4: Contradiction-driven seeds
    for pair in inp.contradiction_statements:
        seeds.append(ScenarioSeed(
            seed_id=f"contra_a_{idx}", source="contradiction",
            premise=pair[0], probability=0.5, novelty=0.5, risk_relevance=0.7,
        ))
        idx += 1
        seeds.append(ScenarioSeed(
            seed_id=f"contra_b_{idx}", source="contradiction",
            premise=pair[1], probability=0.5, novelty=0.5, risk_relevance=0.7,
        ))
        idx += 1

    return seeds


def rank_seeds(
    seeds: List[ScenarioSeed],
    w_prob: float = 0.40,
    w_novelty: float = 0.35,
    w_risk: float = 0.25,
) -> List[ScenarioSeed]:
    """Rank seeds by composite score."""
    scored = [
        (s, w_prob * s.probability + w_novelty * s.novelty + w_risk * s.risk_relevance)
        for s in seeds
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored]


# =====================================================================
# Phase 1: Temperature & Depth  (pure)
# =====================================================================


def compute_uncertainty_drive(
    e_disintegration: float,
    e_ambiguity: float,
    nt_variance: float,
    omega_1: float = 0.35,
    omega_2: float = 0.35,
    omega_3: float = 0.30,
) -> float:
    """κ_uncertainty(t)."""
    return omega_1 * e_disintegration + omega_2 * e_ambiguity + omega_3 * nt_variance


def integrate_uncertainty(
    kappa: float,
    prev_delta: float,
    tau: float,
) -> float:
    """δ_uncertainty(t+1) — leaky integration."""
    if tau < 1e-6:
        return kappa
    decay = math.exp(-1.0 / tau)
    return prev_delta * decay + kappa * (1.0 - decay)


def compute_forecast_temperature(
    T_0: float,
    alpha_temp: float,
    delta_uncertainty: float,
) -> float:
    """T(t) = T₀ × (1 + α × δ_uncertainty)."""
    return max(0.01, T_0 * (1.0 + alpha_temp * delta_uncertainty))


def compute_recursion_depth(
    delta_0: int,
    delta_1: int,
    delta_2: int,
    phi_tg: float,
    vol_symbolic: float,
    D_min: int = 2,
    D_max: int = 10,
) -> int:
    """D_rec(t) = ⌊δ₀ + δ₁·Φ_θγ − δ₂·Vol_Sym⌋, clamped."""
    raw = delta_0 + delta_1 * phi_tg - delta_2 * vol_symbolic
    return max(D_min, min(D_max, int(raw)))


def compute_reward_volatility(phi_tg: float) -> float:
    """Vol_d(t) ∝ 1/(1 + Φ_θγ)."""
    return 1.0 / (1.0 + phi_tg)


# =====================================================================
# Phase 2: Branching Expansion  (pure helpers)
# =====================================================================


def compute_branch_count(
    temperature: float,
    depth: int,
    D_rec: int,
    B_base: int = 3,
    B_temp: float = 3.0,
    B_depth: float = 2.0,
    B_min: int = 2,
    B_max: int = 8,
) -> int:
    """Compute number of branches at a given depth."""
    depth_fraction = depth / max(D_rec, 1)
    raw = B_base + B_temp * temperature - B_depth * depth_fraction
    return max(B_min, min(B_max, int(raw)))


def score_branch_quality(
    rng: np.random.Generator,
    seed_premise: str,
    parent_consistency: float,
    reward_scores: Optional[Dict[str, float]],
    memory_relevance: float,
    branch_index: int,
    total_branches: int,
    cfg: SBConfig,
    sigma: float = 0.02,
) -> float:
    """
    Score a candidate branch's quality.

    In a full system this would use NLP to score coherence, consistency etc.
    Here we use a heuristic based on available signals + stochastic variation.
    """
    # Base quality from parent consistency
    consistency = parent_consistency * (0.8 + 0.2 * rng.random())

    # Coherence: first branches are more coherent (closer to parent logic)
    coherence = 1.0 - (branch_index / max(total_branches, 1)) * 0.5

    # Reward alignment
    reward = 0.5
    if reward_scores:
        reward = sum(reward_scores.values()) / max(len(reward_scores), 1)

    # Novelty: later branches are more novel
    novelty = (branch_index / max(total_branches - 1, 1)) * 0.8 + 0.2

    quality = (
        cfg.w_consistency * consistency
        + cfg.w_coherence * coherence
        + cfg.w_reward * reward
        + cfg.w_novelty * novelty
        + cfg.w_memory * memory_relevance
    )

    # Add noise
    noise = float(rng.normal(0.0, sigma))
    return max(0.01, quality + noise)


# =====================================================================
# Phase 3: Evaluation  (pure)
# =====================================================================


def evaluate_leaf(
    node: ScenarioNode,
    contradiction_count: int,
    fallacy_count: int,
    reward_scores: Optional[Dict[str, float]],
    alpha_plaus: float = 0.35,
    alpha_util: float = 0.40,
    alpha_consist: float = 0.25,
) -> float:
    """Score a leaf node."""
    # Consistency
    max_issues = max(contradiction_count + fallacy_count, 1)
    consistency = node.consistency_score

    # Plausibility
    plausibility = node.cumulative_probability * consistency

    # Utility from reward alignment
    if node.reward_alignment:
        utility = sum(node.reward_alignment.values()) / max(len(node.reward_alignment), 1)
    elif reward_scores:
        utility = sum(reward_scores.values()) / max(len(reward_scores), 1)
    else:
        utility = 0.5

    return (
        alpha_plaus * _clamp(plausibility)
        + alpha_util * _clamp(utility)
        + alpha_consist * _clamp(consistency)
    )


# =====================================================================
# Phase 4: Synthesis  (pure)
# =====================================================================


def compute_risk_profile(
    outcomes: List[ScenarioOutcome],
    theta_disaster: float = 0.15,
) -> RiskProfile:
    """Compute risk metrics from outcome distribution."""
    if not outcomes:
        return RiskProfile()

    utilities = [o.utility for o in outcomes]
    probabilities = [o.probability for o in outcomes]

    best = max(utilities)
    worst = min(utilities)
    modal_idx = max(range(len(outcomes)), key=lambda i: probabilities[i])
    modal_prob = probabilities[modal_idx]

    expected = sum(p * u for p, u in zip(probabilities, utilities))
    variance = sum(p * (u - expected) ** 2 for p, u in zip(probabilities, utilities))

    # Tail risk
    tail = sum(p for p, u in zip(probabilities, utilities) if u < theta_disaster)

    return RiskProfile(
        best_case_utility=best,
        worst_case_utility=worst,
        modal_probability=modal_prob,
        expected_utility=expected,
        outcome_variance=variance,
        tail_risk=tail,
    )


def compute_outcome_entropy(outcomes: List[ScenarioOutcome]) -> float:
    """Shannon entropy of outcome probability distribution."""
    if not outcomes:
        return 0.0
    probs = [o.probability for o in outcomes]
    total = sum(probs)
    if total < 1e-12:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def compute_recommendation(
    outcomes: List[ScenarioOutcome],
) -> Tuple[str, float, Tuple[float, float]]:
    """
    Select recommended action from outcomes.
    Returns (action, confidence, 95% CI).
    """
    if not outcomes:
        return "defer", 0.0, (0.0, 1.0)

    # Best outcome by expected utility
    best = max(outcomes, key=lambda o: o.probability * o.utility)
    expected = sum(o.probability * o.utility for o in outcomes)
    variance = sum(o.probability * (o.utility - expected) ** 2 for o in outcomes)
    std = math.sqrt(max(variance, 0))

    action = best.description if best.description else "proceed"
    confidence = _clamp(best.probability * best.utility)
    ci = (_clamp(expected - 1.96 * std), _clamp(expected + 1.96 * std))

    return action, confidence, ci


def export_uncertainty(
    outcomes: List[ScenarioOutcome],
    D_rec: int,
    D_max: int,
) -> Dict[str, float]:
    """Export uncertainty metrics for Engine 26."""
    if not outcomes:
        return {"branch_uncertainty": 1.0, "outcome_entropy": 0.0, "forecast_horizon_reliability": 0.0}

    max_prob = max(o.probability for o in outcomes)
    entropy = compute_outcome_entropy(outcomes)
    reliability = D_rec / max(D_max, 1)

    return {
        "branch_uncertainty": 1.0 - max_prob,
        "outcome_entropy": entropy,
        "forecast_horizon_reliability": reliability,
    }


# =====================================================================
# Neurochemical computation  (pure)
# =====================================================================


def compute_simulation_neurochem(
    expected_utility: float,
    tail_risk: float,
    worst_severity: float,
    D_rec: int,
    D_base: int,
    D_max: int,
    temperature: float,
    T_0: float,
    outcome_convergence: float,
    cfg: SBConfig,
) -> SimulationNeurochem:
    """Compute NT deltas from simulation results."""
    delta_da = 0.0
    delta_ne = 0.0
    delta_cor = 0.0
    delta_ach = 0.0
    delta_cb1 = 0.0
    delta_5ht = 0.0
    delta_gaba = 0.0
    tg = 0.0
    beta = 0.0
    gamma = 0.0

    # Positive forecast
    if expected_utility > cfg.theta_positive:
        delta_da = cfg.beta_da_positive * (expected_utility - cfg.theta_positive)
        delta_5ht = cfg.beta_5ht_coherence * outcome_convergence

    # Negative forecast
    if expected_utility < cfg.theta_negative:
        delta_da = -cfg.beta_da_negative * (cfg.theta_negative - expected_utility)
        delta_ne = cfg.beta_ne_threat * tail_risk
        delta_cor = cfg.beta_cor_stress * tail_risk * worst_severity

    # Deep simulation
    if D_rec > D_base:
        delta_ach = cfg.beta_ach_depth * (D_rec - D_base) / max(D_max, 1)
        tg = cfg.psi_theta_gamma * D_rec / max(D_max, 1)

    # Creative branching
    if temperature > T_0 and T_0 > 0:
        delta_cb1 = cfg.beta_cb1_creative * (temperature - T_0) / T_0
        gamma = cfg.psi_gamma * min(temperature / max(T_0 * 3, 0.01), 1.0)

    # Completion signal
    delta_gaba = cfg.beta_gaba_release * 0.1

    return SimulationNeurochem(
        delta_da=delta_da,
        delta_ne=delta_ne,
        delta_cor=delta_cor,
        delta_ach=delta_ach,
        delta_cb1=delta_cb1,
        delta_5ht=delta_5ht,
        delta_gaba=delta_gaba,
        theta_gamma_boost=tg,
        beta_boost=beta,
        gamma_burst=_clamp(gamma),
    )


# =====================================================================
# Engine class
# =====================================================================


class SimulationBrainEngine:
    """
    Engine 13 -- Simulation Brain Engine.

    Four-phase probabilistic scenario simulation:
      Phase 1: Scenario Seeding (intent, interpretation, memory, contradiction)
      Phase 2: Branching Expansion (temperature-controlled softmax tree)
      Phase 3: Evaluation (consistency, reward alignment, plausibility)
      Phase 4: Synthesis (outcome distribution, risk profile, recommendation)

    API
    ---
    configure(mode)                -- set operational mode
    update_neurochem_state(state)  -- inject external NT levels
    process(input)                 -- full scenario simulation
    get_status()                   -- introspection
    """

    engine_id = "simulation_brain_engine"
    cluster   = "reasoning"

    def __init__(
        self,
        config: Optional[SBConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or SBConfig()
        self._rng = rng or np.random.default_rng(42)
        self._mode = OperationalMode.NORMAL
        self._state = SBState()
        self._cycle_count = 0

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        if "da" in state_dict:
            self._state.da_level = _clamp(state_dict["da"])
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "ach" in state_dict:
            self._state.ach_level = _clamp(state_dict["ach"])
        if "5ht" in state_dict:
            self._state._5ht_level = _clamp(state_dict["5ht"])
        if "cb1" in state_dict:
            self._state.cb1_level = _clamp(state_dict["cb1"])
        if "cor" in state_dict:
            self._state.cor_level = _clamp(state_dict["cor"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])

    def _mode_key(self) -> str:
        return self._mode.value

    def _get_mode_param(self, param_dict: Dict, default=0.5):
        return param_dict.get(self._mode_key(), default)

    # ----- Internal tree expansion ----------------------------------------

    def _expand_tree(
        self,
        seeds: List[ScenarioSeed],
        temperature: float,
        D_rec: int,
        max_nodes_limit: int,
        inp: SimulationBrainInput,
    ) -> Tuple[Dict[str, ScenarioNode], int]:
        """
        Expand scenario tree from seeds.
        Returns (nodes_dict, pruned_count).
        """
        cfg = self._cfg
        nodes: Dict[str, ScenarioNode] = {}
        node_counter = 0
        pruned = 0

        theta_pr = self._get_mode_param(cfg.theta_prune, 0.05)
        # NT modulation of pruning
        theta_pr_eff = theta_pr * (1.0 - cfg.mu_ne * self._state.ne_level)
        theta_pr_eff = max(0.001, theta_pr_eff)

        B_base_val = self._get_mode_param(cfg.B_base, 3)
        B_max_val = self._get_mode_param(cfg.B_max, 8)

        # NT modulation of weights
        w_novelty_eff = cfg.w_novelty * (1.0 + cfg.mu_da * self._state.da_level)
        w_consist_eff = cfg.w_consistency * (1.0 + cfg.mu_ach * self._state.ach_level)

        # T modulation
        T_eff = temperature * (1.0 + cfg.mu_cb1 * self._state.cb1_level
                               - cfg.mu_5ht * self._state._5ht_level)
        T_eff = max(0.01, T_eff)

        memory_rel = 0.0
        if inp.memory_relevance_scores:
            memory_rel = sum(inp.memory_relevance_scores) / len(inp.memory_relevance_scores)

        # Create root nodes from seeds
        roots: List[str] = []
        for seed in seeds:
            nid = f"node_{node_counter}"
            node_counter += 1
            nodes[nid] = ScenarioNode(
                node_id=nid,
                parent_id=None,
                depth=0,
                premise=seed.premise,
                action=f"seed:{seed.source}",
                outcome=seed.premise,
                probability=seed.probability,
                cumulative_probability=seed.probability,
                consistency_score=1.0,
                reward_alignment=dict(inp.reward_scores) if inp.reward_scores else {},
                quality_score=seed.probability,
                is_leaf=True,
            )
            roots.append(nid)

        # BFS expansion
        frontier = list(roots)
        while frontier and len(nodes) < max_nodes_limit:
            nid = frontier.pop(0)
            node = nodes[nid]

            if node.depth >= D_rec:
                continue

            n_branches = compute_branch_count(
                T_eff, node.depth, D_rec, B_base_val,
                cfg.B_temp, cfg.B_depth, cfg.B_min, B_max_val,
            )

            # Score branches
            qualities = []
            for bi in range(n_branches):
                q = score_branch_quality(
                    self._rng, node.premise, node.consistency_score,
                    inp.reward_scores, memory_rel, bi, n_branches, cfg,
                    cfg.sigma_quality,
                )
                qualities.append(q)

            # Softmax probabilities
            probs = _softmax(qualities, T_eff)

            children: List[str] = []
            for bi, prob in enumerate(probs):
                if len(nodes) >= max_nodes_limit:
                    break

                cum_prob = node.cumulative_probability * prob

                # Pruning
                prune_noise = float(self._rng.normal(0.0, cfg.sigma_prune))
                if cum_prob < max(0.001, theta_pr_eff + prune_noise):
                    pruned += 1
                    continue

                child_id = f"node_{node_counter}"
                node_counter += 1

                # Consistency decays slightly with depth
                child_consist = node.consistency_score * (0.95 + 0.05 * self._rng.random())
                # Adjust for detected issues
                if inp.contradiction_count > 0:
                    child_consist *= 0.9
                if inp.fallacy_count > 0:
                    child_consist *= 0.95

                child_node = ScenarioNode(
                    node_id=child_id,
                    parent_id=nid,
                    depth=node.depth + 1,
                    premise=node.outcome,
                    action=f"branch_{bi}",
                    outcome=f"scenario_d{node.depth + 1}_b{bi}",
                    probability=prob,
                    cumulative_probability=cum_prob,
                    consistency_score=_clamp(child_consist),
                    reward_alignment=dict(inp.reward_scores) if inp.reward_scores else {},
                    quality_score=qualities[bi],
                    is_leaf=True,
                )
                nodes[child_id] = child_node
                children.append(child_id)
                frontier.append(child_id)

            # Update parent to non-leaf
            if children:
                nodes[nid] = ScenarioNode(
                    node_id=node.node_id,
                    parent_id=node.parent_id,
                    depth=node.depth,
                    premise=node.premise,
                    action=node.action,
                    outcome=node.outcome,
                    probability=node.probability,
                    cumulative_probability=node.cumulative_probability,
                    consistency_score=node.consistency_score,
                    reward_alignment=node.reward_alignment,
                    quality_score=node.quality_score,
                    children_ids=tuple(children),
                    is_leaf=False,
                )

        return nodes, pruned

    def _collect_leaves(self, nodes: Dict[str, ScenarioNode]) -> List[ScenarioNode]:
        """Collect all leaf nodes from the tree."""
        return [n for n in nodes.values() if n.is_leaf]

    def _trace_chain(self, node: ScenarioNode, nodes: Dict[str, ScenarioNode]) -> Tuple[str, ...]:
        """Trace the chain from root to this node."""
        chain = [node.node_id]
        current = node
        while current.parent_id and current.parent_id in nodes:
            chain.append(current.parent_id)
            current = nodes[current.parent_id]
        chain.reverse()
        return tuple(chain)

    # ----- Main process ---------------------------------------------------

    def process(self, inp: SimulationBrainInput) -> SimulationBrainResult:
        t0 = time.perf_counter()
        cfg = self._cfg
        mk = self._mode_key()

        # ==============================================================
        # PHASE 1: SCENARIO SEEDING + TEMPERATURE/DEPTH COMPUTATION
        # ==============================================================

        # Compute uncertainty drive and temperature
        kappa = compute_uncertainty_drive(
            inp.e_disintegration, inp.e_ambiguity, inp.nt_variance,
            cfg.omega_1, cfg.omega_2, cfg.omega_3,
        )
        tau = self._get_mode_param(cfg.tau_forecast, 5.0)
        self._state.delta_uncertainty = integrate_uncertainty(
            kappa, self._state.delta_uncertainty, tau,
        )

        T_0 = self._get_mode_param(cfg.T_0, 0.80)
        alpha_t = self._get_mode_param(cfg.alpha_temp, 1.50)
        temperature = compute_forecast_temperature(T_0, alpha_t, self._state.delta_uncertainty)

        # Emotion modulation of temperature
        emo = inp.emotion_intensities or {}
        T_emotion = temperature * (
            1.0
            + cfg.kappa_curious * emo.get("curious", 0.0)
            + cfg.kappa_creative * emo.get("creative", 0.0)
            - cfg.kappa_confident * emo.get("confident", 0.0)
            - cfg.kappa_anxious * emo.get("anxiety", 0.0) * 0.3
        )
        T_emotion = max(0.01, T_emotion)

        # Compute recursion depth
        d0 = self._get_mode_param(cfg.delta_0, 3)
        d1 = self._get_mode_param(cfg.delta_1, 4)
        d2 = self._get_mode_param(cfg.delta_2, 3)
        d_min = self._get_mode_param(cfg.D_min, 2)
        d_max = self._get_mode_param(cfg.D_max, 10)
        vol_sym = compute_reward_volatility(inp.theta_gamma_coupling)

        D_rec = compute_recursion_depth(
            d0, d1, d2, inp.theta_gamma_coupling, vol_sym, d_min, d_max,
        )

        # Emotion modulation of depth
        D_rec_emo = D_rec + int(
            cfg.zeta_curious * emo.get("curious", 0.0)
            - cfg.zeta_anxious * emo.get("anxiety", 0.0)
            - cfg.zeta_confident * emo.get("confident", 0.0)
        )
        D_rec_emo = max(d_min, min(d_max, D_rec_emo))

        # Generate and rank seeds
        all_seeds = generate_seeds(inp)
        ranked_seeds = rank_seeds(all_seeds, cfg.w_seed_prob, cfg.w_seed_novelty, cfg.w_seed_risk)
        n_seeds = self._get_mode_param(cfg.N_seeds, 3)
        selected_seeds = ranked_seeds[:n_seeds]

        # ==============================================================
        # PHASE 2: BRANCHING EXPANSION
        # ==============================================================

        max_nodes_limit = self._get_mode_param(cfg.max_nodes, 200)
        nodes, pruned = self._expand_tree(
            selected_seeds, T_emotion, D_rec_emo, max_nodes_limit, inp,
        )

        # ==============================================================
        # PHASE 3: EVALUATION
        # ==============================================================

        leaves = self._collect_leaves(nodes)
        max_depth = max((n.depth for n in nodes.values()), default=0)

        # Score leaves
        outcomes: List[ScenarioOutcome] = []
        for leaf in leaves:
            score = evaluate_leaf(
                leaf, inp.contradiction_count, inp.fallacy_count,
                inp.reward_scores, cfg.alpha_plaus, cfg.alpha_util, cfg.alpha_consist,
            )
            chain = self._trace_chain(leaf, nodes)
            outcomes.append(ScenarioOutcome(
                outcome_id=leaf.node_id,
                description=leaf.outcome,
                probability=leaf.cumulative_probability,
                utility=score,
                reward_alignment=leaf.reward_alignment,
                chain=chain,
            ))

        # Normalize probabilities
        total_prob = sum(o.probability for o in outcomes)
        if total_prob > 0:
            outcomes = [
                ScenarioOutcome(
                    outcome_id=o.outcome_id, description=o.description,
                    probability=o.probability / total_prob,
                    utility=o.utility, reward_alignment=o.reward_alignment,
                    chain=o.chain,
                )
                for o in outcomes
            ]

        # Sort by probability
        outcomes.sort(key=lambda o: o.probability, reverse=True)

        # ==============================================================
        # PHASE 4: SYNTHESIS
        # ==============================================================

        risk = compute_risk_profile(outcomes, cfg.theta_disaster)
        entropy = compute_outcome_entropy(outcomes)
        action, confidence, ci = compute_recommendation(outcomes)
        modal = outcomes[0] if outcomes else None
        expected_u = risk.expected_utility

        reliability = D_rec_emo / max(d_max, 1)
        sim_uncertainty = export_uncertainty(outcomes, D_rec_emo, d_max)

        # Outcome convergence: 1 if one outcome dominates
        convergence = max((o.probability for o in outcomes), default=0.0)

        # ==============================================================
        # NEUROCHEMICAL SIGNALS
        # ==============================================================

        worst_severity = 1.0 - risk.worst_case_utility if outcomes else 0.0
        neurochem = compute_simulation_neurochem(
            expected_utility=expected_u,
            tail_risk=risk.tail_risk,
            worst_severity=worst_severity,
            D_rec=D_rec_emo,
            D_base=d0,
            D_max=d_max,
            temperature=T_emotion,
            T_0=T_0,
            outcome_convergence=convergence,
            cfg=cfg,
        )

        # ==============================================================
        # UPDATE STATE
        # ==============================================================

        self._cycle_count += 1
        self._state.total_simulations += 1
        self._state.total_nodes_expanded += len(nodes)
        self._state.total_branches_pruned += pruned

        elapsed = (time.perf_counter() - t0) * 1000.0

        return SimulationBrainResult(
            total_nodes=len(nodes),
            max_depth_reached=max_depth,
            branches_pruned=pruned,
            outcomes=tuple(outcomes),
            modal_outcome=modal,
            expected_utility=expected_u,
            risk_profile=risk,
            forecast_temperature=T_emotion,
            recursion_depth=D_rec_emo,
            outcome_entropy=entropy,
            forecast_reliability=reliability,
            recommended_action=action,
            action_confidence=confidence,
            confidence_interval=ci,
            simulation_uncertainty=sim_uncertainty,
            neurochemical_signals=neurochem,
            processing_time_ms=elapsed,
            engine_id=self.engine_id,
            metadata={
                "mode": self._mode.value,
                "cycle": self._cycle_count,
                "seeds_available": len(all_seeds),
                "seeds_selected": len(selected_seeds),
                "nodes_total": len(nodes),
                "leaves": len(leaves),
                "pruned": pruned,
                "kappa": kappa,
                "delta_uncertainty": self._state.delta_uncertainty,
                "T_raw": temperature,
                "T_emotion": T_emotion,
                "D_rec_raw": D_rec,
                "D_rec_emotion": D_rec_emo,
            },
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "total_simulations": self._state.total_simulations,
            "total_nodes_expanded": self._state.total_nodes_expanded,
            "total_branches_pruned": self._state.total_branches_pruned,
            "delta_uncertainty": self._state.delta_uncertainty,
            "nt_levels": {
                "da": self._state.da_level,
                "ne": self._state.ne_level,
                "ach": self._state.ach_level,
                "5ht": self._state._5ht_level,
                "cb1": self._state.cb1_level,
                "cor": self._state.cor_level,
                "gaba": self._state.gaba_level,
            },
        }
