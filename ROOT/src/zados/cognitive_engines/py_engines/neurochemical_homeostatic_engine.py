"""
Engine 27 -- Neurochemical Activity Homeostatic Engine
======================================================
System health monitor and homeostatic regulator.  Monitors the
neurochemical layer's state and enforces homeostatic bounds -- ensuring
no neurotransmitter or receptor state drifts into pathological territory.

This is the system's "vitals monitor."

Key features:
  * Cognitive Load Estimation: L_cog(t) from symbolic, emotional, urgency
    saturation sub-components.
  * Homeostatic bound monitoring for all 12 NTs.
  * Gradual correction (soft pull toward baseline) with emergency hard-reset
    for extreme violations.
  * GABA reactive burst and 5-HT1A / CB1 regulatory up-modulation when
    overloaded.
  * Continuous background process that runs every cycle.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Enums
# =====================================================================


class BoundViolation(str, Enum):
    """Type of homeostatic bound violation."""
    NONE      = "none"
    ELEVATED  = "elevated"    # Above safe operating range
    DEPLETED  = "depleted"    # Below safe operating range
    CRITICAL  = "critical"    # Far outside bounds, emergency


class CorrectionType(str, Enum):
    """How the engine corrects a violation."""
    GRADUAL    = "gradual"    # Soft pull toward baseline
    AGGRESSIVE = "aggressive" # Faster pull (sustained violation)
    HARD_RESET = "hard_reset" # Snap to baseline (critical)


class HealthStatus(str, Enum):
    """Overall system neurochemical health."""
    HEALTHY   = "healthy"     # All NTs within bounds
    STRESSED  = "stressed"    # Some NTs elevated
    OVERLOADED = "overloaded" # Cognitive load high
    CRITICAL  = "critical"    # Multiple violations


# =====================================================================
# Configuration
# =====================================================================


# Homeostatic bounds per NT: (low_bound, baseline, high_bound, critical_low, critical_high)
_NT_BOUNDS: Dict[str, Tuple[float, float, float, float, float]] = {
    "da":        (0.10, 0.40, 0.75, 0.05, 0.90),
    "5ht":       (0.15, 0.45, 0.70, 0.08, 0.85),
    "ne":        (0.10, 0.35, 0.70, 0.05, 0.85),
    "ach":       (0.10, 0.35, 0.70, 0.05, 0.85),
    "oxt":       (0.15, 0.50, 0.80, 0.08, 0.90),
    "mor":       (0.05, 0.25, 0.60, 0.02, 0.80),
    "cb1":       (0.10, 0.35, 0.65, 0.05, 0.80),
    "cor":       (0.05, 0.20, 0.55, 0.02, 0.75),
    "crh":       (0.05, 0.15, 0.50, 0.02, 0.70),
    "gaba":      (0.15, 0.45, 0.75, 0.08, 0.90),
    "glu":       (0.10, 0.40, 0.70, 0.05, 0.85),
    "histamine": (0.08, 0.30, 0.60, 0.03, 0.80),
}


@dataclass(frozen=True)
class HomeostaticConfig:
    """Immutable configuration for the Homeostatic Engine."""

    # --- Cognitive load estimation ---
    w_symbolic:    float = 1.0   # S_sym weight
    w_emotional:   float = 1.2   # S_emo weight (higher)
    w_urgency:     float = 1.5   # S_urg weight (dominant in overload)

    # --- Overload threshold ---
    overload_threshold:   float = 0.85   # L_cog > this → system modulation

    # --- Correction rates ---
    gradual_rate:     float = 0.05    # Per-cycle pull toward baseline
    aggressive_rate:  float = 0.15    # Faster pull for sustained violations
    hard_reset_snap:  float = 1.00    # Fraction snap to baseline (1.0 = full)

    # --- Escalation ---
    cycles_to_aggressive:  int = 3    # Gradual for N cycles, then aggressive
    cycles_to_hard_reset:  int = 8    # Aggressive for M more cycles, then hard reset

    # --- GABA reactive burst (overload) ---
    beta_gaba_overload:    float = 0.12
    gamma_gaba_alpha:      float = 2.0
    gamma_gaba_theta:      float = 0.30

    # --- Regulatory up-modulation (overload) ---
    rho_5ht1a_containment: float = 0.08
    rho_cb1_dampening:     float = 0.06

    # --- Mode adjustments ---
    rem_dream_tolerance:   float = 1.20  # Wider bounds in dream mode (legacy)
    dev_tolerance:         float = 0.90  # Tighter bounds in dev mode

    # --- Dream mode specific (Spec §8.3) ---
    dream_tolerance_band:      float = 0.25   # Absolute +/- from dream baseline
    dream_runaway_band:        float = 0.30   # Exceeding this = runaway candidate
    dream_runaway_max_ticks:   int   = 3      # Consecutive ticks before runaway alert
    dream_monitored_nts:       tuple = ("ach", "da", "gaba")   # Only these during dream
    dream_floor_nts:           tuple = ("ne", "5ht")           # Must stay near-zero
    dream_floor_ceiling:       float = 0.10   # NE/5-HT cannot exceed this in dream

    # --- Neurochemical coupling (self-monitoring) ---
    beta_ne_violation:     float = 0.08   # NE alert on violation
    beta_cor_chronic:      float = 0.06   # Cortisol for chronic violation
    poisson_lam:           float = 1.5


# =====================================================================
# Data types
# =====================================================================


@dataclass
class NTViolationRecord:
    """Record of a homeostatic violation for a single NT."""
    nt_name:             str                = ""
    current_value:       float              = 0.0
    baseline:            float              = 0.0
    low_bound:           float              = 0.0
    high_bound:          float              = 0.0
    violation_type:      BoundViolation     = BoundViolation.NONE
    correction_type:     CorrectionType     = CorrectionType.GRADUAL
    correction_delta:    float              = 0.0   # How much to adjust
    consecutive_cycles:  int                = 0


@dataclass(frozen=True)
class CognitiveLoadEstimate:
    """Cognitive load estimation output."""
    l_cog:           float = 0.0   # [0, 1] composite
    s_symbolic:      float = 0.0   # Symbolic saturation
    s_emotional:     float = 0.0   # Emotional saturation
    s_urgency:       float = 0.0   # Urgency saturation
    epsilon:         float = 0.0   # Dynamic modifier
    overloaded:      bool  = False


@dataclass(frozen=True)
class HomeostaticNeurochem:
    """
    Neurochemical coupling signals from the Homeostatic Engine.

    Notation (Appendix S2-S3, S9):
        delta_gaba  -> Delta C_GABA(t)     : reactive burst on overload L_cog > 0.85
        delta_5ht1a -> Delta S_5HT1A(t)    : containment / anxiolytic at L_cog > 0.6
        delta_cb1   -> Delta C_CB1(t)       : dampening / neuroprotective at L_cog > 0.6
        delta_ne    -> Delta C_NE(t)        : violation alert on critical bound breach
        delta_cor   -> Delta C_Cortisol(t)  : chronic stress from sustained violations
    """
    delta_gaba:    float = 0.0   # GABA reactive burst
    delta_5ht1a:   float = 0.0   # 5-HT1A containment
    delta_cb1:     float = 0.0   # CB1 dampening
    delta_ne:      float = 0.0   # NE violation alert
    delta_cor:     float = 0.0   # Cortisol for chronic violation


@dataclass(frozen=True)
class HomeostaticInput:
    """Input bundle for one Homeostatic Engine cycle."""
    nt_concentrations:   Dict[str, float] = field(default_factory=dict)   # NT → [0, 1]
    symbolic_saturation: float            = 0.0   # From reasoning engines
    emotional_saturation: float           = 0.0   # From emotion tracker
    urgency_saturation:  float            = 0.0   # From urgency forecast
    dynamic_modifier:    float            = 0.0   # Contradiction spikes, etc.
    active_mode:         OperationalMode  = OperationalMode.NORMAL


@dataclass(frozen=True)
class HomeostaticResult:
    """Full output of one Homeostatic Engine cycle."""
    cognitive_load:        CognitiveLoadEstimate    = field(default_factory=CognitiveLoadEstimate)
    violations:            List[NTViolationRecord]  = field(default_factory=list)
    corrections:           Dict[str, float]         = field(default_factory=dict)   # NT → correction delta
    health_status:         HealthStatus             = HealthStatus.HEALTHY
    neurochemical_signals: HomeostaticNeurochem     = field(default_factory=HomeostaticNeurochem)
    processing_time_ms:    float                    = 0.0
    metadata:              Dict[str, Any]           = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class HomeostaticState:
    """Per-NT violation tracking."""
    violation_counters: Dict[str, int]   = field(default_factory=dict)
    previous_load:      float            = 0.0


# =====================================================================
# Pure helper functions
# =====================================================================


def sigmoid(x: float) -> float:
    """Standard sigmoid, clamped for numerical safety."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def compute_cognitive_load(
    s_sym: float,
    s_emo: float,
    s_urg: float,
    epsilon: float,
    cfg: HomeostaticConfig,
) -> float:
    """
    L_cog(t) = sigma(w1 * S_sym + w2 * S_emo + w3 * S_urg + epsilon)
    """
    raw = cfg.w_symbolic * s_sym + cfg.w_emotional * s_emo + cfg.w_urgency * s_urg + epsilon
    return sigmoid(raw)


def check_nt_bounds(
    nt_name: str,
    value: float,
    mode: OperationalMode,
    cfg: HomeostaticConfig,
) -> Tuple[BoundViolation, float, float, float]:
    """
    Check if NT value is within homeostatic bounds.
    Returns (violation_type, low_bound, high_bound, baseline).
    """
    bounds = _NT_BOUNDS.get(nt_name, (0.10, 0.40, 0.70, 0.05, 0.85))
    low, baseline, high, crit_low, crit_high = bounds

    # Mode tolerance adjustment
    tolerance = 1.0
    if mode == OperationalMode.REM_DREAM:
        tolerance = cfg.rem_dream_tolerance
    elif mode == OperationalMode.DEV:
        tolerance = cfg.dev_tolerance

    # Adjust bounds by tolerance
    adj_low = low / tolerance
    adj_high = min(1.0, high * tolerance)
    adj_crit_low = crit_low / tolerance
    adj_crit_high = min(1.0, crit_high * tolerance)

    if value <= adj_crit_low or value >= adj_crit_high:
        return (BoundViolation.CRITICAL, adj_low, adj_high, baseline)
    if value < adj_low:
        return (BoundViolation.DEPLETED, adj_low, adj_high, baseline)
    if value > adj_high:
        return (BoundViolation.ELEVATED, adj_low, adj_high, baseline)
    return (BoundViolation.NONE, adj_low, adj_high, baseline)


# Dream baselines for dream-specific bound checking (from Spec §2.3)
_DREAM_BASELINES: Dict[str, float] = {
    "ach": 0.85, "ne": 0.05, "5ht": 0.05, "da": 0.65,
    "gaba": 0.65, "glu": 0.70, "cb1": 0.75, "mor": 0.60,
    "crh": 0.05, "cor": 0.05, "histamine": 0.05, "oxt": 0.55,
}


def check_nt_bounds_dream(
    nt_name: str,
    value: float,
    cfg: HomeostaticConfig,
) -> Tuple[BoundViolation, float, float, float]:
    """
    Dream-mode specific bounds check (Spec §8.3).

    Uses absolute tolerance bands around dream state baselines instead
    of the waking bounds. Only ACh, DA, GABA are monitored for runaway.
    NE and 5-HT are checked against floor ceiling.

    Returns (violation_type, low_bound, high_bound, baseline).
    """
    dream_base = _DREAM_BASELINES.get(nt_name, 0.5)

    # Floor NTs: NE and 5-HT must stay near-zero
    if nt_name in cfg.dream_floor_nts:
        if value > cfg.dream_floor_ceiling:
            return (BoundViolation.ELEVATED, 0.0, cfg.dream_floor_ceiling, dream_base)
        return (BoundViolation.NONE, 0.0, cfg.dream_floor_ceiling, dream_base)

    # Monitored NTs: absolute tolerance band around dream baseline
    if nt_name in cfg.dream_monitored_nts:
        low = max(0.0, dream_base - cfg.dream_tolerance_band)
        high = min(1.0, dream_base + cfg.dream_tolerance_band)
        crit_low = max(0.0, dream_base - cfg.dream_runaway_band)
        crit_high = min(1.0, dream_base + cfg.dream_runaway_band)

        if value < crit_low or value > crit_high:
            return (BoundViolation.CRITICAL, low, high, dream_base)
        if value < low:
            return (BoundViolation.DEPLETED, low, high, dream_base)
        if value > high:
            return (BoundViolation.ELEVATED, low, high, dream_base)
        return (BoundViolation.NONE, low, high, dream_base)

    # Non-monitored NTs during dream: skip (no violation)
    return (BoundViolation.NONE, 0.0, 1.0, dream_base)


def compute_correction(
    value: float,
    baseline: float,
    violation: BoundViolation,
    consecutive_cycles: int,
    cfg: HomeostaticConfig,
) -> Tuple[CorrectionType, float]:
    """
    Determine correction type and delta for a violated NT.
    Returns (correction_type, correction_delta).
    """
    if violation == BoundViolation.NONE:
        return (CorrectionType.GRADUAL, 0.0)

    if violation == BoundViolation.CRITICAL or consecutive_cycles >= cfg.cycles_to_hard_reset:
        delta = (baseline - value) * cfg.hard_reset_snap
        return (CorrectionType.HARD_RESET, delta)

    if consecutive_cycles >= cfg.cycles_to_aggressive:
        delta = (baseline - value) * cfg.aggressive_rate
        return (CorrectionType.AGGRESSIVE, delta)

    delta = (baseline - value) * cfg.gradual_rate
    return (CorrectionType.GRADUAL, delta)


def compute_health_status(
    violations: List[NTViolationRecord],
    cognitive_load: CognitiveLoadEstimate,
) -> HealthStatus:
    """Classify overall system health."""
    critical_count = sum(1 for v in violations if v.violation_type == BoundViolation.CRITICAL)
    violation_count = sum(1 for v in violations if v.violation_type != BoundViolation.NONE)

    if critical_count > 0 or violation_count >= 4:
        return HealthStatus.CRITICAL
    if cognitive_load.overloaded:
        return HealthStatus.OVERLOADED
    if violation_count > 0:
        return HealthStatus.STRESSED
    return HealthStatus.HEALTHY


def compute_homeostatic_neurochem(
    cognitive_load: CognitiveLoadEstimate,
    violations: List[NTViolationRecord],
    cfg: HomeostaticConfig,
    rng: np.random.Generator,
) -> HomeostaticNeurochem:
    """
    Neurochemical signals from the homeostatic engine.

    GABA  -- reactive burst when cognitive load exceeds threshold
    5-HT1A -- containment/stability up-modulation
    CB1   -- emotional dampening up-modulation
    NE    -- violation alert
    COR   -- chronic violation stress
    """
    delta_gaba = 0.0
    delta_5ht1a = 0.0
    delta_cb1 = 0.0
    delta_ne = 0.0
    delta_cor = 0.0

    # GABA reactive burst when overloaded
    if cognitive_load.overloaded:
        gaba_noise = float(rng.gamma(cfg.gamma_gaba_alpha, cfg.gamma_gaba_theta))
        delta_gaba = cfg.beta_gaba_overload * cognitive_load.l_cog * gaba_noise

    # 5-HT1A and CB1 regulatory up-modulation
    if cognitive_load.l_cog > 0.6:
        delta_5ht1a = cfg.rho_5ht1a_containment * cognitive_load.l_cog
        delta_cb1 = cfg.rho_cb1_dampening * cognitive_load.l_cog

    # NE violation alert
    critical_violations = [v for v in violations if v.violation_type == BoundViolation.CRITICAL]
    if critical_violations:
        ne_impulse = float(rng.poisson(cfg.poisson_lam))
        delta_ne = cfg.beta_ne_violation * len(critical_violations) * ne_impulse

    # Cortisol for chronic violations
    chronic = [v for v in violations if v.consecutive_cycles >= 5]
    if chronic:
        delta_cor = cfg.beta_cor_chronic * len(chronic) / max(1, len(_NT_BOUNDS))

    return HomeostaticNeurochem(
        delta_gaba=delta_gaba,
        delta_5ht1a=delta_5ht1a,
        delta_cb1=delta_cb1,
        delta_ne=delta_ne,
        delta_cor=delta_cor,
    )


# =====================================================================
# Engine class
# =====================================================================


class NeurochemicalHomeostaticEngine:
    """
    Engine 27 -- Neurochemical Activity Homeostatic Engine.

    Monitors neurochemical layer state and enforces homeostatic bounds.
    Runs as a continuous background process every cycle.

    API
    ---
    configure(mode)   -- set operational mode
    process(h_input)  -- run homeostatic check, return HomeostaticResult
    get_status()      -- introspection
    """

    engine_id = "neurochemical_homeostatic_engine"
    cluster   = "homeostasis"

    def __init__(
        self,
        config: Optional[HomeostaticConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or HomeostaticConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = HomeostaticState()
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """No-op — homeostatic engine reads NT state via ``process()`` input.

        Present for interface compliance; the orchestrator may call this
        uniformly on all engines.
        """
        pass  # NT concentrations arrive as HomeostaticInput fields

    # ----- Main pipeline --------------------------------------------------

    def process(self, h_input: HomeostaticInput) -> HomeostaticResult:
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = h_input.active_mode

        # 1. Cognitive load estimation
        l_cog = compute_cognitive_load(
            h_input.symbolic_saturation,
            h_input.emotional_saturation,
            h_input.urgency_saturation,
            h_input.dynamic_modifier,
            self._cfg,
        )
        overloaded = l_cog > self._cfg.overload_threshold
        cog_load = CognitiveLoadEstimate(
            l_cog=round(l_cog, 4),
            s_symbolic=round(h_input.symbolic_saturation, 4),
            s_emotional=round(h_input.emotional_saturation, 4),
            s_urgency=round(h_input.urgency_saturation, 4),
            epsilon=round(h_input.dynamic_modifier, 4),
            overloaded=overloaded,
        )

        # 2. Check each NT against bounds
        violations: List[NTViolationRecord] = []
        corrections: Dict[str, float] = {}

        for nt_name, value in h_input.nt_concentrations.items():
            if mode == OperationalMode.REM_DREAM:
                violation, low, high, baseline = check_nt_bounds_dream(
                    nt_name, value, self._cfg,
                )
            else:
                violation, low, high, baseline = check_nt_bounds(
                    nt_name, value, mode, self._cfg,
                )

            # Update violation counter
            if violation != BoundViolation.NONE:
                prev = self._state.violation_counters.get(nt_name, 0)
                self._state.violation_counters[nt_name] = prev + 1
            else:
                self._state.violation_counters[nt_name] = 0

            consecutive = self._state.violation_counters.get(nt_name, 0)

            correction_type, correction_delta = compute_correction(
                value, baseline, violation, consecutive, self._cfg,
            )

            if violation != BoundViolation.NONE:
                violations.append(NTViolationRecord(
                    nt_name=nt_name,
                    current_value=round(value, 4),
                    baseline=round(baseline, 4),
                    low_bound=round(low, 4),
                    high_bound=round(high, 4),
                    violation_type=violation,
                    correction_type=correction_type,
                    correction_delta=round(correction_delta, 4),
                    consecutive_cycles=consecutive,
                ))
                corrections[nt_name] = round(correction_delta, 4)

        # 3. Health status
        health = compute_health_status(violations, cog_load)

        # 4. Neurochemical coupling
        neurochem = compute_homeostatic_neurochem(
            cog_load, violations, self._cfg, self._rng,
        )

        self._state.previous_load = l_cog
        elapsed = (time.perf_counter() - t0) * 1000.0

        return HomeostaticResult(
            cognitive_load=cog_load,
            violations=violations,
            corrections=corrections,
            health_status=health,
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "cycle": self._cycle_count,
                "nts_checked": len(h_input.nt_concentrations),
                "violations_found": len(violations),
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "previous_load": self._state.previous_load,
            "active_violations": {
                k: v for k, v in self._state.violation_counters.items() if v > 0
            },
        }
