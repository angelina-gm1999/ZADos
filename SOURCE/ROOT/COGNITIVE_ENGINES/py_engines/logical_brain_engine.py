"""
Engine 12 -- Logical Brain Engine  (``logical_brain_engine``)
=============================================================
The Logic reward domain's "exam mode" -- runs the SAME submodules from
``reward/domains/logic/`` but with **elevated parameters** (diagnostic
sensitivity).

Key design from spec:
  * NOT a theorem prover.  Re-uses existing Logic reward submodules.
  * 7 core + 4 extended evaluations (all 11 submodules from logic domain,
    with some in "extended" mode only when cognitive trace is available).
  * 4-tier weighted aggregate scoring → LogicalBrainVerdict.
  * Heavy neurochemical coupling: NE, ACh, GLU, DA, COR, GABA-A.
  * Bidirectional feedback with neurochemical layer.
  * Requires MemoryContrastPort and CognitiveTracePort (can run without,
    but extended evaluations will be skipped / scored at 0.0).
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
    ProcessedStatement,
)
from zados.reward.base.types import RewardContext, RewardDomainResult, RewardSubscore
from zados.reward.domains.logic.ports import (
    MemoryContrastPort,
    CognitiveTracePort,
    ContrastResult,
    TraceResult,
)

# Submodule classes (we instantiate them in diagnostic mode)
from zados.reward.domains.logic.epistemic_calibration import EpistemicCalibrationSubmodule
from zados.reward.domains.logic.uncertainty_acknowledgment import UncertaintyAcknowledgmentSubmodule
from zados.reward.domains.logic.abstention_appropriateness import AbstentionAppropriatenessSubmodule
from zados.reward.domains.logic.internal_consistency import InternalConsistencySubmodule
from zados.reward.domains.logic.external_consistency import ExternalConsistencySubmodule
from zados.reward.domains.logic.semantic_continuity import SemanticContinuitySubmodule
from zados.reward.domains.logic.concept_continuity import ConceptContinuitySubmodule
from zados.reward.domains.logic.context_fidelity import ContextFidelitySubmodule
from zados.reward.domains.logic.concept_fidelity import ConceptFidelitySubmodule


# =====================================================================
# Enums
# =====================================================================


class VerdictLevel(str, Enum):
    """Logical Brain verdict severity."""
    EXEMPLARY    = "exemplary"     # score ≥ 0.85
    ADEQUATE     = "adequate"      # 0.60 ≤ score < 0.85
    DEFICIENT    = "deficient"     # 0.35 ≤ score < 0.60
    CRITICAL     = "critical"      # score < 0.35


class EvaluationTier(str, Enum):
    """Which tier a submodule belongs to for weighted aggregation."""
    CORE_EPISTEMIC   = "core_epistemic"      # Tier 1: epistemic regulators
    CORE_CONSISTENCY = "core_consistency"     # Tier 2: consistency checks
    EXTENDED_FIDELITY = "extended_fidelity"   # Tier 3: fidelity (requires trace)
    EXTENDED_CONTINUITY = "extended_continuity"  # Tier 4: temporal continuity


# =====================================================================
# Configuration
# =====================================================================


# Diagnostic-mode parameter elevations (multiplied on top of base submodule behavior)
DIAGNOSTIC_ELEVATION = 1.25  # 25% more sensitive than standard reward evaluation


@dataclass(frozen=True)
class LogicalBrainConfig:
    """Immutable configuration for the Logical Brain Engine."""

    # --- Tier weights for aggregate scoring ---
    w_tier1_epistemic:   float = 0.30   # Epistemic calibration, uncertainty, abstention
    w_tier2_consistency: float = 0.30   # Internal + external consistency
    w_tier3_fidelity:    float = 0.20   # Context + concept fidelity (need trace)
    w_tier4_continuity:  float = 0.20   # Semantic + concept continuity

    # --- Verdict thresholds ---
    exemplary_threshold: float = 0.85
    adequate_threshold:  float = 0.60
    deficient_threshold: float = 0.35

    # --- Mode thresholds (min score for "pass") ---
    theta_normal:     float = 0.50
    theta_dev:        float = 0.30   # Most lenient in dev
    theta_learning:   float = 0.45
    theta_reflective: float = 0.40
    theta_rem_normal: float = 0.50
    theta_rem_dream:  float = 0.60

    # --- Diagnostic elevation factor ---
    diagnostic_elevation: float = DIAGNOSTIC_ELEVATION

    # --- Neurochemical coupling ---
    # NE -- logical vigilance
    beta_ne_vigilance:    float = 0.12
    lambda_ne_poisson:    float = 2.0
    # ACh -- sustained analytical attention
    beta_ach_analysis:    float = 0.15
    gamma_ach_alpha:      float = 2.0
    gamma_ach_theta:      float = 0.35
    # GLU -- excitatory integration signal
    beta_glu_integration: float = 0.10
    # DA -- prediction error (positive for good logic, negative for bad)
    beta_da_rpe:          float = 0.12
    gamma_da_alpha:       float = 2.0
    gamma_da_theta:       float = 0.30
    # Cortisol -- logical failure stress
    beta_cor_failure:     float = 0.10
    cor_failure_threshold: float = 0.40  # Score below this triggers COR
    # GABA-A -- inhibition of irrelevant processing during analysis
    beta_gaba_inhibit:    float = 0.08
    # Oscillatory
    psi_beta_analysis:    float = 0.08   # Beta boost during analysis
    psi_gamma_integration: float = 0.06  # Gamma boost for cross-submodule integration


# =====================================================================
# Submodule registry: maps submodule name → tier + weight within tier
# =====================================================================


_SUBMODULE_TIERS: Dict[str, Tuple[EvaluationTier, float]] = {
    # Tier 1: Core epistemic (3 submodules)
    "epistemic_calibration":      (EvaluationTier.CORE_EPISTEMIC,   0.40),
    "uncertainty_acknowledgment": (EvaluationTier.CORE_EPISTEMIC,   0.35),
    "abstention_appropriateness": (EvaluationTier.CORE_EPISTEMIC,   0.25),
    # Tier 2: Core consistency (2 submodules)
    "internal_consistency":       (EvaluationTier.CORE_CONSISTENCY, 0.55),
    "external_consistency":       (EvaluationTier.CORE_CONSISTENCY, 0.45),
    # Tier 3: Extended fidelity (2 submodules)
    "context_fidelity":           (EvaluationTier.EXTENDED_FIDELITY, 0.50),
    "concept_fidelity":           (EvaluationTier.EXTENDED_FIDELITY, 0.50),
    # Tier 4: Extended continuity (2 submodules)
    "semantic_continuity":        (EvaluationTier.EXTENDED_CONTINUITY, 0.50),
    "concept_continuity":         (EvaluationTier.EXTENDED_CONTINUITY, 0.50),
}


# =====================================================================
# Data types -- frozen outputs
# =====================================================================


@dataclass(frozen=True)
class SubmoduleScore:
    """Result of one submodule evaluation in diagnostic mode."""
    name:           str
    tier:           EvaluationTier
    raw_score:      float              # From submodule [0, 1]
    elevated_score: float              # After diagnostic elevation adjustment
    flags:          Dict[str, Any]     = field(default_factory=dict)
    meta:           Dict[str, Any]     = field(default_factory=dict)
    skipped:        bool               = False


@dataclass(frozen=True)
class LogicalBrainVerdict:
    """Aggregate verdict from the Logical Brain Engine."""
    verdict_id:     str               = field(default_factory=lambda: str(uuid.uuid4()))
    verdict_level:  VerdictLevel      = VerdictLevel.ADEQUATE
    aggregate_score: float            = 0.0   # [0, 1]
    tier_scores:    Dict[str, float]  = field(default_factory=dict)
    passed:         bool              = True   # aggregate ≥ mode threshold
    description:    str               = ""
    timestamp:      float             = field(default_factory=time.time)


@dataclass(frozen=True)
class LogicalBrainNeurochem:
    """
    Neurochemical coupling signals from one Logical Brain cycle.

    Notation (Appendix S2-S3, S7-S9):
        delta_ne     -> Delta C_NE(t)       : vigilance / diagnostic alertness
        delta_ach    -> Delta C_ACh(t)      : analytical attention (S9: ACh -> beta)
        delta_glu    -> Delta C_GLU(t)      : integration gating (S9: Glu -> gamma,theta-gamma)
        delta_da     -> Delta C_DA(t)       : RPE from verdict vs expected (S9: DA -> gamma,theta)
        delta_cor    -> Delta C_Cortisol(t) : failure stress on CRITICAL/DEFICIENT verdicts
        delta_gaba_a -> Delta S_GABA-A(t)   : inhibitory when aggregate > exemplary threshold
        beta_boost   -> Delta phi_beta(t)   : analytical focus band (S7, S9: ACh -> beta)
        gamma_boost  -> Delta phi_gamma(t)  : integration band (S7, S9: Glu -> gamma)
    """
    delta_ne:     float = 0.0
    delta_ach:    float = 0.0
    delta_glu:    float = 0.0
    delta_da:     float = 0.0
    delta_cor:    float = 0.0
    delta_gaba_a: float = 0.0
    beta_boost:   float = 0.0
    gamma_boost:  float = 0.0


@dataclass(frozen=True)
class LogicalBrainInput:
    """Input bundle for one Logical Brain evaluation."""
    state:                Dict[str, Any]         = field(default_factory=dict)
    user_input:           Optional[ProcessedStatement] = None
    system_output:        Optional[str]          = None
    contradiction_flags:  List[Any]              = field(default_factory=list)
    fallacy_flags:        List[Any]              = field(default_factory=list)
    memory_context:       Optional[Dict[str, Any]] = None
    active_mode:          OperationalMode        = OperationalMode.NORMAL


@dataclass(frozen=True)
class LogicalBrainResult:
    """Full output of one Logical Brain Engine cycle."""
    verdict:               LogicalBrainVerdict    = field(default_factory=LogicalBrainVerdict)
    submodule_scores:      List[SubmoduleScore]   = field(default_factory=list)
    domain_result:         Optional[RewardDomainResult] = None
    neurochemical_signals: LogicalBrainNeurochem  = field(default_factory=LogicalBrainNeurochem)
    processing_time_ms:    float                  = 0.0
    metadata:              Dict[str, Any]         = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class LogicalBrainState:
    """Running neurochemical state for bidirectional feedback."""
    ne_level:   float = 0.0
    ach_level:  float = 0.0
    da_level:   float = 0.0
    cor_level:  float = 0.0
    gaba_level: float = 0.0


# =====================================================================
# Pure helper functions
# =====================================================================


def apply_diagnostic_elevation(raw_score: float, elevation: float) -> float:
    """
    Apply diagnostic elevation: in diagnostic mode the engine is MORE
    sensitive to problems.  Lower raw scores get amplified.

        elevated = raw_score ^ (1 / elevation)

    This makes the scoring HARSHER for low values (they rise less)
    and slightly more generous for high values.
    """
    if raw_score <= 0.0:
        return 0.0
    if raw_score >= 1.0:
        return 1.0
    return raw_score ** (1.0 / elevation)


def compute_tier_score(
    submodule_scores: List[SubmoduleScore],
    tier: EvaluationTier,
) -> float:
    """
    Weighted average within a single tier.
    Skipped submodules contribute 0 but their weight is redistributed.
    """
    relevant = [s for s in submodule_scores if s.tier == tier]
    if not relevant:
        return 0.0

    active = [(s, _SUBMODULE_TIERS[s.name][1]) for s in relevant if not s.skipped]
    if not active:
        return 0.0

    total_w = sum(w for _, w in active)
    if total_w <= 0.0:
        return 0.0

    return sum(s.elevated_score * w for s, w in active) / total_w


def compute_aggregate_score(
    tier_scores: Dict[str, float],
    cfg: LogicalBrainConfig,
) -> float:
    """4-tier weighted aggregate."""
    return (
        cfg.w_tier1_epistemic   * tier_scores.get("core_epistemic", 0.0)
        + cfg.w_tier2_consistency * tier_scores.get("core_consistency", 0.0)
        + cfg.w_tier3_fidelity    * tier_scores.get("extended_fidelity", 0.0)
        + cfg.w_tier4_continuity  * tier_scores.get("extended_continuity", 0.0)
    )


def classify_verdict(score: float, cfg: LogicalBrainConfig) -> VerdictLevel:
    """Map aggregate score to verdict level."""
    if score >= cfg.exemplary_threshold:
        return VerdictLevel.EXEMPLARY
    if score >= cfg.adequate_threshold:
        return VerdictLevel.ADEQUATE
    if score >= cfg.deficient_threshold:
        return VerdictLevel.DEFICIENT
    return VerdictLevel.CRITICAL


def resolve_pass_threshold(mode: OperationalMode, cfg: LogicalBrainConfig) -> float:
    """Mode-dependent minimum pass threshold."""
    return {
        OperationalMode.NORMAL:     cfg.theta_normal,
        OperationalMode.DEV:        cfg.theta_dev,
        OperationalMode.LEARNING:   cfg.theta_learning,
        OperationalMode.REFLECTIVE: cfg.theta_reflective,
        OperationalMode.REM_NORMAL: cfg.theta_rem_normal,
        OperationalMode.REM_DREAM:  cfg.theta_rem_dream,
    }.get(mode, cfg.theta_normal)


def compute_logical_brain_neurochem(
    aggregate_score: float,
    verdict_level: VerdictLevel,
    n_flags: int,
    cfg: LogicalBrainConfig,
    rng: np.random.Generator,
) -> LogicalBrainNeurochem:
    """
    Neurochemical coupling from Logical Brain output.

    NE  -- logical vigilance, fires during analysis
    ACh -- sustained analytical attention
    GLU -- excitatory integration across submodules
    DA  -- positive RPE for good logic, negative for bad
    COR -- stress when logic fails significantly
    GABA-A -- inhibition of irrelevant processing
    Beta/Gamma -- oscillatory coupling
    """
    # NE: vigilance during logical analysis
    ne_impulse = float(rng.poisson(cfg.lambda_ne_poisson))
    delta_ne = cfg.beta_ne_vigilance * (1.0 - aggregate_score + 0.1) * ne_impulse

    # ACh: sustained analysis attention
    ach_noise = float(rng.gamma(cfg.gamma_ach_alpha, cfg.gamma_ach_theta))
    delta_ach = cfg.beta_ach_analysis * ach_noise

    # GLU: integration signal proportional to analysis complexity
    delta_glu = cfg.beta_glu_integration * (0.5 + 0.5 * (n_flags / max(1, 10)))

    # DA: prediction error -- good logic = positive, bad = negative
    da_noise = float(rng.gamma(cfg.gamma_da_alpha, cfg.gamma_da_theta))
    if aggregate_score >= 0.7:
        delta_da = cfg.beta_da_rpe * (aggregate_score - 0.5) * da_noise
    elif aggregate_score < 0.4:
        delta_da = -cfg.beta_da_rpe * (0.5 - aggregate_score) * 0.5
    else:
        delta_da = 0.0

    # COR: stress when logic fails
    delta_cor = 0.0
    if aggregate_score < cfg.cor_failure_threshold:
        delta_cor = cfg.beta_cor_failure * (cfg.cor_failure_threshold - aggregate_score)

    # GABA-A: inhibition of distractors during analysis
    delta_gaba_a = cfg.beta_gaba_inhibit * (1.0 if aggregate_score > 0.3 else 0.5)

    # Oscillatory
    beta_boost = cfg.psi_beta_analysis
    gamma_boost = cfg.psi_gamma_integration * (n_flags / max(1, 5))

    return LogicalBrainNeurochem(
        delta_ne=delta_ne,
        delta_ach=delta_ach,
        delta_glu=delta_glu,
        delta_da=delta_da,
        delta_cor=delta_cor,
        delta_gaba_a=delta_gaba_a,
        beta_boost=beta_boost,
        gamma_boost=gamma_boost,
    )


# =====================================================================
# Engine class
# =====================================================================


class LogicalBrainEngine:
    """
    Engine 12 -- Logical Brain Engine.

    Runs the Logic reward domain's submodules in DIAGNOSTIC MODE with
    elevated parameters, producing a structured LogicalBrainVerdict.

    API
    ---
    configure(mode)           -- set operational mode
    update_neurochem_state(d) -- inject external NT levels
    process(lb_input)         -- run diagnostic evaluation, return LogicalBrainResult
    get_status()              -- introspection
    """

    engine_id = "logical_brain_engine"
    cluster   = "evaluation"

    def __init__(
        self,
        config: Optional[LogicalBrainConfig] = None,
        rng: Optional[np.random.Generator] = None,
        *,
        memory_contrast: Optional[MemoryContrastPort] = None,
        cognitive_trace: Optional[CognitiveTracePort] = None,
    ) -> None:
        self._cfg = config or LogicalBrainConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = LogicalBrainState()
        self._cycle_count = 0

        self._memory_contrast = memory_contrast
        self._cognitive_trace = cognitive_trace

        # Instantiate all submodules
        self._submodules = self._build_submodules()

    def _build_submodules(self) -> Dict[str, Any]:
        """Build the full submodule registry."""
        mc = self._memory_contrast
        return {
            # Tier 1: Core epistemic
            "epistemic_calibration": EpistemicCalibrationSubmodule(),
            "uncertainty_acknowledgment": UncertaintyAcknowledgmentSubmodule(),
            "abstention_appropriateness": AbstentionAppropriatenessSubmodule(),
            # Tier 2: Core consistency (require memory_contrast)
            "internal_consistency": InternalConsistencySubmodule(memory_contrast=mc),
            "external_consistency": ExternalConsistencySubmodule(memory_contrast=mc),
            # Tier 3: Extended fidelity (require cognitive_trace / memory_contrast)
            "context_fidelity": ContextFidelitySubmodule(memory_contrast=mc),
            "concept_fidelity": ConceptFidelitySubmodule(memory_contrast=mc),
            # Tier 4: Extended continuity (require memory_contrast)
            "semantic_continuity": SemanticContinuitySubmodule(memory_contrast=mc),
            "concept_continuity": ConceptContinuitySubmodule(memory_contrast=mc),
        }

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Inject current neurochemical levels for bidirectional feedback."""
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "ach" in state_dict:
            self._state.ach_level = state_dict["ach"]
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "cor" in state_dict:
            self._state.cor_level = state_dict["cor"]
        if "gaba" in state_dict:
            self._state.gaba_level = state_dict["gaba"]

    # ----- Main pipeline --------------------------------------------------

    def process(self, lb_input: LogicalBrainInput) -> LogicalBrainResult:
        """
        Run all Logic reward submodules in diagnostic mode.

        1. Evaluate each submodule with diagnostic elevation.
        2. Compute per-tier weighted scores.
        3. Compute aggregate 4-tier score.
        4. Classify verdict.
        5. Compute neurochemical coupling.
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = lb_input.active_mode
        pass_threshold = resolve_pass_threshold(mode, self._cfg)

        # Bidirectional: high cortisol → more sensitive
        elevation = self._cfg.diagnostic_elevation
        if self._state.cor_level > 0.5:
            elevation *= 1.10

        # Build reward context for submodule evaluation
        ctx = RewardContext(
            reward_profile=mode.value,
            timestamp=time.time(),
            meta=lb_input.state.get("meta", {}),
        )

        # Evaluate all submodules
        sub_scores: List[SubmoduleScore] = []
        all_flags: Dict[str, Any] = {}

        for sm_name, sm in self._submodules.items():
            tier_info = _SUBMODULE_TIERS.get(sm_name)
            if tier_info is None:
                continue
            tier, _w = tier_info

            # Run submodule evaluation
            result: RewardSubscore = sm.evaluate(lb_input.state, ctx)

            skipped = result.meta.get("skipped", False)
            raw_score = result.score
            elevated = apply_diagnostic_elevation(raw_score, elevation) if not skipped else 0.0

            sub_scores.append(SubmoduleScore(
                name=sm_name,
                tier=tier,
                raw_score=raw_score,
                elevated_score=round(elevated, 4),
                flags=dict(result.flags),
                meta=dict(result.meta),
                skipped=skipped,
            ))

            all_flags.update(result.flags)

        # Compute per-tier scores
        tier_scores: Dict[str, float] = {}
        for tier in EvaluationTier:
            tier_scores[tier.value] = round(compute_tier_score(sub_scores, tier), 4)

        # Aggregate
        aggregate = compute_aggregate_score(tier_scores, self._cfg)
        aggregate = round(_clamp(aggregate), 4)

        # Classify
        verdict_level = classify_verdict(aggregate, self._cfg)
        passed = aggregate >= pass_threshold

        verdict = LogicalBrainVerdict(
            verdict_level=verdict_level,
            aggregate_score=aggregate,
            tier_scores=tier_scores,
            passed=passed,
            description=f"Logical Brain: {verdict_level.value} ({aggregate:.2f})",
        )

        # Also produce a standard RewardDomainResult for compatibility
        subscores_dict = {}
        for ss in sub_scores:
            subscores_dict[ss.name] = RewardSubscore(
                name=ss.name,
                score=ss.elevated_score,
                flags=ss.flags,
                meta=ss.meta,
            )

        domain_result = RewardDomainResult(
            domain="logic_diagnostic",
            general_score=aggregate,
            subscores=subscores_dict,
            flags=all_flags,
            meta={
                "diagnostic_mode": True,
                "elevation": elevation,
                "verdict_level": verdict_level.value,
            },
        )

        # Neurochemical coupling
        neurochem = compute_logical_brain_neurochem(
            aggregate,
            verdict_level,
            len(all_flags),
            self._cfg,
            self._rng,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return LogicalBrainResult(
            verdict=verdict,
            submodule_scores=sub_scores,
            domain_result=domain_result,
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "pass_threshold": round(pass_threshold, 4),
                "elevation": round(elevation, 4),
                "cycle": self._cycle_count,
                "submodules_evaluated": len(sub_scores),
                "submodules_skipped": sum(1 for s in sub_scores if s.skipped),
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "ports": {
                "memory_contrast": self._memory_contrast is not None,
                "cognitive_trace": self._cognitive_trace is not None,
            },
            "state": {
                "ne_level": self._state.ne_level,
                "ach_level": self._state.ach_level,
                "da_level": self._state.da_level,
                "cor_level": self._state.cor_level,
                "gaba_level": self._state.gaba_level,
            },
        }
