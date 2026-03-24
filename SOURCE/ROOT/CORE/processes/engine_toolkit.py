"""
ZA-DOS v0.6 — Engine Toolkit (spec §2.2).

Heart of the v0.6 Matrioshka system.  Defines the Mode × Subject →
EngineTier matrix and resolves the final tier assignment for each engine
in a given (mode, subject) context.

Resolution rules (§2.2.4):
  1. Start with BASE_TIERS[mode][engine].
  2. Apply SUBJECT_PROMOTIONS if subject matches.
  3. Apply SUBJECT_DEMOTIONS if subject matches.
  4. Phantom engines (not yet implemented) are forced to T4.
  5. BUDGET_CAPS limit the total number of T1+T2 engines.
  6. If over budget, demote lowest-priority T2 engines to T3.
  7. Final tier dict returned; caller converts to engine_weights.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from zados.cognitive_engines.constants import ENGINE_IDS
from zados.core.types import EngineTier, SubjectCategory

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Engines that don't exist yet (forced to T4 in all modes)
# ------------------------------------------------------------------

PHANTOM_ENGINES: Set[str] = {
    # E31 reflective_learning_engine — IMPLEMENTED (Appendix §4.1)
    # E32 reflective_identity_engine — IMPLEMENTED (Appendix §4.2)
    "emotional_saturation_engine",          # TODO: implement (spec §4.3)
    # Part 4 §7.2 — new engines identified
    "held_thinking_block_writer_engine",    # emotion-threshold interrupt during thinking
    "overview_log_generator_engine",        # end-of-session summary writer
    "knowledge_map_builder_engine",         # creates/updates KnowledgeMaps (Homework Mode)
    "library_ingestor_engine",              # M5 material chunking + AtomSpace linking
    "notebook_writer_engine",               # academic journal for knowledge/notebook
    "question_dedup_guard_engine",          # prevents recursive question re-generation
    "core_memory_update_gate_engine",       # validates peer-review before core memory apply
}

# Phantom engine IDs (reserved for future implementation)
# NOTE: E31/E32 moved to ENGINE_IDS in constants.py (now implemented)
PHANTOM_ENGINE_IDS: Dict[int, str] = {
    33: "emotional_saturation_engine",
    # Part 4 §7.2 — reserved IDs
    34: "held_thinking_block_writer_engine",
    35: "overview_log_generator_engine",
    36: "knowledge_map_builder_engine",
    37: "library_ingestor_engine",
    38: "notebook_writer_engine",
    39: "question_dedup_guard_engine",
    40: "core_memory_update_gate_engine",
}

# All engine IDs including phantoms
ALL_ENGINE_IDS: Dict[int, str] = {**ENGINE_IDS, **PHANTOM_ENGINE_IDS}

# ------------------------------------------------------------------
# Base tier matrix:  mode → engine_name → EngineTier
# ------------------------------------------------------------------
# Modes: "regular", "M1", "M2", "M3", "M4", "M5"
# (Sleep/Meta-Learning modes use "regular" as base, with overrides)

_T1, _T2, _T3, _T4 = EngineTier.T1, EngineTier.T2, EngineTier.T3, EngineTier.T4

# Default tier for all engines not explicitly listed in a mode
_DEFAULT_TIER = _T3

def _fill_mode(overrides: Dict[str, EngineTier]) -> Dict[str, EngineTier]:
    """Build a complete engine→tier dict, defaulting unlisted engines to T3."""
    result: Dict[str, EngineTier] = {}
    for eid, ename in ALL_ENGINE_IDS.items():
        result[ename] = overrides.get(ename, _DEFAULT_TIER)
    # Force phantom engines to T4
    for pe in PHANTOM_ENGINES:
        result[pe] = _T4
    return result


BASE_TIERS: Dict[str, Dict[str, EngineTier]] = {
    # ------------------------------------------------------------------
    # Regular mode — balanced, all clusters represented
    # ------------------------------------------------------------------
    "regular": _fill_mode({
        "intention_map_engine":              _T1,
        "relevance_scoring_engine":          _T1,
        "input_relevance_evaluation_engine": _T1,
        "emotional_detection_engine":        _T1,
        "contradiction_detection_engine":    _T2,
        "paradox_detection_engine":          _T2,
        "fallacy_detection_engine":          _T2,
        "bias_detection_engine":             _T2,
        "logic_trap_detection_engine":       _T2,
        "heuristic_bias_engine":             _T2,
        "data_analysis_engine":              _T2,
        "pattern_identification_engine":     _T2,
        "atomspace_engine":                  _T2,
        "pln_engine":                        _T2,
        "logical_brain_engine":              _T2,
        "soar_production_engine":            _T3,
        "simulated_opposition_engine":       _T3,
        "socratic_reasoning_engine":         _T3,
        "simulation_brain_engine":           _T3,
        "decision_making_engine":            _T3,
        "ecan_engine":                       _T3,
        "pattern_comparison_engine":         _T3,
        "strategic_decision_engine":         _T3,
        "reward_based_learning_engine":      _T3,
        "contextual_learning_engine":        _T3,
        "recursive_learning_engine":         _T3,
        "uncertainty_pattern_engine":        _T3,
        "neurochemical_homeostatic_engine":  _T2,
        "memory_compression_engine":         _T3,
        "retroactive_alignment_engine":      _T3,
    }),

    # ------------------------------------------------------------------
    # M1 — Human Teaches (receptive, detection reframed to LEARNING)
    # Spec Part 4 §2.1: budget 14, T1* reframed engines run in
    # OperationalMode.LEARNING for comprehension, not adversarial.
    # ------------------------------------------------------------------
    "M1": _fill_mode({
        "intention_map_engine":              _T1,
        "relevance_scoring_engine":          _T1,
        "input_relevance_evaluation_engine": _T1,
        "emotional_detection_engine":        _T1,   # T1* LEARNING reframe
        "contradiction_detection_engine":    _T2,   # T1* LEARNING — structural questions
        "paradox_detection_engine":          _T2,   # T1* LEARNING — flag as questions
        "fallacy_detection_engine":          _T4,   # §2.1: T4 off
        "bias_detection_engine":             _T4,   # §2.1: T4 off
        "logic_trap_detection_engine":       _T4,
        "heuristic_bias_engine":             _T3,
        "data_analysis_engine":              _T2,
        "pattern_identification_engine":     _T1,
        "atomspace_engine":                  _T2,   # §2.1: T2 subject-activated
        "pln_engine":                        _T4,   # §2.1: T4 off
        "logical_brain_engine":              _T3,
        "soar_production_engine":            _T4,
        "simulated_opposition_engine":       _T4,   # §2.1: T4 off
        "socratic_reasoning_engine":         _T2,   # §2.1: T1* LEARNING — clarifying Qs
        "simulation_brain_engine":           _T4,
        "decision_making_engine":            _T4,   # §2.1: T4 off
        "ecan_engine":                       _T2,
        "pattern_comparison_engine":         _T1,
        "strategic_decision_engine":         _T4,
        "reward_based_learning_engine":      _T3,   # §2.1: T3 standby
        "contextual_learning_engine":        _T1,
        "recursive_learning_engine":         _T3,   # §2.1: T3 standby
        "uncertainty_pattern_engine":        _T2,   # §2.1: T2 subject-activated
        "neurochemical_homeostatic_engine":  _T1,
        "memory_compression_engine":         _T1,
        "retroactive_alignment_engine":      _T3,
    }),

    # ------------------------------------------------------------------
    # M2 — Peer Review (analytical, detection at full strength)
    # Spec Part 4 §3.1: budget 16, T1* Contradiction Detection runs
    # in LEARNING mode (comprehension, NOT adversarial against reviewer).
    # ------------------------------------------------------------------
    "M2": _fill_mode({
        "intention_map_engine":              _T1,
        "relevance_scoring_engine":          _T1,
        "input_relevance_evaluation_engine": _T1,
        "emotional_detection_engine":        _T1,
        "contradiction_detection_engine":    _T1,   # T1* LEARNING — self-alignment
        "paradox_detection_engine":          _T3,   # §3.1: T3 standby
        "fallacy_detection_engine":          _T3,   # §3.1: T3 standby
        "bias_detection_engine":             _T2,   # §3.1: T2 — checking OWN bias
        "logic_trap_detection_engine":       _T3,   # §3.1: T3 standby
        "heuristic_bias_engine":             _T2,   # §3.1: T2
        "data_analysis_engine":              _T1,
        "pattern_identification_engine":     _T1,
        "atomspace_engine":                  _T2,   # §3.1: T2 subject-activated (TEC/PHI/HIS)
        "pln_engine":                        _T4,   # §3.1: T4 off
        "logical_brain_engine":              _T3,
        "soar_production_engine":            _T3,
        "simulated_opposition_engine":       _T3,   # §3.1: T3 — can surface for self-defense
        "socratic_reasoning_engine":         _T3,   # §3.1: not listed T1/T2
        "simulation_brain_engine":           _T3,
        "decision_making_engine":            _T2,   # §3.1: T2 subject-activated
        "ecan_engine":                       _T2,
        "pattern_comparison_engine":         _T1,
        "strategic_decision_engine":         _T3,
        "reward_based_learning_engine":      _T1,   # §3.1: T1 always-on
        "contextual_learning_engine":        _T1,   # §3.1: T1 always-on
        "recursive_learning_engine":         _T2,   # §3.1: T2
        "uncertainty_pattern_engine":        _T2,
        "neurochemical_homeostatic_engine":  _T1,   # §3.1: T1 always-on
        "memory_compression_engine":         _T1,   # §3.1: T1 always-on
        "retroactive_alignment_engine":      _T2,
    }),

    # ------------------------------------------------------------------
    # M3 — Learn Together (full dialectic, max budget)
    # ------------------------------------------------------------------
    "M3": _fill_mode({
        "intention_map_engine":              _T1,
        "relevance_scoring_engine":          _T1,
        "input_relevance_evaluation_engine": _T1,
        "emotional_detection_engine":        _T1,
        "contradiction_detection_engine":    _T1,
        "paradox_detection_engine":          _T1,
        "fallacy_detection_engine":          _T1,
        "bias_detection_engine":             _T1,
        "logic_trap_detection_engine":       _T2,
        "heuristic_bias_engine":             _T1,
        "data_analysis_engine":              _T1,
        "pattern_identification_engine":     _T1,
        "atomspace_engine":                  _T1,
        "pln_engine":                        _T1,
        "logical_brain_engine":              _T1,
        "soar_production_engine":            _T2,
        "simulated_opposition_engine":       _T1,
        "socratic_reasoning_engine":         _T1,
        "simulation_brain_engine":           _T2,
        "decision_making_engine":            _T2,
        "ecan_engine":                       _T1,
        "pattern_comparison_engine":         _T1,
        "strategic_decision_engine":         _T2,
        "reward_based_learning_engine":      _T1,
        "contextual_learning_engine":        _T1,
        "recursive_learning_engine":         _T1,
        "uncertainty_pattern_engine":        _T1,
        "neurochemical_homeostatic_engine":  _T2,
        "memory_compression_engine":         _T2,
        "retroactive_alignment_engine":      _T2,
    }),

    # ------------------------------------------------------------------
    # M4 — Learned Questions (reflective, question-driven)
    # Spec Part 4 §5.1: budget 12 (lowest), lightweight question
    # surfacing. Fractal Tokenizer T4 (not input decomposition).
    # ------------------------------------------------------------------
    "M4": _fill_mode({
        "intention_map_engine":              _T1,
        "relevance_scoring_engine":          _T1,
        "input_relevance_evaluation_engine": _T2,   # §5.1: T2 user-present only
        "emotional_detection_engine":        _T1,   # §5.1: T1
        "contradiction_detection_engine":    _T2,   # §5.1: T2 soft
        "paradox_detection_engine":          _T2,   # §5.1: T2 soft
        "fallacy_detection_engine":          _T4,   # §5.1: T4 off
        "bias_detection_engine":             _T2,   # §5.1: T2 subject-activated
        "logic_trap_detection_engine":       _T4,
        "heuristic_bias_engine":             _T2,   # §5.1: T2 subject-activated
        "data_analysis_engine":              _T2,   # §5.1: T2 subject-activated (TEC/HIS/PRA)
        "pattern_identification_engine":     _T1,   # §5.1: T1
        "atomspace_engine":                  _T2,   # §5.1: T2 subject-activated
        "pln_engine":                        _T3,
        "logical_brain_engine":              _T3,
        "soar_production_engine":            _T4,
        "simulated_opposition_engine":       _T3,   # §5.1: T3 standby
        "socratic_reasoning_engine":         _T4,   # §5.1: T4 off
        "simulation_brain_engine":           _T3,
        "decision_making_engine":            _T4,   # §5.1: T4 off
        "ecan_engine":                       _T2,   # §5.1: T2 subject-activated
        "pattern_comparison_engine":         _T3,   # §5.1: T3 standby
        "strategic_decision_engine":         _T4,
        "reward_based_learning_engine":      _T3,   # §5.1: not in T1/T2 list
        "contextual_learning_engine":        _T1,   # §5.1: T1
        "recursive_learning_engine":         _T3,   # §5.1: T3 standby
        "uncertainty_pattern_engine":        _T1,   # §5.1: T1
        "neurochemical_homeostatic_engine":  _T1,   # §5.1: T1
        "memory_compression_engine":         _T4,   # §5.1: T4 off
        "retroactive_alignment_engine":      _T3,
    }),

    # ------------------------------------------------------------------
    # M5 — Independent Study (self-directed, exploratory)
    # Spec Part 4 §6.1: budget 14. No human present — E28, Intention
    # Map, Socratic, SimOpp, RewardBased all T4 OFF. Boredom detected
    # directly from NT dynamics via E27, not E28.
    # ------------------------------------------------------------------
    "M5": _fill_mode({
        "intention_map_engine":              _T4,   # §6.1: T4 off — no user
        "relevance_scoring_engine":          _T1,
        "input_relevance_evaluation_engine": _T1,
        "emotional_detection_engine":        _T4,   # §6.1: T4 off — no user to detect
        "contradiction_detection_engine":    _T2,   # §6.1: T2 soft, flagging only
        "paradox_detection_engine":          _T2,   # §6.1: T2
        "fallacy_detection_engine":          _T3,   # §6.1: T3 standby
        "bias_detection_engine":             _T3,   # §6.1: T3 standby
        "logic_trap_detection_engine":       _T3,   # §6.1: T3 standby
        "heuristic_bias_engine":             _T3,   # §6.1: T3 standby
        "data_analysis_engine":              _T1,   # §6.1: T1 always-on
        "pattern_identification_engine":     _T1,   # §6.1: T1
        "atomspace_engine":                  _T1,   # §6.1: T1
        "pln_engine":                        _T2,   # §6.1: T2 subject-activated (TEC/PHI)
        "logical_brain_engine":              _T3,
        "soar_production_engine":            _T3,
        "simulated_opposition_engine":       _T4,   # §6.1: T4 off
        "socratic_reasoning_engine":         _T4,   # §6.1: T4 off
        "simulation_brain_engine":           _T3,
        "decision_making_engine":            _T2,   # §6.1: T2
        "ecan_engine":                       _T1,   # §6.1: T1
        "pattern_comparison_engine":         _T1,   # §6.1: T1
        "strategic_decision_engine":         _T3,
        "reward_based_learning_engine":      _T4,   # §6.1: T4 off
        "contextual_learning_engine":        _T1,   # §6.1: T1
        "recursive_learning_engine":         _T2,   # §6.1: T2
        "uncertainty_pattern_engine":        _T1,   # §6.1: T1
        "neurochemical_homeostatic_engine":  _T1,   # §6.1: T1 — boredom detection
        "memory_compression_engine":         _T1,   # §6.1: T1
        "retroactive_alignment_engine":      _T3,
    }),

    # ------------------------------------------------------------------
    # Homework — Offline Processing & Integration (Part 5 §3.1)
    # Budget 22 (highest). No user present, no emotional feedback loop.
    # NT layer read-only (diagnostic for deficit profiling, NOT modulated).
    # Full adversarial weight on detection + dialectic engines.
    # ------------------------------------------------------------------
    "homework": _fill_mode({
        # T1 always-on — core analysis + processing engines (18)
        "data_analysis_engine":              _T1,   # §3.1: T1 — content decomposition
        "pattern_identification_engine":     _T1,   # §3.1: T1 — pattern analysis
        "pattern_comparison_engine":         _T1,   # §3.1: T1 — MemoryContrast proxy
        "contextual_learning_engine":        _T1,   # §3.1: T1 — context awareness
        "memory_compression_engine":         _T1,   # §3.1: T1 — compression policy
        "atomspace_engine":                  _T1,   # §3.1: T1 — knowledge substrate
        "ecan_engine":                       _T1,   # §3.1: T1 — attention allocation
        "contradiction_detection_engine":    _T1,   # §3.1: T1 — full adversarial
        "paradox_detection_engine":          _T1,   # §3.1: T1 — full adversarial
        "recursive_learning_engine":         _T1,   # §3.1: T1 — meta-learning + reflective
        "uncertainty_pattern_engine":        _T1,   # §3.1: T1 — uncertainty mapping
        "neurochemical_homeostatic_engine":  _T1,   # §3.1: T1 — read-only diagnostics
        "pln_engine":                        _T1,   # §3.1: T1 — confidence weighting
        "decision_making_engine":            _T1,   # §3.1: T1 — lesson validation
        "relevance_scoring_engine":          _T1,   # §3.1: T1 — entry scoring
        "retroactive_alignment_engine":      _T1,   # §3.1: T1 — alignment checks
        "reward_based_learning_engine":      _T1,   # §3.1: T1 — RPE integration
        "logical_brain_engine":              _T1,   # §3.1: T1 — logic verification

        # T2 base — dialectic + detection engines (6)
        "fallacy_detection_engine":          _T2,   # §3.1: T2 — fallacy sweep
        "bias_detection_engine":             _T2,   # §3.1: T2 — bias sweep
        "logic_trap_detection_engine":       _T2,   # §3.1: T2 — trap detection
        "simulated_opposition_engine":       _T2,   # §3.1: T2 — red-team stress-test
        "socratic_reasoning_engine":         _T2,   # §3.1: T2 — dialectic probe
        "heuristic_bias_engine":             _T2,   # §3.1: T2 — heuristic check

        # T3 standby — contextual / situational (4)
        "emotional_detection_engine":        _T3,   # §3.1: T3 — no user to detect
        "intention_map_engine":              _T3,   # §3.1: T3 — no live input
        "strategic_decision_engine":         _T3,   # §3.1: T3 — standby
        "simulation_brain_engine":           _T3,   # §3.1: T3 — standby

        # T4 off (1)
        "input_relevance_evaluation_engine": _T4,   # §3.1: T4 — no live input stream
        "soar_production_engine":            _T3,   # §3.1: T3 — standby
    }),

    # ------------------------------------------------------------------
    # Reflective — Meta-Reflective Identity + Learning Analysis (Appendix §4)
    # Budget 12. No user present, no emotional feedback loop.
    # E31 (reflective_learning) and E32 (reflective_identity) are T1.
    # Detection engines T3 standby (not adversarial — observational only).
    # Learning engines T1 (feed E31 analysis). Memory + homeostasis T1.
    # ------------------------------------------------------------------
    "reflective": _fill_mode({
        # T1 always-on — core reflective engines + knowledge substrate (12)
        "reflective_learning_engine":        _T1,   # §4.1: T1 — primary meta-learning
        "reflective_identity_engine":        _T1,   # §4.2: T1 — primary identity coherence
        "pattern_identification_engine":     _T1,   # T1 — pattern feed for E31
        "pattern_comparison_engine":         _T1,   # T1 — comparison feed for E31
        "contextual_learning_engine":        _T1,   # T1 — context awareness
        "recursive_learning_engine":         _T1,   # T1 — meta-learning feed
        "reward_based_learning_engine":      _T1,   # T1 — RPE history for E31
        "uncertainty_pattern_engine":        _T1,   # T1 — uncertainty for coherence
        "neurochemical_homeostatic_engine":  _T1,   # T1 — NT snapshots for E31/E32
        "memory_compression_engine":         _T1,   # T1 — compression for journal writes
        "retroactive_alignment_engine":      _T1,   # T1 — identity alignment scans
        "atomspace_engine":                  _T2,   # T2 — knowledge substrate

        # T2 supporting — analysis engines (4)
        "data_analysis_engine":              _T2,   # T2 — structural analysis
        "ecan_engine":                       _T2,   # T2 — attention allocation
        "pln_engine":                        _T2,   # T2 — confidence weighting
        "relevance_scoring_engine":          _T2,   # T2 — scoring

        # T3 standby — detection + dialectic (observational only)
        "contradiction_detection_engine":    _T3,   # T3 — not adversarial in reflective
        "paradox_detection_engine":          _T3,
        "fallacy_detection_engine":          _T3,
        "bias_detection_engine":             _T3,
        "logic_trap_detection_engine":       _T3,
        "heuristic_bias_engine":             _T3,
        "simulated_opposition_engine":       _T3,
        "socratic_reasoning_engine":         _T3,
        "logical_brain_engine":              _T3,

        # T4 off — not applicable in reflective mode
        "intention_map_engine":              _T4,   # no live user input
        "input_relevance_evaluation_engine": _T4,   # no live input stream
        "emotional_detection_engine":        _T4,   # no user to detect
        "decision_making_engine":            _T4,   # not decision-making
        "soar_production_engine":            _T4,
        "simulation_brain_engine":           _T4,
        "strategic_decision_engine":         _T4,
    }),
}


# ------------------------------------------------------------------
# Subject-specific tier adjustments
# ------------------------------------------------------------------

SUBJECT_PROMOTIONS: Dict[SubjectCategory, Dict[str, EngineTier]] = {
    SubjectCategory.TECHNICAL: {
        "logical_brain_engine":      _T1,
        "soar_production_engine":    _T2,
        "data_analysis_engine":      _T1,
        "pln_engine":                _T1,
    },
    SubjectCategory.SCIENTIFIC: {
        "data_analysis_engine":      _T1,
        "pattern_identification_engine": _T1,
        "pln_engine":                _T1,
        "simulation_brain_engine":   _T2,
    },
    SubjectCategory.PHILOSOPHICAL: {
        "socratic_reasoning_engine": _T1,
        "simulated_opposition_engine": _T1,
        "paradox_detection_engine":  _T1,
        "uncertainty_pattern_engine": _T1,
    },
    SubjectCategory.SOCIAL: {
        "emotional_detection_engine": _T1,
        "bias_detection_engine":     _T1,
        "contextual_learning_engine": _T1,
    },
    SubjectCategory.CREATIVE: {
        "simulation_brain_engine":   _T1,
        "ecan_engine":               _T1,
        "pattern_comparison_engine": _T1,
    },
    SubjectCategory.PRACTICAL: {
        "strategic_decision_engine": _T2,
        "decision_making_engine":    _T2,
        "soar_production_engine":    _T2,
    },
}

SUBJECT_DEMOTIONS: Dict[SubjectCategory, Dict[str, EngineTier]] = {
    SubjectCategory.SOCIAL: {
        "logical_brain_engine":      _T3,
        "soar_production_engine":    _T4,
    },
    SubjectCategory.CREATIVE: {
        "logic_trap_detection_engine": _T4,
        "soar_production_engine":    _T4,
    },
    SubjectCategory.PRACTICAL: {
        "paradox_detection_engine":  _T4,
        "socratic_reasoning_engine": _T4,
    },
}


# ------------------------------------------------------------------
# Budget caps per mode  (max T1+T2 engines)
# ------------------------------------------------------------------

BUDGET_CAPS: Dict[str, int] = {
    "regular": 20,
    "M1": 14,
    "M2": 16,
    "M3": 18,
    "M4": 12,
    "M5": 14,
    "homework": 22,  # Part 5 §3.1 — highest budget, full adversarial processing
    "reflective": 12,  # Appendix §4 — focused on E31/E32 + learning/memory engines
}


# ------------------------------------------------------------------
# EngineTier → engine_weight conversion
# ------------------------------------------------------------------

TIER_TO_WEIGHT: Dict[EngineTier, float] = {
    EngineTier.T1: 1.0,
    EngineTier.T2: 1.0,
    EngineTier.T3: 0.5,
    EngineTier.T4: 0.0,
}


# ------------------------------------------------------------------
# EngineToolkit
# ------------------------------------------------------------------

class EngineToolkit:
    """Resolves the final engine tier + weight for a (mode, subject) pair.

    Usage
    -----
    >>> tk = EngineToolkit()
    >>> tiers = tk.resolve("M3", SubjectCategory.PHILOSOPHICAL)
    >>> weights = tk.tiers_to_weights(tiers)
    """

    def resolve(
        self,
        mode: str,
        subject: SubjectCategory = SubjectCategory.MIXED,
    ) -> Dict[str, EngineTier]:
        """Resolve final engine tiers for the given mode and subject.

        Parameters
        ----------
        mode : str
            "regular", "M1".."M5".  Unknown modes fall back to "regular".
        subject : SubjectCategory
            Detected subject category.

        Returns
        -------
        Dict[str, EngineTier]
            engine_name → final tier.
        """
        mode_key = mode if mode in BASE_TIERS else "regular"
        tiers = dict(BASE_TIERS[mode_key])  # copy

        # Rule 2: subject promotions (only promote — never demote via this)
        if subject in SUBJECT_PROMOTIONS:
            for eng, tier in SUBJECT_PROMOTIONS[subject].items():
                if eng in tiers and tier.value < tiers[eng].value:
                    tiers[eng] = tier

        # Rule 3: subject demotions (only demote — never promote via this)
        if subject in SUBJECT_DEMOTIONS:
            for eng, tier in SUBJECT_DEMOTIONS[subject].items():
                if eng in tiers and tier.value > tiers[eng].value:
                    tiers[eng] = tier

        # Rule 4: phantom engines forced to T4
        for pe in PHANTOM_ENGINES:
            if pe in tiers:
                tiers[pe] = _T4

        # Rule 5-6: budget cap enforcement
        budget = BUDGET_CAPS.get(mode_key, 20)
        active = [
            (eng, t) for eng, t in tiers.items()
            if t in (_T1, _T2)
        ]
        if len(active) > budget:
            # Sort by tier (T2 before T1 = demote T2 first), then name
            t2_engines = sorted(
                [eng for eng, t in active if t == _T2],
            )
            excess = len(active) - budget
            for eng in t2_engines[:excess]:
                tiers[eng] = _T3

        return tiers

    @staticmethod
    def tiers_to_weights(tiers: Dict[str, EngineTier]) -> Dict[str, float]:
        """Convert tier dict to engine_weights dict for InputBundle.

        Parameters
        ----------
        tiers : Dict[str, EngineTier]
            From resolve().

        Returns
        -------
        Dict[str, float]
            engine_name → weight (1.0 / 0.5 / 0.0).
        """
        return {eng: TIER_TO_WEIGHT[t] for eng, t in tiers.items()}

    @staticmethod
    def tiers_to_weights_by_id(tiers: Dict[str, EngineTier]) -> Dict[str, float]:
        """Like tiers_to_weights but keyed by engine number (str).

        The existing AnswerPipeline uses str(engine_number) as weight keys.
        """
        name_to_id = {v: str(k) for k, v in ALL_ENGINE_IDS.items()}
        weights: Dict[str, float] = {}
        for eng_name, tier in tiers.items():
            eng_id = name_to_id.get(eng_name)
            if eng_id is not None:
                weights[eng_id] = TIER_TO_WEIGHT[tier]
        return weights
