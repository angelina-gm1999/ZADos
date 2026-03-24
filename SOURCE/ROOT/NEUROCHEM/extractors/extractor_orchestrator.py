"""
Extractor Orchestrator — Top-level sequencer for all stochastic extractors.

Cognitive engines call ``orchestrator.step()`` once per tick. The orchestrator
sequences all 4 extractors and returns a single ``ExtractorResult`` containing
everything needed to drive the neurochemical engine:

1. Assemble evaluation vector E(t) from domain results
2. Step emotion tracker from emotion inputs
3. Split emotions into 4M (modulatory) + 4R (reactive) pathways
4. Add 4M modulatory adjustments to E(t)
5. Step regulatory modulator with adjusted E(t)
6. Compute oscillatory envelope modulation
7. Compute stochastic burst deltas from reactivity matrix
8. Merge 4R reactive signals via emotion_profile_to_signals()
9. Package everything into ExtractorResult

Usage
-----
>>> from zados.neurochem.extractors import ExtractorOrchestrator
>>> orchestrator = ExtractorOrchestrator(rng=engine.rng)
>>> result = orchestrator.step(domain_results, emotion_inputs, osc_state, dt=0.01)
>>> engine.step(result.modulation_signals)
>>> engine.apply_feedback(result.feedback_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from zados.neurochem.extractors.evaluation_vector import (
    EvaluationVectorConfig,
    DEFAULT_EVALUATION_CONFIG,
    assemble_evaluation_vector,
)
from zados.neurochem.extractors.stochastic_impulse import sample_impulse
from zados.neurochem.extractors.reactivity_matrix import (
    ReactivityMatrixConfig,
    DEFAULT_REACTIVITY_CONFIG,
    compute_stochastic_burst_deltas,
    burst_deltas_to_modulation_signals,
)
from zados.neurochem.extractors.leaky_integrator import LeakyIntegratorState
from zados.neurochem.extractors.regulatory_modulator import (
    RegulatoryModulatorConfig,
    RegulatoryModulatorState,
    OscillationEnvelopeRule,
    DEFAULT_REGULATORY_CONFIG,
    DEFAULT_ENVELOPE_RULES,
    step_regulatory_modulator,
    compute_oscillation_envelope,
)
from zados.neurochem.extractors.emotion_tracker import (
    EmotionTrackerConfig,
    EmotionTrackerState,
    DEFAULT_EMOTION_TRACKER_CONFIGS,
    step_emotion_tracker,
    get_dominant_emotion,
    get_emotion_saturations,
)
from zados.neurochem.extractors.emotion_splitter import (
    EmotionSplitConfig,
    DEFAULT_EMOTION_SPLIT_CONFIGS,
    split_emotion_effects,
)
from zados.neurochem.extractors.urgency_forecast import (
    UrgencyForecastConfig,
    UrgencyForecastState,
    DEFAULT_URGENCY_FORECAST_CONFIG,
    step_urgency_forecast,
)
from zados.neurochem.utils.emotion_interface import emotion_profile_to_signals
from zados.neurochem.state.oscillation_state import OscillationState


# =====================================================================
# State
# =====================================================================

@dataclass
class ExtractorState:
    """
    Combined state for all extractors.

    Attributes
    ----------
    prev_evaluation_vector : dict, optional
        E(t-dt) from previous tick (for volatility computation).
    regulatory_state : RegulatoryModulatorState, optional
        Current regulatory modulator integrator states.
    emotion_tracker_state : EmotionTrackerState, optional
        Current emotion saturation integrator states.
    """
    prev_evaluation_vector: Optional[Dict[str, float]] = None
    regulatory_state: Optional[RegulatoryModulatorState] = None
    emotion_tracker_state: Optional[EmotionTrackerState] = None
    urgency_forecast_state: Optional[UrgencyForecastState] = None

    @classmethod
    def initialize(
        cls,
        regulatory_config: Optional[RegulatoryModulatorConfig] = None,
        emotion_ids: Optional[List[str]] = None,
        urgency_forecast_config: Optional[UrgencyForecastConfig] = None,
    ) -> ExtractorState:
        """
        Create initial state with all integrators at baseline.

        Parameters
        ----------
        regulatory_config : RegulatoryModulatorConfig, optional
            Config for regulatory modulator. Defaults to DEFAULT_REGULATORY_CONFIG.
        emotion_ids : list of str, optional
            Emotions to track. Defaults to all 12.
        urgency_forecast_config : UrgencyForecastConfig, optional
            Config for urgency forecast. Defaults to DEFAULT_URGENCY_FORECAST_CONFIG.

        Returns
        -------
        ExtractorState
            Fully initialized state.
        """
        reg_cfg = regulatory_config or DEFAULT_REGULATORY_CONFIG
        urg_cfg = urgency_forecast_config or DEFAULT_URGENCY_FORECAST_CONFIG
        return cls(
            prev_evaluation_vector=None,
            regulatory_state=RegulatoryModulatorState.from_config(reg_cfg),
            emotion_tracker_state=EmotionTrackerState.from_emotion_ids(emotion_ids),
            urgency_forecast_state=UrgencyForecastState.from_config(urg_cfg),
        )

    def as_dict(self) -> dict:
        """Export to dictionary for checkpointing."""
        return {
            "prev_evaluation_vector": self.prev_evaluation_vector,
            "regulatory_state": (
                self.regulatory_state.as_dict()
                if self.regulatory_state else None
            ),
            "emotion_tracker_state": (
                self.emotion_tracker_state.as_dict()
                if self.emotion_tracker_state else None
            ),
            "urgency_forecast_state": (
                self.urgency_forecast_state.as_dict()
                if self.urgency_forecast_state else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExtractorState:
        """Restore from dictionary."""
        reg = data.get("regulatory_state")
        emo = data.get("emotion_tracker_state")
        urg = data.get("urgency_forecast_state")
        return cls(
            prev_evaluation_vector=data.get("prev_evaluation_vector"),
            regulatory_state=(
                RegulatoryModulatorState.from_dict(reg) if reg else None
            ),
            emotion_tracker_state=(
                EmotionTrackerState.from_dict(emo) if emo else None
            ),
            urgency_forecast_state=(
                UrgencyForecastState.from_dict(urg) if urg else None
            ),
        )


# =====================================================================
# Result
# =====================================================================

@dataclass
class ExtractorResult:
    """
    Output of one orchestrator tick.

    Attributes
    ----------
    evaluation_vector : dict
        Final E(t) after all adjustments. Maps axis → float [0,1].
    modulation_signals : dict
        Merged signals for engine.step(). {nt_name: {signal_key: value}}.
    feedback_params : dict
        For engine.apply_feedback(). Same format as compute_reward_feedback().
    oscillation_update : OscillationState, optional
        New oscillation state if envelope modulation was computed.
    emotion_saturations : dict
        Per-emotion saturation levels.
    dominant_emotion : tuple
        (emotion_id, saturation_value).
    burst_deltas : dict
        Per-NT stochastic burst deltas. {nt_name: float}.
    """
    evaluation_vector: Dict[str, float] = field(default_factory=dict)
    modulation_signals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feedback_params: Dict = field(default_factory=dict)
    oscillation_update: Optional[OscillationState] = None
    emotion_saturations: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: Tuple[str, float] = ("none", 0.0)
    burst_deltas: Dict[str, float] = field(default_factory=dict)
    urgency_risk: float = 0.0


# =====================================================================
# Orchestrator
# =====================================================================

class ExtractorOrchestrator:
    """
    Top-level orchestrator for all stochastic extractors.

    Sequences Extractor 1 (evaluation vector), Extractor 2 (reactivity
    matrix / stochastic impulse), Extractor 3 (regulatory modulator +
    oscillation envelope), and Extractor 4M/4R (emotion tracker + split).

    Dual-pathway design: The evaluation vector E(t) and emotion recipes
    are independent input streams. E(t) is assembled from reward domain
    subscores; emotion saturations are tracked separately from external
    inputs. The two streams merge at signal level: 4M modulatory
    adjustments shift E(t) before regulatory processing, while 4R
    reactive signals are added to the burst-delta modulation signals
    via emotion_profile_to_signals().

    Parameters
    ----------
    rng : np.random.Generator, optional
        Numpy RNG for stochastic sampling. If None, stochastic components
        still run but with non-deterministic randomness.
    evaluation_config : EvaluationVectorConfig, optional
        Config for evaluation vector assembler.
    reactivity_config : ReactivityMatrixConfig, optional
        Config for reactivity matrix.
    regulatory_config : RegulatoryModulatorConfig, optional
        Config for regulatory modulator.
    envelope_rules : tuple of OscillationEnvelopeRule, optional
        Oscillation envelope modulation rules.
    emotion_tracker_configs : dict, optional
        Emotion tracker per-emotion configs.
    emotion_split_configs : dict, optional
        4M/4R split configs.
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        evaluation_config: Optional[EvaluationVectorConfig] = None,
        reactivity_config: Optional[ReactivityMatrixConfig] = None,
        regulatory_config: Optional[RegulatoryModulatorConfig] = None,
        envelope_rules: Optional[Tuple[OscillationEnvelopeRule, ...]] = None,
        emotion_tracker_configs: Optional[Dict[str, EmotionTrackerConfig]] = None,
        emotion_split_configs: Optional[Dict[str, EmotionSplitConfig]] = None,
        emotion_ids: Optional[List[str]] = None,
        urgency_forecast_config: Optional[UrgencyForecastConfig] = None,
    ):
        self.rng = rng
        self.evaluation_config = evaluation_config or DEFAULT_EVALUATION_CONFIG
        self.reactivity_config = reactivity_config or DEFAULT_REACTIVITY_CONFIG
        self.regulatory_config = regulatory_config or DEFAULT_REGULATORY_CONFIG
        self.envelope_rules = envelope_rules or DEFAULT_ENVELOPE_RULES
        self.emotion_tracker_configs = emotion_tracker_configs or DEFAULT_EMOTION_TRACKER_CONFIGS
        self.emotion_split_configs = emotion_split_configs or DEFAULT_EMOTION_SPLIT_CONFIGS
        self.urgency_forecast_config = urgency_forecast_config or DEFAULT_URGENCY_FORECAST_CONFIG

        # Mutable state
        self._state = ExtractorState.initialize(
            regulatory_config=self.regulatory_config,
            emotion_ids=emotion_ids,
            urgency_forecast_config=self.urgency_forecast_config,
        )

    @property
    def state(self) -> ExtractorState:
        """Current extractor state (for checkpointing)."""
        return self._state

    @state.setter
    def state(self, new_state: ExtractorState):
        """Set state (for restoring from checkpoint)."""
        self._state = new_state

    def step(
        self,
        domain_results: Dict,
        emotion_inputs: Optional[Dict[str, float]] = None,
        current_oscillations: Optional[OscillationState] = None,
        dt: float = 0.01,
    ) -> ExtractorResult:
        """
        Execute one full extractor tick.

        Sequence:
        1. Assemble E(t) from domain_results
        2. Step emotion tracker
        3. Split emotions → 4M + 4R
        4. Add 4M modulatory adjustments to E(t)
        4.5. Urgency forecast (smooth → predict → breach → reactive/modulatory)
        5. Step regulatory modulator with adjusted E(t)
        5.5. Merge urgency feedback into regulatory feedback
        6. Compute oscillation envelope
        7. Compute stochastic burst deltas
        7.5. Merge urgency burst deltas
        8. Compute 4R reactive signals
        9. Merge all into ExtractorResult

        Parameters
        ----------
        domain_results : dict
            Maps domain_name → RewardDomainResult.
        emotion_inputs : dict, optional
            Maps emotion_id → strength [0,1]. None → empty.
        current_oscillations : OscillationState, optional
            Current oscillation state for envelope modulation.
        dt : float
            Timestep.

        Returns
        -------
        ExtractorResult
            Full result with all signals, feedback, and state info.
        """
        if emotion_inputs is None:
            emotion_inputs = {}

        # ---- 1. Assemble evaluation vector E(t) ----
        evaluation_vector = assemble_evaluation_vector(
            domain_results,
            config=self.evaluation_config,
            rng=self.rng,
        )

        # ---- 2. Step emotion tracker ----
        if self._state.emotion_tracker_state is not None:
            self._state.emotion_tracker_state = step_emotion_tracker(
                self._state.emotion_tracker_state,
                emotion_inputs,
                dt,
                configs=self.emotion_tracker_configs,
            )
        emotion_saturations = (
            get_emotion_saturations(self._state.emotion_tracker_state)
            if self._state.emotion_tracker_state else {}
        )
        dominant_emotion = (
            get_dominant_emotion(self._state.emotion_tracker_state)
            if self._state.emotion_tracker_state else ("none", 0.0)
        )

        # ---- 3. Split emotions → 4M + 4R ----
        modulatory_adjustments: Dict[str, float] = {}
        reactive_profile: Dict[str, float] = {}
        if self._state.emotion_tracker_state is not None:
            modulatory_adjustments, reactive_profile = split_emotion_effects(
                self._state.emotion_tracker_state,
                self.emotion_split_configs,
            )

        # ---- 4. Add 4M modulatory adjustments to E(t) ----
        adjusted_eval = dict(evaluation_vector)
        for axis, adj in modulatory_adjustments.items():
            if axis in adjusted_eval:
                adjusted_eval[axis] = max(0.0, min(1.0, adjusted_eval[axis] + adj))
            else:
                adjusted_eval[axis] = max(0.0, min(1.0, adj))

        # ---- 4.5. Urgency forecast ----
        urgency_risk = 0.0
        urgency_burst_deltas: Dict[str, float] = {}
        urgency_feedback: Dict[str, Dict] = {"neurotransmitters": {}, "receptors": {}}
        if self._state.urgency_forecast_state is not None:
            (self._state.urgency_forecast_state,
             urgency_risk,
             urgency_burst_deltas,
             urgency_feedback) = step_urgency_forecast(
                self._state.urgency_forecast_state,
                adjusted_eval,
                dt,
                config=self.urgency_forecast_config,
                rng=self.rng,
            )

        # ---- 5. Step regulatory modulator ----
        feedback_params: Dict = {"neurotransmitters": {}, "receptors": {}}
        if self._state.regulatory_state is not None:
            self._state.regulatory_state, feedback_params = step_regulatory_modulator(
                self._state.regulatory_state,
                adjusted_eval,
                self.regulatory_config,
                dt,
            )

        # ---- 5.5. Merge urgency feedback into regulatory feedback ----
        # Multiplier params multiply; delta/additive params add.
        for section in ("neurotransmitters", "receptors"):
            for key, params in urgency_feedback.get(section, {}).items():
                if key not in feedback_params[section]:
                    feedback_params[section][key] = {}
                for k, v in params.items():
                    if k.endswith("_multiplier"):
                        feedback_params[section][key][k] = (
                            feedback_params[section][key].get(k, 1.0) * v
                        )
                    else:
                        feedback_params[section][key][k] = (
                            feedback_params[section][key].get(k, 0.0) + v
                        )

        # ---- 6. Compute oscillation envelope ----
        oscillation_update = None
        if (
            current_oscillations is not None
            and self._state.regulatory_state is not None
        ):
            oscillation_update = compute_oscillation_envelope(
                self._state.regulatory_state,
                current_oscillations,
                self.envelope_rules,
            )

        # ---- 7. Compute stochastic burst deltas ----
        burst_deltas = compute_stochastic_burst_deltas(
            adjusted_eval,
            self._state.prev_evaluation_vector,
            dt,
            config=self.reactivity_config,
            rng=self.rng,
        )

        # ---- 7.5. Merge urgency burst deltas ----
        for nt_name, delta in urgency_burst_deltas.items():
            burst_deltas[nt_name] = burst_deltas.get(nt_name, 0.0) + delta

        # ---- 8. Compute 4R reactive signals ----
        reactive_signals = {}
        if reactive_profile:
            reactive_signals = emotion_profile_to_signals(reactive_profile)

        # ---- 9. Merge all modulation signals ----
        # Start with burst deltas → modulation signals format
        modulation_signals = burst_deltas_to_modulation_signals(burst_deltas)
        # Merge in reactive emotion signals
        for nt_name, signals in reactive_signals.items():
            if nt_name not in modulation_signals:
                modulation_signals[nt_name] = {}
            for signal_key, value in signals.items():
                if signal_key in modulation_signals[nt_name]:
                    modulation_signals[nt_name][signal_key] += value
                else:
                    modulation_signals[nt_name][signal_key] = value

        # Store current E(t) as previous for next tick
        self._state.prev_evaluation_vector = adjusted_eval

        return ExtractorResult(
            evaluation_vector=adjusted_eval,
            modulation_signals=modulation_signals,
            feedback_params=feedback_params,
            oscillation_update=oscillation_update,
            emotion_saturations=emotion_saturations,
            dominant_emotion=dominant_emotion,
            burst_deltas=burst_deltas,
            urgency_risk=urgency_risk,
        )
