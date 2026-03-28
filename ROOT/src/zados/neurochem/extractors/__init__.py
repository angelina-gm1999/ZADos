"""
ZADOS Stochastic Extractors Package.

Bridges cognitive/emotional evaluation to the neurochemical layer via:
- Evaluation vector assembly from reward domain subscores (Extractor 1)
- Stochastic impulse generators and reactivity matrix (Extractor 2)
- Leaky-integrator regulatory modulation with oscillatory envelope (Extractor 3)
- Emotion saturation tracking with 4M/4R split (Extractor 4M/4R)
- Urgency threshold forecasting with reactive/modulatory outputs (Extractor 5)
- Top-level orchestrator for cognitive engine integration
"""

# Extractor 1: Evaluation Vector
from .evaluation_vector import (
    EvaluationAxisConfig,
    EvaluationVectorConfig,
    DEFAULT_EVALUATION_CONFIG,
    extract_axis_value,
    inject_noise,
    assemble_evaluation_vector,
)

# Extractor 2: Stochastic Impulse + Reactivity Matrix
from .stochastic_impulse import (
    sample_gamma_impulse,
    sample_poisson_impulse,
    sample_lognormal_impulse,
    sample_impulse,
)
from .reactivity_matrix import (
    ReactivityEntry,
    ReactivityMatrixConfig,
    DEFAULT_REACTIVITY_CONFIG,
    apply_threshold_gating,
    compute_stochastic_burst_deltas,
    burst_deltas_to_modulation_signals,
)

# Extractor 3: Leaky Integrator + Regulatory Modulator
from .leaky_integrator import (
    LeakyIntegratorState,
    leaky_integrator_step,
    exponential_moving_average_step,
    batch_leaky_integrator_step,
)
from .regulatory_modulator import (
    RegulatoryPathwayConfig,
    RegulatoryModulatorConfig,
    RegulatoryModulatorState,
    OscillationEnvelopeRule,
    DEFAULT_REGULATORY_CONFIG,
    DEFAULT_ENVELOPE_RULES,
    step_regulatory_modulator,
    compute_oscillation_envelope,
)

# Extractor 4: Emotion Tracker + 4M/4R Split
from .emotion_tracker import (
    EmotionTrackerConfig,
    EmotionTrackerState,
    DEFAULT_EMOTION_TRACKER_CONFIGS,
    step_emotion_tracker,
    get_dominant_emotion,
    get_saturation,
    get_emotion_saturations,
)
from .emotion_splitter import (
    EmotionSplitConfig,
    DEFAULT_EMOTION_SPLIT_CONFIGS,
    compute_modulatory_adjustments,
    compute_reactive_signals,
    split_emotion_effects,
)

# Extractor 5: Urgency Forecast
from .urgency_forecast import (
    UrgencyAxisSourceDef,
    UrgencyAxisConfig,
    UrgencyForecastConfig,
    DEFAULT_URGENCY_FORECAST_CONFIG,
    UrgencyForecastState,
    compute_urgency_axis_value,
    forecast_peak,
    detect_breach,
    compute_urgency_risk,
    compute_reactive_burst,
    compute_modulatory_feedback,
    step_urgency_forecast,
)

# Orchestrator
from .extractor_orchestrator import (
    ExtractorState,
    ExtractorResult,
    ExtractorOrchestrator,
)

__all__ = [
    # Extractor 1: Evaluation Vector
    "EvaluationAxisConfig",
    "EvaluationVectorConfig",
    "DEFAULT_EVALUATION_CONFIG",
    "extract_axis_value",
    "inject_noise",
    "assemble_evaluation_vector",
    # Extractor 2: Stochastic Impulse
    "sample_gamma_impulse",
    "sample_poisson_impulse",
    "sample_lognormal_impulse",
    "sample_impulse",
    # Extractor 2: Reactivity Matrix
    "ReactivityEntry",
    "ReactivityMatrixConfig",
    "DEFAULT_REACTIVITY_CONFIG",
    "apply_threshold_gating",
    "compute_stochastic_burst_deltas",
    "burst_deltas_to_modulation_signals",
    # Extractor 3: Leaky Integrator
    "LeakyIntegratorState",
    "leaky_integrator_step",
    "exponential_moving_average_step",
    "batch_leaky_integrator_step",
    # Extractor 3: Regulatory Modulator
    "RegulatoryPathwayConfig",
    "RegulatoryModulatorConfig",
    "RegulatoryModulatorState",
    "OscillationEnvelopeRule",
    "DEFAULT_REGULATORY_CONFIG",
    "DEFAULT_ENVELOPE_RULES",
    "step_regulatory_modulator",
    "compute_oscillation_envelope",
    # Extractor 4: Emotion Tracker
    "EmotionTrackerConfig",
    "EmotionTrackerState",
    "DEFAULT_EMOTION_TRACKER_CONFIGS",
    "step_emotion_tracker",
    "get_dominant_emotion",
    "get_saturation",
    "get_emotion_saturations",
    # Extractor 4: Emotion Splitter
    "EmotionSplitConfig",
    "DEFAULT_EMOTION_SPLIT_CONFIGS",
    "compute_modulatory_adjustments",
    "compute_reactive_signals",
    "split_emotion_effects",
    # Extractor 5: Urgency Forecast
    "UrgencyAxisSourceDef",
    "UrgencyAxisConfig",
    "UrgencyForecastConfig",
    "DEFAULT_URGENCY_FORECAST_CONFIG",
    "UrgencyForecastState",
    "compute_urgency_axis_value",
    "forecast_peak",
    "detect_breach",
    "compute_urgency_risk",
    "compute_reactive_burst",
    "compute_modulatory_feedback",
    "step_urgency_forecast",
    # Orchestrator
    "ExtractorState",
    "ExtractorResult",
    "ExtractorOrchestrator",
]
