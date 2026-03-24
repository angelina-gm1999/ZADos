from .bands import OscillationState
from .band_associations import (
    NT_BAND_ASSOCIATIONS,
    BAND_MODULATION_DEFAULTS,
    get_primary_bands,
    get_secondary_bands,
    get_all_associated_bands,
    get_nts_for_band,
)
from .oscillation_modulation import (
    modulate_K_d,
    modulate_K_d_multiband,
    modulate_release,
    modulate_noise,
    modulate_noise_multiband,
    modulate_reuptake,
    modulate_tonic_baseline,
    compute_g_chi,
    compute_effective_signaling_proxy,
)
from .transition_modulation import (
    TransitionBandSpec,
    ThresholdBandSpec,
    compute_transition_multiplier,
    modulate_threshold,
)
from .generators import derive_oscillation_state, DEFAULT_BAND_DERIVATION_RULES

# modulation_links is LEGACY — only used by the old batch Dopamine class
# and SimulationRunner. New code should use oscillation_modulation.py instead.

__all__ = [
    # State
    "OscillationState",
    # Band associations (authoritative NT→band map)
    "NT_BAND_ASSOCIATIONS",
    "BAND_MODULATION_DEFAULTS",
    "get_primary_bands",
    "get_secondary_bands",
    "get_all_associated_bands",
    "get_nts_for_band",
    # Pure modulation functions (PDF Appendix I)
    "modulate_K_d",
    "modulate_K_d_multiband",
    "modulate_release",
    "modulate_noise",
    "modulate_noise_multiband",
    "modulate_reuptake",
    "modulate_tonic_baseline",
    "compute_g_chi",
    "compute_effective_signaling_proxy",
    # Transition rate + threshold modulation (Appendix H.7)
    "TransitionBandSpec",
    "ThresholdBandSpec",
    "compute_transition_multiplier",
    "modulate_threshold",
    # State-derived oscillation generator (Appendix I)
    "derive_oscillation_state",
    "DEFAULT_BAND_DERIVATION_RULES",
]
