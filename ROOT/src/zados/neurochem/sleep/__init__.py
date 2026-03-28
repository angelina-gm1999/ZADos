"""
ZADOS Neurochemical Sleep Extension.

Provides prescribed state vectors, pharmacodynamic transitions,
containment monitoring, and neurosymbolic triggers for the three
sleep sub-states: TRIAGE, REM_PROCESSING, and COMPUTATIONAL_DREAMING.
"""

from .state_vectors import (
    SleepPhase,
    SleepNTStateVector,
    TRIAGE_STATE_VECTOR,
    REM_PROCESSING_STATE_VECTOR,
    DREAM_STATE_VECTOR,
    SLEEP_STATE_VECTORS,
)
from .transitions import (
    TransitionConfig,
    DEFAULT_TRANSITION_CONFIG,
    compute_transition_step,
    transition_nt_baselines,
    transition_osc_config,
    check_triage_to_rem_conditions,
    check_rem_to_dream_conditions,
)
from .containment import (
    check_containment,
    check_dream_state_validity,
    check_ne_upregulation_cap,
)
from .state_manager import SleepNeurochemicalStateManager
from .sleep_triggers import DEFAULT_SLEEP_TRIGGERS

__all__ = [
    # State vectors
    "SleepPhase",
    "SleepNTStateVector",
    "TRIAGE_STATE_VECTOR",
    "REM_PROCESSING_STATE_VECTOR",
    "DREAM_STATE_VECTOR",
    "SLEEP_STATE_VECTORS",
    # Transitions
    "TransitionConfig",
    "DEFAULT_TRANSITION_CONFIG",
    "compute_transition_step",
    "transition_nt_baselines",
    "transition_osc_config",
    "check_triage_to_rem_conditions",
    "check_rem_to_dream_conditions",
    # Containment
    "check_containment",
    "check_dream_state_validity",
    "check_ne_upregulation_cap",
    # State manager
    "SleepNeurochemicalStateManager",
    # Triggers
    "DEFAULT_SLEEP_TRIGGERS",
]
