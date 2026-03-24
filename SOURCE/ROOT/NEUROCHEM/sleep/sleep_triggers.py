"""
Sleep-mode neurosymbolic triggers (Spec §7).

Six TriggerDefinition instances for sleep entry, phase transitions,
dream events, containment monitoring, consolidation windows, and
sleep exit. Reuses the existing trigger evaluation infrastructure
from neurosymbolic.triggers.
"""

from __future__ import annotations

from typing import List

from zados.neurochem.neurosymbolic.triggers import TriggerDefinition


# §7.1 — Sleep Mode Entry
SLEEP_ENTRY_TRIGGER = TriggerDefinition(
    condition_str="GABA > 0.55 AND NE < 0.45 AND histamine < 0.35",
    actions=(
        "GABA~GABA_A:inhibition_up",
        "NE~NE_alpha1:binding_down",
    ),
    activate_mode="TriageMode",
)

# §7.2 — NREM-REM Transition (REM Processing -> Dream Mode)
NREM_REM_TRANSITION_TRIGGER = TriggerDefinition(
    condition_str=(
        "5HT < 0.40 AND ACh > 0.60 "
        "AND phi_delta_sigma < 0.20 AND phi_theta > 0.65"
    ),
    actions=(
        "INT_5HT1A",
        "ACh~mAChR_M1:activation_up",
        "CB1~CB1:disinhibition_up",
    ),
    activate_mode="DreamMode",
)

# §7.3 — Dream Scene Shift (PGO-Analog Event)
DREAM_SCENE_SHIFT_TRIGGER = TriggerDefinition(
    condition_str="ACh > 0.75 AND phi_theta_gamma > 0.50",
    actions=(
        "GLU~NMDA:plasticity_up",
        "DA~D3:salience_up",
    ),
    activate_mode="DreamSceneShift",
)

# §7.4 — Dreambox Containment Check
CONTAINMENT_CHECK_TRIGGER = TriggerDefinition(
    condition_str="S_GABA_A < 0.55",
    actions=("ALERT_DreamboxIntegrity",),
    activate_mode="DreamAbort",
)

# §7.5 — Consolidation Window Open (SWR analog during REM Processing)
CONSOLIDATION_WINDOW_TRIGGER = TriggerDefinition(
    condition_str="phi_delta_sigma > 0.55 AND S_GLU_NMDA > 0.55",
    actions=(
        "GLU~NMDA:plasticity_up",
        "AMPA~AMPA:burst_up",
    ),
    activate_mode="ConsolidationReplayWindow",
)

# §7.6 — Sleep Exit
SLEEP_EXIT_TRIGGER = TriggerDefinition(
    condition_str="SleepProcessComplete == 1",
    actions=(
        "NE~NE_beta1:sensitization_up",
        "histamine~H1:arousal_up",
    ),
    activate_mode="WakingReturn",
)

# All sleep triggers as an ordered list
DEFAULT_SLEEP_TRIGGERS: List[TriggerDefinition] = [
    SLEEP_ENTRY_TRIGGER,
    NREM_REM_TRANSITION_TRIGGER,
    DREAM_SCENE_SHIFT_TRIGGER,
    CONTAINMENT_CHECK_TRIGGER,
    CONSOLIDATION_WINDOW_TRIGGER,
    SLEEP_EXIT_TRIGGER,
]
