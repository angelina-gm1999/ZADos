"""
Prescribed neurochemical state vectors for sleep sub-states.

Each sleep phase (TRIAGE, REM_PROCESSING, DREAM) is defined by a
neurochemical state vector that overrides waking tonic baselines.
These are NOT live-computed — they are SET at mode entry and held
constant, with concept-specific deviations applied on top.

Values from ZADOS Neurochemical Sleep Spec §2.1–§2.3, §4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class SleepPhase(Enum):
    """Sleep processing phase."""
    WAKING = "waking"
    TRIAGE = "triage"
    REM_PROCESSING = "rem_processing"
    DREAM = "dream"


@dataclass(frozen=True)
class SleepNTStateVector:
    """Prescribed NT tonic baselines for a sleep phase.

    Attributes
    ----------
    nt_baselines : dict
        Map of NT name → C_tonic target value (all [0, 1]).
    oscillatory_config : dict
        Map of band name → amplitude target (all [0, 1]).
    """
    nt_baselines: Dict[str, float]
    oscillatory_config: Dict[str, float]


# =====================================================================
# Triage Phase — Light NREM Analog (N1-N2)  [Spec §2.1]
# =====================================================================

TRIAGE_NT_BASELINES: Dict[str, float] = {
    "ACh":       0.45,
    "NE":        0.40,
    "5HT":       0.55,
    "DA":        0.40,
    "GABA":      0.60,
    "GLU":       0.45,
    "CB1":       0.45,
    "MOR":       0.45,
    "CRH":       0.30,
    "cortisol":  0.30,
    "histamine": 0.30,
    "OXT":       0.50,
}

TRIAGE_OSC_CONFIG: Dict[str, float] = {
    "delta": 0.35,
    "theta": 0.40,
    "alpha": 0.50,
    "beta":  0.35,
    "gamma": 0.30,
    "sigma": 0.50,
}

TRIAGE_STATE_VECTOR = SleepNTStateVector(
    nt_baselines=TRIAGE_NT_BASELINES,
    oscillatory_config=TRIAGE_OSC_CONFIG,
)

# =====================================================================
# REM Processing Phase — Deep NREM / SWS Analog (N2-N3)  [Spec §2.2]
# =====================================================================

REM_PROCESSING_NT_BASELINES: Dict[str, float] = {
    "ACh":       0.20,
    "NE":        0.25,
    "5HT":       0.60,
    "DA":        0.30,
    "GABA":      0.80,
    "GLU":       0.60,
    "CB1":       0.50,
    "MOR":       0.50,
    "CRH":       0.15,
    "cortisol":  0.15,
    "histamine": 0.15,
    "OXT":       0.50,
}

REM_PROCESSING_OSC_CONFIG: Dict[str, float] = {
    "delta": 0.80,
    "theta": 0.15,
    "alpha": 0.45,
    "beta":  0.25,
    "gamma": 0.25,
    "sigma": 0.70,
}

REM_PROCESSING_STATE_VECTOR = SleepNTStateVector(
    nt_baselines=REM_PROCESSING_NT_BASELINES,
    oscillatory_config=REM_PROCESSING_OSC_CONFIG,
)

# =====================================================================
# Computational Dreaming Phase — REM Analog  [Spec §2.3]
# =====================================================================

DREAM_NT_BASELINES: Dict[str, float] = {
    "ACh":       0.85,
    "NE":        0.05,
    "5HT":       0.05,
    "DA":        0.65,
    "GABA":      0.65,
    "GLU":       0.70,
    "CB1":       0.75,
    "MOR":       0.60,
    "CRH":       0.05,
    "cortisol":  0.05,
    "histamine": 0.05,
    "OXT":       0.55,
}

DREAM_OSC_CONFIG: Dict[str, float] = {
    "delta": 0.10,
    "theta": 0.85,
    "alpha": 0.30,
    "beta":  0.20,
    "gamma": 0.65,
    "sigma": 0.05,
}

DREAM_STATE_VECTOR = SleepNTStateVector(
    nt_baselines=DREAM_NT_BASELINES,
    oscillatory_config=DREAM_OSC_CONFIG,
)

# =====================================================================
# Phase → State Vector Lookup
# =====================================================================

SLEEP_STATE_VECTORS: Dict[SleepPhase, SleepNTStateVector] = {
    SleepPhase.TRIAGE: TRIAGE_STATE_VECTOR,
    SleepPhase.REM_PROCESSING: REM_PROCESSING_STATE_VECTOR,
    SleepPhase.DREAM: DREAM_STATE_VECTOR,
}
