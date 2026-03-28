"""
Per-NT oscillation band association map.

Defines the authoritative mapping from neurotransmitter systems to
their primary and secondary oscillation bands, per PDF Appendix I
(Oscillatory-Neurochemical Coupling Matrix).

Primary bands have strong, direct modulation effects.
Secondary bands have weaker or indirect effects.

Usage
-----
>>> from zados.neurochem.oscillations.band_associations import (
...     NT_BAND_ASSOCIATIONS,
...     get_primary_bands,
...     get_all_associated_bands,
... )
>>> get_primary_bands("DA")
['gamma', 'theta']
>>> get_all_associated_bands("5HT")
['theta', 'alpha', 'delta']
"""

from __future__ import annotations

from typing import Dict, List


# =====================================================================
# Authoritative NT -> Oscillation Band Association Map
# =====================================================================
# Source: Neurochemical Layer PDF, Appendix I
#
# Structure per NT:
#   "primary": bands with strong direct coupling
#   "secondary": bands with weaker/indirect coupling
# =====================================================================

NT_BAND_ASSOCIATIONS: Dict[str, Dict[str, List[str]]] = {

    # DA: Gamma drives phasic release, theta modulates receptor affinity
    "DA": {
        "primary": ["gamma", "theta"],
        "secondary": ["beta"],
    },

    # 5-HT: Theta drives release (slow, tonic), alpha for noise suppression
    "5HT": {
        "primary": ["theta", "alpha"],
        "secondary": ["delta"],
    },

    # NE: Beta drives release (precision/arousal), alpha for noise suppression
    "NE": {
        "primary": ["beta"],
        "secondary": ["gamma"],
    },

    # ACh: Beta drives release (attention/precision)
    "ACh": {
        "primary": ["beta"],
        "secondary": [],
    },

    # OXT: Theta drives release (social/empathic rhythm)
    "OXT": {
        "primary": ["theta"],
        "secondary": ["alpha"],
    },

    # MOR: Delta modulates tonic baseline (slow, hedonic)
    "MOR": {
        "primary": ["delta"],
        "secondary": ["theta"],
    },

    # CB1: Delta modulates tonic, alpha-beta cross-frequency for coupling
    "CB1": {
        "primary": ["delta"],
        "secondary": ["alpha_beta"],
    },

    # Cortisol: Beta drives release (stress/arousal), delta for slow dynamics
    "cortisol": {
        "primary": ["beta"],
        "secondary": ["delta"],
    },

    # CRH: Beta drives release (acute stress)
    "CRH": {
        "primary": ["beta"],
        "secondary": [],
    },

    # GABA: Alpha drives release (inhibitory gating), delta for tonic modulation
    "GABA": {
        "primary": ["alpha", "delta"],
        "secondary": ["theta"],
    },

    # GLU: Gamma and theta-gamma coupling drive release (fast excitation)
    "GLU": {
        "primary": ["gamma", "theta_gamma"],
        "secondary": [],
    },

    # Histamine: Beta drives release (arousal/wakefulness), gamma secondary
    "histamine": {
        "primary": ["beta"],
        "secondary": ["gamma"],
    },
}


# =====================================================================
# Band -> default modulation target
# =====================================================================
# Describes the general role of each band in neurochemical modulation.
# Per-NT modules override with their specific OscillationCouplingRules.
# =====================================================================

BAND_MODULATION_DEFAULTS: Dict[str, str] = {
    "gamma": "release",           # Gamma boosts phasic release
    "theta": "K_d",               # Theta modulates receptor affinity
    "alpha": "noise",             # Alpha suppresses noise / gating
    "beta":  "desensitization",   # Beta accelerates desensitization
    "delta": "tonic",             # Delta modulates tonic baseline
}


# =====================================================================
# Accessor helpers
# =====================================================================

def get_primary_bands(nt_name: str) -> List[str]:
    """
    Get primary oscillation bands for a neurotransmitter.

    Parameters
    ----------
    nt_name : str
        NT identifier (e.g., "DA", "5HT")

    Returns
    -------
    list of str
        Primary band names, or empty list if NT not found.
    """
    entry = NT_BAND_ASSOCIATIONS.get(nt_name)
    if entry is None:
        return []
    return list(entry.get("primary", []))


def get_secondary_bands(nt_name: str) -> List[str]:
    """
    Get secondary oscillation bands for a neurotransmitter.

    Parameters
    ----------
    nt_name : str
        NT identifier

    Returns
    -------
    list of str
        Secondary band names, or empty list if NT not found.
    """
    entry = NT_BAND_ASSOCIATIONS.get(nt_name)
    if entry is None:
        return []
    return list(entry.get("secondary", []))


def get_all_associated_bands(nt_name: str) -> List[str]:
    """
    Get all associated oscillation bands (primary + secondary).

    Parameters
    ----------
    nt_name : str
        NT identifier

    Returns
    -------
    list of str
        All associated band names (primary first, then secondary).
    """
    return get_primary_bands(nt_name) + get_secondary_bands(nt_name)


def get_nts_for_band(band: str) -> List[str]:
    """
    Get all NTs that have a given band as primary.

    Parameters
    ----------
    band : str
        Oscillation band name (e.g., "gamma", "theta")

    Returns
    -------
    list of str
        NT names that have this band as primary.
    """
    result = []
    for nt_name, entry in NT_BAND_ASSOCIATIONS.items():
        if band in entry.get("primary", []):
            result.append(nt_name)
    return sorted(result)
