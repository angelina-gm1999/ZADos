from __future__ import annotations

from typing import Dict, Optional
from enum import Enum


class NeurotransmitterTag(Enum):
    """
    Symbolic tags for neurotransmitters.
    
    Maps to concentration components from state layer.
    """
    DA = "dopamine"
    GABA = "gaba"
    GLU = "glutamate"
    SEROTONIN = "5-HT"
    NE = "norepinephrine"
    ACH = "acetylcholine"
    OXT = "oxytocin"
    MOR = "mu-opioid"
    CB1 = "cannabinoid"
    CRH = "corticotropin"
    CORTISOL = "cortisol"
    HISTAMINE = "histamine"


class ReceptorTag(Enum):
    """
    Symbolic tags for receptor subtypes.
    
    Format: NT_SUBTYPE (e.g., DA_D1, SEROTONIN_2A)
    """
    # Dopamine receptors
    DA_D1 = "DA_D1"
    DA_D2 = "DA_D2"
    DA_D3 = "DA_D3"
    DA_D4 = "DA_D4"
    DA_D5 = "DA_D5"
    
    # GABA receptors
    GABA_A = "GABA_A"
    GABA_B = "GABA_B"
    
    # Glutamate receptors
    GLU_NMDA = "GLU_NMDA"
    GLU_AMPA = "GLU_AMPA"
    GLU_KAINATE = "GLU_KAINATE"
    GLU_mGluR = "GLU_mGluR"
    
    # Serotonin receptors
    SEROTONIN_1A = "5HT_1A"
    SEROTONIN_1B = "5HT_1B"
    SEROTONIN_2A = "5HT_2A"
    SEROTONIN_2C = "5HT_2C"
    SEROTONIN_3 = "5HT_3"
    
    # Norepinephrine receptors
    NE_ALPHA1 = "NE_alpha1"
    NE_ALPHA2 = "NE_alpha2"
    NE_BETA1 = "NE_beta1"
    NE_BETA2 = "NE_beta2"
    
    # Acetylcholine receptors
    ACH_NICOTINIC = "ACh_nicotinic"
    ACH_MUSCARINIC = "ACh_muscarinic"
    
    # Oxytocin receptor
    OXTR = "OXTR"
    
    # Opioid receptors
    MOR_MU = "MOR_mu"
    
    # Cannabinoid receptors
    CB1_RECEPTOR = "CB1"
    
    # CRH receptor
    CRH_R1 = "CRH_R1"

    # Histamine receptors
    HIST_H1 = "HIST_H1"
    HIST_H2 = "HIST_H2"
    HIST_H3 = "HIST_H3"
    HIST_H4 = "HIST_H4"


class OscillationBandTag(Enum):
    """
    Symbolic tags for oscillation bands.
    """
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


class ModifierTag(Enum):
    """
    Symbolic tags for receptor/transmitter modifiers.
    
    Used in neurosymbolic encoding grammar (Appendix K):
    [NT] → [R] : [M]
    """
    # Density modifiers
    UP_DENSITY = "↑density"
    DOWN_DENSITY = "↓density"
    
    # Sensitivity modifiers
    UP_SENSITIVITY = "↑sensitivity"
    DOWN_SENSITIVITY = "↓sensitivity"
    
    # State modifiers
    DESENSITIZED = "desensitized"
    INTERNALIZED = "internalized"
    UPREGULATED = "upregulated"
    ACTIVE = "active"
    
    # Affinity modifiers
    UP_AFFINITY = "↑affinity"
    DOWN_AFFINITY = "↓affinity"
    
    # Release modifiers
    UP_RELEASE = "↑release"
    DOWN_RELEASE = "↓release"
    
    # Reuptake modifiers
    UP_REUPTAKE = "↑reuptake"
    DOWN_REUPTAKE = "↓reuptake"

    # Coupling efficacy modifiers (K.1.4a)
    UP_COUPLING = "↑coupling"
    DOWN_COUPLING = "↓coupling"

    # Availability modifiers (K.1.4a)
    UP_AVAILABILITY = "↑availability"
    DOWN_AVAILABILITY = "↓availability"

    # Activation gain modifiers (K.7.2C)
    UP_ACT = "↑activation"
    DOWN_ACT = "↓activation"

    # Noise modifiers (K.7.2C)
    UP_NOISE = "↑noise"
    DOWN_NOISE = "↓noise"

    # Recovery modifiers (K.7.2C)
    UP_RECOV = "↑recovery"
    DOWN_RECOV = "↓recovery"

    # Saturation/binding modifiers (K.1.4d)
    UP_SATURATION = "↑S"
    DOWN_SATURATION = "↓S"
    BIND = "bind"
    UNBIND = "unbind"


class ConcentrationComponentTag(Enum):
    """
    Tags for concentration components (tonic vs phasic).
    """
    TONIC = "tonic"
    PHASIC = "phasic"
    TOTAL = "total"


class GateTag(Enum):
    """
    Oscillation gate tokens for mastergrid encoding (K.7.2D).

    Includes single-band gates and cross-frequency coupling gates.
    """
    DELTA = "DELTA"
    THETA = "THETA"
    ALPHA = "ALPHA"
    BETA = "BETA"
    GAMMA = "GAMMA"
    THETA_GAMMA = "THETA_GAMMA"
    ALPHA_BETA = "ALPHA_BETA"


class PhasicTonicMarker(Enum):
    """
    Phasic vs tonic signal markers (K.3).

    Phasic (NT•/NT.P): burst component C_i^phasic(t).
    Tonic  (NT~/NT.T): baseline component C_i^tonic(t).
    """
    PHASIC = "phasic"
    TONIC = "tonic"


# ---------------------------------------------------------------------------
# Mastergrid modifier mapping (K.7.2C)
# Maps compact ASCII tokens used in mastergrid cells to ModifierTag members.
# ---------------------------------------------------------------------------

MASTERGRID_MODIFIER_MAP: Dict[str, ModifierTag] = {
    "UP_DENS": ModifierTag.UP_DENSITY,
    "DOWN_DENS": ModifierTag.DOWN_DENSITY,
    "UP_SENS": ModifierTag.UP_SENSITIVITY,
    "DOWN_SENS": ModifierTag.DOWN_SENSITIVITY,
    "DESENS": ModifierTag.DESENSITIZED,
    "INTERNAL": ModifierTag.INTERNALIZED,
    "UPREG": ModifierTag.UPREGULATED,
    "UP_AFF": ModifierTag.UP_AFFINITY,
    "DOWN_AFF": ModifierTag.DOWN_AFFINITY,
    "UP_ACT": ModifierTag.UP_ACT,
    "DOWN_ACT": ModifierTag.DOWN_ACT,
    "UP_NOISE": ModifierTag.UP_NOISE,
    "DOWN_NOISE": ModifierTag.DOWN_NOISE,
    "UP_RECOV": ModifierTag.UP_RECOV,
    "DOWN_RECOV": ModifierTag.DOWN_RECOV,
}


def encode_neurosymbolic_triplet(
    neurotransmitter: NeurotransmitterTag,
    receptor: ReceptorTag,
    modifier: ModifierTag,
    oscillation_gate: Optional[OscillationBandTag] = None,
    phasic_tonic: Optional[PhasicTonicMarker] = None,
) -> str:
    """
    Encode a neurosymbolic triplet in the grammar from Appendix K.

    Format: [NT] → [R] : [M]
    With optional phasic/tonic marker (K.3): NT• or NT~ prefix
    With optional oscillatory gating: k{[NT]→[R]:Δ}

    Parameters
    ----------
    neurotransmitter : NeurotransmitterTag
        Neurotransmitter identifier
    receptor : ReceptorTag
        Receptor subtype identifier
    modifier : ModifierTag
        Modification type
    oscillation_gate : OscillationBandTag, optional
        Oscillation band gating this triplet
    phasic_tonic : PhasicTonicMarker, optional
        If PHASIC, appends '•' to NT name (burst component).
        If TONIC, appends '~' to NT name (baseline component).

    Returns
    -------
    str
        Encoded triplet string

    Examples
    --------
    >>> encode_neurosymbolic_triplet(
    ...     NeurotransmitterTag.DA,
    ...     ReceptorTag.DA_D1,
    ...     ModifierTag.UP_DENSITY
    ... )
    'DA→D1:↑density'

    >>> encode_neurosymbolic_triplet(
    ...     NeurotransmitterTag.DA,
    ...     ReceptorTag.DA_D1,
    ...     ModifierTag.UP_RELEASE,
    ...     phasic_tonic=PhasicTonicMarker.PHASIC
    ... )
    'DA•→D1:↑release'

    >>> encode_neurosymbolic_triplet(
    ...     NeurotransmitterTag.GLU,
    ...     ReceptorTag.GLU_NMDA,
    ...     ModifierTag.UP_AFFINITY,
    ...     OscillationBandTag.GAMMA
    ... )
    'gamma{GLU→NMDA:↑affinity}'
    """
    # Extract short names
    nt_short = neurotransmitter.name
    receptor_short = receptor.value.split("_")[-1]
    modifier_short = modifier.value

    # Append phasic/tonic marker (K.3)
    if phasic_tonic is PhasicTonicMarker.PHASIC:
        nt_short += "•"
    elif phasic_tonic is PhasicTonicMarker.TONIC:
        nt_short += "~"

    base = f"{nt_short}→{receptor_short}:{modifier_short}"

    if oscillation_gate:
        return f"{oscillation_gate.value}{{{base}}}"

    return base


def parse_neurosymbolic_triplet(encoded: str) -> dict:
    """
    Parse an encoded neurosymbolic triplet.

    Parameters
    ----------
    encoded : str
        Encoded triplet string

    Returns
    -------
    dict
        Dictionary with keys: 'neurotransmitter', 'receptor', 'modifier',
        'oscillation_gate', 'phasic_tonic'

    Examples
    --------
    >>> parse_neurosymbolic_triplet("DA→D1:↑density")
    {'neurotransmitter': 'DA', 'receptor': 'D1', 'modifier': '↑density', 'oscillation_gate': None, 'phasic_tonic': None}

    >>> parse_neurosymbolic_triplet("DA•→D1:↑release")
    {'neurotransmitter': 'DA', 'receptor': 'D1', 'modifier': '↑release', 'oscillation_gate': None, 'phasic_tonic': 'phasic'}
    """
    oscillation_gate = None

    # Check for oscillation gating
    if "{" in encoded and "}" in encoded:
        gate_part, rest = encoded.split("{", 1)
        oscillation_gate = gate_part.strip()
        encoded = rest.rstrip("}")

    # Parse triplet: NT→R:M
    parts = encoded.split("→")
    if len(parts) != 2:
        raise ValueError(f"Invalid triplet format: {encoded}")

    neurotransmitter = parts[0].strip()

    # Detect phasic/tonic marker (K.3)
    phasic_tonic = None
    if neurotransmitter.endswith("•"):
        phasic_tonic = "phasic"
        neurotransmitter = neurotransmitter[:-1]
    elif neurotransmitter.endswith("~"):
        phasic_tonic = "tonic"
        neurotransmitter = neurotransmitter[:-1]

    receptor_modifier = parts[1].split(":", 1)
    if len(receptor_modifier) != 2:
        raise ValueError(f"Invalid triplet format: {encoded}")

    receptor = receptor_modifier[0].strip()
    modifier = receptor_modifier[1].strip()

    return {
        "neurotransmitter": neurotransmitter,
        "receptor": receptor,
        "modifier": modifier,
        "oscillation_gate": oscillation_gate,
        "phasic_tonic": phasic_tonic,
    }