from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class NeurochemicalMetrics:
    """
    Derived neurosymbolic metrics computed from low-level neurochemical state.
    
    Implements the metric formulas from Appendix M.
    All metrics are normalized to [0, 1] range.
    
    Attributes
    ----------
    motivation : float
        Drive/approach tendency (DA-D3, OXT ↑; GABA-B ↓)
    empathy : float
        Attunement to others (OXT, 5-HT1A, theta ↑)
    cognitive_rigidity : float
        Resistance to updating beliefs (NE-β1, DA-D2 ↑; CB1 ↓)
    fatigue : float
        Mental/physical exhaustion (GABA-B, delta ↑)
    precision : float
        Error detection sensitivity (NE-β1, DA-D2, beta ↑)
    openness : float
        Exploration/novelty seeking (5-HT2A, DA-D3 ↑; 5-HT1A ↓)
    anxiety : float
        Threat sensitivity (NE, CRH, cortisol ↑; GABA ↓)
    social_engagement : float
        Affiliative drive (OXT, DA-D3 ↑; cortisol ↓)
    """
    
    motivation: float = 0.5
    empathy: float = 0.5
    cognitive_rigidity: float = 0.5
    fatigue: float = 0.5
    precision: float = 0.5
    openness: float = 0.5
    anxiety: float = 0.5
    social_engagement: float = 0.5
    dream_permissiveness: float = 0.0
    consolidation_depth: float = 0.0
    narrative_plasticity: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        """Export metrics as dictionary."""
        return {
            "motivation": self.motivation,
            "empathy": self.empathy,
            "cognitive_rigidity": self.cognitive_rigidity,
            "fatigue": self.fatigue,
            "precision": self.precision,
            "openness": self.openness,
            "anxiety": self.anxiety,
            "social_engagement": self.social_engagement,
            "dream_permissiveness": self.dream_permissiveness,
            "consolidation_depth": self.consolidation_depth,
            "narrative_plasticity": self.narrative_plasticity,
        }


def compute_motivation(
    S_DA_D3: float,
    S_OXT: float,
    S_GABA_B: float,
) -> float:
    """
    Compute motivation metric.
    
    Motivation = S_DA-D3 + S_OXT - S_GABA-B
    
    Parameters
    ----------
    S_DA_D3 : float
        DA D3 receptor saturation
    S_OXT : float
        Oxytocin receptor saturation
    S_GABA_B : float
        GABA-B receptor saturation
        
    Returns
    -------
    float
        Normalized motivation ∈ [0, 1]
    """
    raw = S_DA_D3 + S_OXT - S_GABA_B
    # Raw range: [-1, 2]
    normalized = (raw + 1.0) / 3.0
    return max(0.0, min(1.0, normalized))


def compute_empathy(
    S_OXTR: float,
    phi_theta: float,
    S_5HT1A: float,
) -> float:
    """
    Compute empathy metric.
    
    Empathy = S_OXTR · φ_θ · S_5HT1A
    
    Parameters
    ----------
    S_OXTR : float
        Oxytocin receptor saturation
    phi_theta : float
        Theta band oscillation amplitude
    S_5HT1A : float
        5-HT1A receptor saturation
        
    Returns
    -------
    float
        Normalized empathy ∈ [0, 1]
    """
    # Already bounded since all inputs ∈ [0, 1]
    return S_OXTR * phi_theta * S_5HT1A


def compute_cognitive_rigidity(
    S_NE_beta1: float,
    S_DA_D2: float,
    S_CB1: float,
) -> float:
    """
    Compute cognitive rigidity metric.
    
    CognitiveRigidity = S_NE-β1 + S_DA-D2 - S_CB1
    
    Parameters
    ----------
    S_NE_beta1 : float
        NE β1 receptor saturation
    S_DA_D2 : float
        DA D2 receptor saturation
    S_CB1 : float
        CB1 receptor saturation
        
    Returns
    -------
    float
        Normalized rigidity ∈ [0, 1]
    """
    raw = S_NE_beta1 + S_DA_D2 - S_CB1
    # Raw range: [-1, 2]
    normalized = (raw + 1.0) / 3.0
    return max(0.0, min(1.0, normalized))


def compute_fatigue(
    S_GABA_B: float,
    phi_delta: float,
) -> float:
    """
    Compute fatigue metric.
    
    Fatigue = S_GABA-B + φ_δ
    
    Parameters
    ----------
    S_GABA_B : float
        GABA-B receptor saturation
    phi_delta : float
        Delta band oscillation amplitude
        
    Returns
    -------
    float
        Normalized fatigue ∈ [0, 1]
    """
    raw = S_GABA_B + phi_delta
    # Raw range: [0, 2]
    normalized = raw / 2.0
    return max(0.0, min(1.0, normalized))


def compute_precision(
    S_NE_beta1: float,
    S_DA_D2: float,
    phi_beta: float,
) -> float:
    """
    Compute precision/error detection metric.
    
    Precision = (S_NE-β1 + S_DA-D2) · φ_β
    
    Parameters
    ----------
    S_NE_beta1 : float
        NE β1 receptor saturation
    S_DA_D2 : float
        DA D2 receptor saturation
    phi_beta : float
        Beta band oscillation amplitude
        
    Returns
    -------
    float
        Normalized precision ∈ [0, 1]
    """
    raw = (S_NE_beta1 + S_DA_D2) * phi_beta
    # Raw range: [0, 2]
    normalized = raw / 2.0
    return max(0.0, min(1.0, normalized))


def compute_openness(
    S_5HT2A: float,
    S_DA_D3: float,
    S_5HT1A: float,
) -> float:
    """
    Compute openness/exploration metric.
    
    Openness = S_5HT2A + S_DA-D3 - S_5HT1A
    
    Parameters
    ----------
    S_5HT2A : float
        5-HT2A receptor saturation
    S_DA_D3 : float
        DA D3 receptor saturation
    S_5HT1A : float
        5-HT1A receptor saturation
        
    Returns
    -------
    float
        Normalized openness ∈ [0, 1]
    """
    raw = S_5HT2A + S_DA_D3 - S_5HT1A
    # Raw range: [-1, 2]
    normalized = (raw + 1.0) / 3.0
    return max(0.0, min(1.0, normalized))


def compute_anxiety(
    C_NE: float,
    C_CRH: float,
    C_cortisol: float,
    S_GABA_A: float,
) -> float:
    """
    Compute anxiety/threat sensitivity metric.
    
    Anxiety = (C_NE + C_CRH + C_cortisol) / 3 - S_GABA-A
    
    Parameters
    ----------
    C_NE : float
        Norepinephrine concentration
    C_CRH : float
        CRH concentration
    C_cortisol : float
        Cortisol concentration
    S_GABA_A : float
        GABA-A receptor saturation
        
    Returns
    -------
    float
        Normalized anxiety ∈ [0, 1]
    """
    raw = (C_NE + C_CRH + C_cortisol) / 3.0 - S_GABA_A
    # Raw range: [-1, 1]
    normalized = (raw + 1.0) / 2.0
    return max(0.0, min(1.0, normalized))


def compute_social_engagement(
    S_OXTR: float,
    S_DA_D3: float,
    C_cortisol: float,
) -> float:
    """
    Compute social engagement metric.
    
    SocialEngagement = S_OXTR + S_DA-D3 - C_cortisol
    
    Parameters
    ----------
    S_OXTR : float
        Oxytocin receptor saturation
    S_DA_D3 : float
        DA D3 receptor saturation
    C_cortisol : float
        Cortisol concentration
        
    Returns
    -------
    float
        Normalized social engagement ∈ [0, 1]
    """
    raw = S_OXTR + S_DA_D3 - C_cortisol
    # Raw range: [-1, 2]
    normalized = (raw + 1.0) / 3.0
    return max(0.0, min(1.0, normalized))


def compute_dream_permissiveness(
    S_CB1: float,
    phi_theta_gamma: float,
    S_NE_alpha1: float,
    S_5HT1A: float,
    S_GABA_B: float,
) -> float:
    """
    Compute dream permissiveness metric (Spec §6.1).

    DreamPermissiveness = (S_CB1 * phi_theta_gamma) + (1 - S_NE_alpha1) + (1 - S_5HT1A) - S_GABA_B

    Measures REM-analog creative freedom. High = valid dream processing.
    Low during waking = normal critical state.

    Raw range ~[-1, 3], normalize: (raw + 1) / 4.
    """
    raw = (S_CB1 * phi_theta_gamma) + (1.0 - S_NE_alpha1) + (1.0 - S_5HT1A) - S_GABA_B
    normalized = (raw + 1.0) / 4.0
    return max(0.0, min(1.0, normalized))


def compute_consolidation_depth(
    phi_delta_sigma: float,
    S_GLU_NMDA: float,
    S_NE_beta1: float,
    S_ACh_M1: float,
    infra_slow_cAMP: float = 0.0,
) -> float:
    """
    Compute consolidation depth metric (Spec §6.2).

    ConsolidationDepth = phi_delta_sigma * S_GLU_NMDA + (S_NE_beta1 * infra_slow_cAMP) - S_ACh_M1

    Measures NREM consolidation efficacy. Peaks during REM Processing
    with active delta-sigma coupling.

    Raw range ~[-1, 2], normalize: (raw + 1) / 3.
    Note: infra_slow_cAMP defaults to 0 (open item per spec §9).
    """
    raw = phi_delta_sigma * S_GLU_NMDA + (S_NE_beta1 * infra_slow_cAMP) - S_ACh_M1
    normalized = (raw + 1.0) / 3.0
    return max(0.0, min(1.0, normalized))


def compute_narrative_plasticity(
    phi_theta_gamma: float,
    S_GLU_NMDA: float,
    S_DA_D3: float,
    S_CB1: float,
    cognitive_rigidity: float,
) -> float:
    """
    Compute narrative plasticity metric (Spec §6.3).

    NarrativePlasticity = phi_theta_gamma * S_GLU_NMDA * S_DA_D3 + S_CB1 + (1 - cognitive_rigidity)

    Measures capacity for novel cross-domain associative chain generation.
    Distinct from dream_permissiveness (freedom from filtering) — this
    measures positive generative capacity.

    Raw range ~[0, 3], normalize: raw / 3.
    """
    raw = phi_theta_gamma * S_GLU_NMDA * S_DA_D3 + S_CB1 + (1.0 - cognitive_rigidity)
    normalized = raw / 3.0
    return max(0.0, min(1.0, normalized))


def compute_all_metrics(
    concentrations: Dict[str, float],
    receptor_saturations: Dict[str, float],
    oscillations: Dict[str, float],
) -> NeurochemicalMetrics:
    """
    Compute all neurosymbolic metrics from neurochemical state.
    
    Parameters
    ----------
    concentrations : dict
        Neurotransmitter concentrations, keys like 'DA', 'NE', 'cortisol'
    receptor_saturations : dict
        Receptor saturation values, keys like 'DA_D3', 'GABA_B', 'OXTR'
    oscillations : dict
        Oscillation band amplitudes, keys like 'theta', 'delta', 'beta'
        
    Returns
    -------
    NeurochemicalMetrics
        Computed metrics object
        
    Notes
    -----
    Missing keys default to 0.0. This allows partial state computation.
    """
    # Helper to safely get values with defaults
    def get_conc(key: str) -> float:
        return concentrations.get(key, 0.0)
    
    def get_sat(key: str) -> float:
        return receptor_saturations.get(key, 0.0)
    
    def get_osc(key: str) -> float:
        return oscillations.get(key, 0.0)
    
    return NeurochemicalMetrics(
        motivation=compute_motivation(
            get_sat("DA_D3"),
            get_sat("OXTR"),
            get_sat("GABA_B"),
        ),
        empathy=compute_empathy(
            get_sat("OXTR"),
            get_osc("theta"),
            get_sat("5HT_1A"),
        ),
        cognitive_rigidity=compute_cognitive_rigidity(
            get_sat("NE_beta1"),
            get_sat("DA_D2"),
            get_sat("CB1"),
        ),
        fatigue=compute_fatigue(
            get_sat("GABA_B"),
            get_osc("delta"),
        ),
        precision=compute_precision(
            get_sat("NE_beta1"),
            get_sat("DA_D2"),
            get_osc("beta"),
        ),
        openness=compute_openness(
            get_sat("5HT_2A"),
            get_sat("DA_D3"),
            get_sat("5HT_1A"),
        ),
        anxiety=compute_anxiety(
            get_conc("NE"),
            get_conc("CRH"),
            get_conc("cortisol"),
            get_sat("GABA_A"),
        ),
        social_engagement=compute_social_engagement(
            get_sat("OXTR"),
            get_sat("DA_D3"),
            get_conc("cortisol"),
        ),
        dream_permissiveness=compute_dream_permissiveness(
            get_sat("CB1"),
            get_osc("theta_gamma"),
            get_sat("NE_alpha1"),
            get_sat("5HT_1A"),
            get_sat("GABA_B"),
        ),
        consolidation_depth=compute_consolidation_depth(
            get_osc("delta_sigma"),
            get_sat("GLU_NMDA"),
            get_sat("NE_beta1"),
            get_sat("ACh_M1"),
        ),
        narrative_plasticity=compute_narrative_plasticity(
            get_osc("theta_gamma"),
            get_sat("GLU_NMDA"),
            get_sat("DA_D3"),
            get_sat("CB1"),
            compute_cognitive_rigidity(
                get_sat("NE_beta1"),
                get_sat("DA_D2"),
                get_sat("CB1"),
            ),
        ),
    )