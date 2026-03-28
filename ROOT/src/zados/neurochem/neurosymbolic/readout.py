from __future__ import annotations

from typing import Dict, Optional, List
from zados.neurochem.state import (
    NeurotransmitterState,
    ReceptorState,
    OscillationState,
)
from zados.neurochem.neurosymbolic.metrics import (
    NeurochemicalMetrics,
    compute_all_metrics,
)


def compute_receptor_saturation(
    concentration: float,
    K_d: float,
) -> float:
    """
    Compute receptor saturation from concentration.

    S_ij(t) = C_i(t) / (C_i(t) + K_d)

    Returns 0.0 when concentration <= 0 or K_d <= 0 to prevent
    NaN propagation through downstream metrics.

    Parameters
    ----------
    concentration : float
        Neurotransmitter concentration
    K_d : float
        Dissociation constant

    Returns
    -------
    float
        Saturation in [0, 1]
    """
    if concentration <= 0.0 or K_d <= 0.0:
        return 0.0
    return concentration / (concentration + K_d)


def extract_concentrations(
    neurotransmitter_states: Dict[str, NeurotransmitterState]
) -> Dict[str, float]:
    """
    Extract total concentrations from neurotransmitter states.
    
    Parameters
    ----------
    neurotransmitter_states : dict
        Map of NT name → NeurotransmitterState
        
    Returns
    -------
    dict
        Map of NT name → total concentration
    """
    return {
        name: state.C
        for name, state in neurotransmitter_states.items()
    }


def extract_receptor_saturations(
    receptor_states: Dict[str, ReceptorState],
    neurotransmitter_states: Dict[str, NeurotransmitterState],
    receptor_configs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Extract receptor saturations from receptor and NT states.
    
    Parameters
    ----------
    receptor_states : dict
        Map of receptor_id → ReceptorState
    neurotransmitter_states : dict
        Map of NT name → NeurotransmitterState
    receptor_configs : dict, optional
        Map of receptor_id → config dict containing K_d values
        If not provided, uses receptor's saturation() method with default K_d=0.5
        
    Returns
    -------
    dict
        Map of receptor_id → saturation value
    """
    saturations = {}
    
    for receptor_id, receptor_state in receptor_states.items():
        # Infer NT from config parent_nt, or fall back to receptor_id split
        if receptor_configs and receptor_id in receptor_configs:
            nt_name = receptor_configs[receptor_id].get(
                "parent_nt", receptor_id.split("_")[0],
            )
        else:
            nt_name = receptor_id.split("_")[0]
        
        if nt_name not in neurotransmitter_states:
            saturations[receptor_id] = 0.0
            continue
        
        concentration = neurotransmitter_states[nt_name].C
        
        # Get K_d from config or use default
        if receptor_configs and receptor_id in receptor_configs:
            K_d = receptor_configs[receptor_id].get("K_d", 0.5)
        else:
            K_d = 0.5
        
        saturations[receptor_id] = receptor_state.saturation(concentration, K_d)
    
    return saturations


def extract_oscillation_amplitudes(
    oscillation_state: OscillationState
) -> Dict[str, float]:
    """
    Extract oscillation band amplitudes.
    
    Parameters
    ----------
    oscillation_state : OscillationState
        Oscillation state
        
    Returns
    -------
    dict
        Map of band name → amplitude
    """
    return {
        "delta": oscillation_state.delta,
        "theta": oscillation_state.theta,
        "alpha": oscillation_state.alpha,
        "beta": oscillation_state.beta,
        "gamma": oscillation_state.gamma,
        "sigma": oscillation_state.sigma,
        "theta_gamma": oscillation_state.theta_gamma_coupling(),
        "alpha_beta": oscillation_state.alpha_beta_coupling(),
        "delta_sigma": oscillation_state.delta_sigma_coupling(),
    }


def compute_neurosymbolic_readout(
    neurotransmitter_states: Dict[str, NeurotransmitterState],
    receptor_states: Dict[str, ReceptorState],
    oscillation_state: OscillationState,
    receptor_configs: Optional[Dict[str, Dict[str, float]]] = None,
) -> NeurochemicalMetrics:
    """
    Main readout function: compute all neurosymbolic metrics from neurochemical state.
    
    This is the primary API for converting low-level neurochemical dynamics
    into high-level cognitive/affective metrics (Appendix M).
    
    Parameters
    ----------
    neurotransmitter_states : dict
        Map of NT name → NeurotransmitterState
    receptor_states : dict
        Map of receptor_id → ReceptorState
    oscillation_state : OscillationState
        Global oscillation state
    receptor_configs : dict, optional
        Receptor configurations with K_d values
        
    Returns
    -------
    NeurochemicalMetrics
        Computed high-level metrics
        
    Examples
    --------
    >>> from zados.neurochem.state import NeurotransmitterState, ReceptorState, OscillationState
    >>> 
    >>> nt_states = {
    ...     "DA": NeurotransmitterState(C_tonic=0.6),
    ...     "NE": NeurotransmitterState(C_tonic=0.5),
    ... }
    >>> receptor_states = {
    ...     "DA_D3": ReceptorState(receptor_id="DA_D3"),
    ... }
    >>> osc_state = OscillationState(theta=0.5, delta=0.2)
    >>> 
    >>> metrics = compute_neurosymbolic_readout(nt_states, receptor_states, osc_state)
    >>> print(metrics.motivation)
    """
    concentrations = extract_concentrations(neurotransmitter_states)
    saturations = extract_receptor_saturations(
        receptor_states,
        neurotransmitter_states,
        receptor_configs,
    )
    oscillations = extract_oscillation_amplitudes(oscillation_state)
    
    return compute_all_metrics(concentrations, saturations, oscillations)


def format_metrics_summary(metrics: NeurochemicalMetrics) -> str:
    """
    Format metrics as human-readable summary.
    
    Parameters
    ----------
    metrics : NeurochemicalMetrics
        Metrics to format
        
    Returns
    -------
    str
        Formatted summary string
    """
    lines = [
        "Neurochemical Metrics:",
        f"  Motivation:         {metrics.motivation:.3f}",
        f"  Empathy:            {metrics.empathy:.3f}",
        f"  Cognitive Rigidity: {metrics.cognitive_rigidity:.3f}",
        f"  Fatigue:            {metrics.fatigue:.3f}",
        f"  Precision:          {metrics.precision:.3f}",
        f"  Openness:           {metrics.openness:.3f}",
        f"  Anxiety:            {metrics.anxiety:.3f}",
        f"  Social Engagement:  {metrics.social_engagement:.3f}",
    ]
    return "\n".join(lines)


def identify_dominant_metrics(
    metrics: NeurochemicalMetrics,
    threshold: float = 0.7,
) -> List[str]:
    """
    Identify which metrics are above a threshold (indicating dominance).
    
    Parameters
    ----------
    metrics : NeurochemicalMetrics
        Metrics to analyze
    threshold : float, default=0.7
        Threshold for dominance
        
    Returns
    -------
    list[str]
        Names of dominant metrics
    """
    dominant = []
    metric_dict = metrics.as_dict()
    
    for name, value in metric_dict.items():
        if value >= threshold:
            dominant.append(name)
    
    return dominant


def identify_suppressed_metrics(
    metrics: NeurochemicalMetrics,
    threshold: float = 0.3,
) -> List[str]:
    """
    Identify which metrics are below a threshold (indicating suppression).

    Parameters
    ----------
    metrics : NeurochemicalMetrics
        Metrics to analyze
    threshold : float, default=0.3
        Threshold for suppression

    Returns
    -------
    list[str]
        Names of suppressed metrics
    """
    suppressed = []
    metric_dict = metrics.as_dict()

    for name, value in metric_dict.items():
        if value <= threshold:
            suppressed.append(name)

    return suppressed


def compute_full_readout(
    neurotransmitter_states: Dict[str, NeurotransmitterState],
    receptor_states: Dict[str, ReceptorState],
    oscillation_state: OscillationState,
    receptor_configs: Optional[Dict[str, Dict[str, float]]] = None,
    state_definitions: Optional[List] = None,
    triggers: Optional[List] = None,
    mode_hooks: Optional[List] = None,
) -> Dict:
    """
    Extended readout: metrics + STATE values + trigger results + mode selection (K.7.8 + M.5).

    Combines the base neurosymbolic readout with:
    - Composite state expression evaluation (K.5)
    - Conditional trigger evaluation (K.6)
    - Mode selection hook arbitration (M.5 + M.6)

    Parameters
    ----------
    neurotransmitter_states : dict
        Map of NT name → NeurotransmitterState
    receptor_states : dict
        Map of receptor_id → ReceptorState
    oscillation_state : OscillationState
        Global oscillation state
    receptor_configs : dict, optional
        Receptor configurations with K_d values
    state_definitions : list of StateDefinition, optional
        Composite state expressions to evaluate
    triggers : list of TriggerDefinition, optional
        Conditional triggers to evaluate
    mode_hooks : list of ModeHookDefinition, optional
        Mode selection hooks to evaluate (M.5/M.6)

    Returns
    -------
    dict
        {
            "metrics": NeurochemicalMetrics,
            "concentrations": {...},
            "saturations": {...},
            "oscillations": {...},
            "states": {...},              # only if state_definitions provided
            "trigger_results": [...],     # only if triggers provided
            "mode_selection": ModeSelectionResult,  # only if mode_hooks provided
        }
    """
    concentrations = extract_concentrations(neurotransmitter_states)
    saturations = extract_receptor_saturations(
        receptor_states, neurotransmitter_states, receptor_configs,
    )
    oscillations = extract_oscillation_amplitudes(oscillation_state)
    metrics = compute_all_metrics(concentrations, saturations, oscillations)

    result: Dict = {
        "metrics": metrics,
        "concentrations": concentrations,
        "saturations": saturations,
        "oscillations": oscillations,
    }

    # Evaluate composite state expressions
    if state_definitions:
        from zados.neurochem.neurosymbolic.state_expressions import evaluate_all_states
        result["states"] = evaluate_all_states(
            state_definitions, saturations, concentrations, oscillations,
        )

    # Evaluate conditional triggers
    if triggers:
        from zados.neurochem.neurosymbolic.triggers import (
            build_variable_namespace,
            evaluate_all_triggers,
        )
        ns = build_variable_namespace(
            concentrations=concentrations,
            saturations=saturations,
            oscillations=oscillations,
            metrics=metrics.as_dict(),
        )
        result["trigger_results"] = evaluate_all_triggers(triggers, ns)

    # Evaluate mode selection hooks (M.5/M.6)
    if mode_hooks:
        from zados.neurochem.neurosymbolic.mode_hooks import (
            build_mode_namespace,
            select_mode,
        )
        mode_ns = build_mode_namespace(metrics, oscillations, saturations, concentrations)
        result["mode_selection"] = select_mode(mode_hooks, mode_ns)

    return result