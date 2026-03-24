"""
Default neurochemical configurations for all neurotransmitter systems.

Provides biologically-inspired kinetic parameters adapted to the
ZADOS engine's dict-based config format.

Config dict keys consumed by engine._update_neurotransmitter():
    C_baseline, theta_tonic, theta_phasic, sigma_tonic, sigma_phasic,
    u_base, d_base, c_base

Config dict keys consumed by engine._update_receptor():
    K_d, parent_nt, thresholds, exposure_tau

Usage
-----
>>> from zados.neurochem.neurotransmitters.configs import register_all_neurotransmitters
>>> engine = NeurochemicalEngine(dt=0.01, seed=42)
>>> register_all_neurotransmitters(engine)
>>> engine.step()
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


# =====================================================================
# Default NT configs
# =====================================================================
# Each dict has the 8 keys the engine expects.
# NT name keys MUST match those used in metrics.py, adapter mapping.py,
# and feedback modulator.py.
# =====================================================================

DEFAULT_NT_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ── Dopamine ─────────────────────────────────────────────────
    # Fast phasic bursts, moderate tonic baseline.
    # Primary role: motivation, reward prediction, novelty seeking.
    "DA": {
        "C_baseline": 0.5,
        "theta_tonic": 0.1,
        "theta_phasic": 1.0,
        "sigma_tonic": 0.05,
        "sigma_phasic": 0.1,
        "u_base": 0.1,       # DAT reuptake
        "d_base": 0.05,      # COMT/MAO degradation
        "c_base": 0.02,      # diffusion clearance
        "fatigue_rate": 0.001,
    },

    # ── Serotonin (5-HT) ────────────────────────────────────────
    # Slow tonic, stable baseline, low phasic.
    # Primary role: affect regulation, ambiguity buffering,
    # long-horizon weighting.
    "5HT": {
        "C_baseline": 0.55,
        "theta_tonic": 0.05,   # Slow reversion (tonic stability)
        "theta_phasic": 0.5,   # Moderate phasic decay
        "sigma_tonic": 0.03,   # Low noise (stable system)
        "sigma_phasic": 0.06,
        "u_base": 0.08,        # SERT reuptake
        "d_base": 0.03,        # MAO degradation
        "c_base": 0.015,
        "fatigue_rate": 0.001,
    },

    # ── Norepinephrine (NE) ──────────────────────────────────────
    # Fast, responsive, moderate baseline.
    # Primary role: arousal, salience, contradiction detection,
    # load-responsive gain control.
    "NE": {
        "C_baseline": 0.45,
        "theta_tonic": 0.15,   # Fast reversion (alertness modulation)
        "theta_phasic": 1.2,   # Fast phasic decay
        "sigma_tonic": 0.06,
        "sigma_phasic": 0.12,  # Higher noise for salience variability
        "u_base": 0.12,        # NET reuptake
        "d_base": 0.04,        # COMT/MAO
        "c_base": 0.02,
        "fatigue_rate": 0.001,
    },

    # ── Acetylcholine (ACh) ──────────────────────────────────────
    # Very fast kinetics.
    # Primary role: precision, attention, rule fidelity.
    "ACh": {
        "C_baseline": 0.5,
        "theta_tonic": 0.12,
        "theta_phasic": 1.5,   # Very fast phasic (cholinergic bursts)
        "sigma_tonic": 0.04,
        "sigma_phasic": 0.08,
        "u_base": 0.15,        # AChE hydrolysis (very fast clearance)
        "d_base": 0.03,
        "c_base": 0.01,
        "fatigue_rate": 0.001,
    },

    # ── Oxytocin (OXT) ──────────────────────────────────────────
    # Slow, sustained peptide dynamics.
    # Primary role: social bonding, trust resonance, attunement.
    "OXT": {
        "C_baseline": 0.4,
        "theta_tonic": 0.03,   # Very slow reversion (sustained effects)
        "theta_phasic": 0.3,   # Slow phasic decay
        "sigma_tonic": 0.02,   # Low noise (peptide stability)
        "sigma_phasic": 0.04,
        "u_base": 0.05,        # Peptidase clearance (slow)
        "d_base": 0.02,
        "c_base": 0.01,
        "fatigue_rate": 0.001,
    },

    # ── Endogenous Opioid / μ-Opioid (MOR) ──────────────────────
    # Moderate, slow persistence.
    # Primary role: hedonic tone, comfort, affective buffering.
    "MOR": {
        "C_baseline": 0.35,
        "theta_tonic": 0.04,   # Slow tonic (endorphin persistence)
        "theta_phasic": 0.6,
        "sigma_tonic": 0.03,
        "sigma_phasic": 0.07,
        "u_base": 0.06,        # Peptidase + diffusion
        "d_base": 0.03,
        "c_base": 0.015,
        "fatigue_rate": 0.001,
    },

    # ── Endocannabinoid (CB1) ────────────────────────────────────
    # Very slow lipid signaling, retrograde modulatory.
    # Primary role: flexibility, filter inhibition, affective continuity.
    "CB1": {
        "C_baseline": 0.4,
        "theta_tonic": 0.03,   # Very slow (lipid signaling)
        "theta_phasic": 0.4,
        "sigma_tonic": 0.025,
        "sigma_phasic": 0.05,
        "u_base": 0.04,        # FAAH degradation (slow)
        "d_base": 0.02,
        "c_base": 0.01,
        "fatigue_rate": 0.001,
    },

    # ── Cortisol ─────────────────────────────────────────────────
    # Very slow HPA axis dynamics.
    # Primary role: time-horizon pressure, tradeoff enforcement,
    # stress-conditioned weighting.
    "cortisol": {
        "C_baseline": 0.3,
        "theta_tonic": 0.02,   # Very slow (HPA axis inertia)
        "theta_phasic": 0.2,   # Slow phasic
        "sigma_tonic": 0.015,  # Very low noise
        "sigma_phasic": 0.03,
        "u_base": 0.03,        # 11β-HSD metabolism
        "d_base": 0.015,
        "c_base": 0.01,
        "fatigue_rate": 0.001,
    },

    # ── CRH (Corticotropin-Releasing Hormone) ────────────────────
    # Moderate speed, acute stress trigger.
    # Primary role: acute stress drive, pressure scaling.
    "CRH": {
        "C_baseline": 0.25,
        "theta_tonic": 0.08,   # Moderate reversion
        "theta_phasic": 0.8,   # Fast phasic (acute stress)
        "sigma_tonic": 0.04,
        "sigma_phasic": 0.09,
        "u_base": 0.1,         # Peptidase clearance
        "d_base": 0.04,
        "c_base": 0.02,
        "fatigue_rate": 0.001,
    },

    # ── GABA ─────────────────────────────────────────────────────
    # Fast inhibition, high baseline.
    # Primary role: suppression, gating, stabilization.
    "GABA": {
        "C_baseline": 0.6,
        "theta_tonic": 0.1,    # Moderate reversion
        "theta_phasic": 1.0,   # Fast phasic decay
        "sigma_tonic": 0.04,
        "sigma_phasic": 0.08,
        "u_base": 0.12,        # GAT reuptake
        "d_base": 0.04,        # GABA transaminase
        "c_base": 0.02,
        "fatigue_rate": 0.001,
    },

    # ── Glutamate (GLU) ─────────────────────────────────────────
    # Fast excitation, tightly regulated.
    # Primary role: fast signal propagation, high-resolution integration.
    "GLU": {
        "C_baseline": 0.55,
        "theta_tonic": 0.12,   # Fast reversion (tight regulation)
        "theta_phasic": 1.3,   # Very fast phasic
        "sigma_tonic": 0.05,
        "sigma_phasic": 0.1,
        "u_base": 0.15,        # EAAT reuptake (very fast)
        "d_base": 0.05,        # Glutamine synthetase
        "c_base": 0.02,
        "fatigue_rate": 0.001,
    },

    # ── Histamine ──────────────────────────────────────────────
    # Moderate speed, wake-promoting neuromodulator.
    # Primary role: arousal, wakefulness, attention gating,
    # cognitive readiness.
    "histamine": {
        "C_baseline": 0.35,
        "theta_tonic": 0.08,   # Moderate reversion (wake-state)
        "theta_phasic": 0.9,   # Fast phasic decay
        "sigma_tonic": 0.04,
        "sigma_phasic": 0.08,
        "u_base": 0.1,         # Histamine N-methyltransferase
        "d_base": 0.04,        # MAO-B / diamine oxidase
        "c_base": 0.02,
        "fatigue_rate": 0.001,
    },
}


# =====================================================================
# Default receptor configs
# =====================================================================
# Each has K_d, parent_nt, exposure_tau.
# Receptor IDs MUST match those used in metrics.py (get_sat("DA_D3"),
# get_sat("OXTR"), get_sat("CB1"), get_sat("5HT_1A"), etc.).
# =====================================================================

DEFAULT_RECEPTOR_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ── DA receptors ─────────────────────────────────────────────
    "DA_D1": {
        "K_d": 0.4, "parent_nt": "DA", "exposure_tau": 10.0,
        "kd_band_coefficients": {"sigma": 0.1},   # D1 salience windows in sigma Up-states
    },
    "DA_D2": {"K_d": 0.3, "parent_nt": "DA", "exposure_tau": 12.0},
    "DA_D3": {"K_d": 0.2, "parent_nt": "DA", "exposure_tau": 15.0},
    "DA_D4": {"K_d": 0.35, "parent_nt": "DA", "exposure_tau": 10.0},
    "DA_D5": {"K_d": 0.45, "parent_nt": "DA", "exposure_tau": 10.0},

    # ── 5-HT receptors ──────────────────────────────────────────
    "5HT_1A": {"K_d": 0.3, "parent_nt": "5HT", "exposure_tau": 15.0},
    "5HT_1B": {"K_d": 0.35, "parent_nt": "5HT", "exposure_tau": 12.0},
    "5HT_2A": {"K_d": 0.4, "parent_nt": "5HT", "exposure_tau": 10.0},
    "5HT_2C": {"K_d": 0.35, "parent_nt": "5HT", "exposure_tau": 12.0},
    "5HT_3":  {"K_d": 0.5, "parent_nt": "5HT", "exposure_tau": 8.0},

    # ── NE receptors ─────────────────────────────────────────────
    "NE_alpha1": {"K_d": 0.5, "parent_nt": "NE", "exposure_tau": 10.0},
    "NE_alpha2": {"K_d": 0.25, "parent_nt": "NE", "exposure_tau": 12.0},
    "NE_beta1": {
        "K_d": 0.4, "parent_nt": "NE", "exposure_tau": 10.0,
        "kd_band_coefficients": {"sigma": 0.2},   # Infra-slow cAMP consolidation gating
    },
    "NE_beta2":  {"K_d": 0.45, "parent_nt": "NE", "exposure_tau": 10.0},

    # ── ACh receptors ────────────────────────────────────────────
    "ACh_nicotinic":  {"K_d": 0.5, "parent_nt": "ACh", "exposure_tau": 8.0},
    "ACh_muscarinic": {"K_d": 0.4, "parent_nt": "ACh", "exposure_tau": 12.0},

    # ── OXT receptor ─────────────────────────────────────────────
    # parent_nt override required: split("_")[0] gives "OXTR", not "OXT"
    "OXTR": {"K_d": 0.35, "parent_nt": "OXT", "exposure_tau": 20.0},

    # ── MOR receptor ─────────────────────────────────────────────
    "MOR_mu": {"K_d": 0.3, "parent_nt": "MOR", "exposure_tau": 15.0},

    # ── CB1 receptor ─────────────────────────────────────────────
    # Same name as NT — config merge handled in register_neurotransmitter()
    "CB1": {"K_d": 0.4, "parent_nt": "CB1", "exposure_tau": 20.0},

    # ── CRH receptor ────────────────────────────────────────────
    "CRH_R1": {"K_d": 0.45, "parent_nt": "CRH", "exposure_tau": 10.0},

    # ── GABA receptors ───────────────────────────────────────────
    "GABA_A": {
        "K_d": 0.5, "parent_nt": "GABA", "exposure_tau": 8.0,
        "kd_band_coefficients": {"sigma": 0.3},   # TRN GABA bursting drives spindle rhythm
    },
    "GABA_B": {
        "K_d": 0.4, "parent_nt": "GABA", "exposure_tau": 15.0,
        "kd_band_coefficients": {"sigma": 0.15},  # Slow IPSPs during spindle down-phase
    },

    # ── GLU receptors ────────────────────────────────────────────
    "GLU_NMDA":    {"K_d": 0.5, "parent_nt": "GLU", "exposure_tau": 10.0},
    "GLU_AMPA": {
        "K_d": 0.6, "parent_nt": "GLU", "exposure_tau": 6.0,
        "kd_band_coefficients": {"sigma": 0.25},  # TC relay burst-release during spindles
    },
    "GLU_KAINATE": {"K_d": 0.55, "parent_nt": "GLU", "exposure_tau": 8.0},
    "GLU_mGluR":   {"K_d": 0.35, "parent_nt": "GLU", "exposure_tau": 15.0},

    # ── Histamine receptors ─────────────────────────────────────
    "HIST_H1": {"K_d": 0.45, "parent_nt": "histamine", "exposure_tau": 10.0},
    "HIST_H2": {"K_d": 0.4,  "parent_nt": "histamine", "exposure_tau": 12.0},
    "HIST_H3": {"K_d": 0.25, "parent_nt": "histamine", "exposure_tau": 15.0},
    "HIST_H4": {"K_d": 0.5,  "parent_nt": "histamine", "exposure_tau": 10.0},
}


# =====================================================================
# NT → Receptor mapping
# =====================================================================

NT_RECEPTOR_MAP: Dict[str, List[str]] = {
    "DA":       ["DA_D1", "DA_D2", "DA_D3", "DA_D4", "DA_D5"],
    "5HT":      ["5HT_1A", "5HT_1B", "5HT_2A", "5HT_2C", "5HT_3"],
    "NE":       ["NE_alpha1", "NE_alpha2", "NE_beta1", "NE_beta2"],
    "ACh":      ["ACh_nicotinic", "ACh_muscarinic"],
    "OXT":      ["OXTR"],
    "MOR":      ["MOR_mu"],
    "CB1":      ["CB1"],
    "cortisol": [],              # concentration-only (no receptor dynamics)
    "CRH":      ["CRH_R1"],
    "GABA":     ["GABA_A", "GABA_B"],
    "GLU":      ["GLU_NMDA", "GLU_AMPA", "GLU_KAINATE", "GLU_mGluR"],
    "histamine": ["HIST_H1", "HIST_H2", "HIST_H3", "HIST_H4"],
}


# =====================================================================
# Registration helpers
# =====================================================================

def register_neurotransmitter(
    engine,
    nt_name: str,
    nt_config_overrides: Optional[Dict[str, Any]] = None,
    receptor_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    include_receptors: bool = True,
) -> None:
    """
    Register a single NT and its default receptors in the engine.

    Parameters
    ----------
    engine : NeurochemicalEngine
        Engine instance to register into.
    nt_name : str
        NT name (must be a key in DEFAULT_NT_CONFIGS).
    nt_config_overrides : dict, optional
        Override specific config values for the NT.
    receptor_config_overrides : dict, optional
        Override specific config values for receptors.
        Keys are receptor_ids.
    include_receptors : bool, default=True
        Whether to also register the NT's receptors.

    Raises
    ------
    KeyError
        If nt_name is not in DEFAULT_NT_CONFIGS.
    """
    if nt_name not in DEFAULT_NT_CONFIGS:
        raise KeyError(
            f"Unknown NT: {nt_name!r}. "
            f"Available: {sorted(DEFAULT_NT_CONFIGS)}"
        )

    # Build NT config
    config = dict(DEFAULT_NT_CONFIGS[nt_name])
    if nt_config_overrides:
        config.update(nt_config_overrides)

    engine.add_neurotransmitter(nt_name, config=config)

    # Register receptors
    if include_receptors:
        for receptor_id in NT_RECEPTOR_MAP.get(nt_name, []):
            r_config = dict(DEFAULT_RECEPTOR_CONFIGS[receptor_id])
            if receptor_config_overrides and receptor_id in receptor_config_overrides:
                r_config.update(receptor_config_overrides[receptor_id])

            # Handle CB1 name collision: receptor_id == nt_name
            # Merge NT config keys into receptor config so registry._configs
            # contains both sets of keys (they are disjoint).
            if receptor_id == nt_name:
                merged = dict(config)  # start with NT config
                merged.update(r_config)  # overlay receptor keys
                engine.add_receptor(receptor_id, config=merged)
            else:
                engine.add_receptor(receptor_id, config=r_config)


def register_all_neurotransmitters(
    engine,
    include_receptors: bool = True,
) -> None:
    """
    Register all 12 NT systems with their default receptors.

    Parameters
    ----------
    engine : NeurochemicalEngine
        Engine instance.
    include_receptors : bool, default=True
        Whether to also register receptors.
    """
    for nt_name in DEFAULT_NT_CONFIGS:
        register_neurotransmitter(
            engine, nt_name, include_receptors=include_receptors,
        )


def register_all_receptor_modules_on_engine(engine) -> None:
    """
    Register all receptor family modules on the engine.

    Populates the static ReceptorModuleRegistry, then registers each
    module on the engine so that _update_receptor computes family-specific
    effective signaling (A_ij) with per-subtype weights.

    Parameters
    ----------
    engine : NeurochemicalEngine
        Engine instance.
    """
    from zados.neurochem.receptors.receptor_registry import (
        ReceptorModuleRegistry,
        register_all_receptor_modules,
    )
    register_all_receptor_modules()
    for module in ReceptorModuleRegistry.get_all().values():
        engine.register_receptor_module(module)
