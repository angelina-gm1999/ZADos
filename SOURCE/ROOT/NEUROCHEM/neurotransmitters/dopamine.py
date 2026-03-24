from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional

from zados.neurochem.oscillations.modulation_links import modulate_parameters
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule,
    ReleaseDriveSpec,
    OscillationCouplingRule,
)

class Dopamine:
    """
    LEGACY: Batch/offline dopamine simulator for SimulationRunner.

    .. deprecated::
        For online/real-time simulation, use ``DAModule`` (below) with the
        ``NeurochemicalEngine``. This class is retained only for backward
        compatibility with ``SimulationRunner`` and ``test_dopamine_module.py``.

    Stochastic dopamine module simulating extracellular DA concentration
    with fatigue-modulated reuptake, novelty/reward-dependent release,
    and multiplicative vesicular noise.
    """

    def __init__(self, params: dict, rng: Optional[np.random.Generator] = None):
        """
        Initialize dopamine system.

        Parameters
        ----------
        params : dict
            Dictionary of kinetic parameters. Must include:
                R0, beta_nov, beta_rew,
                ku0, gamma, epsilon,
                kd, kl, alpha
        rng : np.random.Generator, optional
            RNG for reproducible noise
        """
        self.params = params
        self.rng = rng or np.random.default_rng()

        self.C = params.get("C0", 0.5)  # initial DA concentration
        self.F = params.get("F0", 0.0)  # initial fatigue

    def release(self, novelty: float, rpe: float) -> float:
        p = self.params
        return p["R0"] + p["beta_nov"] * novelty + p["beta_rew"] * rpe

    def reuptake(self) -> float:
        p = self.params
        ku = p["ku0"] * (1 - p["gamma"] * self.F)
        return ku * self.C

    def diffusion(self) -> float:
        return self.params["kd"] * self.C

    def degradation(self) -> float:
        return self.params["kl"] * self.C

    def fatigue_update(self, dt: float):
        eps = self.params["epsilon"]
        self.F += eps * (self.C - self.F) * dt
        self.F = np.clip(self.F, 0.0, 1.0)

    def noise(self) -> float:
        alpha = self.params["alpha"]
        return alpha * np.sqrt(max(self.C, 0.0)) * self.rng.normal(0, 1)
    
    def step(self, novelty: float, rpe: float, dt: float, oscillations: dict) -> float:
        """
        Run one Euler–Maruyama update step with oscillatory modulation.

        Parameters
        ----------
        novelty : float
            Novelty input N(t)
        rpe : float
            Reward prediction error RPE(t)
        dt : float
            Time step (e.g., 0.01)
        oscillations : dict
            Oscillation band power levels (e.g., {'gamma': 0.7})

        Returns
        -------
        float
            Updated dopamine concentration
        """
        # 1. Modulate parameters based on oscillations
        modulated_params = modulate_parameters(self.params, oscillations)

        # Temporarily swap params for component calculations
        original_params = self.params
        self.params = modulated_params

        # 2. Compute components
        R = self.release(novelty, rpe)
        U = self.reuptake()
        D = self.diffusion()
        L = self.degradation()
        noise_term = self.noise()

        # Restore original params
        self.params = original_params

        # 3. Update concentration
        self.C += (R - U - D - L) * dt + noise_term * np.sqrt(dt)
        self.C = max(self.C, 0.0)

        # 4. Update fatigue
        self.fatigue_update(dt)

        return self.C

    def state(self) -> dict:
        """Return current concentration and fatigue."""
        return {"C": self.C, "F": self.F}


# =====================================================================
# DAModule: Per-NT behavior descriptor for the engine dispatch system
# =====================================================================

class DAModule(NeurotransmitterModule):
    """
    Dopamine behavior specification for the NeurochemicalEngine.

    Release drivers:
        novelty: Novelty detection signal (phasic burst)
        rpe: Reward prediction error (positive = better than expected)
        effort: Task demand / effort signal
        emotion_drive: Emotion layer modulation signal

    Oscillation coupling (per PDF Appendix I):
        gamma -> release boost (phasic burst amplification)
        theta -> K_d modulation (receptor affinity increase)
        alpha -> noise suppression (sigma reduction)

    Primary role: motivation, reward prediction, novelty seeking.
    """

    @property
    def name(self) -> str:
        return "DA"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["novelty", "rpe", "effort", "emotion_drive"],
            weights=[0.35, 0.30, 0.15, 0.20],
            threshold=0.0,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            # Gamma boosts phasic release (PDF Appendix E.2.2, H.4)
            OscillationCouplingRule(
                target="release", band="gamma", coefficient=0.5,
            ),
            # Theta modulates receptor affinity via K_d (PDF E.2.1, H.3)
            OscillationCouplingRule(
                target="K_d", band="theta", coefficient=-0.3,
                formula="multiplicative",
            ),
            # Alpha suppresses noise (PDF H.6)
            OscillationCouplingRule(
                target="sigma_tonic", band="alpha", coefficient=-0.4,
            ),
            OscillationCouplingRule(
                target="sigma_phasic", band="alpha", coefficient=-0.4,
            ),
        ]

    def compute_release_drive(
        self,
        modulation_signals: Dict[str, float],
    ) -> float:
        """
        DA-specific release drive.

        DA uniquely allows negative RPE to REDUCE release drive
        (reward prediction error can be negative), but the total
        drive is still clamped to non-negative.
        """
        spec = self.release_spec
        total = 0.0
        for key, weight in zip(spec.signal_keys, spec.weights):
            total += weight * modulation_signals.get(key, 0.0)
        # DA allows sub-threshold to just be 0 (no burst),
        # but does NOT have a hard threshold — RPE can be negative
        return max(0.0, total)
