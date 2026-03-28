"""
Sleep Neurochemical State Manager (Spec §8.1).

Orchestrates sleep-mode neurochemical state: phase transitions,
baseline management, containment monitoring, and waking return.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from .state_vectors import (
    SleepPhase,
    SleepNTStateVector,
    SLEEP_STATE_VECTORS,
)
from .transitions import (
    TransitionConfig,
    DEFAULT_TRANSITION_CONFIG,
    FAST_COLLAPSE_NTS,
    compute_transition_step,
    transition_nt_baselines,
    transition_osc_config,
    check_triage_to_rem_conditions,
    check_rem_to_dream_conditions,
)
from .containment import (
    check_containment,
    check_dream_state_validity,
)


class SleepNeurochemicalStateManager:
    """Manages neurochemical state throughout sleep processing.

    Responsibilities (Spec §8.1):
    - Hold prescribed state vectors for each sleep phase
    - Smooth transitions between phases using pharmacodynamic rules
    - Monitor containment integrity during dream mode
    - Execute post-sleep reconciliation

    Parameters
    ----------
    transition_config : TransitionConfig, optional
        Rate constants for transitions.
    state_vectors : dict, optional
        Phase -> SleepNTStateVector mapping. Defaults to spec values.
    """

    def __init__(
        self,
        transition_config: Optional[TransitionConfig] = None,
        state_vectors: Optional[Dict[SleepPhase, SleepNTStateVector]] = None,
    ) -> None:
        self._config = transition_config or DEFAULT_TRANSITION_CONFIG
        self._state_vectors = state_vectors or SLEEP_STATE_VECTORS

        self._phase: SleepPhase = SleepPhase.WAKING
        self._current_nt_baselines: Dict[str, float] = {}
        self._current_osc_config: Dict[str, float] = {}
        self._waking_nt_baselines: Dict[str, float] = {}
        self._waking_osc_config: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> SleepPhase:
        """Current sleep phase."""
        return self._phase

    @property
    def current_baselines(self) -> Dict[str, float]:
        """Current NT tonic baselines."""
        return dict(self._current_nt_baselines)

    @property
    def current_osc_config(self) -> Dict[str, float]:
        """Current oscillatory configuration."""
        return dict(self._current_osc_config)

    def is_active(self) -> bool:
        """True if in any sleep phase (not waking)."""
        return self._phase != SleepPhase.WAKING

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def enter_sleep(
        self,
        waking_baselines: Dict[str, float],
        waking_osc: Dict[str, float],
    ) -> None:
        """Enter sleep mode from waking state.

        Saves waking baselines for later restoration, sets phase to
        TRIAGE, and initializes current values to waking (transitions
        happen via step_transition).

        Parameters
        ----------
        waking_baselines : dict
            Current waking NT baselines to save.
        waking_osc : dict
            Current waking oscillatory config to save.
        """
        self._waking_nt_baselines = deepcopy(waking_baselines)
        self._waking_osc_config = deepcopy(waking_osc)
        self._current_nt_baselines = deepcopy(waking_baselines)
        self._current_osc_config = deepcopy(waking_osc)
        self._phase = SleepPhase.TRIAGE

    def step_transition(self, dt: float = 1.0) -> Dict:
        """Apply one transition step toward current phase targets.

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        dict
            {
                "nt_baselines": {...},
                "osc_config": {...},
                "phase": SleepPhase,
                "alerts": [...],
            }
        """
        if self._phase == SleepPhase.WAKING:
            return {
                "nt_baselines": self.current_baselines,
                "osc_config": self.current_osc_config,
                "phase": self._phase,
                "alerts": [],
            }

        target_vector = self._get_target_vector()
        k = self._get_current_rate()

        # Determine if fast collapse applies (only rem -> dream transition)
        use_fast = self._phase == SleepPhase.DREAM
        fast_nts = FAST_COLLAPSE_NTS if use_fast else frozenset()

        self._current_nt_baselines = transition_nt_baselines(
            self._current_nt_baselines,
            target_vector.nt_baselines,
            k, dt,
            fast_nts=fast_nts,
            k_fast=self._config.k_fast,
        )

        self._current_osc_config = transition_osc_config(
            self._current_osc_config,
            target_vector.oscillatory_config,
            k, dt,
        )

        alerts = self.monitor_containment({})

        return {
            "nt_baselines": self.current_baselines,
            "osc_config": self.current_osc_config,
            "phase": self._phase,
            "alerts": alerts,
        }

    def try_advance_phase(
        self,
        desensitization_flag: bool = False,
        stagnated_queue_nonempty: bool = False,
    ) -> bool:
        """Try to advance to the next sleep phase.

        Parameters
        ----------
        desensitization_flag : bool
            5-HT1A desensitization flag (for rem -> dream).
        stagnated_queue_nonempty : bool
            Whether stagnated concept queue has items.

        Returns
        -------
        bool
            True if phase advanced.
        """
        if self._phase == SleepPhase.TRIAGE:
            if check_triage_to_rem_conditions(
                self._current_nt_baselines,
                self._current_osc_config,
            ):
                self._phase = SleepPhase.REM_PROCESSING
                return True

        elif self._phase == SleepPhase.REM_PROCESSING:
            if check_rem_to_dream_conditions(
                self._current_nt_baselines,
                self._current_osc_config,
                desensitization_flag=desensitization_flag,
                stagnated_queue_nonempty=stagnated_queue_nonempty,
            ):
                self._phase = SleepPhase.DREAM
                return True

        return False

    def monitor_containment(
        self,
        saturations: Dict[str, float],
    ) -> List[str]:
        """Monitor dream-mode containment integrity.

        Only active during DREAM phase.

        Parameters
        ----------
        saturations : dict
            Receptor saturation values (e.g. {"GABA_A": 0.7}).

        Returns
        -------
        list[str]
            Alert strings (empty = safe).
        """
        alerts: List[str] = []

        if self._phase != SleepPhase.DREAM:
            return alerts

        # GABA-A containment check
        gaba_a_sat = saturations.get("GABA_A", 1.0)
        if not check_containment(gaba_a_sat):
            alerts.append(
                f"CONTAINMENT_BREACH: GABA-A saturation {gaba_a_sat:.3f} "
                f"below threshold 0.55"
            )

        # NE/5-HT floor check
        ne_val = self._current_nt_baselines.get("NE", 0.0)
        sht_val = self._current_nt_baselines.get("5HT", 0.0)
        if not check_dream_state_validity(ne_val, sht_val):
            alerts.append(
                f"DREAM_STATE_INVALID: NE={ne_val:.3f}, 5HT={sht_val:.3f} "
                f"(ceiling=0.10)"
            )

        return alerts

    def exit_sleep(self) -> None:
        """Exit sleep mode, targeting saved waking baselines.

        Sets phase to WAKING. The actual transition back to waking
        values should be driven by calling step_transition with
        dt until convergence, but the target is now waking baselines.
        """
        self._phase = SleepPhase.WAKING
        self._current_nt_baselines = deepcopy(self._waking_nt_baselines)
        self._current_osc_config = deepcopy(self._waking_osc_config)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict:
        """Serialize state for persistence."""
        return {
            "phase": self._phase.value,
            "current_nt_baselines": dict(self._current_nt_baselines),
            "current_osc_config": dict(self._current_osc_config),
            "waking_nt_baselines": dict(self._waking_nt_baselines),
            "waking_osc_config": dict(self._waking_osc_config),
            "transition_config": {
                "k_enter": self._config.k_enter,
                "k_phase": self._config.k_phase,
                "k_exit": self._config.k_exit,
                "k_fast": self._config.k_fast,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SleepNeurochemicalStateManager":
        """Restore from serialized state."""
        config = TransitionConfig(**data.get("transition_config", {}))
        mgr = cls(transition_config=config)
        mgr._phase = SleepPhase(data.get("phase", "waking"))
        mgr._current_nt_baselines = dict(data.get("current_nt_baselines", {}))
        mgr._current_osc_config = dict(data.get("current_osc_config", {}))
        mgr._waking_nt_baselines = dict(data.get("waking_nt_baselines", {}))
        mgr._waking_osc_config = dict(data.get("waking_osc_config", {}))
        return mgr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_target_vector(self) -> SleepNTStateVector:
        """Get the target state vector for the current phase."""
        return self._state_vectors[self._phase]

    def _get_current_rate(self) -> float:
        """Get the transition rate for the current phase context."""
        if self._phase == SleepPhase.TRIAGE:
            return self._config.k_enter
        return self._config.k_phase
