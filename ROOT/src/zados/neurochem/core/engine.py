from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from zados.neurochem.core.registry import NeurochemicalRegistry
from zados.neurochem.state import (
    NeurotransmitterState,
    ReceptorState,
    OscillationState,
)
from zados.neurochem.kinetics.mass_balance import (
    compute_drift_term,
    compute_diffusion_term,
    compute_effective_reversion_rate,
)
from zados.neurochem.kinetics.release_drives import (
    compute_combined_release_drive,
    apply_fatigue_gating,
    apply_oscillatory_gating,
    compute_phasic_burst_amplitude,
)
from zados.neurochem.stochastic_modulation.euler_maruyama import (
    euler_maruyama_step_bounded,
)
from zados.neurochem.kinetics.receptor_dynamics import (
    step_receptor_dynamics,
    compute_saturation,
    compute_effective_concentration,
)
from zados.neurochem.receptors.base import ReceptorFamilyModule
from zados.neurochem.neurosymbolic.readout import (
    compute_neurosymbolic_readout,
)
from zados.neurochem.neurotransmitters.base import NeurotransmitterModule
from zados.neurochem.oscillations.oscillation_modulation import (
    modulate_noise,
    modulate_noise_multiband,
    modulate_K_d,
    modulate_K_d_multiband,
    compute_g_chi,
    compute_effective_signaling_proxy,
)


class NeurochemicalEngine:
    """
    Real-time neurochemical simulator for online integration with reward system.
    
    This is the ONLINE/REAL-TIME simulator that steps once per call with
    instantaneous modulation signals. Use this when integrating with the
    reward system during operation.
    
    For batch/offline simulation (run entire t=0→T with functions of time),
    use NeurochemicalSimulation instead.
    
    Key differences:
    - Online: step(signals) → updates state → step(signals) → repeat
    - Batch: run() → returns full history
    
    Coordinates state updates across neurotransmitters, receptors, and oscillations.
    Applies kinetic equations, stochastic integration, and computes derived metrics.
    
    Attributes
    ----------
    registry : NeurochemicalRegistry
        Central registry of all neurochemical components
    current_time : float
        Current simulation time
    dt : float
        Integration time step
    
    Examples
    --------
    >>> sim = NeurochemicalSimulator(dt=0.01)
    >>> sim.add_neurotransmitter("DA", config={"C_baseline": 0.5})
    >>> 
    >>> # Step once with reward adapter signals
    >>> signals = adapter.transform(domain_results)
    >>> sim.step(signals)
    >>> 
    >>> # Get current metrics
    >>> metrics = sim.get_neurosymbolic_readout()
    >>> print(metrics["motivation"])
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        seed: Optional[int] = None,
        use_lambda_loc_routing: bool = False,
        oscillation_mode: str = "static",
    ):
        """
        Initialize simulator.

        Parameters
        ----------
        dt : float, default=0.01
            Integration time step
        seed : int, optional
            Random seed for reproducibility
        use_lambda_loc_routing : bool, default=False
            When True, receptors see concentration routed by lambda_loc
            (presynaptic/synaptic/extrasynaptic). When False (default),
            all receptors see total C = C_tonic + C_phasic.
        oscillation_mode : str, default="static"
            How oscillation state is updated each step:
            - "static": oscillations are set externally (default, backward compat)
            - "state_derived": derive from NT concentrations each step
            - "external": same as static (placeholder for future external driver)
        """
        self.registry = NeurochemicalRegistry()
        self.current_time = 0.0
        self.dt = dt
        self.use_lambda_loc_routing = use_lambda_loc_routing
        self.oscillation_mode = oscillation_mode
        self._nt_modules: Dict[str, NeurotransmitterModule] = {}
        self._receptor_modules: Dict[str, ReceptorFamilyModule] = {}
        self._step_number: int = 0
        self.scheduler = None  # Optional[SparseUpdateScheduler] for N.2

        # RNG: use numpy Generator (PCG64) for reproducible stochastic integration
        self.rng: np.random.Generator = np.random.default_rng(seed)
    
    def add_neurotransmitter(
        self,
        name: str,
        initial_state: Optional[NeurotransmitterState] = None,
        config: Optional[dict] = None,
    ):
        """
        Add a neurotransmitter to the simulation.
        
        Parameters
        ----------
        name : str
            Neurotransmitter identifier (e.g., "DA", "5-HT", "NE")
        initial_state : NeurotransmitterState, optional
            Initial state (default: baseline)
        config : dict, optional
            Configuration parameters (theta, sigma, baseline, etc.)
        """
        if initial_state is None:
            initial_state = NeurotransmitterState()
        
        if config is None:
            config = {}
        
        self.registry.register_neurotransmitter(name, initial_state, config)
    
    def add_receptor(
        self,
        receptor_id: str,
        initial_state: Optional[ReceptorState] = None,
        config: Optional[dict] = None,
    ):
        """
        Add a receptor to the simulation.
        
        Parameters
        ----------
        receptor_id : str
            Receptor identifier (e.g., "DA_D1", "5HT_2A")
        initial_state : ReceptorState, optional
            Initial state
        config : dict, optional
            Configuration parameters (K_d, etc.)
        """
        if initial_state is None:
            initial_state = ReceptorState(receptor_id=receptor_id)
        
        if config is None:
            config = {}
        
        self.registry.register_receptor(receptor_id, initial_state, config)
    
    def set_oscillation_state(self, oscillation_state: OscillationState):
        """
        Set the global oscillation state.

        Parameters
        ----------
        oscillation_state : OscillationState
            Oscillation state
        """
        self.registry.set_oscillations(oscillation_state)

    def register_nt_module(self, module: NeurotransmitterModule) -> None:
        """
        Register a per-NT behavior module.

        When a module is registered for an NT, the engine uses the
        module's release logic and oscillation coupling rules instead
        of the generic update path.

        Parameters
        ----------
        module : NeurotransmitterModule
            Module instance whose .name matches a registered NT.
        """
        self._nt_modules[module.name] = module

    def register_receptor_module(self, module: ReceptorFamilyModule) -> None:
        """
        Register a receptor family behavior module.

        When registered, the engine uses the module's effective signaling
        computation (including per-subtype weights) instead of the generic
        proxy. Also enables emotion plasticity and subtype switching for
        the receptor family.

        Parameters
        ----------
        module : ReceptorFamilyModule
            Module instance whose .parent_nt identifies the family.
        """
        self._receptor_modules[module.parent_nt] = module

    def step(
        self,
        modulation_signals: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Advance simulation by one time step.
        
        Parameters
        ----------
        modulation_signals : dict, optional
            Structured modulation signals from reward adapter
            Format: {
                "DA": {"novelty": float, "rpe": float, "effort": float},
                "NE": {"precision": float, "uncertainty": float},
                ...
            }
        """
        if modulation_signals is None:
            modulation_signals = {}
        
        # Update each neurotransmitter (fast — every tick)
        for nt_name in self.registry.neurotransmitter_names():
            nt_signals = modulation_signals.get(nt_name, {})
            self._update_neurotransmitter(nt_name, nt_signals)

        # State-derived oscillation update (Phase 28)
        # Gated by scheduler when present (N.2)
        _do_osc = self.oscillation_mode == "state_derived"
        if _do_osc and self.scheduler is not None:
            _do_osc = self.scheduler.should_update("oscillation", self._step_number)
        if _do_osc:
            self._derive_oscillations()

        # Update receptors (Phase 2)
        # Gated by scheduler when present (N.2)
        _do_receptors = True
        if self.scheduler is not None:
            _do_receptors = self.scheduler.should_update("receptor", self._step_number)
        if _do_receptors:
            for receptor_id in self.registry.receptor_ids():
                self._update_receptor(receptor_id)

            # Subtype switching (Phase 23) — only when receptors update
            self._apply_subtype_switching()


        # Increment time and step counter
        self.current_time += self.dt
        self._step_number += 1
    
    def get_neurosymbolic_readout(self) -> dict:
        """
        Compute high-level cognitive/affective metrics from current state.

        Returns
        -------
        dict
            Neurosymbolic metrics (motivation, empathy, fatigue, etc.)
        """
        # Collect neurotransmitter states
        neurotransmitter_states = {}
        for nt_name in self.registry.neurotransmitter_names():
            neurotransmitter_states[nt_name] = self.registry.get_neurotransmitter(nt_name)

        # Collect receptor states and configs
        receptor_states = {}
        receptor_configs = {}

        for receptor_id in self.registry.receptor_ids():
            receptor_states[receptor_id] = self.registry.get_receptor(receptor_id)
            config = self.registry.get_config(receptor_id)
            K_d = config.get("K_d", 0.5)
            receptor_configs[receptor_id] = {"K_d": K_d}

        # Get oscillation state
        oscillation_state = self.registry.get_oscillations()

        # Compute readout
        metrics = compute_neurosymbolic_readout(
            neurotransmitter_states=neurotransmitter_states,
            receptor_states=receptor_states,
            oscillation_state=oscillation_state,
            receptor_configs=receptor_configs,
        )
        return metrics.as_dict()
    
    def _update_neurotransmitter(
        self,
        nt_name: str,
        modulation_signals: Dict[str, float],
    ):
        """
        Update a single neurotransmitter's concentration.

        Dispatches to per-NT module if registered, otherwise falls
        back to the generic update path.

        Parameters
        ----------
        nt_name : str
            Neurotransmitter name
        modulation_signals : dict
            Modulation signals for this NT (novelty, rpe, effort, etc.)
        """
        module = self._nt_modules.get(nt_name)
        if module is not None:
            self._update_nt_with_module(nt_name, modulation_signals, module)
        else:
            self._update_nt_generic(nt_name, modulation_signals)

    def _update_nt_generic(
        self,
        nt_name: str,
        modulation_signals: Dict[str, float],
    ):
        """
        Generic NT update (fallback for NTs without a registered module).

        Uses DA-centric release drives (novelty, rpe, effort) and
        PDF-aligned oscillation coupling via pure functions:
        alpha → noise suppression, gamma → release boost.
        """
        state = self.registry.get_neurotransmitter(nt_name)
        config = self.registry.get_config(nt_name)
        oscillations = self.registry.get_oscillations()

        # Get parameters
        C_baseline = config.get("C_baseline", 0.5)
        theta_tonic = config.get("theta_tonic", 0.1)
        theta_phasic = config.get("theta_phasic", 1.0)
        sigma_tonic = config.get("sigma_tonic", 0.05)
        sigma_phasic = config.get("sigma_phasic", 0.1)
        u_base = config.get("u_base", 0.1)
        d_base = config.get("d_base", 0.05)
        c_base = config.get("c_base", 0.02)
        fatigue_rate = config.get("fatigue_rate", 0.001)

        # Oscillation coupling — noise modulation (PDF Appendix H.6):
        # Multi-band: sigma_mod = sigma * max(floor, 1 - Σ s_k*φ_k + Σ a_k*φ_k)
        # Legacy: alpha-only noise suppression
        noise_band_coefficients = config.get("noise_band_coefficients")
        if oscillations and noise_band_coefficients:
            osc_amps = self._build_osc_amplitudes(oscillations)
            suppression = noise_band_coefficients.get("suppression", {})
            amplification = noise_band_coefficients.get("amplification", {})
            sigma_tonic = modulate_noise_multiband(
                sigma_tonic, osc_amps, suppression, amplification,
            )
            sigma_phasic = modulate_noise_multiband(
                sigma_phasic, osc_amps, suppression, amplification,
            )
        elif oscillations:
            sigma_tonic = modulate_noise(sigma_tonic, oscillations.alpha)
            sigma_phasic = modulate_noise(sigma_phasic, oscillations.alpha)

        # Compute release drive from modulation signals
        burst_amplitude = 0.0
        if modulation_signals:
            novelty = modulation_signals.get("novelty", 0.0)
            rpe = modulation_signals.get("rpe", 0.0)
            effort = modulation_signals.get("effort", 0.0)
            emotion_drive = modulation_signals.get("emotion_drive", 0.0)

            release_drive = compute_combined_release_drive(
                novelty,
                rpe,
                effort,
            )

            # Fold in emotion-driven release (from emotion_profile_to_signals).
            # emotion_drive is additive to the combined release drive so
            # that reactive emotions modulate every NT, not just those
            # with dedicated per-NT modules.
            release_drive += emotion_drive

            # Apply fatigue gating
            release_drive = apply_fatigue_gating(
                release_drive,
                state.F,
                fatigue_threshold=0.7,
            )

            # Apply oscillatory gating
            if oscillations:
                # Gamma boosts phasic release (PDF Appendix I)
                release_drive = apply_oscillatory_gating(
                    release_drive,
                    oscillations.gamma,
                    band_preference=0.5,
                )

            # Compute phasic burst amplitude
            burst_amplitude = compute_phasic_burst_amplitude(
                release_drive,
                max_burst=1.0,
                receptor_sensitivity=2.0,
            )

        # Shared integration pipeline
        new_state = self._integrate_nt_state(
            state=state,
            C_baseline=C_baseline,
            theta_tonic=theta_tonic,
            theta_phasic=theta_phasic,
            sigma_tonic=sigma_tonic,
            sigma_phasic=sigma_phasic,
            u_base=u_base,
            d_base=d_base,
            c_base=c_base,
            burst_amplitude=burst_amplitude,
            fatigue_rate=fatigue_rate,
        )

        # Update registry
        self.registry.register_neurotransmitter(nt_name, new_state, config)

    def _update_nt_with_module(
        self,
        nt_name: str,
        modulation_signals: Dict[str, float],
        module: NeurotransmitterModule,
    ):
        """
        Module-driven NT update.

        Uses the per-NT module's release logic and oscillation coupling
        rules instead of the generic hardcoded path.

        Parameters
        ----------
        nt_name : str
            Neurotransmitter name
        modulation_signals : dict
            Modulation signals for this NT
        module : NeurotransmitterModule
            Per-NT behavior module
        """
        state = self.registry.get_neurotransmitter(nt_name)
        config = self.registry.get_config(nt_name)
        oscillations = self.registry.get_oscillations()

        # Build params dict from config
        C_baseline = config.get("C_baseline", 0.5)
        params = {
            "theta_tonic": config.get("theta_tonic", 0.1),
            "theta_phasic": config.get("theta_phasic", 1.0),
            "sigma_tonic": config.get("sigma_tonic", 0.05),
            "sigma_phasic": config.get("sigma_phasic", 0.1),
            "u_base": config.get("u_base", 0.1),
            "d_base": config.get("d_base", 0.05),
            "c_base": config.get("c_base", 0.02),
        }

        # Step 1: Module applies oscillation coupling to kinetic params
        if oscillations:
            osc_amplitudes = {
                "delta": oscillations.delta,
                "theta": oscillations.theta,
                "alpha": oscillations.alpha,
                "beta": oscillations.beta,
                "gamma": oscillations.gamma,
                "theta_gamma": oscillations.theta_gamma_coupling(),
                "alpha_beta": oscillations.alpha_beta_coupling(),
            }
            params = module.apply_oscillation_coupling(params, osc_amplitudes)

        # Step 2: Module computes release drive from signals
        release_drive = module.compute_release_drive(modulation_signals)

        # Step 3: Apply fatigue gating
        release_drive = apply_fatigue_gating(
            release_drive,
            state.F,
            fatigue_threshold=0.7,
        )

        # Step 4: Apply oscillatory gating using module's primary release band
        if oscillations:
            primary_band = module.get_primary_release_band()
            primary_coeff = module.get_primary_release_coefficient()
            if primary_band is not None:
                band_amp = osc_amplitudes.get(primary_band, 0.0)
                release_drive = apply_oscillatory_gating(
                    release_drive,
                    band_amp,
                    band_preference=primary_coeff,
                )

        # Step 5: Compute phasic burst amplitude
        burst_amplitude = compute_phasic_burst_amplitude(
            release_drive,
            max_burst=1.0,
            receptor_sensitivity=2.0,
        )

        # Steps 6-10: Shared integration pipeline
        new_state = self._integrate_nt_state(
            state=state,
            C_baseline=C_baseline,
            theta_tonic=params["theta_tonic"],
            theta_phasic=params["theta_phasic"],
            sigma_tonic=params["sigma_tonic"],
            sigma_phasic=params["sigma_phasic"],
            u_base=params["u_base"],
            d_base=params["d_base"],
            c_base=params["c_base"],
            burst_amplitude=burst_amplitude,
            fatigue_rate=config.get("fatigue_rate", 0.001),
        )
        self.registry.register_neurotransmitter(nt_name, new_state, config)

    def _integrate_nt_state(
        self,
        state: NeurotransmitterState,
        C_baseline: float,
        theta_tonic: float,
        theta_phasic: float,
        sigma_tonic: float,
        sigma_phasic: float,
        u_base: float,
        d_base: float,
        c_base: float,
        burst_amplitude: float,
        fatigue_rate: float = 0.001,
    ) -> NeurotransmitterState:
        """
        Shared integration pipeline: drift → diffusion → EM step → fatigue.

        Called by both ``_update_nt_generic`` and ``_update_nt_with_module``
        after they have computed their respective burst amplitudes and
        (possibly oscillation-modulated) kinetic parameters.

        Parameters
        ----------
        state : NeurotransmitterState
            Current state
        C_baseline : float
            Tonic baseline target
        theta_tonic, theta_phasic : float
            Mean-reversion rates (pre-oscillation-modulation)
        sigma_tonic, sigma_phasic : float
            Noise amplitudes (possibly already oscillation-modulated)
        u_base, d_base, c_base : float
            Clearance rate coefficients
        burst_amplitude : float
            Phasic burst amplitude (0.0 if no burst);
            injected into phasic drift as burst_amplitude / dt
        fatigue_rate : float, default=0.001
            Rate of fatigue accumulation per unit time

        Returns
        -------
        NeurotransmitterState
            Updated state after one EM integration step
        """
        # Fatigue-modulated reversion rates.
        # Tonic uses higher fatigue_scaling (0.5) because sustained
        # baseline maintenance is more energetically costly, so fatigue
        # slows tonic reversion more. Phasic uses lower scaling (0.3)
        # because transient bursts are less affected by metabolic fatigue.
        theta_tonic_eff = compute_effective_reversion_rate(
            theta_tonic,
            state.F,
            fatigue_scaling=0.5,
        )
        theta_phasic_eff = compute_effective_reversion_rate(
            theta_phasic,
            state.F,
            fatigue_scaling=0.3,
        )

        # Drift terms
        drift_tonic = compute_drift_term(
            state.C_tonic,
            C_baseline,
            theta_tonic_eff,
            state.eta_u,
            u_base,
            d_base,
            c_base,
        )

        drift_phasic = compute_drift_term(
            state.C_phasic,
            0.0,  # Phasic decays to zero
            theta_phasic_eff,
            state.eta_u,
            u_base,
            d_base,
            c_base,
        )
        drift_phasic += burst_amplitude / self.dt

        # Diffusion terms
        diffusion_tonic = compute_diffusion_term(
            state.C_tonic,
            sigma_tonic,
            multiplicative=True,
        )
        diffusion_phasic = compute_diffusion_term(
            state.C_phasic,
            sigma_phasic,
            multiplicative=True,
        )

        # Euler-Maruyama integration (dW from numpy RNG for reproducibility)
        sqrt_dt = math.sqrt(self.dt)
        C_tonic_new = euler_maruyama_step_bounded(
            state.C_tonic,
            drift_tonic,
            diffusion_tonic,
            self.dt,
            lower_bound=0.0,
            upper_bound=1.0,
            dW=float(self.rng.normal(0.0, sqrt_dt)),
        )
        C_phasic_new = euler_maruyama_step_bounded(
            state.C_phasic,
            drift_phasic,
            diffusion_phasic,
            self.dt,
            lower_bound=0.0,
            upper_bound=1.0,
            dW=float(self.rng.normal(0.0, sqrt_dt)),
        )

        # Fatigue update (slow accumulation)
        F_new = state.F + fatigue_rate * self.dt
        F_new = max(0.0, min(1.0, F_new))

        return NeurotransmitterState(
            C_tonic=C_tonic_new,
            C_phasic=C_phasic_new,
            F=F_new,
            eta_u=state.eta_u,
        )

    def _build_osc_amplitudes(
        self,
        oscillations: OscillationState,
    ) -> Dict[str, float]:
        """
        Build a flat dict of oscillation band amplitudes including CFC.

        Parameters
        ----------
        oscillations : OscillationState
            Current oscillation state

        Returns
        -------
        dict
            Band name -> amplitude (includes theta_gamma, alpha_beta CFC)
        """
        return {
            "delta": oscillations.delta,
            "theta": oscillations.theta,
            "alpha": oscillations.alpha,
            "beta": oscillations.beta,
            "gamma": oscillations.gamma,
            "sigma": oscillations.sigma,
            "theta_gamma": oscillations.theta_gamma_coupling(),
            "alpha_beta": oscillations.alpha_beta_coupling(),
            "delta_sigma": oscillations.delta_sigma_coupling(),
        }

    def _derive_oscillations(self) -> None:
        """
        Derive oscillation state from current NT concentrations.

        Called in step() when oscillation_mode == "state_derived".
        Collects all registered NT states and passes them to
        derive_oscillation_state().
        """
        from zados.neurochem.oscillations.generators import derive_oscillation_state

        nt_states = {}
        for nt_name in self.registry.neurotransmitter_names():
            nt_states[nt_name] = self.registry.get_neurotransmitter(nt_name)
        new_osc = derive_oscillation_state(nt_states)
        self.registry.set_oscillations(new_osc)

    def _update_receptor(self, receptor_id: str):
        """
        Update a single receptor's state using CTMC transition dynamics.

        Parameters
        ----------
        receptor_id : str
            Receptor identifier (e.g., "DA_D1")
        """
        receptor_state = self.registry.get_receptor(receptor_id)

        try:
            receptor_config = self.registry.get_config(receptor_id)
        except KeyError:
            receptor_config = {}

        # Infer parent NT: "DA_D1" -> "DA", "5HT_2A" -> "5HT"
        parent_nt = receptor_config.get("parent_nt", receptor_id.split("_")[0])

        # Get NT concentration
        try:
            nt_state = self.registry.get_neurotransmitter(parent_nt)
            if self.use_lambda_loc_routing:
                concentration = compute_effective_concentration(
                    nt_state.C_tonic, nt_state.C_phasic,
                    receptor_state.lambda_loc,
                )
            else:
                concentration = nt_state.C
        except KeyError:
            concentration = 0.0

        K_d = receptor_config.get("K_d", 0.5)

        # Get oscillation state
        oscillations = self.registry.get_oscillations()
        beta_amplitude = oscillations.beta if oscillations else 0.0

        # K_d oscillation modulation (PDF Appendix H.3)
        # Multi-band: K_d(t) = K_d * (1 - Σ α_k * φ_k)
        # Legacy: theta-only modulation
        kd_band_coefficients = receptor_config.get("kd_band_coefficients")
        if oscillations and kd_band_coefficients:
            osc_amps = self._build_osc_amplitudes(oscillations)
            K_d = modulate_K_d_multiband(K_d, osc_amps, kd_band_coefficients)
        elif oscillations and oscillations.theta > 0:
            K_d = modulate_K_d(K_d, oscillations.theta)

        thresholds = receptor_config.get("thresholds", None)
        exposure_tau = receptor_config.get("exposure_tau", 10.0)

        # Multi-band CTMC transition modulation (Phase 26/27)
        transition_specs = receptor_config.get("transition_band_specs")
        threshold_mod_specs = receptor_config.get("threshold_modulation_specs")
        osc_amps_for_ctmc = None
        if oscillations and (transition_specs or threshold_mod_specs):
            osc_amps_for_ctmc = self._build_osc_amplitudes(oscillations)

        new_state = step_receptor_dynamics(
            receptor_state=receptor_state,
            concentration=concentration,
            K_d=K_d,
            dt=self.dt,
            thresholds=thresholds,
            beta_amplitude=beta_amplitude,
            exposure_tau=exposure_tau,
            osc_amplitudes=osc_amps_for_ctmc,
            transition_specs=transition_specs,
            threshold_modulation_specs=threshold_mod_specs,
        )

        self.registry.register_receptor(receptor_id, new_state, receptor_config)

        # Compute effective signaling A_ij
        sat = compute_saturation(concentration, K_d)
        module = self._receptor_modules.get(parent_nt)
        if module is not None:
            a_ij = module.compute_effective_signaling(
                receptor_id,
                new_state.rho,
                new_state.sigma,
                new_state.chi.name,
                sat,
                gamma_gprotein=new_state.gamma_gprotein,
            )
        else:
            g_chi = compute_g_chi(new_state.chi.name)
            a_ij = compute_effective_signaling_proxy(
                new_state.rho, new_state.sigma, g_chi, sat,
                gamma_gprotein=new_state.gamma_gprotein,
            )
        self.registry.set_effective_signaling(receptor_id, a_ij)

    # ------------------------------------------------------------------
    # Emotion plasticity
    # ------------------------------------------------------------------

    def apply_emotion_event(
        self,
        emotion_id: str,
        intensity: float = 1.0,
    ) -> None:
        """
        Apply emotion-driven plasticity to all registered receptors.

        Looks up emotion_plasticity_rules from registered receptor family
        modules and applies sigma/rho deltas scaled by intensity.

        Parameters
        ----------
        emotion_id : str
            Emotion identifier (e.g., "joy", "fear", "curiosity")
        intensity : float, default=1.0
            Scaling factor for plasticity deltas
        """
        from zados.neurochem.receptors.plasticity import (
            compute_plasticity_deltas,
            apply_plasticity_delta,
        )
        deltas = compute_plasticity_deltas(emotion_id, self._receptor_modules)
        for receptor_id, receptor_deltas in deltas.items():
            try:
                state = self.registry.get_receptor(receptor_id)
            except KeyError:
                continue
            new_state = apply_plasticity_delta(state, receptor_deltas, intensity)
            try:
                config = self.registry.get_config(receptor_id)
            except KeyError:
                config = {}
            self.registry.register_receptor(receptor_id, new_state, config)

    def _apply_subtype_switching(self) -> None:
        """
        Apply subtype switching rules from registered receptor modules.

        For each module with non-empty subtype_switch_rules, compute density
        transfer deltas and apply them to the registry. Called at end of step()
        after all receptor updates.
        """
        from zados.neurochem.receptors.subtype_switching import (
            compute_subtype_switch_deltas,
            apply_subtype_switch_deltas,
        )
        for module in self._receptor_modules.values():
            rules = module.subtype_switch_rules
            if not rules:
                continue
            # Collect current receptor states for this module's receptors
            receptor_ids = set()
            for rule in rules:
                receptor_ids.add(rule.source_receptor_id)
                receptor_ids.add(rule.target_receptor_id)
            receptor_states = {}
            for rid in receptor_ids:
                try:
                    receptor_states[rid] = self.registry.get_receptor(rid)
                except KeyError:
                    continue
            if not receptor_states:
                continue
            deltas = compute_subtype_switch_deltas(receptor_states, rules, self.dt)
            if not deltas:
                continue
            new_states = apply_subtype_switch_deltas(receptor_states, deltas)
            for rid, new_state in new_states.items():
                if rid in deltas:
                    try:
                        config = self.registry.get_config(rid)
                    except KeyError:
                        config = {}
                    self.registry.register_receptor(rid, new_state, config)

    # ------------------------------------------------------------------
    # Reward feedback loop (closes synthesis → neurochemical loop)
    # ------------------------------------------------------------------

    def apply_feedback(
        self,
        feedback_params: Dict[str, Dict[str, Dict[str, float]]],
    ):
        """
        Apply reward-conditioned feedback to modulate neurochemical baselines.

        Accepts the output of ``compute_reward_feedback()`` and updates
        registered NT/receptor configs accordingly.  Unregistered NTs
        or receptors are silently skipped (future-proof for Phase 9).

        Parameters
        ----------
        feedback_params : dict
            Output of ``compute_reward_feedback()``.  Structure::

                {
                    "neurotransmitters": {
                        "OXT": {"C_baseline_delta": float},
                        "CB1": {"C_baseline_delta": float},
                        "NE":  {"u_base_multiplier": float},
                    },
                    "receptors": {
                        "GABA_B": {"K_d_multiplier": float},
                    },
                }
        """
        nt_feedback = feedback_params.get("neurotransmitters", {})
        for nt_name, params in nt_feedback.items():
            self._apply_nt_feedback(nt_name, params)

        receptor_feedback = feedback_params.get("receptors", {})
        for receptor_id, params in receptor_feedback.items():
            self._apply_receptor_feedback(receptor_id, params)

    def _apply_nt_feedback(
        self,
        nt_name: str,
        params: Dict[str, float],
    ):
        """
        Apply feedback to a single neurotransmitter's config.

        Supported keys in *params*:

        - ``C_baseline_delta`` — additive shift to C_baseline,
          result clamped to [0.0, 1.0].
        - ``u_base_multiplier`` — multiplicative factor for u_base,
          result clamped to [0.01, ∞).

        Silently returns if the NT is not registered.
        """
        if nt_name not in self.registry._neurotransmitters:
            return

        config = self.registry.get_config(nt_name)

        if "C_baseline_delta" in params:
            current = config.get("C_baseline", 0.5)
            new_val = current + params["C_baseline_delta"]
            config["C_baseline"] = max(0.0, min(1.0, new_val))

        if "u_base_multiplier" in params:
            current = config.get("u_base", 0.1)
            new_val = current * params["u_base_multiplier"]
            config["u_base"] = max(0.01, new_val)

        # Re-register with updated config (state unchanged)
        state = self.registry.get_neurotransmitter(nt_name)
        self.registry.register_neurotransmitter(nt_name, state, config)

    def _apply_receptor_feedback(
        self,
        receptor_id: str,
        params: Dict[str, float],
    ):
        """
        Apply feedback to a single receptor's config.

        Supported keys in *params*:

        - ``K_d_multiplier`` — multiplicative factor for K_d,
          result clamped to [0.01, 10.0].

        Silently returns if the receptor is not registered.
        """
        if receptor_id not in self.registry._receptors:
            return

        try:
            config = self.registry.get_config(receptor_id)
        except KeyError:
            return

        if "K_d_multiplier" in params:
            current = config.get("K_d", 0.5)
            new_val = current * params["K_d_multiplier"]
            config["K_d"] = max(0.01, min(10.0, new_val))

        # Re-register with updated config (state unchanged)
        state = self.registry.get_receptor(receptor_id)
        self.registry.register_receptor(receptor_id, state, config)