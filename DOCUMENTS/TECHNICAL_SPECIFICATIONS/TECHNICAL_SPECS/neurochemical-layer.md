# Neurochemical Layer — Consolidated Technical Specification

**Project:** ZADOS  
**Version:** 2.0  
**Date:** March 23, 2026  
**Status:** Code-Verified Against 6015-Test Codebase

---

## Consolidated Summary

The neurochemical layer defines a continuous, time-resolved simulation framework for neurotransmitter dynamics, receptor interactions, pharmacodynamic adaptation, and oscillatory modulation. It operates through explicitly defined state variables, kinetic equations, receptor binding models, and plasticity rules, forming a unified biochemical substrate that modulates all downstream computational processes in ZADOS.

At its core, the layer models twelve neurotransmitters, each represented by a synaptic concentration variable decomposed into tonic and phasic components. These concentrations evolve according to stochastic differential equations (Ornstein-Uhlenbeck with losses) integrated via Euler-Maruyama with reflecting boundaries. Release is governed by per-transmitter module specifications with weighted signal keys (novelty, RPE, effort, emotion drive, etc.), while clearance operates through three channels: transporter reuptake, enzymatic degradation, and diffusion/spillover.

Each neurotransmitter interacts with one or more of 33 receptor subtypes. Receptors are modeled as structured bundles of continuous parameters (density, sensitivity, localization bias, G-protein coupling efficacy) plus a discrete functional state evolving via a continuous-time Markov chain (CTMC) with four states: ACTIVE, DESENSITIZED, INTERNALIZED, and UPREGULATED. Emotion-driven plasticity rules modify receptor parameters in response to affective state, and homeostatic subtype switching transfers density between receptor subtypes under sustained exposure.

Oscillatory modulation operates through six canonical frequency bands (delta, theta, alpha, beta, gamma, sigma) plus three cross-frequency couplings (theta-gamma, alpha-beta, delta-sigma). A bidirectional closure derives band amplitudes from neurotransmitter concentrations and feeds them back as multiplicative modulators on kinetic parameters: binding affinity, release rate, noise amplitude, reuptake rate, and receptor state transition rates.

Eleven derived neurochemical metrics (motivation, empathy, cognitive rigidity, fatigue, precision, openness, anxiety, social engagement, dream permissiveness, consolidation depth, narrative plasticity) summarize system-level chemical states as algebraic combinations of receptor saturations and oscillatory amplitudes, bridging the continuous biochemical substrate to the symbolic cognitive architecture.

---

## 1. Neurotransmitters (Modeled Set)

The neurochemical layer models twelve neurotransmitter systems. Each system consists of a neurotransmitter module (defining release drives and oscillation coupling) and one or more receptor family modules (defining receptor subtypes with binding, plasticity, and switching rules).

### Glutamate (GLU)

Primary excitatory transmitter for fast signal propagation and high-resolution integration. Receptor families: ionotropic AMPA, NMDA, Kainate, plus metabotropic mGluR. Clearance: EAAT reuptake transporter (`u_base = 0.15`).

### GABA

Primary inhibitory transmitter for suppression, gating, and stabilization. Receptor families: GABA_A (ionotropic, fast) and GABA_B (metabotropic, slow). Clearance: GAT reuptake transporter (`u_base = 0.12`).

### Dopamine (DA)

Core neuromodulator for reward prediction, novelty salience, motivation, and drive regulation. Receptor subtypes: D1–D5 (D1/D5 excitatory D1-like; D2/D3/D4 inhibitory D2-like). Clearance: DAT reuptake (`u_base = 0.1`) + COMT/MAO degradation (`d_base = 0.05`).

### Serotonin (5-HT)

Neuromodulator for affect regulation, ambiguity buffering, long-horizon weighting, and response-mode stabilization under uncertainty. Receptor subtypes: 5-HT1A, 5-HT1B (inhibitory), 5-HT2A (excitatory), 5-HT2C (modulatory), 5-HT3 (ionotropic excitatory). Clearance: SERT reuptake (`u_base = 0.08`).

### Norepinephrine (NE)

Neuromodulator for arousal, salience, contradiction detection, and load-responsive gain control. Receptor subtypes: alpha1, alpha2 (adrenergic), beta1, beta2. Clearance: NET reuptake (`u_base = 0.12`).

### Acetylcholine (ACh)

Precision and attention modulator for salience filtering and rule fidelity. Receptor families: nicotinic alpha7 (ionotropic) and muscarinic M1–M5. Clearance: AChE hydrolysis (`u_base = 0.15`).

### Oxytocin (OXT)

Social-affective neuromodulator for trust resonance, bonding-weighting, and relational attunement. Receptor: OXTR. Clearance: peptidase (`u_base = 0.05`).

### Endorphins / Opioid Peptides (MOR)

Comfort and affective buffering system. Primary receptor: mu-opioid (MOR). Clearance: peptidase (`u_base = 0.06`).

### Endocannabinoids (CB1)

Flexibility and filtering modulator. Primary receptor: CB1. Clearance: FAAH degradation (`u_base = 0.04`).

### Cortisol (simulated via GR/NR3C1)

Stress-axis proxy for time-horizon pressure, tradeoff enforcement, and stress-conditioned weighting. No dedicated receptor in the receptor registry (signals via direct modulation). Clearance: 11-beta-HSD (`u_base = 0.03`).

### CRH (Corticotropin-Releasing Hormone)

Acute stress drive and pressure scaling signal. Receptor: CRH_R1 (excitatory). Clearance: peptidase (`u_base = 0.1`).

### Histamine

Arousal, wakefulness, and cognitive readiness modulator. Receptor subtypes: H1 (excitatory), H2 (excitatory), H3 (inhibitory/autoreceptor), H4 (modulatory). Clearance: HNMT (`u_base = 0.1`).

---

## 2. Neurochemical State Variables

### 2.1 Per-Transmitter State

For each neurotransmitter `NT_i`, the simulation maintains the following state variables:

- `C_tonic(t)` — tonic (baseline) concentration component, range `[0, 1]`
- `C_phasic(t)` — phasic (burst) concentration component, range `[0, 1]`
- `C(t) = C_tonic(t) + C_phasic(t)` — total synaptic concentration (derived)
- `F(t)` — fatigue level, range `[0, 1]`
- `eta_u(t)` — transporter efficiency, range `[0, 1]`

The tonic component represents baseline signaling; the phasic component represents transient burst events. In the neurosymbolic notation, tonic quantities are annotated with `~` and phasic with a bullet marker.

### 2.2 Neurotransmitter Configuration Parameters

Each neurotransmitter is parameterized by a configuration record with 9 kinetic parameters:

| NT | C_base | th_ton | th_pha | sig_ton | sig_pha | u_base | d_base | c_base | fat_rate |
|----|--------|--------|--------|---------|---------|--------|--------|--------|----------|
| DA | 0.50 | 0.10 | 1.00 | 0.05 | 0.10 | 0.10 | 0.05 | 0.02 | 0.001 |
| 5HT | 0.55 | 0.05 | 0.50 | 0.03 | 0.06 | 0.08 | 0.03 | 0.015 | 0.001 |
| NE | 0.45 | 0.15 | 1.20 | 0.06 | 0.12 | 0.12 | 0.04 | 0.02 | 0.001 |
| ACh | 0.50 | 0.12 | 1.50 | 0.04 | 0.08 | 0.15 | 0.03 | 0.01 | 0.001 |
| OXT | 0.40 | 0.03 | 0.30 | 0.02 | 0.04 | 0.05 | 0.02 | 0.01 | 0.001 |
| MOR | 0.35 | 0.04 | 0.60 | 0.03 | 0.07 | 0.06 | 0.03 | 0.015 | 0.001 |
| CB1 | 0.40 | 0.03 | 0.40 | 0.025 | 0.05 | 0.04 | 0.02 | 0.01 | 0.001 |
| cortisol | 0.30 | 0.02 | 0.20 | 0.015 | 0.03 | 0.03 | 0.015 | 0.01 | 0.001 |
| CRH | 0.25 | 0.08 | 0.80 | 0.04 | 0.09 | 0.10 | 0.04 | 0.02 | 0.001 |
| GABA | 0.60 | 0.10 | 1.00 | 0.04 | 0.08 | 0.12 | 0.04 | 0.02 | 0.001 |
| GLU | 0.55 | 0.12 | 1.30 | 0.05 | 0.10 | 0.15 | 0.05 | 0.02 | 0.001 |
| histamine | 0.35 | 0.08 | 0.90 | 0.04 | 0.08 | 0.10 | 0.04 | 0.02 | 0.001 |

> **Legend:** `C_base` = baseline concentration; `th_ton` = theta_tonic (OU mean-reversion rate, tonic); `th_pha` = theta_phasic (phasic decay rate); `sig_ton` = sigma_tonic (tonic noise amplitude); `sig_pha` = sigma_phasic (phasic noise amplitude); `u_base` = reuptake coefficient; `d_base` = degradation rate; `c_base` = clearance/diffusion rate; `fat_rate` = fatigue accumulation rate.

### 2.3 Per-Receptor State

For each receptor subtype `R_ij`, the simulation maintains:

- `rho_ij(t)` — receptor density (expression level), range `[0, 1]`
- `sigma_ij(t)` — receptor sensitivity (gain per binding event), range `[0, 1]`
- `lambda_ij` — synaptic localization bias, range `[0, 1]` (0 = presynaptic, 0.5 = synaptic, 1.0 = extrasynaptic)
- `gamma_ij(t)` — G-protein coupling efficacy, range `[0.05, 1.0]`
- `chi_ij(t)` — functional state (CTMC), one of `{ACTIVE, DESENSITIZED, INTERNALIZED, UPREGULATED}`
- `exposure_trace_ij(t)` — slow-decaying integral of saturation history
- `time_in_state_ij(t)` — duration in current functional state

### 2.4 Oscillation State

The oscillatory state consists of six bounded amplitude envelopes, per-band phase angles, and three cross-frequency coupling terms.

**Band amplitudes (all in `[0, 1]`):**

- `phi_delta(t)` — delta band (0.5–4 Hz) — recovery / reset dynamics
- `phi_theta(t)` — theta band (4–8 Hz) — simulation / narrative expansion
- `phi_alpha(t)` — alpha band (8–12 Hz) — inhibitory gating / attentional filtering
- `phi_beta(t)` — beta band (13–30 Hz) — precision / executive control
- `phi_gamma(t)` — gamma band (30–100+ Hz) — binding / cross-domain integration
- `phi_sigma(t)` — sigma band (12–15 Hz) — sleep spindles (0 during wake)

**Cross-frequency couplings (product form, all in `[0, 1]`):**

- `phi_theta_gamma(t) = phi_theta(t) * phi_gamma(t)`
- `phi_alpha_beta(t) = phi_alpha(t) * phi_beta(t)`
- `phi_delta_sigma(t) = phi_delta(t) * phi_sigma(t)` — NREM consolidation window

**Phase angles:**

- `psi_k(t)` in `[0, 2*pi)` per band `k`, used when phase-aware oscillator mode is active

---

## 3. Neurochemical Kinetics

### 3.1 Core Evolution Equation

For each neurotransmitter `NT_i`, synaptic concentration `C_i(t)` evolves according to a mass-balance kinetics equation with release and multiple loss channels:

```
dC_i(t)/dt = r_i(t) - u_i(t) - d_i(t) - c_i(t)
```

Where:

- `r_i(t)` = release drive (input term)
- `u_i(t)` = reuptake loss
- `d_i(t)` = enzymatic degradation loss
- `c_i(t)` = diffusion / clearance loss

### 3.2 Drift Term (Ornstein-Uhlenbeck with Losses)

The drift term uses an Ornstein-Uhlenbeck mean-reversion formulation combined with the loss channels. The tonic and phasic components are integrated separately:

**Tonic drift:**
```
mu_tonic = -theta_tonic * (C_tonic - C_baseline) - L_total(C_tonic, eta_u)
```

**Phasic drift:**
```
mu_phasic = -theta_phasic * (C_phasic - 0.0) - L_total(C_phasic, eta_u)
```

The phasic component reverts to zero (not to baseline), reflecting its transient burst nature. The total loss across all three channels is:

```
L_total(C, eta_u) = u_base * eta_u * C + d_base * C + c_base * (C - C_extracell)
```

Where `C_extracell` defaults to `0.0` in the single-compartment model.

**Fatigue-modulated reversion rate:**
```
theta_eff = theta_base * (1 - 0.5 * F)
```

This slows mean-reversion under high fatigue (`fatigue_scaling = 0.5` default).

### 3.3 Release Drives

Release drives are computed per-transmitter via the NT module system. Each module defines a `ReleaseDriveSpec` with weighted signal keys:

```
R_total = max(0, sum_j(w_j * signal_j) - threshold)
```

Concrete drive computation functions:

- **Novelty drive:** `N(t) = sensitivity * max(0, stimulus_novelty - threshold)`
- **RPE drive:** `RPE(t) = gain * reward_prediction_error` (may be negative for DA)
- **Effort drive:** `E(t) = willingness * max(0, task_demand - threshold)`
- **Emotion drive:** derived from emotion NT recipes (Section 10)

Post-processing applied to raw release:

- **Fatigue gating:** `R_gated = R * (1 - suppression * max(0, F - 0.7))`
- **Oscillatory gating:** `R_osc = R * (1 + band_coefficient * phi_k)`
- **Phasic burst amplitude:** `A_burst = max_burst * (1 - exp(-sensitivity * drive))` — saturating exponential

### 3.4 Loss Channels

Three independent clearance mechanisms operate on each neurotransmitter:

**Reuptake (transporter-mediated):**
```
u_i(t) = u_base * eta_u(t) * C_i(t)
```

Where `eta_u` is the fatigue-modulated transporter efficiency:
```
eta_u = max(0, 1 - gamma_fatigue * F)
```
Default `gamma_fatigue = 0.1`.

**Enzymatic degradation:**
```
d_i(t) = d_base * C_i(t)
```

**Diffusion / clearance:**
```
c_i(t) = c_base * (C_i(t) - C_extracell)
```

### 3.5 Fatigue Dynamics

Fatigue accumulates proportionally to neurotransmitter activity and decays naturally:

```
dF/dt = epsilon * C(t) - decay * F(t)
```

Default parameters:

- `epsilon = 0.01` — fatigue accumulation rate from NT activity
- `decay = 0.001` — natural fatigue decay per timestep
- `F` clamped to `[0, 1]`

### 3.6 Numerical Integration (Euler-Maruyama)

Concentration dynamics are integrated as a stochastic differential equation using the Euler-Maruyama scheme:

```
C(t+dt) = C(t) + mu(C,t) * dt + sigma(C,t) * sqrt(dt) * xi
xi ~ N(0, 1) i.i.d. per timestep
```

**Noise model (multiplicative, concentration-scaled):**
```
sigma(C,t) = alpha * sqrt(C(t))
```

The square-root dependence reflects Poisson-like vesicular release variability. The noise amplitude coefficients `sigma_tonic` and `sigma_phasic` are per-transmitter configurable, and are further modulated by oscillatory state (Section 5.6).

**Reflecting boundary enforcement:**

After each Euler-Maruyama step, concentrations are bounded to `[0, 1]` using a reflecting boundary via modular folding:

```
domain_width = upper - lower  (= 1.0)
period = 2 * domain_width     (= 2.0)
shifted = (C_next - lower) mod period
if shifted > domain_width: shifted = period - shifted
C_bounded = lower + shifted
```

This O(1) reflection method avoids the infinite-loop risk of iterative clamping while preserving the stochastic variance structure.

> **Default integration step:** `dt = 0.01`

### 3.7 Per-Tick Integration Procedure

At each simulation tick `n`, for each neurotransmitter `NT_i`:

1. Compute modulation signals from upstream (emotion events, engine feedback, domain subscores)
2. Compute release drive `R_i(t)` from module spec and signals
3. Apply oscillatory gating to release and kinetic parameters
4. Compute drift `mu_i(C, t)` = OU reversion + release − losses
5. Compute diffusion `sigma_i(C, t)` = noise amplitude (oscillation-modulated)
6. Sample `xi ~ N(0,1)`
7. Euler-Maruyama step: `C_next = C + mu*dt + sigma*sqrt(dt)*xi`
8. Apply reflecting boundary to `[0, 1]`
9. Update fatigue: `F_next = F + (epsilon*C - decay*F)*dt`
10. Update transporter efficiency: `eta_u = max(0, 1 - gamma*F)`

---

## 4. Receptors and Pharmacodynamics

### 4.1 Receptor Definitions

For each receptor subtype `R_ij` (receptor `j` targeted by neurotransmitter `NT_i`), the simulation represents receptors as a structured bundle of continuous parameters plus a discrete functional state:

- **Density (`rho_ij`):** expression level / available receptor sites. Scaling factor on maximal achievable binding impact. Primary slow variable for upregulation/downregulation.
- **Sensitivity (`sigma_ij`):** responsiveness per unit binding. Gain term for downstream signaling strength. Primary target for tolerance/sensitization adaptation.
- **Synaptic localization bias (`lambda_ij`):** compartmental weighting (0=presynaptic, 0.5=synaptic, 1.0=extrasynaptic). Determines which concentration pool (tonic vs phasic) drives the receptor.
- **G-protein coupling efficacy (`gamma_ij`):** coupling strength to downstream transduction machinery. Degrades under sustained high saturation, recovers when saturation drops.
- **Functional state (`chi_ij`):** discrete CTMC state governing receptor availability and signaling mode.

### 4.2 Receptor Configuration Parameters

| Receptor | Parent | K_d | Exp_tau | Signal Type | Weight | Iono |
|----------|--------|-----|---------|-------------|--------|------|
| DA_D1 | DA | 0.40 | 10.0 | excitatory | 1.00 | No |
| DA_D2 | DA | 0.30 | 12.0 | inhibitory | 0.90 | No |
| DA_D3 | DA | 0.20 | 15.0 | inhibitory | 0.80 | No |
| DA_D4 | DA | 0.35 | 10.0 | modulatory | 0.70 | No |
| DA_D5 | DA | 0.45 | 10.0 | excitatory | 0.75 | No |
| 5HT_1A | 5HT | 0.30 | 15.0 | inhibitory | 1.00 | No |
| 5HT_1B | 5HT | 0.35 | 12.0 | inhibitory | 0.85 | No |
| 5HT_2A | 5HT | 0.40 | 10.0 | excitatory | 1.00 | No |
| 5HT_2C | 5HT | 0.35 | 12.0 | modulatory | 0.90 | No |
| 5HT_3 | 5HT | 0.50 | 8.0 | excitatory | 0.80 | Yes |
| NE_a1 | NE | 0.50 | 10.0 | excitatory | 1.00 | No |
| NE_a2 | NE | 0.25 | 12.0 | inhibitory | 0.90 | No |
| NE_b1 | NE | 0.40 | 10.0 | excitatory | 0.95 | No |
| NE_b2 | NE | 0.45 | 10.0 | excitatory | 0.85 | No |
| ACh_nic | ACh | 0.50 | 8.0 | excitatory | 1.00 | Yes |
| ACh_mus | ACh | 0.40 | 12.0 | modulatory | 0.90 | No |
| OXTR | OXT | 0.35 | 20.0 | excitatory | 1.00 | No |
| MOR_mu | MOR | 0.30 | 15.0 | inhibitory | 1.00 | No |
| CB1 | CB1 | 0.40 | 20.0 | inhibitory | 1.00 | No |
| CRH_R1 | CRH | 0.45 | 10.0 | excitatory | 1.00 | No |
| GABA_A | GABA | 0.50 | 8.0 | inhibitory | 1.00 | Yes |
| GABA_B | GABA | 0.40 | 15.0 | inhibitory | 0.90 | No |
| GLU_NMDA | GLU | 0.50 | 10.0 | excitatory | 1.00 | Yes |
| GLU_AMPA | GLU | 0.60 | 6.0 | excitatory | 1.00 | Yes |
| GLU_KA | GLU | 0.55 | 8.0 | excitatory | 0.85 | Yes |
| GLU_mGluR | GLU | 0.35 | 15.0 | modulatory | 0.90 | No |
| HIST_H1 | hist | 0.45 | 10.0 | excitatory | 1.00 | No |
| HIST_H2 | hist | 0.40 | 12.0 | excitatory | 0.80 | No |
| HIST_H3 | hist | 0.25 | 15.0 | inhibitory | 0.60 | No |
| HIST_H4 | hist | 0.50 | 10.0 | modulatory | 0.40 | No |

> **Legend:** `K_d` = dissociation constant (lower = higher affinity); `Exp_tau` = exposure trace time constant; `Weight` = effective_signaling_weight; `Iono` = ionotropic. 33 total receptor subtypes across 12 NT families.

### 4.3 Receptor Binding (Michaelis-Menten Saturation)

```
S_ij(t) = C_eff(t) / (C_eff(t) + K_d,ij)
```

Where the effective concentration seen by the receptor depends on localization bias:

```
C_eff = C_tonic + (1 - lambda_loc) * C_phasic
```

Interpretation: presynaptic receptors (`lambda=0`) see full phasic; extrasynaptic receptors (`lambda=1`) see only tonic.

### 4.4 Effective Signaling Proxy

The effective signaling output of receptor `R_ij` combines all pharmacodynamic factors:

```
A_ij(t) = rho_ij * sigma_ij * gamma_ij * g(chi_ij) * S_ij(t) * w_ij
```

Where `g(chi)` is the functional state gate:

| Functional State | g(chi) | Interpretation |
|------------------|--------|----------------|
| ACTIVE | 1.0 | Full signal transduction |
| DESENSITIZED | 0.5 | Reduced responsiveness |
| INTERNALIZED | 0.1 | Near-zero availability |
| UPREGULATED | 1.2 | Enhanced availability |

> **Note:** The values `0.1` and `1.2` in `oscillation_modulation.py` differ slightly from `0.3` and `1.3` used in some receptor code paths. The oscillation modulation module uses the more conservative values (`0.1` / `1.2`).

### 4.5 Receptor CTMC State Transitions

Receptor functional states evolve via a continuous-time Markov chain (CTMC) with the following transition rules and concrete threshold parameters:

| Transition | Condition | Threshold | Time Req |
|------------|-----------|-----------|----------|
| ACTIVE → DESENSITIZED | S_ij > theta_desens | 0.7 | t0 = 5.0 |
| DESENSITIZED → INTERNALIZED | exposure_trace > theta_intern | 15.0 | — |
| DESENSITIZED → ACTIVE | S_ij < epsilon_recovery | 0.3 | t = 10.0 |
| ACTIVE → UPREGULATED | S_ij < epsilon_upreg | 0.1 | t0 = 20.0 |
| UPREGULATED → ACTIVE | S_ij > theta_upreg_exit | 0.4 | t = 5.0 |
| INTERNALIZED → ACTIVE | automatic recycling | — | t_recycle = 50.0 |

**State-entry parameter modifications:**

| On Entry To | Parameter Change |
|-------------|-----------------|
| DESENSITIZED | `sigma *= 0.5` |
| INTERNALIZED | `rho *= 0.7`, `sigma *= 0.3` |
| UPREGULATED | `sigma *= 1.3`, `rho *= 1.2` |
| ACTIVE (recovery) | `sigma` recovers at rate `0.05 * (1 - sigma)` per `dt` |

### 4.6 Exposure Trace

A slow-decaying integral of saturation history drives long-timescale plasticity:

```
E(t+dt) = E(t) * exp(-dt / tau) + S(t) * dt
```

Default `tau = 10.0` (transmitter-specific via `exposure_tau` config). High exposure trace (> 15.0) triggers internalization.

### 4.7 G-Protein Coupling Dynamics

G-protein coupling efficacy degrades under sustained high saturation and recovers when saturation drops:

```
If S > 0.5:  gamma -= 0.02 * S * dt
If S <= 0.5: gamma += 0.01 * (1 - gamma) * dt
gamma clamped to [0.05, 1.0]
```

### 4.8 Sensitivity Recovery

While in the ACTIVE state, receptor sensitivity gradually recovers toward 1.0:

```
sigma(t+dt) = sigma(t) + 0.05 * (1 - sigma(t)) * dt
```

### 4.9 Emotion-Driven Receptor Plasticity

Emotion events modify receptor parameters (sensitivity and density) according to per-receptor plasticity rules. When an emotion is detected with intensity `I`:

```
sigma_new = sigma + sigma_delta * I
rho_new   = rho   + rho_delta   * I
```

Each receptor subtype defines its own emotion plasticity rule dictionary mapping `emotion_id` to `{sigma_delta, rho_delta}` values. See Section 7 for per-receptor plasticity tables.

### 4.10 Subtype Switching (Homeostatic Compensation)

Under sustained exposure, receptor density can transfer between subtypes within a family to maintain homeostasis:

| Switch Rule | Threshold | Rate (fwd) | Rate (rev) | Max/Step |
|-------------|-----------|------------|------------|----------|
| DA D1 ↔ D2 | 20.0 | 0.005 | 0.003 | 0.018 |
| 5HT 1A ↔ 2A | 20.0 | 0.004 | 0.004 | 0.018 |
| NE alpha1 ↔ alpha2 | 20.0 | 0.004 | 0.003 | 0.018 |
| ACh nic ↔ mus | 20.0 | 0.004 | 0.003 | 0.018 |
| GABA A ↔ B | 20.0 | 0.004 | 0.003 | 0.018 |
| GLU NMDA ↔ AMPA | 20.0 | 0.004 | 0.003 | 0.018 |

Conservation law: `source_delta + target_delta = 0` (total density is conserved).

---

## 5. Oscillatory Modulation

### 5.1 Oscillatory Variables

Six bounded amplitude envelopes represent canonical electrophysiological frequency bands:

```
phi_k(t) in [0, 1],  k in {delta, theta, alpha, beta, gamma, sigma}
```

| Band | Freq Range | Functional Role | Primary NT Drivers |
|------|------------|----------------|-------------------|
| delta | 0.5–4 Hz | Recovery / reset / slow inhibitory stabilization | GABA-B, MOR, CB1 |
| theta | 4–8 Hz | Simulation / narrative expansion / associative traversal | DA-D3, 5HT-2A, OXT |
| alpha | 8–12 Hz | Inhibitory gating / attentional filtering / noise suppression | GABA-A, 5HT-1A, OXT |
| beta | 13–30 Hz | Precision / executive control / contradiction detection | NE-beta1, DA-D2, ACh, cortisol |
| gamma | 30–100+ Hz | Binding / cross-domain integration / insight | DA-D3, NMDA, OXT |
| sigma | 12–15 Hz | Sleep spindles (always 0 during wake) | GABA (TRN), GLU (TC rebound) |

### 5.2 Band Derivation from NT Concentrations (Closure)

Band amplitudes are derived from neurotransmitter concentrations via weighted sums, creating a bidirectional closure between NTs and oscillations:

| Band | NT Source | Component | Weight |
|------|-----------|-----------|--------|
| gamma | DA | C_phasic | 0.4 |
| gamma | GLU | C_phasic | 0.3 |
| gamma | ACh | C_phasic | 0.3 |
| theta | OXT | C_tonic | 0.4 |
| theta | 5HT | C_tonic | 0.3 |
| theta | DA | C_tonic | 0.3 |
| alpha | GABA | C_tonic | 0.5 |
| alpha | 5HT | C_tonic | 0.3 |
| alpha | CB1 | C_tonic | 0.2 |
| beta | NE | C_total | 0.4 |
| beta | cortisol | C_total | 0.3 |
| beta | ACh | C_total | 0.3 |
| delta | MOR | C_tonic | 0.4 |
| delta | CB1 | C_tonic | 0.3 |
| delta | GABA | C_tonic | 0.3 |
| sigma | GABA | C_tonic | 0.5 |
| sigma | GLU | C_phasic | 0.3 |
| sigma | NE | C_tonic | -0.3 |

> All band amplitudes are clamped to `[0, 1]` after weighted summation. The sigma band uses a negative NE weight (NE suppresses sleep spindles). This derivation operates when `oscillation_mode="state_derived"`; in `"static"` mode, oscillations are set externally.

### 5.3 Cross-Frequency Coupling

Three cross-frequency coupling envelopes are computed as products of their constituent bands:

```
phi_theta_gamma(t) = phi_theta(t) * phi_gamma(t)
phi_alpha_beta(t)  = phi_alpha(t) * phi_beta(t)
phi_delta_sigma(t) = phi_delta(t) * phi_sigma(t)
```

These coupling terms enter modulation rules identically to primary band envelopes and remain bounded in `[0, 1]` by construction.

### 5.4 Binding Affinity Modulation

```
K_d,ij(t) = K_d,ij * (1 - alpha_theta * phi_theta(t))
```

Default coefficient: `alpha_theta = 0.3`. Floor: `K_d >= 0.01` to prevent division by zero in saturation computation. Higher theta increases effective affinity (lower K_d) for the same concentration.

**Multi-band extension:**
```
K_d,ij(t) = K_d,ij * (1 - sum_k alpha_k,ij * phi_k(t))
```

With per-receptor band coefficients stored in `kd_band_coefficients` config.

### 5.5 Release Rate Modulation

```
r_i(t) = r_base(t) * (1 + beta_gamma * phi_gamma(t))
```

Default coefficient: `beta_gamma = 0.5` for DA; varies per-NT module. Gamma boosts phasic release, reflecting high-throughput binding integration.

### 5.6 Noise Modulation

```
sigma_eff(t) = sigma_base * max(0.1, 1 - chi_alpha * phi_alpha(t))
```

Default coefficient: `chi_alpha = 0.4`. A floor of `0.1` (10% of base) is enforced to maintain numerical stability. Alpha suppresses stochastic noise in precision-sensitive channels.

### 5.7 Reuptake Modulation

```
u_eff(t) = u_base * (1 + zeta_beta * phi_beta(t))
```

Default coefficient: `zeta_beta = 0.3`. Beta speeds up clearance during precision-demanding executive control.

### 5.8 Tonic Baseline Modulation

```
C_baseline_mod = C_baseline * (1 - coeff_delta * phi_delta(t))
```

Clamped to `[0.01, 1.0]`. Delta modulates the OU reversion target during recovery/reset phases.

### 5.9 Receptor State Transition Modulation

Oscillatory envelopes scale receptor CTMC transition rates as multiplicative factors. The concrete implementation uses beta to accelerate desensitization:

```
beta_desens_scaling = 0.3
```

General form: `q_AB(t) = q_AB_base(t) * m_AB(t)`, where `m_AB` is an oscillation-dependent multiplier. The full CTMC modulation framework supports per-transition, per-band coefficients as described in the mathematical appendix.

### 5.10 NT-Oscillatory Associations

| NT | Primary Bands | Secondary |
|----|--------------|-----------|
| DA | gamma, theta | beta |
| 5HT | theta, alpha | delta |
| NE | beta | gamma |
| ACh | beta | — |
| OXT | theta | alpha |
| MOR | delta | theta |
| CB1 | delta | alpha-beta |
| cortisol | beta | delta |
| CRH | beta | — |
| GABA | alpha, delta | theta |
| GLU | gamma, theta-gamma | — |
| histamine | beta | gamma |

---

## 6. Per-Neurotransmitter Module Specifications

Each neurotransmitter is governed by a module that defines (a) its release drive specification (weighted signal keys) and (b) its oscillation coupling rules. These modules are registered with the `NeurochemicalEngine` at initialization.

### 6.1 Dopamine (DAModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| novelty | 0.35 |
| rpe | 0.3 |
| effort | 0.15 |
| emotion_drive | 0.2 |
| threshold | 0.0 |

> Allows negative RPE (custom compute override)

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | gamma | 0.5 |
| K_d | theta | -0.3 |
| sigma_tonic | alpha | -0.4 |
| sigma_phasic | alpha | -0.4 |

> All rules use multiplicative formula unless noted.

### 6.2 Serotonin (SerotoninModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| mood_stability | 0.3 |
| ambiguity | 0.25 |
| horizon_weight | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | theta | 0.4 |
| sigma_tonic | alpha | -0.4 |
| sigma_phasic | alpha | -0.3 |

> All rules use multiplicative formula unless noted.

### 6.3 Norepinephrine (NEModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| precision | 0.3 |
| uncertainty | 0.25 |
| contradiction | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | beta | 0.3 |
| sigma_tonic | alpha | -0.4 |
| sigma_phasic | alpha | -0.3 |

> All rules use multiplicative formula unless noted.

### 6.4 Acetylcholine (AChModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| attention_demand | 0.3 |
| rule_fidelity | 0.25 |
| precision_weight | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | beta | 0.3 |
| release | gamma | 0.25 |
| sigma_tonic | alpha | -0.3 |

> All rules use multiplicative formula unless noted.

### 6.5 Oxytocin (OXTModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| empathy | 0.3 |
| social_engagement | 0.25 |
| trust | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | theta | 0.4 |
| sigma_tonic | alpha | -0.2 |

> All rules use multiplicative formula unless noted.

### 6.6 Opioid (MORModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| hedonic_tone | 0.3 |
| comfort | 0.25 |
| satisfaction | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| theta_tonic | delta | -0.2 |
| sigma_tonic | delta | -0.15 |

> All rules use multiplicative formula unless noted.

### 6.7 Endocannabinoid (CB1Module)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| flexibility | 0.3 |
| filter_suppression | 0.25 |
| continuity | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| theta_tonic | delta | -0.2 |
| sigma_tonic | alpha_beta | -0.2 |

> All rules use multiplicative formula unless noted.

### 6.8 Cortisol (CORTModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| stress_level | 0.3 |
| time_pressure | 0.25 |
| tradeoff_load | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | beta | 0.25 |
| theta_tonic | delta | -0.15 |

> All rules use multiplicative formula unless noted.

### 6.9 CRH (CRHModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| acute_stress | 0.4 |
| pressure_scaling | 0.3 |
| emotion_drive | 0.3 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | beta | 0.3 |

> All rules use multiplicative formula unless noted.

### 6.10 GABA (GABAModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| inhibition | 0.3 |
| boundary_proximity | 0.25 |
| suppression | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | alpha | 0.35 |
| theta_tonic | delta | -0.2 |

> All rules use multiplicative formula unless noted.

### 6.11 Glutamate (GLUModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| excitation | 0.3 |
| integration_demand | 0.25 |
| signal_propagation | 0.2 |
| emotion_drive | 0.25 |
| threshold | 0.05 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | gamma | 0.35 |
| release | beta | 0.2 |
| sigma_tonic | alpha | -0.3 |

> All rules use multiplicative formula unless noted.

### 6.12 Histamine (HistamineModule)

**Release Drive Specification:**

| Signal Key | Weight |
|------------|--------|
| wakefulness | 0.35 |
| attention_demand | 0.25 |
| arousal | 0.2 |
| emotion_drive | 0.2 |
| threshold | 0.0 |

**Oscillation Coupling Rules:**

| Target | Band | Coefficient |
|--------|------|-------------|
| release | beta | 0.4 |
| sigma_tonic | alpha | -0.3 |
| sigma_phasic | alpha | -0.3 |

> All rules use multiplicative formula unless noted.

---

## 7. Per-Receptor Family Specifications

Each receptor family module defines subtypes with signaling characteristics and emotion-driven plasticity rules. Plasticity applies when emotion events are detected: sigma and rho are adjusted by `delta * intensity`.

### 7.1 Dopamine Receptors

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| DA_D1 | excitatory | 1.0 | No | joy: s+0.1; excitement: s+0.08; fear: s-0.05 |
| DA_D2 | inhibitory | 0.9 | No | fear: s-0.1; caution: s+0.08 |
| DA_D3 | inhibitory | 0.8 | No | contentment: s+0.05 |
| DA_D4 | modulatory | 0.7 | No | curiosity: s+0.12, r+0.05 |
| DA_D5 | excitatory | 0.75 | No | excitement: s+0.08 |

> Subtype switching: D1↔D2 — threshold=20.0, fwd=0.005, rev=0.003

### 7.2 Serotonin Receptors

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| 5HT_1A | inhibitory | 1.0 | No | calm: s+0.1; serenity: s+0.08; anxiety: s-0.1; distress: s-0.08 |
| 5HT_1B | inhibitory | 0.85 | No | calm: s+0.05 |
| 5HT_2A | excitatory | 1.0 | No | openness: s+0.12, r+0.05; curiosity: s+0.08; rigidity: s-0.1 |
| 5HT_2C | modulatory | 0.9 | No | anxiety: s+0.1; fear: s+0.08; calm: s-0.05 |
| 5HT_3 | excitatory | 0.8 | Yes | nausea_distress: s+0.1 |

> Subtype switching: 1A↔2A — threshold=20.0, rate=0.004, max=0.018

### 7.3 Norepinephrine Receptors

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| NE_alpha1 | excitatory | 1.0 | No | alertness: s+0.1, r+0.05; vigilance: s+0.08; calm: s-0.05 |
| NE_alpha2 | inhibitory | 0.9 | No | calm: s+0.1; focus: s+0.06; panic: s-0.1 |
| NE_beta1 | excitatory | 0.95 | No | fear: s+0.12; threat: s+0.1; safety: s-0.08 |
| NE_beta2 | excitatory | 0.85 | No | arousal: s+0.06 |

> Subtype switching: alpha1↔alpha2 — threshold=20.0, fwd=0.004, rev=0.003

### 7.4 Acetylcholine Receptors

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| ACh_nic | excitatory | 1.0 | Yes | focus: s+0.1, r+0.05; attention: s+0.08; fatigue: s-0.1, r-0.05 |
| ACh_mus | modulatory | 0.9 | No | curiosity: s+0.08; engagement: s+0.06; fatigue: s-0.08 |

> Subtype switching: nic↔mus — threshold=20.0, fwd=0.004, rev=0.003

### 7.5 Oxytocin Receptor

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| OXTR | excitatory | 1.0 | No | trust: s+0.12, r+0.08; bonding: s+0.1; empathy: s+0.08, r+0.05; compassion: s+0.06; betrayal: s-0.12, r-0.05; isolation: s-0.08 |

> Subtype switching: No switching

### 7.6 Opioid Receptor

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| MOR_mu | inhibitory | 1.0 | No | contentment: s+0.1, r+0.05; comfort: s+0.08; pleasure: s+0.06, r+0.04; pain: s-0.1; distress: s-0.08 |

> Subtype switching: No switching

### 7.7 Cannabinoid Receptor

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| CB1 | inhibitory | 1.0 | No | flexibility: s+0.1, r+0.05; openness: s+0.08; relaxation: s+0.06, r+0.04; rigidity: s-0.1; control: s-0.06 |

> Subtype switching: No switching

### 7.8 CRH Receptor

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| CRH_R1 | excitatory | 1.0 | No | stress: s+0.12, r+0.06; threat: s+0.1; anxiety: s+0.08, r+0.05; safety: s-0.1; calm: s-0.08 |

> Subtype switching: No switching

### 7.9 GABA Receptors

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| GABA_A | inhibitory | 1.0 | Yes | calm: s+0.1, r+0.05; safety: s+0.08; anxiety: s-0.12; panic: s-0.15; restraint: s+0.06 |
| GABA_B | inhibitory | 0.9 | No | restraint: s+0.1; caution: s+0.08; calm: s+0.05; impulsivity: s-0.1 |

> Subtype switching: A↔B — threshold=20.0, fwd=0.004, rev=0.003

### 7.10 Glutamate Receptors

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| GLU_NMDA | excitatory | 1.0 | Yes | learning: s+0.12, r+0.05; engagement: s+0.08; overwhelm: s-0.1; fatigue: s-0.06 |
| GLU_AMPA | excitatory | 1.0 | Yes | alertness: s+0.08; overwhelm: s-0.12; excitotoxic_stress: s-0.15 |
| GLU_KA | excitatory | 0.85 | Yes | engagement: s+0.06 |
| GLU_mGluR | modulatory | 0.9 | No | focus: s+0.1; plasticity: s+0.08, r+0.04; rigidity: s-0.06 |

> Subtype switching: NMDA↔AMPA — threshold=20.0, fwd=0.004, rev=0.003

### 7.11 Histamine Receptors

| Subtype | Signal | Weight | Iono | Emotion Plasticity (s=sigma, r=rho) |
|---------|--------|--------|------|-------------------------------------|
| HIST_H1 | excitatory | 1.0 | No | focus: s+0.1, r+0.05; curiosity: r+0.06; calm: s-0.08; sadness: s-0.06 |
| HIST_H2 | excitatory | 0.8 | No | focus: s+0.06; anxiety: s+0.04; calm: s-0.05 |
| HIST_H3 | inhibitory | 0.6 | No | focus: s-0.08; calm: s+0.06; sadness: s+0.05 |
| HIST_H4 | modulatory | 0.4 | No | anxiety: s+0.04 |

> Subtype switching: No switching

---

## 8. Neurosymbolic Encoding

### 8.1 Syntax

The neurosymbolic encoding provides a compact, machine-parsable notation for neurotransmitter-receptor interactions:

**Base form:**
```
[NT_i] -> [R_ij] : Delta_state
```

**Oscillatory-gated form:**
```
k{ [NT_i] -> [R_ij] : Delta }
k in {delta, theta, alpha, beta, gamma}
```

**Cross-frequency gated:**
```
theta_gamma{ [NT_i] -> [R_ij] : Delta }
```

**Signal markers:**

- `NT_i` (bullet) — phasic component dominant (`C_phasic / C_total >= tau_phasic`)
- `NT_i` (tilde) — tonic component dominant (`C_phasic / C_total <= tau_tonic`)

### 8.2 State-Change Vocabulary

**Functional state tags:** `active`, `desensitized`, `internalized`, `upregulated`

**Parameter direction tags:** `up-rho`, `down-rho`, `up-sigma`, `down-sigma`, `up-gamma`, `down-gamma`

**Binding direction tags:** `up-S`, `down-S`, `S>theta`, `S<epsilon`

**Timing tags:** `[dt > t0]` (appended when a window condition applies)

**Examples:**

- `gamma{ DA(phasic) -> D1 : up-S }` — Gamma-gated dopamine phasic burst increases D1 saturation
- `theta{ OXT(tonic) -> OXTR : active }` — Theta-gated oxytocin maintains OXTR in active state
- `DA -> D2 : desensitized [dt > 5.0]` — Sustained DA binding desensitizes D2 after 5 time units

---

## 9. Derived Neurochemical Metrics

Eleven metrics are computed as algebraic combinations of receptor saturations and oscillatory amplitudes. All metrics are normalized to `[0, 1]`.

### 9.1 Core Metrics (8)

**Motivation:**
```
(S_DA_D3 + S_OXTR - S_GABA_B + 1) / 3
```
Higher DA D3 and OXT drive → higher motivation; GABA-B inhibition reduces it.

**Empathy:**
```
S_OXTR * phi_theta * S_5HT_1A
```
Triple product requiring simultaneous oxytocin receptor activation, theta-band engagement, and serotonergic tone.

**Cognitive Rigidity:**
```
(S_NE_beta1 + S_DA_D2 - S_CB1 + 1) / 3
```
NE and DA D2 increase rigidity; CB1 endocannabinoid promotes flexibility.

**Fatigue:**
```
(S_GABA_B + phi_delta) / 2
```
GABA-B inhibition and delta-band dominance signal system fatigue.

**Precision:**
```
((S_NE_beta1 + S_DA_D2) * phi_beta) / 2
```
Noradrenergic and dopaminergic D2 activation gated by beta-band executive control.

**Openness:**
```
(S_5HT_2A + S_DA_D3 - S_5HT_1A + 1) / 3
```
5HT-2A and DA D3 promote openness; 5HT-1A inhibitory tone reduces it.

**Anxiety:**
```
((C_NE + C_CRH + C_cortisol)/3 - S_GABA_A + 1) / 2
```
Stress axis concentrations (NE, CRH, cortisol) minus GABAergic inhibition.

**Social Engagement:**
```
(S_OXTR + S_DA_D3 - C_cortisol + 1) / 3
```
Oxytocin and DA D3 promote social engagement; cortisol stress suppresses it.

### 9.2 Sleep-State Metrics (3)

**Dream Permissiveness:**
```
((S_CB1 * phi_theta_gamma) + (1 - S_NE_alpha1) + (1 - S_5HT_1A) - S_GABA_B + 1) / 4
```
High when CB1 under theta-gamma coupling is active, NE and 5HT tone are low, and GABA-B is not dominant.

**Consolidation Depth:**
```
(phi_delta_sigma * S_GLU_NMDA + S_NE_beta1 * infra_slow_cAMP - S_ACh_M1 + 1) / 3
```
NREM spindle-delta coupling drives NMDA-dependent consolidation. `infra_slow_cAMP` defaults to `0.0` (open integration point).

**Narrative Plasticity:**
```
(phi_theta_gamma * S_GLU_NMDA * S_DA_D3 + S_CB1 + (1 - cognitive_rigidity)) / 3
```
Theta-gamma binding with glutamate and dopamine D3, plus flexibility (inverse of rigidity).

---

## 10. Emotion Interface

The emotion interface maps discrete emotion events to neurotransmitter modulation signals. Each emotion is defined as an `EmotionNTRecipe` specifying which NTs are driven and by how much.

### 10.1 Default Emotion Recipes (13)

| Emotion | NT Drives (positive = release boost, negative = suppression) |
|---------|--------------------------------------------------------------|
| joy | DA(0.8), 5HT(0.6), MOR(0.5), OXT(0.3) |
| curiosity | DA(0.7), 5HT(0.4), ACh(0.5), CB1(0.4), GLU(0.4), hist(0.4) |
| anxiety | NE(0.7), CRH(0.6), cort(0.5), DA(0.3), GABA(-0.3) |
| calm | 5HT(0.7), GABA(0.6), MOR(0.4), NE(-0.3) |
| empathy | OXT(0.8), 5HT(0.4), MOR(0.3) |
| focus | ACh(0.8), NE(0.5), DA(0.3), GABA(0.3), hist(0.6) |
| sadness | 5HT(-0.4), DA(-0.3), MOR(0.4), OXT(0.3) |
| anger | NE(0.8), DA(0.5), CRH(0.4), GABA(-0.4) |
| trust | OXT(0.8), 5HT(0.5), MOR(0.3), DA(0.3), GABA(0.3) |
| surprise | NE(0.7), DA(0.5), ACh(0.4), GLU(0.3), hist(0.5) |
| contentment | 5HT(0.7), MOR(0.6), GABA(0.4), DA(0.2) |
| fear | NE(0.9), CRH(0.8), cort(0.6), GABA(-0.5), DA(-0.2) |

> Negative drive values (e.g., `GABA -0.3` in anxiety) represent active suppression of that NT channel. Signal values are clamped to `[-1.0, 1.0]` per key.

### 10.2 Emotion Profile to Modulation Signals

The `emotion_profile_to_signals()` function converts an emotion profile (dict of `emotion_id` to intensity) into the `modulation_signals` dict consumed by `NeurochemicalEngine.step()`. Each emotion recipe is weighted by its intensity, and per-NT signal keys are aggregated and clamped.

---

## 11. Subsystem Interfaces

### 11.1 Extractor Orchestrator

The `ExtractorOrchestrator` sequences four extractors that bridge between domain evaluation results and neurochemical modulation:

1. Evaluation vector assembly from domain subscores
2. Emotion tracker stepping (temporal dynamics of emotion state)
3. Emotion splitting into modulatory (4M, tonic) and reactive (4R, phasic) pathways
4. Modulatory adjustments applied to evaluation vector
5. Urgency forecasting from current state
6. Regulatory modulation (bounds enforcement)
7. Urgency feedback merging
8. Oscillation envelope computation
9. Stochastic burst delta computation and signal merging

**Output:** `ExtractorResult` containing `evaluation_vector`, `modulation_signals`, `feedback_params`, `oscillation_update`, `emotion_saturations`, `dominant_emotion`, `burst_deltas`, `urgency_risk`.

### 11.2 Inference Matrix (Bidirectional NT ↔ Engine Coupling)

**Engine to NT** (evaluation results → NT modulation signals):

Maps cognitive engine evaluation results (confidence, contradictions, social_resonance, risk, novelty, domain_scores) to per-NT modulation signals:

| NT | Signal Keys Derived From |
|----|--------------------------|
| DA | novelty_detected → novelty; quality → rpe; emotion → emotion_drive |
| NE | quality → precision; 1-confidence → uncertainty; contradictions → contradiction |
| OXT | social_resonance → empathy, social_engagement |
| 5HT | quality → mood_stability; emotion → emotion_drive |
| GABA | risk → inhibition; 1-risk → boundary_proximity |
| cortisol | risk → stress_level |
| CRH | risk → acute_stress |
| ACh | quality → attention_demand, rule_fidelity |
| MOR | quality → hedonic_tone; 1-risk → comfort |
| CB1 | quality → flexibility |
| GLU | quality → integration_demand; quality → excitation |

**NT to Engine** (neurosymbolic metrics → engine priority weights):

Maps derived metrics to cognitive engine priority weights:

| Engine Priority | Formula |
|-----------------|---------|
| exploration | (motivation + openness - rigidity + 1) / 3 |
| verification | (precision + rigidity - fatigue + 1) / 3 |
| attunement | (empathy + social_engagement) / 2 |
| safety | (anxiety + (1 - openness)) / 2 |
| integration | ((1 - rigidity) + (1 - fatigue)) / 2 |

### 11.3 Domain NT Profiles

Domain-specific NT mapping modules translate domain evaluation subscores into NT modulation signals. Each domain defines:

- `target_nts` — which NTs the domain influences
- `signal_mappings` — per-subscore list of `NTSignalMapping` (`nt_name`, `signal_key`, `weight`, `invert`, `offset`)

**Implemented domains:**

- **Ethics:** targets GABA (inhibition), cortisol (time_pressure), CRH (acute_stress)
- **Logic:** targets NE, ACh, DA
- **Innovation:** targets DA, CB1, GLU
- **Human Attunement:** targets OXT, 5HT, MOR

### 11.4 Sleep Neurochemical State Management

The `SleepNeurochemicalStateManager` orchestrates pharmacodynamic transitions across sleep phases:

- **WAKE → TRIAGE:** save waking baselines, begin transition to sleep targets
- **TRIAGE → REM:** conditions: 5HT>0.50, ACh<0.30, delta>0.60, sigma>0.55
- **REM → DREAM:** additional conditions including desensitization flags
- **DREAM → CONSOLIDATION → WAKE:** gradual restoration of waking baselines

Transition dynamics use exponential approach:
```
C_next = C_current + k * (C_target - C_current) * dt
```

Transition rates: `k_enter=0.2`, `k_phase=0.25`, `k_exit=0.1`, `k_fast=0.4` (for NE/5HT fast collapse).

---

## 12. Engine Architecture

### 12.1 NeurochemicalEngine

The `NeurochemicalEngine` is the central online/real-time simulator. Key initialization parameters:

- `dt = 0.01` — integration timestep
- `seed = None` — RNG seed for reproducibility
- `use_lambda_loc_routing = False` — when `True`, effective concentration uses localization bias
- `oscillation_mode = "static"` — or `"state_derived"` for bidirectional closure

### 12.2 Step Procedure

The engine `step()` method executes the following sequence each tick:

1. For each registered NT: update concentration via module or generic pathway
2. Derive oscillations from NT concentrations (if `state_derived` mode)
3. For each registered receptor: update saturation, exposure trace, CTMC state transitions, G-protein coupling, sensitivity recovery
4. Apply subtype switching rules from receptor modules

### 12.3 Readout Pipeline

The `get_neurosymbolic_readout()` method produces a `NeurochemicalMetrics` object by:

1. Extracting concentrations from all NT states
2. Computing receptor saturations (`S = C/(C+K_d)`) for all receptors
3. Extracting oscillation amplitudes including CFC terms
4. Computing all 11 derived metrics via the formulas in Section 9

Extended readout (`compute_full_readout`) additionally evaluates neurosymbolic state expressions, trigger conditions, and mode selection hooks.

### 12.4 Additional Engine Methods

- `apply_emotion_event(emotion_id, intensity)` — triggers emotion-driven receptor plasticity across all registered receptor modules
- `apply_feedback(feedback_params)` — applies reward-conditioned parameter adjustments (`C_baseline_delta`, `u_base_multiplier`, `K_d_multiplier`)

---

## 13. Complete NT-Receptor Map

| Neurotransmitter | Receptor Subtypes |
|------------------|-------------------|
| DA | DA_D1, DA_D2, DA_D3, DA_D4, DA_D5 |
| 5HT | 5HT_1A, 5HT_1B, 5HT_2A, 5HT_2C, 5HT_3 |
| NE | NE_alpha1, NE_alpha2, NE_beta1, NE_beta2 |
| ACh | ACh_nicotinic, ACh_muscarinic |
| OXT | OXTR |
| MOR | MOR_mu |
| CB1 | CB1 |
| cortisol | *(no receptor in registry; direct modulation)* |
| CRH | CRH_R1 |
| GABA | GABA_A, GABA_B |
| GLU | GLU_NMDA, GLU_AMPA, GLU_KAINATE, GLU_mGluR |
| histamine | HIST_H1, HIST_H2, HIST_H3, HIST_H4 |

> **Total:** 12 neurotransmitter systems, 33 receptor subtypes (30 with dedicated receptor configs + 3 additional in family modules).
