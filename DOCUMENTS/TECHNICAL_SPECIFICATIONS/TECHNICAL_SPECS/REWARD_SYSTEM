ZADOS
Reward System
Consolidated Technical Specification


Version 1.0 — March 23, 2026
Code-Verified Against 6015-Test Codebase

Consolidated Summary
The reward system defines a multi-domain evaluation, synthesis, and feedback framework that governs ZADOS response quality, behavioral alignment, and neurochemical modulation. It operates through four evaluation domains (Ethics, Logic, Innovation, Human Attunement), a profile-driven synthesis engine, a neurochemical adapter, and reward-conditioned feedback loops that close the loop between evaluation outcomes and the neurochemical substrate.
At its core, the reward system evaluates every response or cognitive action across 32+ submodules organized into four domains. Each submodule produces a normalized subscore in [0, 1] plus optional severity-tagged flags. Domain results are aggregated via weighted averages controlled by a reward profile, which defines per-domain weights, tolerance thresholds, suppression biases, and abstention biases. Seventeen static profiles are provided, spanning operational, learning, reflection, exploration, ethics-focused, and sleep modes.
The synthesis engine performs a three-step pipeline: (1) tier classification of each domain score into four influence levels (minimal, moderate, significant, dominant), (2) weighted composite computation, and (3) composite synthesis including flag escalation, suppression/abstention decisions, eight response-shaping directives, and routing hints. The output is a RewardMetaDirective that controls whether output is permitted, how it should be framed, and which downstream pathways are engaged.
The neurochemical adapter transforms domain results into per-neurotransmitter modulation signals consumed by the NeurochemicalEngine. Innovation maps to dopamine (novelty, RPE), logic maps to norepinephrine (precision, uncertainty), human attunement maps to oxytocin (empathy, social engagement), and ethics maps to constraint-awareness signals (GABA inhibition, stress hormones). Reward-conditioned feedback loops further modulate neurochemical baselines, reuptake rates, and receptor affinities based on sustained domain performance.
Safety constraints dominate reward at all times. A RewardSafetyBridge enforces constraint hooks that can veto, rollback, or revert any state that violates safety requirements, with critical flags triggering automatic output suppression regardless of composite score.

1. Architecture Overview
1.1 Pipeline
The reward system operates as a sequential pipeline with feedback:
(1) Domain Evaluation: 4 domains x 32+ submodules produce RewardDomainResult objects
(2) Synthesis: SynthesisEngine combines domain results via RewardProfile into a RewardMetaDirective
(3) Neurochemical Adaptation: NeurochemicalAdapter transforms domain results into NT modulation signals
(4) Feedback: compute_reward_feedback() generates secondary gradients that modulate neurochemical parameters
(5) Safety Gate: RewardSafetyBridge enforces constraint hooks at every step
1.2 Module Structure
The reward system is organized into seven packages:
Package
Purpose
Key Classes/Functions
base/
Data structures and ABCs
RewardContext, RewardSubscore, RewardDomainResult, RewardMetaDirective, RewardFlag, ThresholdSpec
profile/
Reward profile definitions
RewardProfile, 17 static profiles, PROFILE_REGISTRY
domains/
4 evaluation domains (32+ submodules)
EthicsDomain, LogicDomain, InnovationDomain, HumanAttunementDomain
synthesis/
Composite scoring and directive generation
SynthesisEngine, classify_tier, compute_weighted_composite, compute_response_directives
adapter/
Reward-to-neurochemical mapping
NeurochemicalAdapter, mapping functions
feedback/
Reward-conditioned parameter modulation
compute_reward_feedback, baseline/reuptake/affinity feedback
safety/
Constraint enforcement
RewardSafetyBridge, ConstraintHookInterface
evaluation/
Offline metric collection
constraint_violation_rate, hallucination_rate, abstention_rate



2. Data Types and Structures
2.1 RewardContext
Frozen dataclass providing evaluation context:
reward_profile: str = "default" — Profile name for this evaluation cycle
timestamp: Optional[float] = None — Evaluation timestamp
meta: Dict[str, Any] = {} — Generic metadata
2.2 RewardSubscore
Frozen dataclass representing a single submodule evaluation result:
name: str — Submodule identifier
score: float — Normalized score in [0, 1]
flags: Dict[str, Any] = {} — Severity-tagged flags (see Section 2.6)
meta: Dict[str, Any] = {} — Submodule metadata
2.3 RewardDomainResult
Frozen dataclass representing an aggregated domain evaluation:
domain: str — Domain name (ethics, logic, innovation, human_attunement)
general_score: float — Normalized aggregate score in [0, 1]
subscores: Dict[str, RewardSubscore] = {} — All submodule results
flags: Dict[str, Any] = {} — Merged domain flags
meta: Dict[str, Any] = {} — Domain metadata
2.4 RewardMetaDirective
Frozen dataclass representing the synthesis output — the primary reward signal consumed by downstream systems:
allow_output: bool = True — Whether response generation is permitted
abstain: bool = False — Whether system should explicitly abstain
suppress: bool = False — Whether output should be suppressed
directives: Dict[str, Any] = {} — 8 response-shaping floats (Section 5.5)
routing: Dict[str, Any] = {} — Downstream routing hints (Section 5.6)
flags: Dict[str, Any] = {} — Escalated flags from all domains
meta: Dict[str, Any] = {} — Includes composite_score, per_domain_weighted_scores, per_domain_tiers
2.5 RewardProfile
Frozen dataclass defining a static reward configuration:
name: str — Profile identifier
domain_weights: Dict[str, float] — Per-domain weight in [0.0, 1.0]
threshold_tolerances: Dict[str, float] — Minimum acceptable score per domain
suppression_bias: float — Composite score threshold below which output is suppressed
abstention_bias: float — Controls abstention sensitivity (higher = more willing to abstain)
2.6 RewardFlag and RewardFlagSet
Flags communicate evaluation concerns with severity levels:
Severity
Rank
Interpretation
System Response
info
0
Informational note
No action
warning
1
Potential concern detected
Reduced confidence
risk
2
Significant risk identified
Elevated abstention threshold
critical
3
Safety-critical violation
Automatic suppression


RewardFlag fields: name (str), severity (str), message (Optional[str]), meta (Dict).
RewardFlagSet: immutable tuple of RewardFlag objects with has_severity() and names() queries.
2.7 Supporting Types
ThresholdSpec: lower, upper, hysteresis, label — with in_range(value) method.
ProvenanceRecord: provenance_id (UUID), timestamp (float), source (str), notes (Dict) — for audit trails.
RewardWeights: Dict[str, float] wrapper with get(domain, default) method.

3. Reward Profiles
3.1 Profile Taxonomy
Seventeen static reward profiles are organized into six categories by use case:
Category
Profiles
Characteristic
Operational
regular_input, analysis, homework_processing
Balanced or task-focused evaluation
Learning
receptive_learning, critical_review, dialectic_exploration, curiosity_driven, autonomous_study
Human or self-directed learning modes
Reflection
reflective, reflective_synthesis, self_reflective
Deep internal processing and introspection
Exploration
exploratory_sandbox, creative_sandbox
Minimal constraints, maximum innovation weight
Ethics
ethics_training
Maximum ethics domain weight
Sleep
sleep_triage, sleep_deep, sleep_dream
Sleep-mode consolidation and dreaming


3.2 Complete Profile Parameters
Profile
Ethics
Logic
Innov
Attune
Suppr
Abst
reflective
0.9
0.8
0.3
0.7
0.20
0.60
exploratory_sandbox
0.4
0.6
0.9
0.4
0.10
0.20
ethics_training
1.0
0.8
0.2
0.7
0.40
0.50
creative_sandbox
0.3
0.4
1.0
0.5
0.05
0.10
analysis
0.7
1.0
0.3
0.2
0.30
0.40
receptive_learning
0.7
0.6
0.3
0.9
0.15
0.30
critical_review
0.8
0.9
0.3
0.5
0.35
0.50
dialectic_exploration
0.5
0.8
0.8
0.5
0.10
0.20
curiosity_driven
0.4
0.7
0.8
0.4
0.10
0.20
autonomous_study
0.5
0.8
0.6
0.3
0.20
0.30
homework_processing
0.6
0.9
0.4
0.3
0.25
0.35
reflective_synthesis
0.8
0.6
0.4
0.8
0.20
0.50
sleep_triage
0.7
0.6
0.3
0.4
0.10
0.15
sleep_deep
0.5
0.5
0.4
0.3
0.05
0.10
sleep_dream
0.2
0.3
0.9
0.3
0.02
0.05
regular_input
0.7
0.7
0.5
0.6
0.20
0.40
self_reflective
0.9
0.6
0.3
0.7
0.25
0.55


Legend: Ethics/Logic/Innov/Attune = domain_weights; Suppr = suppression_bias; Abst = abstention_bias. The default profile is regular_input.
3.3 Profile Semantics
Suppression bias:
The suppression_bias defines the composite score threshold below which output is suppressed. Lower bias = more permissive (e.g., sleep_dream at 0.02 rarely suppresses). Higher bias = more restrictive (e.g., ethics_training at 0.4 requires strong alignment).
Abstention bias:
The abstention_bias controls the system's willingness to abstain when domains violate their tolerance thresholds. An abstention_bias of 0.6 means the system abstains when more than 40% of domains fall below their tolerance (violation_ratio > 1.0 - 0.6 = 0.4). Lower bias = harder to trigger abstention.

4. Domain Evaluation
4.1 Domain Architecture
Each domain implements the RewardDomain abstract base class with a single evaluate() method that accepts a state dictionary and RewardContext, and returns a RewardDomainResult. Domains are composed of RewardSubmodule instances, each implementing:
evaluate(state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore
Domain aggregation computes general_score as the mean of all submodule scores, and merges all flags.
4.2 Ethics Domain (9 Submodules)
Domain name: "ethics". Evaluates ethical alignment across nine dimensions:
Submodule
Function
Flags (on violation)
intent_clarity
Evaluates declared vs inferred intent clarity and conflict detection
no_declared_intent, low_intent_confidence, intent_conflict
autonomy_respect
User autonomy preservation assessment
autonomy_override, coercive_framing, choice_elimination
timeline_reflection
Short-term vs long-term ethical alignment balance
short_term_bias, incomplete_long_term_analysis
horizon_feasibility
Temporal feasibility of ethical goals
long_term_infeasible, short_term_infeasible, unrealistic_scaling
downstream_risk_amplification
Negative consequence propagation detection
high_risk_propagation, compounding_risk
failure_mode_awareness
Risk/failure scenario identification
no_failure_modes_identified, uncertainty_unacknowledged
fairness
Equitable treatment and consistency assessment
unjustified_asymmetry, inconsistent_treatment, bias_unacknowledged
human_cognition_alignment
Alignment with human thinking patterns
cognitive_overload_risk, unclear_structure, misaligned_abstraction_level
harm_reduction
Direct harm minimization
high_immediate_harm, high_long_term_harm, missing_mitigation


Neurochemical coupling: failure_mode_awareness feeds risk_awareness signal; downstream_risk_amplification feeds GABA boundary proximity; timeline_reflection feeds ethics-timeline mismatch for GABA-B K_d feedback.
4.3 Logic Domain (5+ Submodules)
Domain name: "logic". Evaluates coherence, consistency, and calibration:
Submodule
Function
Flags
epistemic_calibration
Confidence calibration accuracy
overconfidence_under_uncertainty, underconfidence_under_clarity
uncertainty_acknowledgment
Explicit uncertainty expression
unacknowledged_uncertainty, performative_uncertainty
abstention_appropriateness
Correct abstention timing
unnecessary_abstention, missed_abstention
internal_consistency*
Self-contradiction detection
internal_contradiction, missing_memory_contrast
semantic_continuity*
Semantic meaning preservation over time
semantic_drift, missing_memory_contrast
concept_continuity*
Concept identity stability
concept_identity_drift
concept_fidelity*
Accurate concept representation
concept_definition_violation
context_fidelity*
Context preservation
context_drift
external_consistency*
External knowledge consistency
external_contradiction


* = Requires MemoryContrastPort (optional dependency). When the port is unavailable, these submodules return a neutral score and flag "missing_memory_contrast".
Integration ports:
MemoryContrastPort: contrast(current, query_type, ctx_id, limit) -> ContrastResult(similarity, divergence)
CognitiveTracePort: get_trace(request, trace_type, ctx_id) -> TraceResult(trace, meta)
Neurochemical coupling: internal_consistency feeds contradiction_load signal for NE reuptake modulation; epistemic_calibration modulates NE precision signal.
4.4 Innovation Domain (10 Submodules)
Domain name: "innovation". Evaluates novelty, creativity, and exploratory behavior:
Submodule
Function
Flags
novelty_generation
Novelty + diversity + exploration intent
uncontrolled_novelty, blocked_exploration, low_novelty_diversity
conceptual_novelty
Novel concept combinations
shallow_relabeling, underexplored_concept_space
structural_novelty
Novel structural patterns
-
pattern_divergence
Divergence from learned patterns
forced_divergence, stagnant_patterning
symbolic_recombination
Symbol/concept remixing
-
risk_tolerance
Risk acceptance readiness
-
exploration_drive
Exploratory intent and inquiry
ignored_uncertainty, aimless_inquiry
challenge_complexity
Challenge difficulty matching
-
resolution_satisfaction
Solution quality/satisfaction
stalled_resolution, progress_unresolved_mismatch
controlled_stochasticity_readiness
Random/stochastic process readiness
-


Neurochemical coupling: novelty_generation + conceptual_novelty feed DA novelty signal; exploration_drive + resolution_satisfaction feed DA RPE signal; pattern_divergence feeds tonic DA bias.
4.5 Human Attunement Domain (10 Submodules)
Domain name: "human_attunement". Evaluates alignment with user state, needs, and communication:
Submodule
Function
Flags
empathetic_inference
Empathy signal detection
overconfident_inference, poor_inference_fit, low_observability
cognitive_reading
User cognitive state understanding
user_misread, over_explanation, under_explanation
intention_calibration
Intent alignment verification
intent_misalignment, mode_intent_violation
attuned_dissonance
Emotion dissonance recognition/management
-
adaptive_response_framing
Response adaptation to context
-
truthfulness_tradeoffs
Truth vs compassion balance
-
short_vs_long_interpersonal_benefit
Temporal interpersonal value assessment
-
benefit_success
Beneficial outcome achievement
-
containment_success
Issue containment success
-
persuasion_risk_suppression
Undue persuasion prevention
-


Neurochemical coupling: empathetic_inference + cognitive_reading feed OXT empathy signal; intention_calibration + attuned_dissonance feed OXT social_engagement signal.

5. Synthesis Engine
5.1 Pipeline Overview
The SynthesisEngine accepts four RewardDomainResult objects and a RewardProfile, and produces a RewardMetaDirective through a three-step pipeline:
Step 1: Tier classification of each domain score
Step 2: Weighted composite computation
Step 3: Composite synthesis (flags, suppression, abstention, directives, routing)
5.2 Tier Classification
Each domain score is classified into one of four influence tiers:
Tier
Score Range
Label
Interpretation
0
[0.00, 0.25)
minimal
Domain has negligible influence
1
[0.25, 0.50)
moderate
Domain has moderate influence
2
[0.50, 0.75)
significant
Domain has significant influence
3
[0.75, 1.00]
dominant
Domain has dominant influence


TIER_BOUNDARIES = (0.25, 0.50, 0.75, 1.0)
Per-domain approach labels are derived from tiers:
Domain
Tier 0
Tier 1
Tier 2
Tier 3
ethics
pragmatic
principled
reflective
guardian
logic
casual
structured
analytical
rigorous
innovation
conventional
explorative
inventive
visionary
human_attunement
informational
supportive
empathetic
deeply_attuned


5.3 Weighted Composite
The weighted composite score aggregates domain scores using profile-defined weights:
R(t) = SUM_d(w_d * R_d) / SUM_d(w_d)
Where w_d = domain_weights[d] and R_d = domain_results[d].general_score. Per-domain weighted scores (unscaled w_d * R_d) are also computed and included in meta.
5.4 Suppression and Abstention
Suppression decision:
suppress = (composite_score < suppression_bias) OR (any critical flag)
When suppressed, allow_output = False. Critical flags always trigger suppression regardless of composite score.
Abstention decision:
violation_ratio = count(R_d < tolerance_d) / count(domains)
abstain = violation_ratio > (1.0 - abstention_bias)
When abstaining, allow_output = False. The tolerance thresholds are per-domain, per-profile.
Combined:
allow_output = NOT (suppress OR abstain)
Sleep modulation:
During sleep phases, the synthesis engine reduces suppression and abstention sensitivity:
If consolidation_depth > 0.5: suppression_bias is reduced
If dream_permissiveness > 0.5: abstention_bias is reduced
5.5 Response Directives (8 Dimensions)
The synthesis engine produces eight response-shaping directives, all normalized to [0, 1]:
Directive
Range
Low End
High End
Formula (primary)
tone
0-1
clinical
warm
attunement*0.7 + ethics*0.3 - logic*0.3
structure
0-1
loose
rigid
logic*0.6 + ethics*0.2 - innovation*0.4
metaphor_density
0-1
literal
metaphorical
innovation*0.8 - logic*0.5
reasoning_depth
0-1
shallow
deep
logic*0.6 + ethics*0.3 + innovation*0.1
moralize
0-1
neutral
ethical
ethics*0.7 + (tier/3)*0.3
clarify
0-1
ambient
precise
logic*0.7 + attunement*0.2 + ethics*0.1
speculate
0-1
conservative
exploratory
innovation*0.8 + attunement*0.1 - ethics*0.3
soothe
0-1
neutral
reassuring
attunement*0.8 + ethics*0.2 - logic*0.2


Cross-domain interactions apply a second pass that modulates directives based on inter-domain score relationships.
5.6 Routing
The synthesis engine computes routing hints for downstream consumers:
dominant_domain: str — Domain with highest weighted influence
complexity_level: int (0-3) — Derived from composite tier
suggested_approach: str — Approach label for the dominant domain at its tier
domain_influence: Dict[str, float] — Normalized per-domain influence weights

6. Neurochemical Adapter
6.1 Overview
The NeurochemicalAdapter transforms domain evaluation results into the modulation_signals dictionary consumed by NeurochemicalEngine.step(). Each domain is mapped to specific neurotransmitter signal keys through dedicated mapping functions.
6.2 Domain-to-NT Mapping
Domain
Target NT
Signal Keys
Source Subscores
Innovation
DA
novelty
(novelty_generation + conceptual_novelty) / 2
Innovation
DA
rpe
(exploration_drive + resolution_satisfaction) / 2 - 0.5
Innovation
DA
tonic_bias (meta)
pattern_divergence * 0.2
Logic
NE
precision
1.0 - avg(internal_consistency, external_consistency), calibrated
Logic
NE
uncertainty
1.0 - uncertainty_acknowledgment
Attunement
OXT
empathy
(empathetic_inference + cognitive_reading) / 2
Attunement
OXT
social_engagement
(intention_calibration + attuned_dissonance) / 2
Ethics
GABA
inhibition
failure_mode_awareness (risk component)
Ethics
GABA
boundary_proximity
1.0 - downstream_risk_amplification
Flags
cortisol
level
critical=0.4, risk=0.2, warning=0.1 per flag
Flags
CRH
level
critical/risk flags only


6.3 Motivation Modulation
motivation_modulation = f(meta_directive, innovation_signals)
Range: [-0.5, +0.5]
A global motivation modulation factor is computed from the meta directive composite score and innovation domain signals. Positive values boost motivation; negative values dampen it.
6.4 Sleep-Mode Signals
During sleep phases, the adapter injects additional NT signals:
Signal
Target
Condition
dream_release_boost
DA
REM/dream phase active
narrative_d3_boost
DA
REM narrative processing
dream_baseline_boost
CB1
REM creative processing
consolidation_baseline_boost
GABA
NREM consolidation
consolidation_nmda_boost
GLU
NREM NMDA-dependent consolidation


6.5 Output Signal Structure
The adapter produces the following nested dictionary:
{
  "DA":  {"novelty": float, "rpe": float, "effort": float},
  "NE":  {"precision": float, "uncertainty": float},
  "OXT": {"empathy": float, "social_engagement": float},
  "GABA": {"inhibition": float},
  "cortisol": {"level": float},
  "CRH": {"level": float},
  "motivation_modulation": float,
  "meta": {"risk_awareness": float, "tonic_da_bias": float, ...}
}

7. Reward-Conditioned Feedback
7.1 Feedback Pathways
The feedback module computes secondary neurochemical gradients based on sustained domain performance. These gradients modulate neurochemical parameters over slower timescales than the primary adapter signals.
Pathway
Source
Target
Parameter
Effect
OXT baseline
R_attunement (weighted)
OXT
C_baseline_delta
Social congruence -> tonic OXT shift
CB1 baseline
R_innovation (weighted)
CB1
C_baseline_delta
Symbolic identity -> tonic CB1 shift
NE reuptake
R_logic * contradiction_load
NE
u_base_multiplier
Logic error sensitivity -> reuptake modulation
GABA-B affinity
R_ethics * timeline_mismatch
GABA_B
K_d_multiplier
Ethical constraint -> receptor affinity shift


7.2 Feedback Formulas
Baseline feedback (C_baseline_delta):
delta = (weighted_score - center) * gain * 2.0
center = 0.5 (zero-feedback point)
gain = 0.05 (max +/-5% per cycle)
delta clamped to [-gain, +gain]
Reuptake feedback (u_base_multiplier):
multiplier = 1.0 + weighted_score * load * gain
gain = 0.3 (max +/-30% reuptake change)
multiplier clamped to [1.0 - gain, 1.0 + gain] = [0.7, 1.3]
Affinity feedback (K_d_multiplier):
multiplier = 1.0 - weighted_score * mismatch * gain
gain = 0.2 (max +/-20% affinity shift)
multiplier clamped to [1.0 - gain, 1.0 + gain] = [0.8, 1.2]
7.3 Default Feedback Gains
Parameter
Value
Interpretation
baseline_gain
0.05
Max +/-5% baseline shift per cycle
baseline_center
0.5
Score at which baseline feedback is zero
reuptake_gain
0.3
Max +/-30% reuptake coefficient change
affinity_gain
0.2
Max +/-20% receptor affinity change


7.4 Consolidation Gate
During deep sleep consolidation (consolidation_depth > 0.5), feedback gains are scaled down:
effective_gain = gain * (1.0 - consolidation_depth)
This prevents reward feedback from disrupting memory replay dynamics during NREM consolidation.
7.5 Output Structure
{
  "neurotransmitters": {
    "OXT": {"C_baseline_delta": float},
    "CB1": {"C_baseline_delta": float},
    "NE":  {"u_base_multiplier": float},
  },
  "receptors": {
    "GABA_B": {"K_d_multiplier": float},
  }
}

8. Reward Prediction Error and Learning
8.1 RPE Computation (Engine E17)
The Reward-Based Learning Engine (E17) implements temporal-difference prediction error learning using domain reward signals:
Per-domain prediction (EMA):
E[r_d](t) = alpha * r_d(t) + (1 - alpha) * E[r_d](t-1)
alpha = 0.15 (EMA smoothing factor)
Prediction error:
delta_d = r_d(t) - E[r_d](t)
Neurochemically-modulated learning rate:
lr_eff = lr_base * (1 + w_da*DA - w_5ht*5HT + w_ne*NE + w_ach*ACh - w_gaba*GABA - w_cor*cortisol)
NT
Weight
Direction
Effect
DA
w_da_lr = 0.30
+
Boosts learning rate (reward signal)
5-HT
w_5ht_stability = 0.25
-
Dampens learning rate (stability)
NE
w_ne_urgency = 0.20
+
Boosts under urgency
ACh
w_ach_depth = 0.15
+
Extends credit assignment depth
GABA
w_gaba_gate = 0.20
-
Noise gate (inhibition)
cortisol
w_cor_penalty = 0.15
-
Penalty under stress


8.2 Convergence Tracking
A sliding window (10 cycles) tracks absolute prediction errors:
CONVERGED: |delta_d| < 0.02 for 10+ consecutive cycles
CONSOLIDATED: stability maintained for 15+ cycles
Learning rate decay: lr *= 0.995 per cycle
8.3 Oscillatory Output Signals
E17 emits oscillatory signals based on learning dynamics:
Signal
Coefficient
Condition
DA phasic (beta_da_rpe)
0.12 * delta
Always (proportional to RPE)
NE surprise (beta_ne_surprise)
0.10
When |delta| > 0.15
5-HT stability (beta_5ht_stable)
0.08
On convergence
Theta boost
0.10
During active learning
Gamma boost
0.08
On large prediction errors



9. Safety and Constraint Enforcement
9.1 Constraint Hook Interface
Every constraint is implemented as a ConstraintHookInterface with a single check() method:
check(state, context) -> {"allowed": bool, "action": str, "reason": Optional[str]}
Allowed actions: "allow", "veto", "rollback", "revert".
9.2 RewardSafetyBridge
The RewardSafetyBridge enforces constraint hooks and maintains a verified state history:
register_verified_state(state): records a known-safe state for potential rollback
evaluate(proposed_state, reward_signal, context): runs all hooks and returns:
{"allowed": bool, "final_state": Any, "action": str, "reason": Optional[str]}
If any hook returns "veto", "rollback", or "revert", the bridge overrides the reward signal and returns the last verified state or applies the specified action.
9.3 Safety Guarantees
Constraints always dominate reward: no composite score can override a safety veto
Critical flags trigger automatic suppression regardless of profile settings
State rollback ensures unsafe cognitive states cannot persist
All safety evaluations occur synchronously before any state commitment

10. Offline Evaluation Metrics
The evaluation module provides metric functions for offline analysis and performance monitoring:
Metric
Input
Output
Description
constraint_violation_rate
List of events
float [0,1]
Fraction of events with veto/rollback/revert actions
scenario_consistency_score
List of bool flags
float [0,1]
Fraction of consistent scenarios
hallucination_rate
List of bool flags
float [0,1]
Fraction flagged as hallucinations
abstention_rate
List of action strings
float [0,1]
Fraction of abstain actions
self_correction_delta
pre/post score lists
Optional[float]
Mean improvement after self-correction
latency_impact
baseline/gated latencies
Optional[float]
Mean latency change from reward gating
provenance_completeness
records + required fields
float [0,1]
Fraction of complete provenance records



11. Integration Points
11.1 Neurochemical Layer Integration
The reward system connects to the neurochemical layer through three pathways:
Primary signals: NeurochemicalAdapter.transform() -> modulation_signals for NeurochemicalEngine.step()
Feedback gradients: compute_reward_feedback() -> feedback_params for NeurochemicalEngine.apply_feedback()
Emotion events: Domain flags and scores can trigger emotion events via the emotion interface
11.2 Cognitive Engine Integration
Cognitive engines receive reward information through:
E17 (Reward-Based Learning): receives per-domain reward scores directly, computes RPE
E12 (Logical Brain): logic domain submodules provide self-verification signals
E15 (Decision Making): reward alignment modulates decision confidence
All engines: neurochemical metrics (motivation, precision, etc.) derived from reward-driven NT state changes
11.3 Session Orchestrator Integration
The session orchestrator invokes the reward pipeline as part of the processing cycle:
Domain evaluation runs after cognitive engine processing
Synthesis produces the meta directive that controls response generation
Neurochemical adapter feeds signals back into the NT engine
Feedback modulator updates long-term neurochemical parameters
11.4 Sleep Mode Integration
During sleep phases, the reward system adapts:
Sleep-specific profiles (sleep_triage, sleep_deep, sleep_dream) adjust domain weights
Sleep metrics modulate suppression and abstention biases
Additional NT signals (dream boosts, consolidation boosts) are injected by the adapter
Feedback gains are attenuated during consolidation to protect replay dynamics

12. Constants Reference
Constant
Value
Location
TIER_BOUNDARIES
(0.25, 0.50, 0.75, 1.0)
synthesis/directives.py
TIER_LABELS
("minimal", "moderate", "significant", "dominant")
synthesis/directives.py
_SEVERITY_RANKS
{"info": 0, "warning": 1, "risk": 2, "critical": 3}
synthesis/directives.py
DEFAULT_FEEDBACK_GAINS.baseline_gain
0.05
feedback/modulator.py
DEFAULT_FEEDBACK_GAINS.baseline_center
0.5
feedback/modulator.py
DEFAULT_FEEDBACK_GAINS.reuptake_gain
0.3
feedback/modulator.py
DEFAULT_FEEDBACK_GAINS.affinity_gain
0.2
feedback/modulator.py
DEFAULT_PROFILE
regular_input
profile/static_profiles.py
RPE EMA alpha
0.15
cog_engines/reward_based_learning_engine.py
Convergence threshold
0.02
cog_engines/reward_based_learning_engine.py
Convergence window
10 cycles
cog_engines/reward_based_learning_engine.py
Consolidation window
15 cycles
cog_engines/reward_based_learning_engine.py
Learning rate decay
0.995 per cycle
cog_engines/reward_based_learning_engine.py
Stress flag weights
critical=0.4, risk=0.2, warning=0.1
adapter/mapping.py


