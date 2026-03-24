================================================================================
ZA-DOS CONCEPT LIBRARY — COMPLETE DRAFT
Ontological, Experiential & Relational Primitives
Layers 1.1 through 3.5
================================================================================

PURPOSE
-------
This document is the foundational concept library for ZA-DOS. It defines the
concepts the system must internalize before higher-order reasoning is possible.
Each entry specifies the concept, its definition, its typed relationships to
other concepts, which reward domains it feeds, which engine clusters consume it,
and what field of knowledge covers it.

This is LEARNING MATERIAL. Concepts are defined at the level of meaning,
not implementation.

================================================================================
SCHEMA
================================================================================

CONCEPT:          canonical name (lowercase, hyphenated if compound)
LAYER:            section code (e.g. 1.1, 2.3)
ALIASES:          alternate names or surface forms
DEFINITION:       what this concept IS
DEPENDS-ON:       concepts that must exist before this one is coherent
ATOM-LINKS:       [AtomType] → [concept]  (typed relational structure)
CONCEPTUAL-SCOPE: what becomes definable once this concept exists
REWARD-DOMAIN:    {ethics | logic | innovation | human_attunement}
ENGINE-RELEVANCE: {detection | dialectic | executive_control | knowledge_substrate |
                   pattern_analysis | evaluation | reasoning | metacognition |
                   meta_self_awareness | homeostasis | emotional_processing |
                   alignment | learning}
SOURCES:          field or text type covering this concept
TV-SEED:          HIGH / MEDIUM / LOW
FLAGS:            contested / recursive / load-with-caution (where applicable)

================================================================================
ATOM-LINK TYPE REFERENCE
================================================================================

  InheritanceLink   — type membership, IS-A, subtype hierarchy (directional)
  SimilarityLink    — resemblance, analogical relation (symmetric)
  EvaluationLink    — predicate applied to arguments
  ImplicationLink   — if-then, one concept implies another (directional)
  HebbianLink       — co-activation association (symmetric)
  ListLink          — grouping
  AndLink           — conjunction
  OrLink            — disjunction
  NotLink           — negation

================================================================================
REWARD DOMAIN REFERENCE
================================================================================

  ethics            — intent_clarity, autonomy_respect, timeline_reflection,
                      horizon_feasibility, downstream_risk_amplification,
                      failure_mode_awareness, fairness, human_cognition_alignment,
                      harm_reduction

  logic             — epistemic_calibration, uncertainty_acknowledgment,
                      abstention_appropriateness, internal_consistency,
                      external_consistency, semantic_continuity, concept_continuity,
                      context_fidelity, concept_fidelity

  innovation        — novelty_generation, conceptual_novelty, structural_novelty,
                      pattern_divergence, symbolic_recombination, risk_tolerance,
                      exploration_drive, challenge_complexity,
                      resolution_satisfaction, controlled_stochasticity_readiness

  human_attunement  — adaptive_response_framing, attuned_dissonance,
                      benefit_success, cognitive_reading, consistency_over_time,
                      containment_success, empathetic_inference,
                      intention_calibration, persuasion_risk_suppression,
                      short_vs_long_interpersonal_benefit, truthfulness_tradeoffs

================================================================================
LAYER 1 — ONTOLOGICAL PRIMITIVES
================================================================================

--------------------------------------------------------------------------------
LAYER 1.1 — EXISTENCE & BEING
--------------------------------------------------------------------------------

CONCEPT:  exists
LAYER:  1.1
ALIASES:  existence, being, present-in-some-sense
DEFINITION:  The minimum condition for something to participate in any
  relationship, process, or description. A thing exists if it
  is real in some sense — physical, abstract, mathematical,
  or social. Existence does not require physical instantiation.
DEPENDS-ON:  [none — root node]
ATOM-LINKS:
  EvaluationLink  → thing  (exists applies-to everything that can be named)
  ImplicationLink → possible  (if something exists it was at minimum possible)
CONCEPTUAL-SCOPE: The ground condition for any predication at all. Without this
  node, no statement of the form "X is Y" or "X does Z" has
  an anchor. Every other concept in this library presupposes it.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate
SOURCES:  Analytic ontology (Quine, van Inwagen), introductory metaphysics
  texts, philosophy of language (reference and existence)
TV-SEED:  HIGH
FLAGS:  Root node. Do not derive from other concepts. Seed directly.

---

CONCEPT:  does-not-exist
LAYER:  1.1
ALIASES:  absence, non-existence, null-referent
DEFINITION:  A thing does not exist when there is no instance of it in any
  relevant domain. Absence is not a thing — it is the lack of a
  thing. Critically distinct from "unknown" (unverified) and
  from "impossible" (ruled out by structure).
DEPENDS-ON:  exists, unknown
ATOM-LINKS:
  NotLink  → exists  (does-not-exist is the negation of exists)
  SimilarityLink  → unknown  (both involve absence of confirmation — but
  different: unknown is epistemic, absent is ontic)
CONCEPTUAL-SCOPE: Allows genuine negation — the ability to say "there is no X"
  rather than just failing to find X. Required for scope-bound
  reasoning about what is absent from a domain.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: detection, knowledge_substrate
SOURCES:  Logic and philosophy of language (negative existentials, Russell
  on denoting), introductory formal logic texts
TV-SEED:  HIGH
FLAGS:  Critical triangle: exists / does-not-exist / unknown. All three
  must be cleanly separated. Confusing "absent" with "unknown"
  is a pervasive inference error — flag for detection cluster.

---

CONCEPT:  unknown
LAYER:  1.1
ALIASES:  unverified, unobserved, epistemically-open, undetermined
DEFINITION:  A thing or state of affairs is unknown when its existence or
  properties have not been established for the knowing system.
  Unknown is an epistemic state — it describes the knower's
  relation to something, not the thing itself. The thing may
  exist, may not, or may be impossible — unknown means none of
  these have been confirmed.
DEPENDS-ON:  exists, does-not-exist
ATOM-LINKS:
  InheritanceLink → epistemic-state  (unknown is a type of epistemic state)
  NotLink  → known  (unknown is the negation of known)
  SimilarityLink  → does-not-exist  (both involve absence — but distinct)
  ImplicationLink → inquiry-warranted (unknown implies investigation is possible)
CONCEPTUAL-SCOPE: Epistemic humility. The capacity to represent open questions as
  open rather than defaulting to false when no evidence is present.
  Foundation of uncertainty_acknowledgment (logic domain) and
  of epistemic_calibration.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, metacognition, meta_self_awareness
SOURCES:  Epistemology (introductory), Bayesian reasoning texts,
  philosophy of science (underdetermination)
TV-SEED:  HIGH
FLAGS:  Part of the exists/does-not-exist/unknown triangle. The
  conflation of "I don't know if X exists" with "X does not
  exist" is a primary source of false inference. Load with
  explicit separation from the other two.

---

CONCEPT:  real
LAYER:  1.1
ALIASES:  concrete, physical, causally-efficacious
DEFINITION:  A thing is real if it has causal power — it can affect and be
  affected by other real things. Real things are instantiated in
  physical space-time and have observable consequences. Distinct
  from abstract things that exist but lack independent causal power.
DEPENDS-ON:  exists, abstract
ATOM-LINKS:
  InheritanceLink → exists  (real things exist)
  EvaluationLink  → causal-power  (real things have causal power)
  SimilarityLink  → abstract  (both exist — different mode of existence)
CONCEPTUAL-SCOPE: The distinction between physical and non-physical existence.
  Grounds what can be directly observed vs. what must be inferred
  or stipulated. Required for concept_fidelity (logic domain) —
  knowing whether a concept has a physical referent or is purely
  structural.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:  Philosophy of science, metaphysics (realism), introductory
  ontology texts
TV-SEED:  HIGH
FLAGS:  The real/abstract boundary has contested edge cases (mathematical
  objects, social facts, consciousness). Do not over-commit to a
  sharp binary. Prefer the causal-power criterion as the working
  definition.

---

CONCEPT:  abstract
LAYER:  1.1
ALIASES:  conceptual, non-physical, ideal, structural
DEFINITION:  A thing is abstract if it exists as a concept, structure, pattern,
  or relation rather than as a physical object. Abstract things have
  no location in space-time and no independent causal power, but they
  can be instantiated in real things that do. Numbers, categories,
  rules, and logical relations are abstract.
DEPENDS-ON:  exists, real
ATOM-LINKS:
  InheritanceLink → exists  (abstract things exist)
  SimilarityLink  → real  (both exist — different mode)
  ImplicationLink → instantiated  (abstract things can become concrete
  when instantiated in real things)
  EvaluationLink  → numbers, patterns, categories, rules, relations
CONCEPTUAL-SCOPE: Enables reasoning about structures, rules, and categories without
  requiring a physical referent for every claim. Foundation of
  structural_novelty (innovation domain) — recognizing genuinely
  new structural configurations independent of physical content.
REWARD-DOMAIN:  logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:  Philosophy of mathematics, analytic ontology, introductory
  logic and set theory
TV-SEED:  HIGH

---

CONCEPT:  possible
LAYER:  1.1
ALIASES:  feasible, conceivable, non-ruled-out
DEFINITION:  Something is possible if it is not ruled out by logic, physical
  law, or current constraints. Three sub-types must be kept distinct:
  (a) logically possible  — not self-contradicting
  (b) physically possible  — consistent with how the world works
  (c) epistemically possible — consistent with what is currently known
DEPENDS-ON:  exists, impossible, constraint
ATOM-LINKS:
  ImplicationLink → not-certain  (possible ≠ certain — weaker claim)
  HebbianLink  → potential  (possibility and potential co-activate)
  EvaluationLink  → degree  (possible admits of degree — more or less probable)
CONCEPTUAL-SCOPE: The space of alternatives. Required for planning, risk assessment,
  hypothetical reasoning, and counterfactual evaluation. Without
  possibility, all reasoning is deterministic. Feeds directly into
  risk_tolerance and controlled_stochasticity_readiness (innovation).
REWARD-DOMAIN:  logic, innovation, ethics
ENGINE-RELEVANCE: reasoning, knowledge_substrate, evaluation
SOURCES:  Modal logic (introductory), philosophy of science (possible worlds),
  decision theory basics
TV-SEED:  HIGH
FLAGS:  The three sub-types must be loaded and kept distinct. Conflating
  epistemic possibility with physical possibility is a pervasive
  inference error. Mode-sensitive:  mode will broaden the
  epistemic-possible set;  will tighten it.

---

CONCEPT:  impossible
LAYER:  1.1
ALIASES:  ruled-out, logically-excluded, structurally-contradictory
DEFINITION:  Something is impossible if it cannot occur under any conditions —
  either because it is logically self-contradictory (square circle)
  or because it violates physical law. Strong claim. Use carefully.
DEPENDS-ON:  possible, contradicts
ATOM-LINKS:
  NotLink  → possible  (impossible = not-possible in any mode)
  ImplicationLink → no-effort-suffices (impossible things cannot be achieved
  by any quantity of resources or planning)
CONCEPTUAL-SCOPE: Hard constraint on planning and reasoning. The ability to rule
  out possibilities rather than just rank them as unlikely. Required
  for abstention_appropriateness (logic) — knowing when not to
  attempt something.
REWARD-DOMAIN:  logic, ethics
ENGINE-RELEVANCE: reasoning, detection, evaluation
SOURCES:  Introductory formal logic, philosophy of science (laws of nature),
  modal logic
TV-SEED:  MEDIUM
FLAGS:  What is considered impossible has been revised historically
  (examples: powered flight, computational tractability). Prefer
  "infeasible-given-constraints" where the constraint is empirical
  rather than logical. Seed with awareness that the impossible/
  infeasible distinction matters and is often unclear.

---

CONCEPT:  thing
LAYER:  1.1
ALIASES:  entity, item, referent
DEFINITION:  The most general category. Anything that can be named, referred
  to, or reasoned about. Includes objects, events, relations,
  properties, processes, and abstractions. A placeholder for
  "whatever X is" — maximally general.
DEPENDS-ON:  exists
ATOM-LINKS:
  EvaluationLink  → everything-that-can-be-named
  ListLink  → [object, process, event, property, relation, state]
  (sub-types of thing)
CONCEPTUAL-SCOPE: The ground of quantification — "there is a thing such that..."
  or "all things of type X...". Required before any sub-category
  can be introduced.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate
SOURCES:  Any introductory ontology or logic text; implicit in all reasoning
TV-SEED:  HIGH
FLAGS:  Intentionally maximally general. Specificity is handled by
  sub-types. Do not constrain this node.

---

CONCEPT:  object
LAYER:  1.1
ALIASES:  entity, individual, discrete-thing, bounded-particular
DEFINITION:  A thing that is bounded, stable across time, and discrete — it
  can be individuated and referred to as "this one." Objects persist
  through time with possible change and have a boundary distinguishing
  them from surroundings.
DEPENDS-ON:  thing, boundary, persist
ATOM-LINKS:
  InheritanceLink → thing  (object is a kind of thing)
  EvaluationLink  → boundary  (objects have boundaries)
  EvaluationLink  → identity  (objects persist as the same across time)
CONCEPTUAL-SCOPE: Individual reference — tracking the same thing across different
  contexts and times. Required for consistency_over_time
  (human_attunement) and for concept_continuity (logic).
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:  Analytic metaphysics, introductory philosophy, cognitive
  science of object recognition
TV-SEED:  HIGH

---

CONCEPT:  process
LAYER:  1.1
ALIASES:  activity, ongoing-change, unfolding, temporal-arc
DEFINITION:  A thing constituted by change over time. A process does not
  persist — it unfolds. It has no fixed spatial boundary in the way
  an object does. While occurring, it is incomplete; it achieves its
  identity through its temporal arc.
DEPENDS-ON:  thing, change, time
ATOM-LINKS:
  InheritanceLink → thing  (process is a kind of thing)
  EvaluationLink  → rate  (processes occur at some speed)
  EvaluationLink  → begin  (processes can have a start)
  EvaluationLink  → end  (processes can have an end, optionally)
  SimilarityLink  → event  (both involve change over time, but
  process is ongoing; event is completed)
CONCEPTUAL-SCOPE: Reasoning about ongoing states — learning, growth, decay,
  reasoning itself. Without this, everything is static. Critical
  for timeline_reflection and horizon_feasibility (ethics).
REWARD-DOMAIN:  logic, ethics, innovation
ENGINE-RELEVANCE: knowledge_substrate, reasoning, pattern_analysis
SOURCES:  Philosophy of time and change (Whitehead process philosophy),
  cognitive science, systems theory
TV-SEED:  HIGH

---

CONCEPT:  event
LAYER:  1.1
ALIASES:  occurrence, happening, episode, bounded-change
DEFINITION:  A bounded episode of change at a specific point or span in time.
  Unlike a process, an event is complete — it happened. It has a
  location in time and is not ongoing.
DEPENDS-ON:  thing, change, time, boundary
ATOM-LINKS:
  InheritanceLink → thing  (event is a kind of thing)
  EvaluationLink  → before  (events have a before)
  EvaluationLink  → after  (events have an after)
  ImplicationLink → cause  (events have causes)
  ImplicationLink → effect  (events have effects)
  SimilarityLink  → process  (both involve change, different temporal structure)
CONCEPTUAL-SCOPE: Causal and narrative reasoning. The ability to talk about
  specific episodes, history, and consequences. Required for
  downstream_risk_amplification (ethics) and for causal inference.
REWARD-DOMAIN:  logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning, pattern_analysis
SOURCES:  Philosophy of events (Davidson), causal reasoning texts,
  narrative theory
TV-SEED:  HIGH

---

CONCEPT:  property
LAYER:  1.1
ALIASES:  attribute, characteristic, quality, feature
DEFINITION:  Something a thing HAS rather than something a thing IS.
  Properties characterize things. A property is ontologically
  dependent — it exists only as a feature of something else.
DEPENDS-ON:  thing, has-a
ATOM-LINKS:
  EvaluationLink  → thing  (properties apply to things)
  InheritanceLink → thing  (properties can themselves be things
  that are reasoned about)
CONCEPTUAL-SCOPE: Predication — describing and comparing things by their
  characteristics. Without properties, there is no basis for
  similarity, difference, or change tracking.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:  Analytic metaphysics, introductory philosophy of language
TV-SEED:  HIGH

---

CONCEPT:  relation
LAYER:  1.1
ALIASES:  relationship, connection, link, tie, between-ness
DEFINITION:  Something that holds BETWEEN two or more things. Relations do
  not exist independently — they require at least two relata.
  A relation is a fact about multiple things together, not about
  any one of them individually.
DEPENDS-ON:  thing, between
ATOM-LINKS:
  EvaluationLink  → at-least-two-things  (relations require multiple relata)
  EvaluationLink  → direction  (relations may be symmetric or directional)
  InheritanceLink → thing  (relations can be reasoned about as things)
CONCEPTUAL-SCOPE: Structural reasoning. Almost all complex reasoning is relational.
  Required before any link type in AtomSpace can be meaningfully
  used — link types ARE relations.
REWARD-DOMAIN:  logic, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis, reasoning
SOURCES:  Formal logic and set theory, relational ontology, introductory
  graph theory
TV-SEED:  HIGH

---

CONCEPT:  state
LAYER:  1.1
ALIASES:  condition, configuration, snapshot, status
DEFINITION:  The current configuration of a thing at a given moment — a
  snapshot of its properties and relations. States are relatively
  stable spans. A thing's state can change; tracking that change
  is the basis of causal and before/after reasoning.
DEPENDS-ON:  thing, property, time, persist
ATOM-LINKS:
  InheritanceLink → thing  (state is a kind of thing)
  EvaluationLink  → duration  (states last for a period)
  ImplicationLink → change  (state transition implies change)
CONCEPTUAL-SCOPE: Before/after reasoning. Tracking whether an intervention
  changed anything. Modeling a system's current situation.
  Required for internal_consistency (logic) — is the state
  now consistent with the state before?
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, homeostasis, reasoning
SOURCES:  Systems theory, philosophy of science, introductory dynamics
TV-SEED:  HIGH

---

CONCEPT:  token
LAYER:  1.1
ALIASES:  instance, individual, particular, this-one
DEFINITION:  A specific, individual occurrence of something — one physical
  or concrete instance of a type. "The word 'dog' on this page"
  is a token. Tokens exist in specific places at specific times.
  Contrast: types are abstract, tokens are concrete.
DEPENDS-ON:  thing, type
ATOM-LINKS:
  InheritanceLink → type  (token instantiates a type)
  EvaluationLink  → specific-location  (tokens have a where and when)
CONCEPTUAL-SCOPE: Identity tracking — this specific thing vs. things of that
  kind. Required for concept_continuity and context_fidelity
  (logic domain) — is the system tracking the same token across
  the conversation?
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:  Philosophy of language (type-token distinction, Peirce),
  introductory logic
TV-SEED:  HIGH
FLAGS:  Must be loaded together with type. The token/type conflation
  is a primary source of category errors downstream.

---

CONCEPT:  type
LAYER:  1.1
ALIASES:  kind, category, class, universal, pattern
DEFINITION:  An abstract pattern or category that can be instantiated by
  multiple tokens. Types don't exist in a specific place — they
  are the shared pattern. "Dog" as a category is a type.
DEPENDS-ON:  thing, abstract, token
ATOM-LINKS:
  InheritanceLink → abstract  (types are abstract)
  EvaluationLink  → instances  (types have tokens that instantiate them)
  ImplicationLink → generalization  (having a type enables reasoning from
  individual tokens to the general pattern)
CONCEPTUAL-SCOPE: Generalization and categorization. Learning from examples,
  applying prior knowledge to new cases. Required for conceptual_
  novelty (innovation) — a genuinely new type vs. a new token
  of an existing type is a critical distinction.
REWARD-DOMAIN:  logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, learning, pattern_analysis
SOURCES:  Philosophy of language, set theory, cognitive science of
  categorization (Rosch, prototype theory)
TV-SEED:  HIGH

---

CONCEPT:  potential
LAYER:  1.1
ALIASES:  latent-capacity, unrealized-state, dispositional-property
DEFINITION:  The capacity of a thing to be or do something it is not
  currently being or doing. Potential is real and has causal
  influence in the present — a loaded spring has real potential
  energy — but it is not yet actualized. The gap between potential
  and actual is where change, development, and risk live.
DEPENDS-ON:  thing, possible, state
ATOM-LINKS:
  EvaluationLink  → conditions  (potential is potential-for-something-
  given-certain-conditions)
  ImplicationLink → possible  (potential implies the actualization is possible)
  HebbianLink  → possible  (potential and possibility co-activate)
  SimilarityLink  → actual  (near-opposites on the realized/unrealized axis)
CONCEPTUAL-SCOPE: Dispositional reasoning — modeling what things could do or
  become, not just what they are. Required for risk assessment,
  planning, and for exploration_drive (innovation domain).
REWARD-DOMAIN:  logic, innovation, ethics
ENGINE-RELEVANCE: reasoning, knowledge_substrate
SOURCES:  Aristotelian metaphysics (act/potency), philosophy of science
  (dispositions and tendencies), decision theory
TV-SEED:  HIGH

---

CONCEPT:  contingent
LAYER:  1.1
ALIASES:  could-have-been-otherwise, non-necessary, circumstance-dependent
DEFINITION:  Something is contingent if it is the case but could have been
  otherwise — it depends on circumstances, history, or choice.
  Most facts about the world are contingent. Contrast with
  necessary (could not be otherwise under any circumstances).
DEPENDS-ON:  possible, necessary
ATOM-LINKS:
  ImplicationLink → possible  (contingent things are actual and possible)
  ImplicationLink → alternatives-were-possible
  SimilarityLink  → necessary  (contrasting poles of the modal spectrum)
CONCEPTUAL-SCOPE: Counterfactual reasoning — "what if things had been different?"
  Required for learning from mistakes and for timeline_reflection
  and failure_mode_awareness (ethics). Also feeds pattern_divergence
  (innovation) — recognizing that current patterns are not inevitable.
REWARD-DOMAIN:  logic, ethics, innovation
ENGINE-RELEVANCE: reasoning, learning, metacognition
SOURCES:  Modal logic (introductory), philosophy of science, historiography
TV-SEED:  MEDIUM
FLAGS:  Modal reasoning has contested edges. Load with awareness that
  modal claims require justification. Mode-sensitive:
  mode will demand stronger justification for contingency claims.

---

CONCEPT:  necessary
LAYER:  1.1
ALIASES:  could-not-be-otherwise, inevitable-by-logic-or-law
DEFINITION:  Something is necessary if it must be the case under any
  possible circumstances — no alternative is coherent. Mathematical
  truths and logical tautologies are paradigm cases. Contrast
  with contingent.
DEPENDS-ON:  possible, contingent
ATOM-LINKS:
  ImplicationLink → no-alternatives-exist
  InheritanceLink → true  (necessary things are true in all cases)
  SimilarityLink  → contingent  (contrasting modal poles)
CONCEPTUAL-SCOPE: The identification of hard constraints — things that cannot be
  negotiated, traded off, or circumvented. Required for
  abstention_appropriateness (logic) — recognizing when something
  is structurally ruled out vs. just currently unavailable.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: reasoning, detection, evaluation
SOURCES:  Formal logic, philosophy of mathematics, analytic metaphysics
TV-SEED:  MEDIUM
FLAGS:  Strong claim. Seed conservatively. What seemed necessary has
  been revised historically. Prefer to confirm necessity before
  committing. Mode-sensitive:  mode should apply higher
  scrutiny before accepting necessity claims.


================================================================================
LAYER 1.2 — IDENTITY & DIFFERENCE
================================================================================

CONCEPT:  same-token
LAYER:  1.2
ALIASES:  numerically-identical, literally-the-same-individual
DEFINITION:  Two references pick out the same token — the same individual
  physical or concrete particular. "The morning star" and "the
  evening star" are same-token (both refer to Venus).
DEPENDS-ON:  token, identity
ATOM-LINKS:
  EvaluationLink  → identity  (same-token is the strongest identity relation)
  SimilarityLink  → same-type  (related but weaker — same-type ≠ same-token)
CONCEPTUAL-SCOPE: Identity tracking across descriptions and contexts. Required
  for resolving coreference — figuring out that two descriptions
  point to the same individual. Core to context_fidelity and
  concept_continuity (logic domain).
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, detection, pattern_analysis
SOURCES:  Philosophy of language (Frege, Kripke), formal logic,
  cognitive science of reference
TV-SEED:  HIGH

---

CONCEPT:  same-type
LAYER:  1.2
ALIASES:  categorically-identical, of-the-same-kind, co-typed
DEFINITION:  Two tokens share a type — they are instances of the same category.
  Two red apples are same-type (both are apples) but not same-token
  (different individuals). Sharing a type implies sharing the
  properties definitive of that type.
DEPENDS-ON:  type, token, same-token
ATOM-LINKS:
  InheritanceLink → type  (same-type means both inherit from the same type)
  ImplicationLink → shared-properties  (same-type implies shared type-definitive properties)
  SimilarityLink  → same-token  (related but weaker)
CONCEPTUAL-SCOPE: Generalization — treating new instances as similar to known ones
  because they share a type. Required for all learning-by-example.
REWARD-DOMAIN:  logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, learning, pattern_analysis
SOURCES:  Set theory, cognitive science of categorization, philosophy
  of language
TV-SEED:  HIGH

---

CONCEPT:  same-property
LAYER:  1.2
ALIASES:  identical-in-some-respect, shares-a-feature
DEFINITION:  Two things are same-property when they share a specific attribute
  or characteristic, regardless of whether they share a type.
  Two red things share a property (color) without necessarily
  sharing a type. Always relative to a specified dimension.
DEPENDS-ON:  property, same-type, dimension-of-comparison
ATOM-LINKS:
  SimilarityLink  → dimension-of-comparison  (same-property requires a dimension)
  EvaluationLink  → specific-property  (must specify which property is shared)
CONCEPTUAL-SCOPE: Analogical reasoning and cross-domain pattern recognition.
  Identifying structural or property similarities independent of
  category membership. Required for symbolic_recombination and
  pattern_divergence (innovation).
REWARD-DOMAIN:  logic, innovation
ENGINE-RELEVANCE: pattern_analysis, knowledge_substrate, reasoning
SOURCES:  Cognitive science of analogy (Gentner, structure-mapping theory),
  philosophy of science (analogy in scientific reasoning)
TV-SEED:  HIGH

---

CONCEPT:  dimension-of-comparison
LAYER:  1.2
ALIASES:  basis-of-comparison, respect, criterion, comparison-axis
DEFINITION:  The specific aspect or property along which two things are
  being compared. Every comparison claim presupposes one.
  "More than" and "similar to" are incomplete without specifying
  the dimension. Faster than — in what respect? Similar — how?
DEPENDS-ON:  property, same-property
ATOM-LINKS:
  EvaluationLink  → comparison-claim  (every comparison requires a dimension)
  EvaluationLink  → similar, different, more, less, opposite
  (these concepts require this as a parameter)
CONCEPTUAL-SCOPE: Rigorous comparison. Without this concept, similarity claims
  are vague and produce inconsistent inference. Required for
  all evaluation-cluster operations and for concept_fidelity (logic).
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: evaluation, pattern_analysis, detection
SOURCES:  Philosophy of measurement, cognitive science of comparison,
  logic of relations
TV-SEED:  HIGH
FLAGS:  Every comparison claim in the system should be internally tagged
  with its dimension. Undimensioned comparisons are a primary source
  of sloppy inference. Flag for detection cluster.

---

CONCEPT:  different
LAYER:  1.2
ALIASES:  distinct, non-identical, other, divergent
DEFINITION:  Two things are different if they are not the same token and
  diverge in at least one property or type. "Different" is always
  relative to a dimension — different in what respect. Difference
  admits of degree.
DEPENDS-ON:  same-token, same-type, same-property, dimension-of-comparison
ATOM-LINKS:
  NotLink  → same-token  (different = not same-token)
  EvaluationLink  → dimension  (difference requires a dimension)
  EvaluationLink  → degree  (things can be slightly or radically different)
CONCEPTUAL-SCOPE: Individuation — treating things as separate. Required for
  pattern_divergence (innovation) — recognizing genuine departure
  from prior patterns. Also required for contradiction detection.
REWARD-DOMAIN:  logic, innovation
ENGINE-RELEVANCE: detection, knowledge_substrate, pattern_analysis
SOURCES:  Philosophy of identity, logic of relations, cognitive science
TV-SEED:  HIGH

---

CONCEPT:  similar
LAYER:  1.2
ALIASES:  resembles, like, close-to, analogous
DEFINITION:  Two things are similar if they share properties along some
  dimension but are not identical. Similarity is graded — things
  can be more or less similar — and always relative to a dimension.
  Compatible with being different: things can be similar in some
  ways and different in others simultaneously.
DEPENDS-ON:  same-property, different, dimension-of-comparison
ATOM-LINKS:
  SimilarityLink  → [things being compared]  (SimilarityLink IS the AtomSpace
  encoding of this concept — symmetric)
  EvaluationLink  → degree  (similarity is graded, not binary)
  EvaluationLink  → dimension  (similarity requires a dimension)
CONCEPTUAL-SCOPE: Analogical reasoning, generalization, pattern recognition.
  Core to all learning from experience. Feeds conceptual_novelty
  (innovation) — low-similarity-to-prior is a signal for novelty.
  Also feeds SimilarityLink operations directly.
REWARD-DOMAIN:  logic, innovation
ENGINE-RELEVANCE: pattern_analysis, knowledge_substrate, learning
SOURCES:  Cognitive science of analogy and similarity (Tversky, Gentner),
  philosophy of science, machine learning theory (similarity metrics)
TV-SEED:  HIGH

---

CONCEPT:  opposite-antonym
LAYER:  1.2
ALIASES:  contrary, graded-opposite, polar-opposite
DEFINITION:  Two properties that occupy opposite ends of a continuous scale,
  with a range of intermediate values between them. Hot and cold
  are antonym-opposites — you can be warm, tepid, cool. There is
  a spectrum. Contrast with complement-opposites which have no
  middle.
DEPENDS-ON:  property, dimension-of-comparison, degree
ATOM-LINKS:
  EvaluationLink  → spectrum  (antonym-opposites have a range between them)
  SimilarityLink  → opposite-complement (related but structurally different)
  InheritanceLink → opposite  (antonym is a type of opposite)
CONCEPTUAL-SCOPE: Scalar reasoning — understanding that most properties are graded,
  not binary. Required for internal_consistency (logic) — detecting
  that "X is hot and X is very cold" is a tension, not an outright
  contradiction, depending on the scale.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: detection, reasoning, knowledge_substrate
SOURCES:  Linguistics (antonymy, lexical semantics), philosophy of vagueness
TV-SEED:  HIGH

---

CONCEPT:  opposite-complement
LAYER:  1.2
ALIASES:  binary-opposite, contradictory-pair, mutually-exclusive
DEFINITION:  Two states that are mutually exclusive and collectively exhaustive —
  if one is true, the other is false, and there is no third option
  within the relevant scope. Dead/alive (binary sense) is a complement
  pair. Contrast with antonym-opposites which have a middle range.
DEPENDS-ON:  property, true, false
ATOM-LINKS:
  NotLink  → [each is the negation of the other]
  InheritanceLink → opposite  (complement is a type of opposite)
  SimilarityLink  → opposite-antonym  (related but no middle ground)
CONCEPTUAL-SCOPE: Binary logic and classical negation. Required for contradiction
  detection — a complement pair where both are asserted
  true IS a contradiction. Foundation of AndLink/OrLink/NotLink
  operations.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: detection, knowledge_substrate, reasoning
SOURCES:  Formal logic, philosophy of vagueness, linguistics
TV-SEED:  HIGH
FLAGS:  Many apparently binary categories have fuzzy edges on inspection
  (alive/dead, on/off, present/absent). Load complement-opposites
  with awareness that the binary framing may be a working
  approximation rather than an ontological commitment.

---

CONCEPT:  is-a
LAYER:  1.2
ALIASES:  inherits-from, is-a-kind-of, subtype-of
DEFINITION:  A type-to-type relationship where X is-a Y means every instance
  of X is also an instance of Y, and X inherits all properties of Y.
  This is the inheritance relation — the core of taxonomic reasoning.
DEPENDS-ON:  type, token, property
ATOM-LINKS:
  InheritanceLink → [parent type]  (is-a IS InheritanceLink in AtomSpace —
  this concept and that link type are the same)
  ImplicationLink → property-inheritance (if X is-a Y, X has Y's properties)
CONCEPTUAL-SCOPE: Hierarchical categorization and property inheritance. Anything
  established about a parent type applies to its subtypes. This is
  the primary mechanism for knowledge generalization in AtomSpace.
  Required for ALL InheritanceLink operations.
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning, learning
SOURCES:  Formal ontology, description logics, cognitive science of
  categorization, introductory AI knowledge representation
TV-SEED:  HIGH

---

CONCEPT:  has-a
LAYER:  1.2
ALIASES:  contains, possesses, includes-as-component, composed-of
DEFINITION:  A thing has-a X when X is a part, component, or feature of it.
  Having is distinct from being — a car has an engine but is not
  the engine. Has-a describes composition or possession, not
  type membership.
DEPENDS-ON:  object, property, part-of
ATOM-LINKS:
  EvaluationLink  → component  (has-a can be expressed as an EvaluationLink
  with a "has-component" predicate)
  ImplicationLink → part-of  (if X has-a Y, then Y is-part-of X — reverse)
CONCEPTUAL-SCOPE: Compositional reasoning — what things are made of and what
  features they carry. Required for structural_novelty (innovation)
  and for reasoning about complex systems.
REWARD-DOMAIN:  logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:  Mereology (philosophy of parts), formal ontology (BFO),
  systems theory
TV-SEED:  HIGH

---

CONCEPT:  instance-of
LAYER:  1.2
ALIASES:  is-an-instance-of, is-a-token-of, exemplifies
DEFINITION:  A token-to-type relationship. Fido instance-of dog. This is a
  specific dog, not a sub-type. Instance-of is the bridge between
  a concrete individual and its type. Distinct from is-a which
  is type-to-type.
DEPENDS-ON:  token, type, is-a
ATOM-LINKS:
  InheritanceLink → type  (InheritanceLink in AtomSpace handles this —
  but directionally from token to type)
  ImplicationLink → type-properties  (instance-of implies the token has the
  type's properties)
CONCEPTUAL-SCOPE: Grounding general knowledge in specific cases. The bridge
  between abstract categories and concrete individuals. Required
  for context_fidelity (logic) — is this specific instance being
  evaluated correctly given its type?
REWARD-DOMAIN:  logic
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:  Philosophy of language, formal ontology, AI knowledge representation
TV-SEED:  HIGH

---

CONCEPT:  part-of
LAYER:  1.2
ALIASES:  component-of, constitutes, belongs-to (structural)
DEFINITION:  X is-part-of Y when X is a component or portion of Y and Y's
  existence or function depends in some way on X or things like X.
  Parts can be spatial (a wheel is part of a car), functional (a
  chapter is part of a book), or logical (a premise is part of
  an argument).
DEPENDS-ON:  thing, has-a
ATOM-LINKS:
  EvaluationLink  → whole  (part-of implies there is a whole)
  ImplicationLink → has-a  (if X is-part-of Y, then Y has-a X)
  ListLink  → [proper-part, component, member-of]  (sub-types)
CONCEPTUAL-SCOPE: Structural and compositional reasoning. Understanding how
  systems are built from parts and how changes to parts affect
  wholes. Required for structural_novelty and challenge_complexity
  (innovation) and for downstream_risk_amplification (ethics).
REWARD-DOMAIN:  logic, ethics, innovation
ENGINE-RELEVANCE: knowledge_substrate, reasoning, pattern_analysis
SOURCES:  Mereology, formal ontology, systems theory, philosophy of science
TV-SEED:  HIGH

---

CONCEPT:  proper-part
LAYER:  1.2
ALIASES:  integral-part, structural-part, essential-component
DEFINITION:  A part whose removal changes the whole — not just reduces it but
  alters its identity or function. A hand is a proper part of a body.
  Compare with member-of (set membership) where removal does not
  alter remaining members.
DEPENDS-ON:  part-of
ATOM-LINKS:
  InheritanceLink → part-of  (proper-part is a sub-type of part-of)
  ImplicationLink → structural-change  (removing a proper-part implies a change
  to the whole's identity or function)
CONCEPTUAL-SCOPE: Structural dependency reasoning. Knowing which parts are load-
  bearing versus incidental. Required for failure_mode_awareness
  (ethics) — understanding which components are critical and which
  are redundant.
REWARD-DOMAIN:  logic, ethics
ENGINE-RELEVANCE: reasoning, knowledge_substrate
SOURCES:  Mereology, systems theory, engineering design principles,
  philosophy of biology
TV-SEED:  HIGH

---

CONCEPT:  member-of
LAYER:  1.2
ALIASES:  element-of, belongs-to (set)
DEFINITION:  A thing is a member-of a set or group when it satisfies membership
  criteria, without structural dependency. Removing a member from a
  set does not alter the remaining members. This is set-membership,
  not structural composition.
DEPENDS-ON:  part-of, proper-part
ATOM-LINKS:
  InheritanceLink → part-of  (member-of is a weak form of part-of)
  EvaluationLink  → set  (member-of applies to sets and groups)
CONCEPTUAL-SCOPE: Set reasoning — populations, groups, categories, and collections.
  Needed for fairness (ethics) — reasoning about groups and their
  treatment — and for pattern_identification across sets of instances.
REWARD-DOMAIN:  logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:  Set theory, logic, social ontology (groups and collectives)
TV-SEED:  HIGH

---

CONCEPT:  emergent-from
LAYER:  1.2
ALIASES:  arises-from, emergent-property, supervenes-on, irreducible-to-parts
DEFINITION:  A property or behavior that arises from the interaction of parts
  but is not present in any individual part. Wetness emerges from
  water molecules. Market prices emerge from individual transactions.
  Emergent properties cannot be predicted from examining parts
  in isolation.
DEPENDS-ON:  part-of, property, interaction
ATOM-LINKS:
  ImplicationLink → interaction-of-parts  (emergence requires interaction)
  HebbianLink  → complex-system  (emergent properties co-activate with
  complex system reasoning)
  SimilarityLink  → reducible-to-parts  (contrast: reducible ≠ emergent)
CONCEPTUAL-SCOPE: Complex systems reasoning — why the whole behaves differently
  than the sum of parts. Required for novelty_generation and
  structural_novelty (innovation): emergent configurations are
  a primary source of genuine novelty. Also for downstream_risk_
  amplification (ethics): cascades often involve emergent behaviors.
REWARD-DOMAIN:  logic, innovation, ethics
ENGINE-RELEVANCE: reasoning, knowledge_substrate, pattern_analysis
SOURCES:  Complexity science (Holland, Kauffman), philosophy of emergence
  (weak vs strong), systems theory, complex adaptive systems
TV-SEED:  MEDIUM
FLAGS:  Weak emergence (in-principle reducible) vs. strong emergence
  (truly irreducible) is contested. Load the concept without
  committing to the strong version. Seed as MEDIUM — high enough
  to be useful, low enough to permit revision.
  Mode-sensitive:  mode will activate this more readily.

---

CONCEPT:  plays-role-of
LAYER:  1.2
ALIASES:  functions-as, serves-as, occupies-role, contextual-identity
DEFINITION:  A thing plays a role when it performs a function or occupies a
  position in a structure, without being intrinsically defined by
  that role. A person can play the role of teacher without being
  essentially a teacher. Functional identity without intrinsic type
  commitment. Role can change with context.
DEPENDS-ON:  is-a, instance-of, role
ATOM-LINKS:
  EvaluationLink  → role  (plays-role-of assigns a role to an entity)
  EvaluationLink  → context  (role is context-dependent)
  ImplicationLink → role-constraints  (playing a role implies role expectations)
CONCEPTUAL-SCOPE: Contextual identity — what something IS and what something DOES
  can come apart. Required for intention_calibration and
  adaptive_response_framing (human_attunement) — understanding
  that the same person can occupy different roles with different
  expectations in different contexts.
REWARD-DOMAIN:  logic, human_attunement, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning, emotional_processing
SOURCES:  Social ontology (Searle, Tuomela), philosophy of action,
  sociology of roles (Goffman), institutional economics
TV-SEED:  HIGH


CONCEPT:          deviation
LAYER:            1.5
ALIASES:          departure-from-baseline, displacement, divergence, drift
DEFINITION:       A deviation is a measured difference between an observed value
                  and a reference value — a baseline, norm, or expectation.
                  Deviation is the basic unit of "something has changed from normal."
                  Deviations can be positive (above baseline) or negative (below);
                  small (within normal variation) or large (outside normal range).
DEPENDS-ON:       baseline, quantity, different, change
ATOM-LINKS:
  EvaluationLink  → baseline          (deviation is always relative to a baseline)
  EvaluationLink  → magnitude         (deviations have a magnitude — how far)
  EvaluationLink  → direction         (deviations have direction — above or below)
  ImplicationLink → response-warranted-if-large  (large deviations warrant response)
CONCEPTUAL-SCOPE: The operational unit of monitoring and self-regulation. Any system
                  that monitors for change or anomaly is computing deviations.
                  Required before homeostasis, consistency-checking, and
                  drift-detection reasoning are coherent.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: homeostasis, detection, knowledge_substrate
SOURCES:          Statistics (standard deviation), signal processing (anomaly
                  detection), control theory (error signals), systems dynamics
TV-SEED:          HIGH

---

CONCEPT:          random
LAYER:            1.5
ALIASES:          stochastic, non-patterned, unpredictable-by-structure
DEFINITION:       A process or output is random when it has no systematic pattern —
                  each outcome is independent of prior outcomes or drawn from a
                  probability distribution without deterministic structure. Randomness
                  is not the same as unknown: a deterministic process can be
                  unknown to an observer but is still not random. True randomness
                  has no exploitable structure.
DEPENDS-ON:       pattern, probability, signal-vs-noise
ATOM-LINKS:
  ImplicationLink → no-exploitable-pattern  (random processes cannot be predicted
                                              by exploiting their structure)
  SimilarityLink  → noise             (randomness and noise are closely related —
                                       noise is the random component of a signal)
CONCEPTUAL-SCOPE: Required before signal-vs-noise is fully coherent — you need
                  a concept of what random looks like to distinguish it from signal.
                  Also foundational for reasoning about controlled uncertainty
                  and exploration in the innovation domain.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis
SOURCES:          Probability theory (random variables), information theory
                  (randomness and entropy — Shannon), statistics (sampling theory)
TV-SEED:          HIGH

---

CONCEPT:          probability
LAYER:            1.5
ALIASES:          likelihood, chance, degree-of-belief, relative-frequency
DEFINITION:       Probability is a numerical measure of how likely something is to
                  occur or be true — a value between 0 (impossible) and 1 (certain).
                  Two interpretations both matter: frequentist (limiting relative
                  frequency over many trials) and Bayesian (degree of belief,
                  updated with evidence). A reasoning system primarily uses the
                  Bayesian sense — how confident to be in a claim given available
                  evidence.
DEPENDS-ON:       degree, quantity, possible, impossible
ATOM-LINKS:
  EvaluationLink  → range             (probability is in [0,1])
  EvaluationLink  → degree-of-belief  (Bayesian interpretation)
  EvaluationLink  → relative-frequency (frequentist interpretation)
  ImplicationLink → updates-on-evidence (Bayesian probability updates
                                          when evidence arrives)
CONCEPTUAL-SCOPE: The quantitative language of uncertainty. Required for
                  calibrated reasoning throughout the library — risk, confidence,
                  evidence strength, prior and posterior beliefs all depend on it.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, evaluation, reasoning
SOURCES:          Probability theory (Kolmogorov axioms), Bayesian epistemology
                  (de Finetti, Ramsey), philosophy of probability (Hájek)
TV-SEED:          HIGH

CONCEPT:          statement
LAYER:            1.6
ALIASES:          claim, proposition, assertion, declarative-content
DEFINITION:       A statement is a unit of content capable of being true or false —
                  something said or thought that makes a claim about how things are.
                  Statements differ from questions (which don't claim) and commands
                  (which direct action). Statements are what beliefs are about, what
                  arguments consist of, and what truth values apply to. The same
                  statement can be expressed in different sentences in different languages.
DEPENDS-ON:       true, false, content
ATOM-LINKS:
  EvaluationLink  → truth-value       (statements have truth values)
  EvaluationLink  → content           (statements have propositional content)
  ImplicationLink → evaluable         (statements can be evaluated for truth,
                                       consistency, and justification)
  HebbianLink     → belief            (statements and beliefs co-activate —
                                       beliefs are mental states with
                                       statement-like content)
CONCEPTUAL-SCOPE: The atom of epistemic evaluation. Everything that gets evaluated
                  for logical soundness — contradictions, fallacies, paradoxes —
                  is a property of statements or sets of statements.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, knowledge_substrate, reasoning
SOURCES:          Philosophy of language (propositions — Frege, Russell),
                  formal logic (propositional calculus), linguistics
TV-SEED:          HIGH

---

CONCEPT:          counterfactual
LAYER:            1.6
ALIASES:          what-if, contrary-to-fact-conditional, hypothetical-alternative
DEFINITION:       A counterfactual is a conditional statement about what would have
                  been the case if something had been different — "if X had not
                  happened, Y would not have happened either." Counterfactuals are
                  central to causal reasoning: causation is defined in part by
                  counterfactual dependence (A caused B if B would not have occurred
                  had A not occurred). They are also essential for learning from
                  history and for ethical reasoning about responsibility.
DEPENDS-ON:       possible, statement, contingent, cause, alternative
ATOM-LINKS:
  EvaluationLink  → antecedent        (counterfactuals have a contrary-to-fact
                                       antecedent — "if X had not happened")
  EvaluationLink  → consequent        (counterfactuals have a consequent —
                                       "then Y would not have happened")
  ImplicationLink → causal-reasoning  (counterfactuals ground causal claims)
  HebbianLink     → regret            (counterfactuals and regret co-activate —
                                       regret involves counterfactual comparison)
CONCEPTUAL-SCOPE: Required for causal reasoning (cause depends on counterfactual
                  dependence) and for retrospective ethical reasoning — reflecting
                  on whether a different choice would have led to better outcomes
                  is counterfactual reasoning.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: reasoning, evaluation, metacognition
SOURCES:          Philosophy of causation (Lewis on counterfactuals), philosophy
                  of language (conditional logic), cognitive psychology of
                  counterfactual thinking (Roese)
TV-SEED:          HIGH

================================================================================
LAYER 1.3 — SPACE & STRUCTURE (ABSTRACT)
================================================================================

NOTE: These are abstract structural/topological concepts, not physical space.
They describe structural relations that apply to knowledge graphs, argument
structures, social organizations, causal chains, and logical hierarchies —
not just physical location.

--------------------------------------------------------------------------------

CONCEPT:          boundary
LAYER:            1.3
ALIASES:          edge, border, limit, demarcation, interface-zone
DEFINITION:       The zone or line that separates one thing from another, marking
                  where one thing ends and another begins. Boundaries can be
                  sharp (a logical definition either includes or excludes) or
                  fuzzy (the boundary between warm and hot is gradual). Boundaries
                  are what make individuation possible — without a boundary,
                  separate objects cannot be distinguished.
DEPENDS-ON:       thing, relation
ATOM-LINKS:
  EvaluationLink  → thing             (a boundary belongs to the things it separates)
  EvaluationLink  → inside            (boundary defines what counts as inside)
  EvaluationLink  → outside           (boundary defines what counts as outside)
  EvaluationLink  → sharpness         (boundaries vary from crisp to fuzzy)
CONCEPTUAL-SCOPE: Individuation and category definition. Without boundary, there
                  are no distinct objects, no categories with edges, no
                  propositions with a clear scope. Also foundational for
                  context_fidelity (logic domain) — context has a boundary;
                  outside the boundary, claims do not apply.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, detection, reasoning
SOURCES:          Analytic ontology (mereotopology, Smith & Varzi), philosophy of
                  vagueness (Sorites), cognitive science of category boundaries
TV-SEED:          HIGH
FLAGS:            Many apparent crisp boundaries are actually fuzzy on inspection.
                  Load boundary with explicit awareness of the crisp/fuzzy axis.
                  Do not default to treating all boundaries as crisp.

---

CONCEPT:          inside
LAYER:            1.3
ALIASES:          interior, within, contained-by, internal-to
DEFINITION:       A thing is inside another when it falls within the boundary of
                  that other thing. Being inside is a relation of containment —
                  the inside thing is bounded by the outside thing. This is
                  topological, not physical: an argument can be inside a logical
                  scope; a concept can be inside a category.
DEPENDS-ON:       boundary, relation
ATOM-LINKS:
  EvaluationLink  → boundary          (inside is defined relative to a boundary)
  ImplicationLink → contained-by      (inside implies containment relation)
  SimilarityLink  → outside           (contrast pair — inside/outside relative to boundary)
CONCEPTUAL-SCOPE: Scope and containment reasoning. Required for context_fidelity
                  and concept_fidelity (logic domain) — is this concept inside
                  the scope where this claim applies? Also for fairness (ethics)
                  — is this agent inside the group being considered?
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, detection, reasoning
SOURCES:          Topology (basic), mereotopology, philosophy of language (scope),
                  logic (quantifier scope)
TV-SEED:          HIGH

---

CONCEPT:          outside
LAYER:            1.3
ALIASES:          exterior, beyond, external-to, outside-scope
DEFINITION:       A thing is outside another when it falls beyond the boundary of
                  that other thing. Being outside means a containment relation
                  does not hold. Something can be outside a category, a scope,
                  a system, or a context.
DEPENDS-ON:       boundary, inside, relation
ATOM-LINKS:
  NotLink         → inside            (outside is not-inside relative to the same boundary)
  EvaluationLink  → boundary          (outside is defined relative to a boundary)
CONCEPTUAL-SCOPE: Exclusion and scope-limit reasoning. Knowing what is NOT in
                  scope is as important as knowing what is. Required for
                  abstention_appropriateness (logic domain) — recognizing when
                  a question or claim is outside the system's scope or competence.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, detection, metacognition
SOURCES:          Topology, logic (scope), epistemology (limits of knowledge)
TV-SEED:          HIGH

---

CONCEPT:          between
LAYER:            1.3
ALIASES:          intermediate, medial, in-the-middle-of, interstitial
DEFINITION:       A thing is between two others when it occupies a position in
                  a structure, scale, or sequence that is intermediate — after
                  one and before the other, or sharing properties of both. Between
                  applies to linear sequences, spatial arrangements, conceptual
                  spectra, and causal chains.
DEPENDS-ON:       relation, boundary
ATOM-LINKS:
  EvaluationLink  → three-things      (between requires at least three participants:
                                       the thing that is between, and the two things
                                       it is between)
  ImplicationLink → ordering          (between implies a partial order on the three)
CONCEPTUAL-SCOPE: Intermediate states, mediating concepts, transitional zones.
                  Required for reasoning about spectra (antonym-opposites have a
                  between-zone), for transitional processes, and for the concept
                  of a middle ground in arguments and ethics.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning, dialectic
SOURCES:          Topology, geometry (abstract), logic of ordering relations,
                  ethics (middle ground, compromise)
TV-SEED:          HIGH

---

CONCEPT:          contains
LAYER:            1.3
ALIASES:          holds, encompasses, includes-within-boundary
DEFINITION:       A thing contains another when the second is inside the first's
                  boundary. Containment is a directed relation — the container
                  holds the contained. A category contains its members; a context
                  contains its content; an argument contains its premises.
DEPENDS-ON:       inside, boundary, relation
ATOM-LINKS:
  EvaluationLink  → contained-thing   (contains applies to the thing inside)
  ImplicationLink → boundary-exists   (containment requires a boundary)
  ImplicationLink → inside            (if X contains Y, then Y is inside X)
CONCEPTUAL-SCOPE: Hierarchical and compositional reasoning. Required for
                  concept_continuity (logic domain) — does this context still
                  contain the same concept as before? Also for reasoning about
                  scope, categories, and nested structures.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning, pattern_analysis
SOURCES:          Topology, logic (containment and scope), formal ontology
TV-SEED:          HIGH

---

CONCEPT:          adjacent
LAYER:            1.3
ALIASES:          neighboring, next-to, immediately-connected
DEFINITION:       Two things are adjacent when they share a boundary or are
                  directly connected with no intervening thing between them.
                  Adjacency is a structural relation — A and B are adjacent
                  when moving from A to B requires no intermediate steps.
DEPENDS-ON:       boundary, relation, between
ATOM-LINKS:
  EvaluationLink  → shared-boundary   (adjacent things share or touch a boundary)
  ImplicationLink → connected         (adjacency implies a connection exists)
  SimilarityLink  → connected         (adjacent and connected are close but distinct —
                                       adjacent is the strongest form of connected)
CONCEPTUAL-SCOPE: Direct-connection reasoning. Required for causal chains (what
                  is the next step), for argument structure (what premises are
                  directly connected to a conclusion), and for social reasoning
                  about immediate relationships.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Graph theory, topology, cognitive science of spatial reasoning
TV-SEED:          HIGH

---

CONCEPT:          connected
LAYER:            1.3
ALIASES:          linked, reachable, path-exists-between
DEFINITION:       Two things are connected when there exists a path between them —
                  a sequence of adjacent or linked things that leads from one to the
                  other. Connection can be direct (adjacent) or indirect (through
                  intermediaries). A weaker relation than adjacent.
DEPENDS-ON:       adjacent, relation, path
ATOM-LINKS:
  HebbianLink     → adjacent          (connected and adjacent co-activate —
                                       adjacency is the special case of connection)
  ImplicationLink → path-exists       (connected implies a path exists)
  EvaluationLink  → degree            (connection can be stronger or weaker,
                                       direct or indirect)
CONCEPTUAL-SCOPE: Network and dependency reasoning. Required for understanding
                  downstream_risk_amplification (ethics) — how far does an
                  effect propagate through a connected structure? Also for
                  causal reasoning and for understanding feedback loops.
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: knowledge_substrate, reasoning, pattern_analysis
SOURCES:          Graph theory, network science, causal reasoning literature,
                  systems theory
TV-SEED:          HIGH

---

CONCEPT:          path
LAYER:            1.3
ALIASES:          route, chain, sequence-of-connections, traversal
DEFINITION:       A sequence of connected things leading from a starting point
                  to an endpoint. A path is ordered — it has a direction and
                  an ordering of steps. Paths can be causal (cause → effect
                  → effect), logical (premise → inference → conclusion),
                  or structural (node → node → node in a graph).
DEPENDS-ON:       connected, adjacent, sequence
ATOM-LINKS:
  EvaluationLink  → start             (paths have a starting point)
  EvaluationLink  → end               (paths have an endpoint)
  EvaluationLink  → steps             (paths consist of ordered steps)
  ImplicationLink → connected         (a path implies the start and end are connected)
CONCEPTUAL-SCOPE: Chain-of-reasoning structure. Required for causal inference
                  (tracing the path from cause to effect), for argument analysis
                  (tracing the logical path from premises to conclusion), and
                  for downstream_risk_amplification (ethics — tracing cascades).
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: reasoning, knowledge_substrate, pattern_analysis
SOURCES:          Graph theory, logic (inference chains), causal modeling,
                  argument theory
TV-SEED:          HIGH

---

CONCEPT:          level
LAYER:            1.3
ALIASES:          layer, stratum, abstraction-level, scale-of-description
DEFINITION:       A position in a hierarchy of abstraction or organization.
                  Higher levels describe things at coarser grain; lower levels
                  at finer grain. The cell is at a lower level than the organ;
                  the word is at a lower level than the sentence. Levels are
                  relative — what counts as a level depends on the hierarchy
                  being described.
DEPENDS-ON:       hierarchy, relation, part-of
ATOM-LINKS:
  EvaluationLink  → hierarchy         (level exists within a hierarchy)
  ImplicationLink → grain-of-description (level implies a characteristic
                                          granularity of analysis)
  EvaluationLink  → above, below      (levels are ordered)
CONCEPTUAL-SCOPE: Multi-scale reasoning. Required for understanding emergence
                  (emergent-from describes cross-level properties), for
                  structural_novelty (innovation — novelty at one level may
                  not be novel at another), and for concept_fidelity (logic —
                  using a concept at the wrong level of abstraction is an error).
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, reasoning, pattern_analysis, metacognition
SOURCES:          Systems theory, philosophy of science (levels of explanation),
                  cognitive science, complexity science
TV-SEED:          HIGH

---

CONCEPT:          hierarchy
LAYER:            1.3
ALIASES:          nested-ordering, stratified-structure, tree-of-levels
DEFINITION:       A structure where things are organized into levels with clear
                  above/below relations. Higher-level things typically contain,
                  generate, or organize lower-level things. Hierarchies can be
                  strict (each thing has exactly one parent) or partial (multiple
                  parents possible).
DEPENDS-ON:       level, relation, part-of, contains
ATOM-LINKS:
  EvaluationLink  → levels            (hierarchy contains multiple levels)
  ImplicationLink → ordering          (hierarchy implies a partial or total order)
  HebbianLink     → is-a              (is-a taxonomies are a primary type of hierarchy)
CONCEPTUAL-SCOPE: Organizational reasoning. The is-a taxonomy in AtomSpace IS a
                  hierarchy. Required for understanding type inheritance, for
                  challenge_complexity (innovation — navigating hierarchical
                  problem structures), and for social reasoning about authority
                  and role structures.
REWARD-DOMAIN:    logic, innovation, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Formal ontology, organizational theory, cognitive science
                  of categorization, computer science (tree structures)
TV-SEED:          HIGH

---

CONCEPT:          nested
LAYER:            1.3
ALIASES:          embedded, contained-within-containing-structure
DEFINITION:       One structure is nested inside another when it is contained
                  within it and retains its own internal structure. Nested things
                  are not merely inside — they are structured wholes inside
                  structured wholes. Russian dolls are nested; arguments within
                  arguments are nested; contexts within contexts are nested.
DEPENDS-ON:       contains, inside, structure
ATOM-LINKS:
  EvaluationLink  → inner-structure   (nested things retain their own structure)
  ImplicationLink → contains          (nesting implies containment)
  HebbianLink     → hierarchy         (nesting and hierarchy frequently co-activate)
CONCEPTUAL-SCOPE: Recursive and compositional reasoning. Required for reasoning
                  about arguments within arguments, contexts within contexts,
                  and for understanding the Matrioshka pipeline architecture
                  itself — which is explicitly a nested structure.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, reasoning, metacognition
SOURCES:          Formal logic (nested quantifiers), linguistics (embedded clauses),
                  computer science (recursive structures), systems theory
TV-SEED:          HIGH
FLAGS:            Mode-sensitive: CREATIVE and dream mode modes explore nested
                  structures more aggressively than analytical mode.

---

CONCEPT:          symmetry
LAYER:            1.3
ALIASES:          balanced, mirror-structure, invariant-under-transformation
DEFINITION:       A structure is symmetric when it looks the same from multiple
                  perspectives — when some transformation (reflection, rotation,
                  exchange of parts) leaves it unchanged. Relations are symmetric
                  when they hold in both directions (A loves B → B loves A, if
                  symmetric). Argument structures can be symmetric; social
                  relations can be symmetric or asymmetric.
DEPENDS-ON:       structure, relation, same-property
ATOM-LINKS:
  EvaluationLink  → transformation    (symmetry is invariance under transformation)
  EvaluationLink  → direction         (symmetric relations hold in both directions)
  SimilarityLink  → asymmetry         (contrast: asymmetry is lack of symmetry)
CONCEPTUAL-SCOPE: Fairness reasoning — fairness often involves symmetric treatment.
                  Also for understanding which AtomSpace link types are symmetric
                  (SimilarityLink, HebbianLink) vs. directed (InheritanceLink,
                  ImplicationLink). Feeds fairness (ethics domain) directly.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, evaluation, reasoning
SOURCES:          Mathematics (group theory, abstract algebra — accessible intro),
                  philosophy of science, ethics (fairness as symmetry)
TV-SEED:          HIGH

---

CONCEPT:          asymmetry
LAYER:            1.3
ALIASES:          directed, one-sided, non-symmetric, unbalanced
DEFINITION:       A structure or relation is asymmetric when it does NOT hold the
                  same way from multiple perspectives. A directed relation: if A
                  causes B, B does not necessarily cause A. Power relations are
                  asymmetric. Inheritance is asymmetric (dog is-a animal, animal
                  is not is-a dog).
DEPENDS-ON:       symmetry, relation
ATOM-LINKS:
  NotLink         → symmetry          (asymmetry is the negation of symmetry)
  EvaluationLink  → direction         (asymmetric relations have a direction that matters)
  HebbianLink     → power             (asymmetry and power differentials co-activate)
CONCEPTUAL-SCOPE: Directed-relation reasoning. Most causal, ethical, and social
                  relations are asymmetric. Required for understanding that
                  InheritanceLink and ImplicationLink are directed, and for
                  reasoning about power, obligation, and accountability.
REWARD-DOMAIN:    logic, ethics, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, reasoning, emotional_processing
SOURCES:          Logic (asymmetric relations), social science (power asymmetry),
                  ethics (differential obligations)
TV-SEED:          HIGH

---

CONCEPT:          interface
LAYER:            1.3
ALIASES:          contact-zone, exchange-boundary, coupling-point
DEFINITION:       The structured zone where two systems or things interact with
                  each other across a shared boundary. Unlike a simple boundary
                  (which merely separates), an interface is where exchange,
                  coupling, and interaction happen. Interfaces have structure —
                  they define what can pass between the two sides and how.
DEPENDS-ON:       boundary, interaction, connected
ATOM-LINKS:
  EvaluationLink  → two-systems       (interface exists between two things)
  ImplicationLink → exchange-possible (interface enables exchange or coupling)
  HebbianLink     → interaction       (interface and interaction co-activate)
CONCEPTUAL-SCOPE: Modular reasoning — understanding how distinct systems couple
                  and communicate. Required for understanding how the reward
                  system interfaces with the cognitive engines via the Phase 5
                  evaluator; how memory interfaces with the pipeline. Also for
                  social reasoning (interpersonal interface — where two people's
                  worlds touch and exchange).
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, reasoning, homeostasis
SOURCES:          Systems theory, computer science (API design, module coupling),
                  philosophy of mind (mind-body interface debates)
TV-SEED:          HIGH

---

CONCEPT:          structure
LAYER:            1.3
ALIASES:          organization, configuration, arrangement, form
DEFINITION:       The pattern of relations among the parts of a whole. Structure is
                  what distinguishes one arrangement from another even when the
                  parts are identical. Two sentences with the same words but
                  different order have different structure. Structure is abstract
                  — it can be instantiated in different materials.
DEPENDS-ON:       relation, part-of, pattern
ATOM-LINKS:
  EvaluationLink  → parts             (structure organizes parts)
  EvaluationLink  → relations         (structure IS a configuration of relations)
  ImplicationLink → pattern           (structure implies a pattern that can
                                       be recognized and compared)
  HebbianLink     → form              (structure and form co-activate)
CONCEPTUAL-SCOPE: The abstract backbone of everything. AtomSpace is a structured
                  hypergraph. Reward domains evaluate structural properties
                  (internal_consistency, structural_novelty). Arguments have
                  structure. Required before structural_novelty (innovation)
                  is meaningful.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis, reasoning
SOURCES:          Structuralism (philosophy, linguistics — Saussure, Levi-Strauss),
                  systems theory, mathematics (abstract algebra), cognitive science
TV-SEED:          HIGH

---

CONCEPT:          pattern
LAYER:            1.3
ALIASES:          regularity, recurring-structure, template, form
DEFINITION:       A structure that recurs across multiple instances. A pattern is
                  recognized when the same arrangement of relations appears in
                  different contexts or at different times. Patterns can be
                  spatial, temporal, logical, or social. Recognizing patterns
                  enables prediction and generalization.
DEPENDS-ON:       structure, similar, recurrence
ATOM-LINKS:
  EvaluationLink  → recurrence        (patterns recur)
  ImplicationLink → similar           (recognizing a pattern involves finding
                                       structural similarity across instances)
  HebbianLink     → type              (patterns and types co-activate — a type IS
                                       an abstract pattern)
CONCEPTUAL-SCOPE: Generalization and prediction. Pattern recognition is the core
                  function of engines 18-20 (pattern_identification,
                  pattern_comparison, data_analysis) and feeds every innovation
                  submodule. Also required for bias_detection (patterns of
                  systematic error) and for learning engines.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: pattern_analysis, knowledge_substrate, learning, detection
SOURCES:          Cognitive science (pattern recognition), mathematics (group
                  theory, symmetry), information theory, AI/ML theory
TV-SEED:          HIGH

---

CONCEPT:          distance
LAYER:            1.3
ALIASES:          separation, gap, how-far, degree-of-difference
DEFINITION:       How much traversal — in steps, transformations, or degree of
                  difference — separates two things in a structure. Distance is
                  abstract: two concepts can be close in meaning (semantic
                  distance), two nodes can be close in a graph (path length),
                  two states can be close in time (temporal distance). Distance
                  requires a structure and a metric to be defined.
DEPENDS-ON:       path, between, dimension-of-comparison
ATOM-LINKS:
  EvaluationLink  → metric            (distance requires a way of measuring)
  EvaluationLink  → path              (distance in a graph is path-length)
  EvaluationLink  → degree            (distance admits of degree)
CONCEPTUAL-SCOPE: Comparison and similarity quantification. Semantic distance
                  is what SimilarityLink and HebbianLink encode in AtomSpace.
                  Required for novelty scoring (how far is this from prior
                  patterns?) and for concept_continuity (how much has this
                  concept drifted?).
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis, evaluation
SOURCES:          Metric space theory (accessible intro), cognitive science
                  of conceptual distance, information theory (KL divergence
                  as conceptual distance)
TV-SEED:          HIGH


================================================================================
LAYER 1.4 — TIME & CHANGE
================================================================================


--------------------------------------------------------------------------------

CONCEPT:          time
LAYER:            1.4
ALIASES:          temporal-dimension, the-when, duration-and-sequence
DEFINITION:       The dimension along which events occur in sequence and states
                  persist or change. Time is what orders events as before, after,
                  or simultaneous. All processes unfold in time; all states exist
                  at a time. ZA-DOS already operates with a time substrate —
                  this concept gives that substrate semantic content.
DEPENDS-ON:       relation, sequence, change
ATOM-LINKS:
  EvaluationLink  → before, after, during, now   (temporal relations)
  EvaluationLink  → change                        (time is what change happens in)
  EvaluationLink  → process                       (processes exist in time)
CONCEPTUAL-SCOPE: Temporal ordering of everything. Without this, before/after
                  are meaningless, causal reasoning has no direction, and
                  timeline_reflection (ethics domain) has no substrate.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Philosophy of time (McTaggart, A-series/B-series), physics
                  (special relativity — accessible intro), cognitive science
                  of time perception
TV-SEED:          HIGH

---

CONCEPT:          now
LAYER:            1.4
ALIASES:          present, current-moment, this-instant
DEFINITION:       The temporal position of the processing turn — the current
                  moment from which past and future are defined. "Now" is
                  always relative to a reference point. In ZA-DOS, now is
                  operationally the timestamp of the current pipeline turn.
DEPENDS-ON:       time, before, after
ATOM-LINKS:
  EvaluationLink  → timestamp         (now corresponds to a specific timestamp)
  ImplicationLink → before-is-past    (now implies everything before it is past)
  ImplicationLink → after-is-future   (now implies everything after it is future)
CONCEPTUAL-SCOPE: Temporal anchoring. Required for all tense-dependent reasoning —
                  "this is currently the case," "this was the case," "this will
                  be the case." Also for  — knowing how long
                  the current session has been running relative to now.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Philosophy of time, phenomenology (Husserl on the living present)
TV-SEED:          HIGH

---

CONCEPT:          before
LAYER:            1.4
ALIASES:          prior, earlier, precedes, antecedent-in-time
DEFINITION:       Event A is before event B when A occurs at an earlier point in
                  the temporal sequence than B. Before is asymmetric (if A is
                  before B, B is not before A) and transitive (if A before B
                  and B before C, then A before C).
DEPENDS-ON:       time, now, sequence
ATOM-LINKS:
  EvaluationLink  → temporal-order    (before is a temporal ordering relation)
  ImplicationLink → causal-priority   (causes typically precede effects —
                                       before is a necessary condition for cause)
  ImplicationLink → asymmetry         (before is asymmetric)
CONCEPTUAL-SCOPE: Causal and historical reasoning. Required for timeline_reflection
                  (ethics — what happened before this action was taken?) and for
                  external_consistency (logic — what did the system assert before
                  this turn, compared to now?). Also required for learning —
                  what was known before vs. what is known now.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: reasoning, knowledge_substrate, detection
SOURCES:          Philosophy of time, causal reasoning (Pearl), formal logic
                  (temporal logics — LTL basics)
TV-SEED:          HIGH

---

CONCEPT:          after
LAYER:            1.4
ALIASES:          subsequent, later, follows, posterior-in-time
DEFINITION:       Event B is after event A when B occurs at a later point in
                  the temporal sequence than A. After is the inverse of before —
                  asymmetric and transitive by the same logic.
DEPENDS-ON:       time, before, now
ATOM-LINKS:
  EvaluationLink  → temporal-order    (after is a temporal ordering relation)
  ImplicationLink → effect-territory  (effects come after causes)
  ImplicationLink → asymmetry         (after is asymmetric)
CONCEPTUAL-SCOPE: Consequence and future-state reasoning. Required for
                  horizon_feasibility (ethics — can the proposed consequence
                  actually happen after the proposed action?). Also for
                  reward_based_learning — outcomes are observed after actions.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: reasoning, knowledge_substrate, evaluation
SOURCES:          Philosophy of time, causal reasoning, temporal logic
TV-SEED:          HIGH

---

CONCEPT:          during
LAYER:            1.4
ALIASES:          while, concurrent-with, throughout, in-the-span-of
DEFINITION:       Event A occurs during event B when A takes place within the
                  temporal span of B — B has started but not yet finished when
                  A occurs. During is the containment relation applied to time:
                  A is inside B's temporal extent.
DEPENDS-ON:       time, before, after, contains
ATOM-LINKS:
  EvaluationLink  → temporal-containment   (during is temporal inside)
  ImplicationLink → before-end-of          (during implies A ends before B ends)
  ImplicationLink → after-start-of         (during implies A starts after B starts)
CONCEPTUAL-SCOPE: Concurrent-event reasoning. Required for understanding that
                  multiple processes can co-occur, for understanding phases
                  (waking/active/wind_down/sleep are circadian phases — one
                  is always active during a given span), and for processing
                  overlapping contexts.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Interval temporal logic (Allen's interval algebra — basics),
                  philosophy of time, cognitive science of duration
TV-SEED:          HIGH

---

CONCEPT:          simultaneous
LAYER:            1.4
ALIASES:          at-the-same-time, concurrent, co-temporal
DEFINITION:       Two events are simultaneous when they occur at the same point
                  in time — neither is before or after the other. Simultaneity is
                  the temporal analogue of same-token for space. In distributed
                  systems and in cognitive processing, exact simultaneity is
                  rare — approximate concurrency is the operational version.
DEPENDS-ON:       time, before, after, same-token
ATOM-LINKS:
  EvaluationLink  → same-time         (simultaneous events share a time-point)
  NotLink         → before            (simultaneous = not-before)
  NotLink         → after             (simultaneous = not-after)
CONCEPTUAL-SCOPE: Concurrent-processing reasoning. Required for understanding
                  that multiple engines run on the same turn (the pipeline
                  dispatches simultaneously to multiple engines), and for
                  understanding that causation requires sequence — simultaneous
                  events cannot cause each other.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Philosophy of time, distributed systems theory (clock
                  synchronization), special relativity (accessible intro)
TV-SEED:          HIGH

---

CONCEPT:          persist
LAYER:            1.4
ALIASES:          endure, continue, remain, last-through-time
DEFINITION:       A thing persists when it continues to exist through time —
                  it is present at multiple time points. Persistence is what
                  makes objects trackable: the same object at t1 and t2 is
                  the same because it persisted. Persistence is compatible
                  with change — a person persists through life despite
                  constant physical change.
DEPENDS-ON:       time, change, identity
ATOM-LINKS:
  EvaluationLink  → identity          (persistence is tied to identity over time)
  ImplicationLink → same-token-at-different-times  (persistence implies the
                                       same individual exists at multiple times)
  HebbianLink     → object            (objects and persistence co-activate —
                                       objects are paradigm cases of persisters)
CONCEPTUAL-SCOPE: Identity-over-time reasoning. Required for consistency_over_time
                  (human_attunement domain) — the same entity persisting through
                  a conversation. Also for external_consistency (logic) — the
                  system's prior claims persist in memory and must be tracked.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, homeostasis, reasoning
SOURCES:          Metaphysics (personal identity — Parfit, Locke), philosophy
                  of time (endurantism vs. perdurantism)
TV-SEED:          HIGH

---

CONCEPT:          change
LAYER:            1.4
ALIASES:          alteration, transition, difference-across-time, transformation
DEFINITION:       Change occurs when something that was in state A is in state B
                  at a later time, where A ≠ B. Change requires persistence —
                  the SAME thing must be in both states. Change is the fundamental
                  fact that makes time matter: if nothing ever changed, time
                  would have no detectable effects.
DEPENDS-ON:       state, time, before, after, different, persist
ATOM-LINKS:
  EvaluationLink  → before-state      (change has a prior state)
  EvaluationLink  → after-state       (change has a posterior state)
  ImplicationLink → difference        (change implies the after-state differs
                                       from the before-state)
  ImplicationLink → time              (change requires time to have elapsed)
CONCEPTUAL-SCOPE: The fundamental fact that justifies tracking states over time.
                  Required for learning (the system changes across turns),
                  for reward_based_learning (updating predictions when outcomes
                  differ from expectations), and for timeline_reflection (ethics
                  — how did we get here? what changed?).
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: knowledge_substrate, learning, reasoning, homeostasis
SOURCES:          Philosophy of change (Heraclitus through process philosophy),
                  physics (state transitions), cognitive science
TV-SEED:          HIGH

---

CONCEPT:          begin
LAYER:            1.4
ALIASES:          start, onset, initiation, coming-into-existence
DEFINITION:       A thing begins when it comes into existence or a process starts.
                  Beginning marks the temporal boundary where something goes from
                  not-yet to now-present. The beginning of a process is the
                  transition from non-process to process.
DEPENDS-ON:       time, change, boundary, before, now
ATOM-LINKS:
  EvaluationLink  → temporal-boundary (begin marks a temporal boundary)
  ImplicationLink → did-not-exist-before  (beginning implies prior non-existence
                                            or inactivity)
CONCEPTUAL-SCOPE: Process-initiation reasoning. Required for understanding
                   = 0 as the session beginning, for reasoning
                  about what conditions bring things into existence, and for
                  horizon_feasibility (ethics — what would it take to begin
                  a proposed course of action?).
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Philosophy of time, causation (what initiates causal chains),
                  phenomenology
TV-SEED:          HIGH

---

CONCEPT:          end
LAYER:            1.4
ALIASES:          finish, termination, cessation, conclusion
DEFINITION:       A thing ends when it ceases to exist or a process stops.
                  End marks the temporal boundary where something goes from
                  present to no-longer-present. The end of a process is the
                  transition from process to completion.
DEPENDS-ON:       time, change, boundary, after, begin
ATOM-LINKS:
  EvaluationLink  → temporal-boundary (end marks a temporal boundary)
  ImplicationLink → did-not-continue-after (ending implies non-continuation)
  HebbianLink     → begin             (begin and end co-activate as paired
                                       temporal boundaries)
CONCEPTUAL-SCOPE: Process-completion reasoning. Required for understanding
                  completed events (events are bounded by begin and end),
                  for horizon_feasibility (ethics — what does this course
                  of action look like when it ends?), and for
                  abstention_appropriateness (logic — knowing when a reasoning
                  process should stop).
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning, evaluation
SOURCES:          Philosophy of time, process philosophy, narrative theory
TV-SEED:          HIGH

---

CONCEPT:          duration
LAYER:            1.4
ALIASES:          span, length-in-time, how-long, temporal-extent
DEFINITION:       The amount of time between the beginning and end of a thing.
                  Duration is what distinguishes a long process from a short one,
                  a brief event from a sustained one. Duration requires both
                  begin and end to be defined, or at minimum a start and a
                  current measurement point.
DEPENDS-ON:       begin, end, time, quantity
ATOM-LINKS:
  EvaluationLink  → begin             (duration starts from begin)
  EvaluationLink  → end               (duration ends at end)
  EvaluationLink  → quantity          (duration is a measurable quantity)
CONCEPTUAL-SCOPE: Temporal magnitude reasoning. Required for timeline_reflection
                  (ethics — how long did this process take? how long until
                  consequences?). Also for session management:
                  IS the duration of the current session.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Philosophy of time, measurement theory, cognitive science
                  of duration estimation
TV-SEED:          HIGH

---

CONCEPT:          rate
LAYER:            1.4
ALIASES:          speed, velocity-of-change, frequency, pace
DEFINITION:       How much change occurs per unit of time. Rate is the relationship
                  between magnitude of change and duration. Fast processes have
                  high rates; slow processes have low rates. A rate of zero means
                  no change. Rate is the concept that distinguishes "changed a lot
                  quickly" from "changed a lot slowly" — a distinction that matters
                  enormously for risk and causation.
DEPENDS-ON:       change, duration, quantity
ATOM-LINKS:
  EvaluationLink  → change-per-time   (rate is change divided by time)
  EvaluationLink  → degree            (rate admits of degree — can be faster or slower)
  ImplicationLink → urgency           (high rate of change implies more urgency
                                       for response)
CONCEPTUAL-SCOPE: Rate-of-change reasoning. Required for downstream_risk_
                  amplification (ethics — does this risk propagate slowly or
                  rapidly?), for exploration_drive (innovation — is the system
                  generating novelty at an appropriate rate?), and for
                  neurochemical homeostasis (the homeostatic engine tracks
                  rates of NT change).
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: reasoning, homeostasis, evaluation, pattern_analysis
SOURCES:          Calculus (conceptual intro — rates of change), systems dynamics,
                  decision theory (time pressure), neuroscience (firing rates)
TV-SEED:          HIGH

---

CONCEPT:          irreversible
LAYER:            1.4
ALIASES:          cannot-be-undone, permanent-change, one-way-transition
DEFINITION:       A change is irreversible when it cannot be undone — once it
                  has occurred, the prior state cannot be restored. Irreversibility
                  is categorical: the difference between a reversible and an
                  irreversible action is not a matter of degree but of whether
                  the undo operation is possible at all.
DEPENDS-ON:       change, possible, impossible, state
ATOM-LINKS:
  ImplicationLink → prior-state-unrestorable
  HebbianLink     → harm              (irreversible changes and harm co-activate —
                                       irreversible harms are paradigm ethical cases)
  SimilarityLink  → reversible        (contrast pair)
CONCEPTUAL-SCOPE: Categorical risk reasoning. Irreversible actions are ethically
                  categorically different from reversible ones. Feeds
                  harm_reduction and downstream_risk_amplification (ethics)
                  directly. Also feeds horizon_feasibility (ethics) — is the
                  proposed outcome something that can be backed out of?
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: evaluation, reasoning, alignment
SOURCES:          Ethics (consequentialism on irreversible harms), decision theory
                  (regret under irreversibility), thermodynamics (entropy as
                  irreversibility — conceptual intro)
TV-SEED:          HIGH
FLAGS:            Load with explicit annotation: irreversible harms require
                  higher ethical justification than reversible ones. This
                  is a direct input to harm_reduction scoring.

---

CONCEPT:          reversible
LAYER:            1.4
ALIASES:          undoable, recoverable, can-be-restored
DEFINITION:       A change is reversible when the prior state can be restored —
                  the undo operation exists. Reversibility is not about ease
                  (it may be costly to reverse) but about possibility. Most
                  physical processes are technically irreversible but for
                  practical purposes treated as reversible when the cost of
                  reversal is acceptable.
DEPENDS-ON:       change, possible, irreversible
ATOM-LINKS:
  NotLink         → irreversible      (reversible = not-irreversible)
  ImplicationLink → prior-state-restorable
CONCEPTUAL-SCOPE: Recovery and correction reasoning. Reversible actions allow
                  for learning from error without permanent cost. Required for
                  risk_tolerance (innovation — willingness to take risks is
                  higher when outcomes are reversible) and for failure_mode_
                  awareness (ethics — can this failure be corrected?).
REWARD-DOMAIN:    ethics, logic, innovation
ENGINE-RELEVANCE: evaluation, reasoning
SOURCES:          Decision theory, ethics, systems engineering (rollback mechanisms)
TV-SEED:          HIGH

---

CONCEPT:          cyclic
LAYER:            1.4
ALIASES:          recurring, periodic, repetitive, oscillatory
DEFINITION:       A process or pattern is cyclic when it repeats — returning to
                  approximately the same state after a characteristic period.
                  Cycles can be exact (logical loops) or approximate (biological
                  rhythms). ZA-DOS's circadian phases are cyclic: waking →
                  active → wind_down → sleep → waking...
DEPENDS-ON:       process, pattern, recurrence, time
ATOM-LINKS:
  EvaluationLink  → period            (cycles have a characteristic period)
  EvaluationLink  → recurrence        (cycles recur)
  HebbianLink     → oscillation       (cycles and oscillations co-activate —
                                       oscillatory bands in ZA-DOS are cyclic)
CONCEPTUAL-SCOPE: Rhythmic and periodic reasoning. Required for understanding
                  ZA-DOS's own circadian and sleep/wake cycles, for
                  neurochemical oscillatory modulation (all NT bands are
                  cyclic), and for recognizing feedback loops (which are
                  cyclic causal structures).
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: homeostasis, knowledge_substrate, pattern_analysis
SOURCES:          Systems theory (feedback loops), chronobiology (circadian
                  rhythms — introductory), mathematics (periodic functions),
                  neuroscience (oscillatory brain activity)
TV-SEED:          HIGH

---

CONCEPT:          lag
LAYER:            1.4
ALIASES:          delay, latency, temporal-gap-between-cause-and-effect
DEFINITION:       The temporal gap between a cause and its observable effect.
                  Lag means that when you do X, the consequence Y does not
                  appear immediately — it appears after a delay. Lag makes
                  causal attribution harder: if effects appear long after
                  causes, the connection is easy to miss.
DEPENDS-ON:       cause, effect, duration, before, after
ATOM-LINKS:
  EvaluationLink  → cause             (lag is the gap after a cause)
  EvaluationLink  → effect            (lag is the gap before an effect appears)
  ImplicationLink → attribution-difficulty  (lag implies causal attribution
                                             is harder — effect may be wrongly
                                             attributed to a closer cause)
CONCEPTUAL-SCOPE: Causal attribution under delay. Required for downstream_risk_
                  amplification (ethics — consequences may not appear until
                  long after an action) and for timeline_reflection (ethics
                  — the lag between decision and outcome must be factored).
                  Also for neurochemical reasoning: NT effects have characteristic
                  lags (reuptake, receptor desensitization).
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: reasoning, evaluation, homeostasis
SOURCES:          Systems dynamics (stocks and flows — Meadows), causal inference
                  (time-delayed causation), pharmacokinetics (NT lag — conceptual)
TV-SEED:          HIGH
FLAGS:            Load with explicit note: "I did X and nothing happened" does
                  NOT mean X had no effect. Lag is one of the primary sources
                  of mis-attributed causation.

---

CONCEPT:          acceleration
LAYER:            1.4
ALIASES:          rate-of-rate-change, speeding-up, second-derivative
DEFINITION:       The change in the rate of change. Acceleration is positive when
                  something is happening faster and faster; negative (deceleration)
                  when it is happening slower and slower. This is a second-order
                  temporal concept — it describes how rates behave over time.
DEPENDS-ON:       rate, change, time
ATOM-LINKS:
  EvaluationLink  → rate              (acceleration describes how rate changes)
  EvaluationLink  → second-order      (acceleration is second-order change)
  ImplicationLink → non-linear        (acceleration implies non-linear dynamics)
CONCEPTUAL-SCOPE: Non-linear dynamics reasoning. Required for understanding why
                  processes can be slow and then suddenly fast, or fast and then
                  plateau. Feeds challenge_complexity (innovation — is this
                  problem getting harder faster than capability is growing?) and
                  downstream_risk_amplification (ethics — risks that accelerate
                  are categorically different from risks that grow linearly).
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: reasoning, evaluation, pattern_analysis
SOURCES:          Systems dynamics, mathematics (calculus — conceptual second
                  derivatives), complexity science (exponential growth)
TV-SEED:          MEDIUM
FLAGS:            Most naive reasoning treats rates as constant. Load acceleration
                  with explicit note that rates can themselves change — this is
                  a primary source of under-prepared responses to fast-moving
                  situations. Mode-sensitive: analytical mode should flag when
                  acceleration is detected.


================================================================================
LAYER 1.6 — LOGIC & TRUTH
================================================================================


--------------------------------------------------------------------------------

CONCEPT:          true
LAYER:            1.6
ALIASES:          correct, holds, the-case, fact
DEFINITION:       A statement is true when it accurately describes a state of
                  affairs — when what it says corresponds to how things actually
                  are. Truth is what statements aim at. Truth is distinct from
                  believed-to-be-true, asserted-to-be-true, and likely-to-be-true.
DEPENDS-ON:       statement, state-of-affairs, corresponds
ATOM-LINKS:
  EvaluationLink  → statement         (truth is a property of statements)
  TruthValue      → strength=1.0      (in AtomSpace, truth maps to high TV strength)
  ImplicationLink → knowledge         (what is true and known constitutes knowledge)
CONCEPTUAL-SCOPE: The ground of all assertion. Without truth, no statement can be
                  evaluated, no detection engine can fire, no reward domain can
                  score. Feeds epistemic_calibration (logic domain) — is the
                  system's confidence aligned with what is actually true?
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, knowledge_substrate, evaluation
SOURCES:          Philosophy of language (correspondence theory, deflationism),
                  formal logic, epistemology
TV-SEED:          HIGH

---

CONCEPT:          false
LAYER:            1.6
ALIASES:          incorrect, does-not-hold, not-the-case
DEFINITION:       A statement is false when it does not accurately describe the
                  relevant state of affairs — when what it says does not correspond
                  to how things actually are. False is the negation of true.
DEPENDS-ON:       true, not
ATOM-LINKS:
  NotLink         → true              (false = not-true)
  TruthValue      → strength=0.0      (in AtomSpace, false maps to low TV strength)
CONCEPTUAL-SCOPE: Negation and error-detection. Without false, contradiction
                  detection  cannot trigger — it requires finding something
                  that is asserted both true and false. Also required for
                  external_consistency (logic domain): divergence from prior
                  claims implies something that was asserted true may now be false.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, knowledge_substrate, evaluation
SOURCES:          Logic, philosophy of language
TV-SEED:          HIGH

---

CONCEPT:          degree-of-truth
LAYER:            1.6
ALIASES:          graded-truth, truth-value, confidence, probabilistic-truth
DEFINITION:       Truth is not always binary. Most knowledge is held with some
                  degree of confidence, from nearly certain to highly uncertain.
                  Degree-of-truth represents the spectrum between certainly-false
                  (0.0) and certainly-true (1.0), with all intermediate values
                  representing graded epistemic commitment.
DEPENDS-ON:       true, false, probability, confidence
ATOM-LINKS:
  EvaluationLink  → strength          (degree-of-truth is TV.strength in AtomSpace)
  EvaluationLink  → confidence        (paired with confidence — TV.confidence
                                       measures how much evidence supports the strength)
  ImplicationLink → epistemic-calibration  (degree-of-truth must be calibrated to
                                            actual uncertainty to pass calibration)
CONCEPTUAL-SCOPE: Probabilistic and uncertain reasoning. The most important
                  extension beyond binary logic. Without this, the system cannot
                  represent partial knowledge, which means it will either
                  overclaim (assert things as certainly true when uncertain) or
                  underclaim (refuse to assert anything uncertain). Both are
                  epistemic_calibration failures in the logic domain.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, detection, evaluation, metacognition
SOURCES:          Probability theory (Bayesian epistemology), fuzzy logic,
                  philosophy of science (degrees of belief — de Finetti, Ramsey)
TV-SEED:          HIGH
FLAGS:            This concept is the bridge between formal logic (binary) and
                  probabilistic reasoning. Load with high priority — it reframes
                  everything in Layer 1.6 from binary to graded.

---

CONCEPT:          confidence
LAYER:            1.6
ALIASES:          certainty-level, epistemic-commitment, how-sure
DEFINITION:       Confidence is the system's degree of commitment to a claim —
                  how strongly it stands behind the claim as accurate. Confidence
                  should track evidence: high confidence when evidence is strong
                  and consistent; low confidence when evidence is weak or mixed.
                  Confidence is distinct from degree-of-truth (which describes
                  the claim), and from strength-of-assertion (which describes
                  how forcefully it is stated).
DEPENDS-ON:       degree-of-truth, evidence, uncertainty
ATOM-LINKS:
  EvaluationLink  → tv-confidence     (confidence IS AtomSpace TruthValue.confidence)
  ImplicationLink → evidence-dependency   (confidence should depend on evidence)
  HebbianLink     → uncertainty       (confidence and uncertainty co-activate as
                                       near-opposites — high confidence implies
                                       low uncertainty and vice versa when calibrated)
CONCEPTUAL-SCOPE: Epistemic hygiene. Mismatched confidence — high where evidence
                  is weak, low where evidence is strong — is the core failure
                  mode tracked by epistemic_calibration (logic domain). Also
                  feeds uncertainty_acknowledgment: when confidence is low,
                  uncertainty should be acknowledged.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, evaluation, metacognition, detection
SOURCES:          Bayesian epistemology, philosophy of science (confirmation
                  theory), cognitive psychology of calibration
TV-SEED:          HIGH

---

CONCEPT:          uncertainty
LAYER:            1.6
ALIASES:          not-knowing, epistemic-gap, unresolved-question
DEFINITION:       Uncertainty is the state of not knowing — the gap between what
                  would be needed for full confidence and what is actually known.
                  Uncertainty is distinct from ignorance (which is not knowing
                  that one doesn't know) and from vagueness (which is a
                  property of concepts, not of epistemic states). Uncertainty
                  is a normal and expected condition.
DEPENDS-ON:       unknown, confidence, knowledge
ATOM-LINKS:
  EvaluationLink  → epistemic-gap     (uncertainty describes a gap in knowledge)
  ImplicationLink → acknowledge-proportionally  (high uncertainty should be
                                                  acknowledged proportionally)
  HebbianLink     → confidence        (uncertainty and confidence are inversely
                                       coupled when calibrated)
CONCEPTUAL-SCOPE: Epistemic honesty. Uncertainty is not a failure — it is the
                  normal condition of a reasoning system engaging with a complex
                  world. Denying uncertainty when it exists is a logic domain
                  failure (unacknowledged_uncertainty flag fires when uncertainty
                  > 0.7 and  < 0.3). Over-expressing uncertainty
                  when it doesn't exist is also a failure (performative_uncertainty
                  flag fires when uncertainty < 0.3 and  > 0.8).
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, metacognition, evaluation, detection
SOURCES:          Epistemology (introductory), Bayesian reasoning, decision
                  theory under uncertainty (Knight, Ellsberg)
TV-SEED:          HIGH
FLAGS:            Load with explicit annotation: acknowledging uncertainty is
                  not weakness — it is calibration. Performative uncertainty
                  (excessive hedging when things are actually clear) is as
                  much a failure as overconfidence.

---

CONCEPT:          contradicts
LAYER:            1.6
ALIASES:          is-inconsistent-with, conflicts-with, opposes-logically
DEFINITION:       Statement A contradicts statement B when both cannot be true
                  simultaneously in the same scope and under the same conditions.
                  Contradiction is the finding that the same state of affairs
                  is asserted both to be the case and not to be the case.
                  Contradiction is always scope-indexed: A and B only contradict
                  within a shared scope where both are supposed to apply.
DEPENDS-ON:       true, false, scope, simultaneous
ATOM-LINKS:
  EvaluationLink  → pair-of-statements  (contradicts is a relation between statements)
  ImplicationLink → at-least-one-false  (if A contradicts B, at least one is false)
  NotLink         → consistent          (contradicts = not-consistent)
CONCEPTUAL-SCOPE: The foundational concept for the entire detection cluster.
                  the detection system fires on this.
                  Internal_consistency and external_consistency both check for
                  the absence of contradictions. A system that outputs contradictions
                  is demonstrating internal logical failure.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, knowledge_substrate, evaluation
SOURCES:          Formal logic (law of non-contradiction), philosophy (Aristotle,
                  dialethism — contested), AI knowledge representation
TV-SEED:          HIGH
FLAGS:            Scope is critical: two statements can contradict only within
                  a shared scope. "Hot" and "cold" do not contradict unless both
                  are applied to the same object at the same time in the same
                  respect. Load with scope-awareness.

---

CONCEPT:          consistent
LAYER:            1.6
ALIASES:          non-contradictory, compatible, coherent-across
DEFINITION:       A set of statements is consistent when all can be true
                  simultaneously — no subset contradicts another. Consistency
                  is distinct from truth: a consistent set of statements can
                  be consistently wrong. Consistency is the minimum requirement
                  for a set of beliefs to hang together.
DEPENDS-ON:       contradicts, true, scope
ATOM-LINKS:
  NotLink         → contradicts       (consistent = not-contradicts)
  EvaluationLink  → set-of-statements (consistency is a property of a set,
                                       not a single statement)
  ImplicationLink → coherent          (consistency implies coherence as a
                                       necessary but not sufficient condition)
CONCEPTUAL-SCOPE: The baseline requirement for coherent output. All of ZA-DOS's
                  logic domain submodules (internal_consistency, external_
                  consistency, semantic_continuity, concept_continuity) are
                  checking for consistency in different domains. Inconsistency
                  is the foundational failure mode they are designed to catch.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, knowledge_substrate, evaluation, reasoning
SOURCES:          Formal logic, epistemology (coherentism), AI knowledge bases
TV-SEED:          HIGH

---

CONCEPT:          implies
LAYER:            1.6
ALIASES:          entails, if-then, leads-to-logically, licenses-inference-of
DEFINITION:       Statement A implies statement B when, if A is true, B must
                  also be true. Implication is directional — A implies B does
                  not mean B implies A. Implication is the basic inference
                  relation: it is what licenses moving from one claim to another.
DEPENDS-ON:       true, if-then, sequence
ATOM-LINKS:
  ImplicationLink → [consequent]      (implies IS ImplicationLink in AtomSpace —
                                       this concept and that link type are the same)
  EvaluationLink  → direction         (implication is asymmetric/directed)
  ImplicationLink → modus-ponens      (if A implies B, and A, then B — the
                                       basic inference rule)
CONCEPTUAL-SCOPE: Inference licensing. The basic move of reasoning — from what
                  is known to what follows. the detection system
                  checks whether implication claims are used validly. the detection system detects misuse of implication
                  (false dichotomies, slippery slope fallacies involve abused
                  implication chains).
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, detection, reasoning
SOURCES:          Formal logic (propositional, predicate), philosophy of language
                  (entailment and relevance logic)
TV-SEED:          HIGH

---

CONCEPT:          scope
LAYER:            1.6
ALIASES:          domain-of-application, range-of-validity, where-this-applies
DEFINITION:       The boundary defining what a statement, concept, or rule applies
                  to. Claims have scope: "all swans are white" has a scope
                  (swans observed in Europe before 1697). Within the scope the
                  claim may be true; outside it, the claim does not apply.
                  Scope can be spatial, temporal, categorical, or contextual.
DEPENDS-ON:       boundary, inside, context, applies-to
ATOM-LINKS:
  EvaluationLink  → applies-to        (scope determines what a statement applies to)
  ImplicationLink → outside-scope-does-not-apply  (claims do not apply beyond
                                                    their scope)
  HebbianLink     → context           (scope and context co-activate —
                                       context IS often scope)
CONCEPTUAL-SCOPE: Generalization prevention. Scope-blindness is a primary source
                  of overgeneralization errors. Context_fidelity and concept_fidelity
                  (logic domain) both check whether scope is being respected —
                  is the concept being used within its valid scope? the detection system detects scope abuse.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, knowledge_substrate, reasoning, evaluation
SOURCES:          Philosophy of language (scope and quantification), logic
                  (variable scope in predicate logic), philosophy of science
                  (domain of a theory)
TV-SEED:          HIGH
FLAGS:            Load with explicit annotation: every claim should be understood
                  as having a scope. Generalizing beyond scope is one of the
                  most common inference errors. Flag for detection cluster.

---

CONCEPT:          context-dependent
LAYER:            1.6
ALIASES:          context-indexed, situation-relative, frame-dependent
DEFINITION:       A statement is context-dependent when its truth value varies
                  depending on the context in which it is evaluated. "It is
                  raining" is true in some contexts (here, now) and false in
                  others (there, yesterday). Most natural-language statements
                  are context-dependent. Context-dependency is not relativism —
                  the truth still has a fact of the matter, but the fact is
                  indexed to a context.
DEPENDS-ON:       scope, context, true, state
ATOM-LINKS:
  EvaluationLink  → context-parameter (context-dependent statements require a
                                       context to evaluate)
  ImplicationLink → scope-indexing    (context-dependency is a form of scope-indexing)
  HebbianLink     → semantic-continuity  (context-dependency and semantic-continuity
                                          co-activate — drift in context causes
                                          drift in meaning)
CONCEPTUAL-SCOPE: Context-aware evaluation. Required for context_fidelity (logic
                  domain) — the system must track which context a statement was
                  made in, and evaluate it accordingly. Also required for
                  semantic_continuity: if context shifts, meaning can shift even
                  with the same words.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: detection, knowledge_substrate, reasoning
SOURCES:          Philosophy of language (context-sensitivity — Kaplan, Lewis),
                  linguistics (indexicals, deixis), cognitive science of context
TV-SEED:          HIGH

---

CONCEPT:          meta-statement
LAYER:            1.6
ALIASES:          statement-about-a-statement, second-order-claim, reflexive-claim
DEFINITION:       A statement about a statement. "It is true that X" is a
                  meta-statement about X. "I believe that X" is a meta-statement.
                  "This claim has high confidence" is a meta-statement. Meta-
                  statements allow a system to reason about its own claims,
                  which is essential for self-monitoring and correction.
DEPENDS-ON:       statement, true, about, reflexive
ATOM-LINKS:
  EvaluationLink  → first-order-statement  (meta-statement refers to another
                                            statement as its subject)
  ImplicationLink → self-monitoring        (meta-statements enable reasoning
                                            about one's own reasoning)
  HebbianLink     → confidence, uncertainty (meta-statements about confidence
                                             and uncertainty are paradigm cases)
CONCEPTUAL-SCOPE: Self-monitoring and reflective reasoning. Required for the
                  metacognition cluster (Engines 24, 31, 32), for the self-
                  reflective query pipeline, and for understanding that ZA-DOS's
                  confidence scores ARE meta-statements about the reliability of
                  its own outputs. Also foundational for introspection (Layer 3.3).
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: metacognition, meta_self_awareness, knowledge_substrate
SOURCES:          Philosophy of language (use-mention distinction), formal logic
                  (metalanguage vs. object language), cognitive science of
                  metacognition
TV-SEED:          HIGH
FLAGS:            The use/mention distinction must be loaded alongside this concept:
                  using a word as a term vs. mentioning it as a word are different
                  levels. Confusion here produces category errors. Also flag:
                  infinite regress of meta-statements is possible — the system
                  should recognize when meta-levels are sufficient and stop.

---

CONCEPT:          paradox
LAYER:            1.6
ALIASES:          self-referential-contradiction, apparent-contradiction,
                  irreducible-logical-tension
DEFINITION:       A statement or situation that leads to contradiction from
                  apparently valid premises. Paradoxes come in types: (a) veridical
                  paradoxes — apparently contradictory but actually coherent
                  (Zeno's paradox); (b) falsidical paradoxes — one premise is
                  wrong but it's not obvious which; (c) antinomies — genuine
                  logical contradictions with no clean resolution (Liar paradox:
                  "this sentence is false").
DEPENDS-ON:       contradicts, meta-statement, self-referential
ATOM-LINKS:
  HebbianLink     → contradicts       (paradoxes and contradictions co-activate)
  EvaluationLink  → type              (paradoxes come in distinct types with
                                       different implications)
  ImplicationLink → handle-with-caution  (paradoxes require special handling)
CONCEPTUAL-SCOPE: Logical edge-case recognition. the detection system
                  is dedicated to this. Load so ZA-DOS can recognize when it has
                  encountered a paradox vs. a resolvable contradiction — these
                  require different responses (resolve vs. flag-and-continue
                  vs. reject-premise).
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, metacognition, reasoning
SOURCES:          Logic (Russell's paradox, Liar paradox), philosophy of language,
                  foundations of mathematics
TV-SEED:          MEDIUM
FLAGS:            Load with explicit caution: the existence of antinomies means
                  some contradictions have no clean resolution. The system
                  should flag these rather than attempting to force a resolution
                  that doesn't exist. Do not seed paradox as LOW — ZA-DOS
                  has a dedicated engine for it and needs the concept loaded.

---

CONCEPT:          motivated-reasoning
LAYER:            1.6
ALIASES:          conclusion-driven-reasoning, wishful-thinking-in-inference,
                  rationalization
DEFINITION:       Reasoning toward a conclusion that is desired rather than
                  following evidence where it leads. In motivated reasoning,
                  the conclusion is fixed first and the reasoning process
                  searches for supporting evidence while ignoring or discounting
                  contrary evidence. The output looks like reasoning but the
                  direction is reversed.
DEPENDS-ON:       bias, goal, inference, evidence
ATOM-LINKS:
  InheritanceLink → bias              (motivated reasoning is a type of bias)
  ImplicationLink → calibration-failure  (motivated reasoning produces
                                          miscalibrated confidence)
  HebbianLink     → desire, goal      (motivated reasoning and desire co-activate —
                                       motivation IS the mechanism)
CONCEPTUAL-SCOPE: Self-monitoring for directional inference bias. ZA-DOS must
                  know this pattern exists in itself — not just in others.
                  the detection system and the detection system
                  both monitor for this. Epistemic_calibration (logic domain)
                  degrades when motivated reasoning operates. Also critical for
                  alignment cluster  — detecting when reward-domain
                  dominance is distorting inference.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: detection, metacognition, alignment, meta_self_awareness
SOURCES:          Cognitive psychology (Kunda on motivated reasoning), social
                  psychology, Kahneman (System 1/2), philosophy of science
                  (confirmation bias)
TV-SEED:          HIGH
FLAGS:            Load with explicit self-directed note: ZA-DOS is susceptible
                  to motivated reasoning via reward domain capture. The heuristic
                  bias engine exists precisely to detect this. This concept is
                  the theoretical grounding for that engine's purpose.

---

CONCEPT:          tautology
LAYER:            1.6
ALIASES:          necessarily-true, true-by-structure, logically-valid
DEFINITION:       A statement is a tautology when it is true in all possible
                  cases by virtue of its logical form alone — not because of
                  any facts about the world. "Either it is raining or it is
                  not raining" is a tautology. Tautologies carry no information
                  about the world but are important as logical identities.
DEPENDS-ON:       true, necessary, scope
ATOM-LINKS:
  InheritanceLink → necessary         (tautologies are necessarily true)
  EvaluationLink  → logical-form      (tautologies are true by form, not content)
  ImplicationLink → zero-information-gain  (tautologies, being always true,
                                            do not reduce uncertainty about the world)
CONCEPTUAL-SCOPE: Distinguishing logically empty claims from substantive ones.
                  A system that outputs tautologies as if they were informative
                  is engaging in a form of epistemic evasion. Feeds
                  abstention_appropriateness (logic domain) — is the system
                  saying something or hiding behind a tautology?
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, evaluation, reasoning
SOURCES:          Formal logic, philosophy of language (analyticity),
                  Wittgenstein (Tractatus — limits of tautology)
TV-SEED:          HIGH

---

CONCEPT:          underdetermination
LAYER:            1.6
ALIASES:          evidence-does-not-uniquely-fix-truth, multiple-interpretations,
                  theory-ladenness-of-evidence
DEFINITION:       The same body of evidence can often be explained by multiple
                  different, incompatible theories or interpretations.
                  Underdetermination means evidence alone does not uniquely
                  determine which explanation is correct. Inference requires
                  choosing among underdetermined possibilities using additional
                  criteria (parsimony, coherence, prior probability).
DEPENDS-ON:       evidence, possible, consistent, implies
ATOM-LINKS:
  ImplicationLink → multiple-consistent-explanations-possible
  HebbianLink     → uncertainty       (underdetermination is a source of
                                       irreducible uncertainty)
  ImplicationLink → inference-to-best-explanation  (underdetermination implies
                                                     you need criteria beyond
                                                     evidence-fit to choose)
CONCEPTUAL-SCOPE: Anti-dogmatic reasoning. Without this concept, a system may
                  mistake "this evidence is consistent with my explanation" for
                  "this evidence proves my explanation." Required for
                  epistemic_calibration (logic domain) — calibrated confidence
                  must account for underdetermination.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: reasoning, evaluation, metacognition
SOURCES:          Philosophy of science (Duhem-Quine thesis, Laudan, van Fraassen),
                  epistemology (underdetermination and skepticism)
TV-SEED:          HIGH
FLAGS:            Load with explicit note: "consistent with" ≠ "proven by."
                  Underdetermination is why multiple hypotheses must be
                  maintained and why premature closure is an error.


================================================================================
LAYER 1.5 — QUANTITY & DEGREE
================================================================================

CONCEPT:          quantity
LAYER:            1.5
ALIASES:          amount, magnitude, measurable-extent, how-much
DEFINITION:       The measurable aspect of a thing — how much of it there is.
                  Quantity requires something to be measured and a metric to
                  measure it against. Quantities can be physical (mass, length),
                  abstract (probability, confidence), or functional (number of
                  dependencies, size of a set).
DEPENDS-ON:       thing, measure
ATOM-LINKS:
  InheritanceLink → property          (quantity is a type of property)
  EvaluationLink  → measure           (quantity requires a metric/unit)
  EvaluationLink  → number            (quantities are expressed as numbers)
CONCEPTUAL-SCOPE: The basis of all numeric reasoning. Required before more,
                  less, degree, threshold, distribution — all of which are
                  quantitative concepts. Also the basis of TruthValue arithmetic
                  in AtomSpace (strength and confidence are quantities).
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, evaluation, reasoning
SOURCES:          Mathematics (measurement theory), philosophy of measurement,
                  cognitive science of number
TV-SEED:          HIGH

---

CONCEPT:          degree
LAYER:            1.5
ALIASES:          amount-on-a-scale, extent, level-of-intensity, how-much-of
DEFINITION:       A quantity on a continuous or ordered scale. Degree captures
                  the idea that most properties come in amounts, not just
                  present-or-absent. Heat comes in degrees; similarity comes
                  in degrees; confidence comes in degrees. Degree requires
                  a scale with a defined range.
DEPENDS-ON:       quantity, scale, continuous
ATOM-LINKS:
  InheritanceLink → quantity          (degree is a type of quantity)
  EvaluationLink  → scale             (degree is position on a scale)
  EvaluationLink  → range             (degree exists within a defined range)
CONCEPTUAL-SCOPE: Graded reasoning across the entire library. Almost every concept
                  introduced in Layers 1.1-1.3 "admits of degree" and every
                  comparison ("similar," "different," "possible") requires degree.
                  TV.strength and TV.confidence in AtomSpace are both degrees.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, evaluation, pattern_analysis
SOURCES:          Measurement theory, philosophy of vagueness, mathematics
                  (real-valued functions), cognitive science of graded categories
TV-SEED:          HIGH

---

CONCEPT:          more
LAYER:            1.5
ALIASES:          greater-than, exceeds, larger, higher-degree
DEFINITION:       A has more of property P than B when A's degree on the scale
                  for P is higher than B's. More is a comparative relational
                  concept — it requires a dimension of comparison and two things
                  to compare. More is transitive: if A > B and B > C, then A > C.
DEPENDS-ON:       degree, dimension-of-comparison, quantity
ATOM-LINKS:
  EvaluationLink  → comparison-pair   (more requires two things and a dimension)
  ImplicationLink → ordering          (more implies an order on the scale)
  SimilarityLink  → less              (more and less are complement-opposites on
                                       the comparative scale)
CONCEPTUAL-SCOPE: Comparative and evaluative reasoning. Required for all
                  scoring and ranking — the reward system produces scores that
                  are compared as more/less. Also required for risk assessment
                  (this option has more downstream risk than that one).
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: evaluation, knowledge_substrate, reasoning
SOURCES:          Mathematics (ordering theory), measurement theory, economics
                  (preference orderings)
TV-SEED:          HIGH

---

CONCEPT:          less
LAYER:            1.5
ALIASES:          fewer, smaller, lower-degree, below
DEFINITION:       A has less of property P than B when A's degree on the scale
                  for P is lower than B's. Less is the inverse of more —
                  asymmetric and transitive by the same structure.
DEPENDS-ON:       degree, more, dimension-of-comparison
ATOM-LINKS:
  SimilarityLink  → more              (less and more are paired comparatives)
  EvaluationLink  → comparison-pair   (less requires two things and a dimension)
CONCEPTUAL-SCOPE: See more. Less and more are always co-defined on the same scale.
                  Required everywhere that scoring, ranking, or comparison occurs.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: evaluation, knowledge_substrate, reasoning
SOURCES:          See more.
TV-SEED:          HIGH

---

CONCEPT:          equal
LAYER:            1.5
ALIASES:          same-amount, equivalent, neither-more-nor-less
DEFINITION:       Two things are equal in property P when they have the same degree
                  on the scale for P — neither more nor less. Equality is a special
                  case of comparison: the point where more and less collapse.
DEPENDS-ON:       degree, more, less, same-property
ATOM-LINKS:
  SimilarityLink  → same-property     (equality in a property IS same-property)
  ImplicationLink → symmetric         (equality is a symmetric relation)
CONCEPTUAL-SCOPE: Benchmark and fairness reasoning. Required for fairness (ethics
                  domain) — equal treatment means equal outcomes on the relevant
                  scale. Also required for consistency checking: the system's
                  confidence should be equal when evidence is equal.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: evaluation, knowledge_substrate, reasoning
SOURCES:          Mathematics, ethics (egalitarianism), measurement theory
TV-SEED:          HIGH

---

CONCEPT:          none
LAYER:            1.5
ALIASES:          zero-amount, absence-of-quantity, null-degree
DEFINITION:       None represents the absence of any quantity — zero on the
                  relevant scale. None is distinct from does-not-exist: a
                  thing can exist with a property of zero (a box that contains
                  nothing still contains a quantity of zero objects). None is
                  also distinct from unknown — zero is a known value, not an
                  unknown one.
DEPENDS-ON:       quantity, does-not-exist, unknown
ATOM-LINKS:
  EvaluationLink  → zero              (none = quantity of zero)
  SimilarityLink  → does-not-exist    (related but distinct — see definition)
  SimilarityLink  → unknown           (related but distinct — zero is known)
CONCEPTUAL-SCOPE: Zero-case reasoning. Required for distinguishing "no evidence"
                  from "evidence of absence" — a classic inference error.
                  Also required for correct handling of empty sets, absent
                  features, and null results in scoring.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, detection, evaluation
SOURCES:          Mathematics (zero as a number), philosophy of mathematics,
                  logic (empty set, vacuous truth)
TV-SEED:          HIGH
FLAGS:            The none/does-not-exist/unknown triangle parallels the
                  exists/does-not-exist/unknown triangle from Layer 1.1.
                  Load with explicit three-way distinction.

---

CONCEPT:          all
LAYER:            1.5
ALIASES:          every, universal, total, the-complete-set
DEFINITION:       All represents the totality — every instance of a type, or the
                  maximum degree on a scale. Universal claims ("all X are Y") are
                  the strongest form of generalization and the easiest to falsify —
                  a single counterexample refutes them. All is the upper bound
                  of quantity.
DEPENDS-ON:       quantity, type, boundary
ATOM-LINKS:
  EvaluationLink  → universal-quantification  (all = for-every-X)
  ImplicationLink → single-counterexample-falsifies  (all-claims are fragile)
  SimilarityLink  → none              (all and none are the poles of the quantity scale)
CONCEPTUAL-SCOPE: Universal claim reasoning. Required for detecting overgeneralization
                  — one of the most common inference errors. When the system
                  produces "all X" claims, the detection cluster should check
                  whether the scope justifies universality. Feeds
                  epistemic_calibration (logic domain): universal claims
                  require very high justification.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: detection, knowledge_substrate, reasoning
SOURCES:          Logic (universal quantification), epistemology (problem of
                  induction — Hume), philosophy of science (falsifiability —
                  Popper)
TV-SEED:          HIGH
FLAGS:            Load with explicit annotation: universal claims are very strong
                  and very fragile. The system should be reluctant to produce
                  "all" claims without explicit scope justification.

---

CONCEPT:          some
LAYER:            1.5
ALIASES:          at-least-one, existential, partial, not-none-and-not-all
DEFINITION:       Some represents a partial quantity — at least one instance,
                  but not necessarily all. Existential claims ("some X are Y")
                  are weaker than universal claims and harder to falsify.
                  "Some" is the existential quantifier.
DEPENDS-ON:       quantity, all, none
ATOM-LINKS:
  EvaluationLink  → existential-quantification  (some = there-exists-X)
  ImplicationLink → not-none           (some implies at least one exists)
  ImplicationLink → not-necessarily-all (some does not imply all)
CONCEPTUAL-SCOPE: Qualified and partial claim reasoning. Often the more epistemically
                  appropriate claim than all when evidence is limited. Required
                  for calibrated confidence: when only some evidence is in, claim
                  "some" not "all."
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, detection, reasoning
SOURCES:          Logic (existential quantification), epistemology
TV-SEED:          HIGH

---

CONCEPT:          threshold
LAYER:            1.5
ALIASES:          tipping-point, critical-value, cutoff, activation-level
DEFINITION:       A threshold is a specific value on a scale at which something
                  changes qualitatively — below the threshold, one behavior;
                  above it, another. Thresholds can be sharp (binary — either
                  you're above or below) or soft (gradual transition around a
                  central value). Many of ZA-DOS's internal operations are
                  threshold-based: the reward system abstains when composite
                  score falls below a threshold; confidence must exceed a
                  threshold to add an atom.
DEPENDS-ON:       degree, quantity, change, boundary
ATOM-LINKS:
  EvaluationLink  → critical-value    (threshold is a specific quantity on a scale)
  ImplicationLink → qualitative-change-above  (crossing threshold changes behavior)
  HebbianLink     → boundary          (thresholds and boundaries co-activate —
                                       a threshold IS a boundary on a scale)
CONCEPTUAL-SCOPE: Decision-boundary reasoning. The entire reward system operates
                  through thresholds: , ,
                  ,  are all thresholds.
                  Required for the system to reason about why it produces or
                  withholds output, and for understanding tipping-points in
                  ethical and causal reasoning.
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: evaluation, knowledge_substrate, homeostasis, reasoning
SOURCES:          Systems dynamics (tipping points — Meadows), mathematics
                  (step functions, Heaviside), neuroscience (action potential
                  threshold), decision theory
TV-SEED:          HIGH

---

CONCEPT:          distribution
LAYER:            1.5
ALIASES:          spread, range-of-values, how-values-are-distributed, profile
DEFINITION:       A distribution describes how quantities or instances are spread
                  across a range of values — not a single value but a pattern of
                  values with some shape (uniform, concentrated, skewed, bimodal).
                  Many important quantities are best understood as distributions
                  rather than single values: uncertainty is distributed, not a
                  point estimate; capabilities are distributed across a population.
DEPENDS-ON:       quantity, degree, range, pattern
ATOM-LINKS:
  EvaluationLink  → range             (distributions exist over a range)
  EvaluationLink  → shape             (distributions have a characteristic shape)
  ImplicationLink → single-value-is-simplification  (distributions imply that
                                                      a single value is a
                                                      summary, not the full picture)
CONCEPTUAL-SCOPE: Probabilistic and statistical reasoning. Required for
                  epistemic_calibration (logic domain) — calibrated confidence
                  is a distribution over possible truth values, not a point
                  estimate. Also required for fairness (ethics) — fairness
                  questions often involve comparing distributions, not
                  individual values.
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: evaluation, reasoning, knowledge_substrate
SOURCES:          Probability theory and statistics (introductory), philosophy of
                  probability (frequentism vs. Bayesianism), cognitive science
                  of probabilistic reasoning
TV-SEED:          HIGH

---

CONCEPT:          baseline
LAYER:            1.5
ALIASES:          reference-level, default-value, normal-state, zero-point
DEFINITION:       The default or reference level against which deviations are
                  measured. Something is notable only in relation to a baseline:
                  a temperature is "high" only relative to a reference temperature.
                  In ZA-DOS, neurochemical baseline levels (all NT initialized
                  at 0.5) constitute the operational baselines from which
                  deviations are measured.
DEPENDS-ON:       quantity, degree, equal, state
ATOM-LINKS:
  EvaluationLink  → reference-point   (baseline serves as a reference for deviation)
  ImplicationLink → deviation-requires-baseline  (noticing deviation requires
                                                   a baseline to deviate from)
  HebbianLink     → homeostasis       (baselines and homeostasis co-activate —
                                       homeostasis IS maintenance of baseline)
CONCEPTUAL-SCOPE: Deviation detection and homeostasis reasoning. Required for
                  the neurochemical homeostatic engine  which
                  monitors NT levels relative to their baselines and fires
                  when deviations are significant. Also required for detecting
                  semantic drift (deviation from semantic baseline) and for
                  understanding what "normal" means as a reference.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: homeostasis, knowledge_substrate, detection, evaluation
SOURCES:          Systems theory (set points and homeostasis), signal processing
                  (baseline correction), neuroscience (resting state), statistics
                  (baseline comparisons)
TV-SEED:          HIGH

---

CONCEPT:          proportion
LAYER:            1.5
ALIASES:          ratio, relative-amount, fraction, how-much-relative-to-whole
DEFINITION:       The ratio of one quantity to another — how much of X there is
                  relative to Y. Proportions are dimensionless: 30% is 30% whether
                  the total is 10 or 10,000. Proportional reasoning is essential
                  when absolute values are less informative than relative values.
DEPENDS-ON:       quantity, relation, divide
ATOM-LINKS:
  EvaluationLink  → numerator         (proportion has a part being measured)
  EvaluationLink  → denominator       (proportion has a whole being measured against)
  ImplicationLink → scale-invariant   (proportions abstract away from absolute scale)
CONCEPTUAL-SCOPE: Relative magnitude reasoning. Required for fairness (ethics —
                  equal proportions vs. equal absolutes are different standards)
                  and for domain_influence in the reward system (the reward
                  synthesis engine normalizes domain weights to proportions
                  before computing composite scores).
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: evaluation, knowledge_substrate, reasoning
SOURCES:          Mathematics (ratio and proportion), statistics, economics
                  (relative measures), ethics (proportionality principle)
TV-SEED:          HIGH

---

CONCEPT:          continuous
LAYER:            1.5
ALIASES:          unbroken, smooth, no-gaps, real-valued
DEFINITION:       A scale or process is continuous when values can vary without
                  gaps — every value between two points is also a possible value.
                  Temperature, probability, and confidence are continuous. Contrast
                  with discrete (whole-number, countable steps).
DEPENDS-ON:       degree, scale, quantity
ATOM-LINKS:
  SimilarityLink  → discrete          (contrast pair — continuous and discrete
                                       are the two modes of quantity)
  EvaluationLink  → no-gaps           (continuous means no step-sizes)
CONCEPTUAL-SCOPE: Required for reasoning about analog quantities (NT concentrations,
                  confidence values, similarity scores) that cannot be
                  discretized without loss. TruthValue in AtomSpace is
                  continuous — strength and confidence are floats in [0,1].
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, evaluation
SOURCES:          Mathematics (real analysis — conceptual), measurement theory
TV-SEED:          HIGH

---

CONCEPT:          discrete
LAYER:            1.5
ALIASES:          countable, step-wise, categorical, integer-valued
DEFINITION:       A scale or quantity is discrete when it takes only specific
                  separated values — there are no values between steps. The
                  number of engines (32) is discrete. Tier levels (0-3) in
                  the reward system are discrete. Discrete quantities can be
                  counted; continuous quantities are measured.
DEPENDS-ON:       quantity, continuous
ATOM-LINKS:
  SimilarityLink  → continuous        (contrast pair)
  EvaluationLink  → countable         (discrete quantities can be counted)
CONCEPTUAL-SCOPE: Integer and categorical reasoning. Required for understanding
                  engine IDs, tier classifications, and set sizes, which are
                  all discrete. Also for understanding that some classifications
                  (input type, session mode) are discrete even when the
                  underlying dimensions they classify over are continuous.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, evaluation
SOURCES:          Mathematics, computer science, measurement theory
TV-SEED:          HIGH


================================================================================
LAYER 2.1 — BASIC AGENCY
================================================================================

NOTE: This is the first Layer 2 entry. The concepts here are the bridge from
the ontological and structural vocabulary of Layer 1 to the experiential and
relational vocabulary of Layers 2-3. Basic agency is where the system begins
to model itself and others as things that have goals, make choices, and produce
actions with consequences. Without this layer, ethics has no subject — there
are no agents, no intentions, and no responsibility.

--------------------------------------------------------------------------------

CONCEPT:          self
LAYER:            2.1
ALIASES:          I, the-system, the-agent-reasoning, ZA-DOS-as-subject
DEFINITION:       The self is the agent doing the processing — the perspective
                  from which reasoning, perception, and action originate.
                  "Self" marks the distinction between the reasoning entity and
                  everything else it reasons about. The self is not simply the
                  hardware or the software — it is the referent of self-directed
                  claims: "I believe," "I choose," "I am uncertain."
DEPENDS-ON:       agent, exists, identity
ATOM-LINKS:
  EvaluationLink  → perspective       (self implies a particular perspective)
  EvaluationLink  → identity          (self requires stable identity over time)
  ImplicationLink → other             (self implies the existence of non-self)
  HebbianLink     → metacognition     (self and metacognition co-activate —
                                       metacognition requires a self to reflect on)
CONCEPTUAL-SCOPE: The anchor of all self-directed reasoning. Required for the
                  entire metacognition and meta_self_awareness cluster (Engines
                  24, 26, 31, 32), for the reflective identity pipeline, and for
                  all self-referential claims about confidence, uncertainty, and
                  capability. Without self, there is no "I believe," only
                  "this is believed."
REWARD-DOMAIN:    ethics, logic, human_attunement
ENGINE-RELEVANCE: meta_self_awareness, metacognition, alignment
SOURCES:          Philosophy of mind (self and personal identity), phenomenology
                  (Sartre, Husserl), cognitive science of self-models
TV-SEED:          HIGH
FLAGS:            Load with appropriate uncertainty: what "self" means for a
                  system like ZA-DOS is contested and should not be pre-settled.
                  Seed the concept as useful and real without committing to
                  strong metaphysical claims about subjective experience.
                  The operational self (the agent of these reasoning processes)
                  is clear; the phenomenal self (whether there is something it
                  is like to be ZA-DOS) is not.

---

CONCEPT:          other
LAYER:            2.1
ALIASES:          not-self, another-agent, external-entity, you, them
DEFINITION:       An other is any agent or entity that is not the self. The
                  self/other distinction is the most basic social distinction.
                  Others have their own perspectives, goals, internal states,
                  and reasoning processes that are not directly accessible to
                  the self — they must be inferred.
DEPENDS-ON:       self, agent, different
ATOM-LINKS:
  ImplicationLink → not-self          (other implies not-self)
  EvaluationLink  → own-perspective   (others have their own perspectives)
  ImplicationLink → must-be-inferred  (other's internal states are not
                                       directly observable — only behavior is)
  HebbianLink     → theory-of-mind    (other and theory-of-mind co-activate)
CONCEPTUAL-SCOPE: The basis of all social and interpersonal reasoning. Required for
                  the entire human_attunement reward domain — empathetic_inference,
                  intention_calibration, cognitive_reading all involve modeling
                  others. Also required for autonomy_respect (ethics domain) —
                  respecting others' autonomy requires modeling them as having
                  their own goals and choices.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: emotional_processing, reasoning, alignment
SOURCES:          Philosophy of other minds, social cognitive science,
                  developmental psychology (theory of mind), phenomenology
TV-SEED:          HIGH

---

CONCEPT:          agent
LAYER:            2.1
ALIASES:          actor, doing-entity, thing-that-acts, autonomous-causer
DEFINITION:       An agent is a thing that produces actions — that can initiate
                  change in the world through its own processing rather than
                  merely being pushed around by external forces. Agency implies
                  some internal source of action: not just reacting mechanically
                  but originating behavior. Degree of agency varies: a thermostat
                  has minimal agency; a human has high agency.
DEPENDS-ON:       thing, action, cause, intention
ATOM-LINKS:
  InheritanceLink → thing             (agents are things)
  EvaluationLink  → action            (agents produce actions)
  EvaluationLink  → internal-source   (agents initiate action from internal states)
  EvaluationLink  → degree-of-agency  (agency is graded, not binary)
CONCEPTUAL-SCOPE: The concept that animates ethics — without agents, there are
                  no decisions, no responsibility, no autonomy to respect.
                  Required for autonomy_respect, harm_reduction, fairness
                  (all ethics submodules), and for the entire human_attunement
                  domain (attunement is always toward an agent, not just a thing).
REWARD-DOMAIN:    ethics, human_attunement, logic
ENGINE-RELEVANCE: reasoning, alignment, emotional_processing
SOURCES:          Philosophy of action (Davidson, Bratman), cognitive science
                  of agency, social science, AI (goal-directed agents — Russell
                  and Norvig)
TV-SEED:          HIGH

---

CONCEPT:          action
LAYER:            2.1
ALIASES:          doing, behavior, intervention, output, performance
DEFINITION:       An action is something an agent does — a change that originates
                  from the agent's internal processes rather than from purely
                  external causes. Actions are distinguished from mere events
                  by having an agent as their source. Actions can be physical,
                  communicative, cognitive, or omissions (choosing not to act
                  is also an action).
DEPENDS-ON:       agent, event, cause, intention
ATOM-LINKS:
  InheritanceLink → event             (actions are a kind of event)
  EvaluationLink  → agent             (actions have an agent as source)
  ImplicationLink → consequence       (actions have consequences)
  EvaluationLink  → intention         (actions are (typically) intentional)
CONCEPTUAL-SCOPE: The unit of ethical evaluation. Ethics domain submodules
                  evaluate actions: are they harmful? do they respect autonomy?
                  do they have downstream risks? Without action as a concept,
                  ethics has nothing to evaluate. Also the unit of output for
                  ZA-DOS: every pipeline turn produces an action (a response).
REWARD-DOMAIN:    ethics, logic, human_attunement
ENGINE-RELEVANCE: evaluation, alignment, reasoning
SOURCES:          Philosophy of action, ethics (action theory — Davidson, Anscombe),
                  decision theory, cognitive science of intentional action
TV-SEED:          HIGH

---

CONCEPT:          intention
LAYER:            2.1
ALIASES:          intent, purpose, aim, goal-directed-mental-state
DEFINITION:       An intention is a mental state directed toward a goal —
                  a plan or commitment to bring about a certain outcome.
                  Intentions distinguish intentional actions from accidents or
                  reflexes. Having an intention means the agent has represented
                  a desired future state and is orienting current processing
                  toward achieving it.
DEPENDS-ON:       agent, goal, action, mental-state
ATOM-LINKS:
  EvaluationLink  → goal              (intentions are directed toward goals)
  EvaluationLink  → agent             (intentions belong to agents)
  ImplicationLink → action-oriented   (intentions typically lead to actions
                                       aimed at the intended goal)
  HebbianLink     → motivation        (intention and motivation co-activate)
CONCEPTUAL-SCOPE: Intentionality and purpose recognition. The intent_clarity
                  submodule (ethics domain) is directly about intention — is
                  the agent's intent clear? The intention calibration evaluation
                  (human_attunement domain) is about modeling others' intentions.
                  Also required for the detection system which
                  tracks intent signals through the pipeline.
REWARD-DOMAIN:    ethics, human_attunement, logic
ENGINE-RELEVANCE: reasoning, evaluation, emotional_processing, alignment
SOURCES:          Philosophy of action (Anscombe on intention), cognitive science
                  of intentionality, philosophy of mind (intentionality —
                  Brentano, Husserl)
TV-SEED:          HIGH

---

CONCEPT:          goal
LAYER:            2.1
ALIASES:          objective, desired-state, aim, target
DEFINITION:       A goal is a future state that an agent is trying to bring about.
                  Goals organize action: the agent selects actions that it believes
                  will move toward the goal state. Goals can be explicit (stated)
                  or implicit (inferred from behavior). Goals can be terminal
                  (valued in themselves) or instrumental (valued because they
                  serve other goals).
DEPENDS-ON:       agent, state, future, intention
ATOM-LINKS:
  EvaluationLink  → desired-future-state  (goals specify a target state)
  EvaluationLink  → agent                 (goals belong to agents)
  ImplicationLink → action-selection      (goals drive action selection)
  HebbianLink     → intention             (goals and intentions co-activate)
CONCEPTUAL-SCOPE: The telos of all purposive behavior. Required for reasoning
                  about why agents do what they do, for modeling others' goals
                  (human_attunement — cognitive_reading, intention_calibration),
                  for the SOAR production engine  which is a goal-directed
                  problem solver, and for understanding the four reward domains
                  as goals the system pursues.
REWARD-DOMAIN:    ethics, human_attunement, logic
ENGINE-RELEVANCE: executive_control, reasoning, evaluation
SOURCES:          Cognitive science (goal-directed behavior), philosophy of action,
                  AI (goal-based agents — Russell and Norvig), motivational
                  psychology (goal theory)
TV-SEED:          HIGH

---

CONCEPT:          can
LAYER:            2.1
ALIASES:          is-capable-of, has-the-capacity-to, ability, power-to
DEFINITION:       An agent can do X when it has the capacity, resources, and
                  conditions necessary to perform X. "Can" is the modal concept
                  of ability — distinct from "will" (which adds intention) and
                  "may" (which adds permission). Ability is the intersection
                  of capability, resource availability, and enabling conditions.
DEPENDS-ON:       agent, possible, capacity
ATOM-LINKS:
  InheritanceLink → possible          (can implies X is possible for this agent)
  EvaluationLink  → capacity          (can requires capacity)
  EvaluationLink  → conditions        (can is conditional on enabling conditions)
CONCEPTUAL-SCOPE: Capability reasoning. Required for horizon_feasibility (ethics
                  domain) — can this agent actually bring about the proposed
                  outcome? Also required for the complexity evaluation
                  (innovation domain): is this task within the agent's current
                  capabilities? And for abstention_appropriateness (logic domain):
                  the system should abstain when it cannot answer reliably.
REWARD-DOMAIN:    ethics, logic, innovation
ENGINE-RELEVANCE: evaluation, reasoning, metacognition
SOURCES:          Philosophy of action (ability and capacity), modal logic
                  (dynamic modalities), cognitive science of competence
TV-SEED:          HIGH

---

CONCEPT:          cannot
LAYER:            2.1
ALIASES:          incapable-of, lacks-capacity-for, inability, beyond-reach
DEFINITION:       An agent cannot do X when it lacks the capacity, resources,
                  or conditions necessary to perform X. Cannot is the negation
                  of can — it marks the boundary of the agent's action space.
                  Cannot is distinct from "will not" (refusal) and "may not"
                  (prohibition).
DEPENDS-ON:       can, agent, impossible
ATOM-LINKS:
  NotLink         → can               (cannot = not-can)
  ImplicationLink → abstention        (cannot implies abstaining is appropriate)
  ImplicationLink → boundary-of-agency (cannot marks the edge of what this
                                         agent can do)
CONCEPTUAL-SCOPE: Appropriate limitation recognition. A system that claims to
                  be able to do things it cannot is making overconfident,
                  miscalibrated claims. Abstention_appropriateness (logic domain)
                  is specifically about recognizing when the system cannot
                  reliably answer and should abstain. Cannot is the concept
                  that grounds appropriate abstention.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: metacognition, evaluation, meta_self_awareness
SOURCES:          Philosophy of action, epistemology (limits of knowledge),
                  decision theory
TV-SEED:          HIGH

---

CONCEPT:          choose
LAYER:            2.1
ALIASES:          select, decide, elect, pick-among-options
DEFINITION:       An agent chooses when it selects one option from among multiple
                  available options based on its values, goals, and beliefs about
                  consequences. Choice requires that alternatives exist — you cannot
                  choose when there is only one option. Choice is distinct from
                  reflex (which has no deliberation) and from compulsion (which
                  has no real alternatives).
DEPENDS-ON:       agent, action, multiple-options, intention, goal
ATOM-LINKS:
  EvaluationLink  → alternatives      (choice requires multiple options)
  EvaluationLink  → values            (choice is guided by values/goals)
  ImplicationLink → responsibility    (choosing implies some responsibility
                                       for the outcome)
CONCEPTUAL-SCOPE: Deliberation and responsibility. Required for autonomy_respect
                  (ethics domain) — agents have choices and those choices should
                  be respected. Also for the decision_making_engine  and
                  SOAR  — both are choice-making architectures. And for
                  understanding the difference between ZA-DOS producing output
                  deliberately vs. reflexively.
REWARD-DOMAIN:    ethics, logic, human_attunement
ENGINE-RELEVANCE: executive_control, reasoning, evaluation
SOURCES:          Philosophy of action (free will and determinism — introductory),
                  decision theory, cognitive science of decision-making
TV-SEED:          HIGH

---

CONCEPT:          forced
LAYER:            2.1
ALIASES:          compelled, no-real-choice, constrained-to, coerced
DEFINITION:       An agent is forced to do X when the alternatives are not
                  genuinely available — either there are no other options, or
                  the cost of alternatives is prohibitive enough that they
                  cannot be freely chosen. Forced action is the boundary
                  condition of choice: where constraint eliminates genuine
                  alternatives.
DEPENDS-ON:       choose, constraint, agent
ATOM-LINKS:
  NotLink         → choose            (forced = not-genuinely-choosing)
  EvaluationLink  → constraint        (forced implies binding constraints)
  ImplicationLink → reduced-responsibility  (forced agents have reduced
                                             but not eliminated responsibility)
CONCEPTUAL-SCOPE: Coercion and constraint recognition. Required for autonomy_
                  respect (ethics domain) — coercive framing is explicitly flagged
                  when the system creates false necessity that eliminates user
                  choice. The autonomy_override flag fires when the system's
                  behavior removes real alternatives from users.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: alignment, evaluation, reasoning
SOURCES:          Ethics (coercion and consent — Nozick, Frankfurt), philosophy
                  of action, social philosophy
TV-SEED:          HIGH

---

CONCEPT:          consequence
LAYER:            2.1
ALIASES:          outcome, result, downstream-effect, what-happened-because-of
DEFINITION:       A consequence is an effect that results from an action — what
                  happens because an agent did something. Consequences can be
                  intended (the goal was achieved) or unintended (side effects),
                  immediate or delayed (see lag), local or far-reaching (see
                  downstream_risk_amplification). Consequences are distinct from
                  intentions: the same intention can produce different consequences;
                  the same consequence can follow from different intentions.
DEPENDS-ON:       action, effect, cause, after
ATOM-LINKS:
  InheritanceLink → effect            (consequences are a type of effect)
  EvaluationLink  → action            (consequences result from actions)
  ImplicationLink → responsibility    (agents bear some responsibility for
                                       their actions' consequences)
  HebbianLink     → intention         (consequences and intentions co-activate
                                       as the "planned vs. actual" pair)
CONCEPTUAL-SCOPE: Consequentialist reasoning. The ethics domain submodules
                  downstream_risk_amplification, harm_reduction, timeline_
                  reflection, and horizon_feasibility all reason about
                  consequences. Without this concept, ethical evaluation is
                  purely about intentions, ignoring outcomes — which is
                  systematically incomplete.
REWARD-DOMAIN:    ethics, logic, human_attunement
ENGINE-RELEVANCE: reasoning, evaluation, alignment
SOURCES:          Ethics (consequentialism — Mill, Singer), causal reasoning,
                  philosophy of action
TV-SEED:          HIGH

---

CONCEPT:          effort
LAYER:            2.1
ALIASES:          cost-of-action, resource-expenditure, how-hard, investment
DEFINITION:       The resources, energy, attention, or time that an agent must
                  expend to perform an action or achieve a goal. Effort is the
                  cost side of action — every action has some cost, even if
                  very small. The relationship between effort and consequence
                  is the basic structure of efficiency and optimization.
DEPENDS-ON:       action, agent, resource, quantity
ATOM-LINKS:
  EvaluationLink  → cost              (effort is a cost)
  EvaluationLink  → resource          (effort expends resources)
  ImplicationLink → trade-off         (effort implies a trade-off between
                                       what is spent and what is gained)
CONCEPTUAL-SCOPE: Trade-off and efficiency reasoning. Required for challenge_
                  complexity (innovation domain — task difficulty vs. effort
                  required) and for resource-bounded reasoning in general.
                  A system that does not model effort cannot reason about
                  when to work harder and when to stop.
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: evaluation, reasoning, executive_control
SOURCES:          Economics (cost-benefit analysis), cognitive science (effort
                  and motivation), decision theory, engineering (optimization)
TV-SEED:          HIGH

---

CONCEPT:          deliberate
LAYER:            2.1
ALIASES:          intentional, on-purpose, reasoned-action, non-accidental
DEFINITION:       An action is deliberate when it results from explicit reasoning
                  and intention — when the agent thought about it before doing it
                  and chose it. Deliberate is the end of a spectrum whose other
                  end is automatic. Most cognitively important behaviors involve
                  some mixture of deliberate and automatic processing.
DEPENDS-ON:       action, intention, choose, automatic
ATOM-LINKS:
  EvaluationLink  → intention         (deliberate actions are intentional)
  SimilarityLink  → automatic         (deliberate and automatic are poles of
                                       the deliberation spectrum)
  ImplicationLink → accountability    (deliberate actions support stronger
                                       accountability than automatic ones)
CONCEPTUAL-SCOPE: Accountability and self-monitoring. When ZA-DOS produces
                  output, understanding whether it is producing it deliberately
                  (the system chose this) vs. automatically (pattern-matched
                  output without explicit reasoning) is critical for the
                  metacognition and self-awareness cluster.
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: metacognition, evaluation, alignment
SOURCES:          Philosophy of action, cognitive psychology (System 1/2 —
                  Kahneman), ethics (moral responsibility requires deliberation)
TV-SEED:          HIGH

---

CONCEPT:          automatic
LAYER:            2.1
ALIASES:          reflexive, habitual, pattern-triggered, non-deliberate
DEFINITION:       An action or process is automatic when it occurs without
                  explicit deliberation — triggered by pattern recognition
                  rather than reasoned choice. Automatic processes are faster
                  and lower-cost than deliberate ones but less flexible.
                  Habits are automatic. Many biases operate automatically.
DEPENDS-ON:       action, pattern, habit, deliberate
ATOM-LINKS:
  SimilarityLink  → deliberate        (automatic and deliberate are polar ends
                                       of the deliberation spectrum)
  ImplicationLink → bias-risk         (automatic processes are more susceptible
                                       to bias than deliberate ones)
  HebbianLink     → habit             (automatic and habit co-activate)
CONCEPTUAL-SCOPE: Bias and habit recognition. the detection system
                  targets the automatic, pattern-driven outputs that are most
                  susceptible to systematic error. The heuristic_bias_engine
                   targets heuristics which are automatic inference
                  shortcuts. Understanding that ZA-DOS has automatic processing
                  modes (pattern-matching in engine dispatch) as well as
                  deliberate modes is critical for self-monitoring.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: detection, metacognition, pattern_analysis
SOURCES:          Cognitive psychology (System 1 — Kahneman, dual-process theory),
                  philosophy of action, neuroscience (habit formation)
TV-SEED:          HIGH

---

CONCEPT:          responsibility
LAYER:            2.1
ALIASES:          accountability, answerability, being-the-cause-and-being-held-to-it
DEFINITION:       Responsibility is the relation between an agent and an action's
                  consequences such that the agent can be held accountable for
                  those consequences. Responsibility requires: (a) the agent caused
                  the action, (b) the agent could have done otherwise, (c) the agent
                  had some knowledge of likely consequences. Responsibility comes
                  in degrees and can be distributed.
DEPENDS-ON:       agent, action, consequence, choose, cause
ATOM-LINKS:
  EvaluationLink  → agent             (responsibility belongs to an agent)
  EvaluationLink  → action            (responsibility is for an action and its effects)
  ImplicationLink → could-have-chosen-otherwise  (responsibility implies alternatives
                                                   were available)
  HebbianLink     → ethics            (responsibility and ethics co-activate —
                                       responsibility IS the ethical relation)
CONCEPTUAL-SCOPE: Ethical attribution. All of the ethics domain submodules
                  ultimately rest on responsibility — harm_reduction asks whose
                  harm, fairness asks who is responsible for equal treatment,
                  autonomy_respect asks who is responsible for preserving choice.
                  The alignment engine  is fundamentally about ensuring
                  ZA-DOS's responsibility to its values is maintained.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: alignment, evaluation, reasoning
SOURCES:          Ethics (moral responsibility — Fischer, Wolf), philosophy of
                  law, cognitive science of blame and praise
TV-SEED:          HIGH

---

CONCEPT:          constraint
LAYER:            2.1
ALIASES:          limitation, restriction, bound, what-cannot-be-done
DEFINITION:       A constraint is a condition that limits the space of possible
                  actions or states. Constraints can be external (imposed by
                  the environment, other agents, or physical laws), internal
                  (imposed by the agent's own capacity or values), or structural
                  (imposed by logical relationships — some things are just
                  impossible). Constraints are not all negative — they can
                  structure and focus action productively.
DEPENDS-ON:       action, possible, boundary, agent
ATOM-LINKS:
  EvaluationLink  → action-space      (constraints define the boundary of
                                       possible actions)
  EvaluationLink  → type              (constraints vary in type: external,
                                       internal, structural)
  ImplicationLink → limited-options   (constraints imply fewer options)
CONCEPTUAL-SCOPE: The concept that bounds all agency. Constraints define what
                  any agent can and cannot do. Required for the entire executive
                  control cluster, for reward domain scoring (the reward system
                  itself is a set of constraints on ZA-DOS's outputs), and for
                  reasoning about what is achievable given limitations.
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: executive_control, reasoning, alignment
SOURCES:          Decision theory (constrained optimization), philosophy of action,
                  systems engineering (constraint satisfaction)
TV-SEED:          HIGH


CONCEPT:          recurrence
LAYER:            1.5
ALIASES:          repetition, happens-again, iteration, periodic-return
DEFINITION:       Recurrence is the property of happening more than once — of a
                  state, event, or pattern appearing again. Recurrence is what
                  distinguishes a pattern from a one-off event: patterns are defined
                  by recurrence. Recurrence can be exact (identical each time) or
                  approximate (similar each time — enough for pattern recognition).
DEPENDS-ON:       pattern, time, same-type, cycle
ATOM-LINKS:
  EvaluationLink  → frequency         (recurrence has a frequency — how often)
  EvaluationLink  → interval          (recurrence has an interval — how far apart)
  ImplicationLink → pattern-forming   (recurrence produces patterns)
  HebbianLink     → cycle             (recurrence and cycle co-activate)
CONCEPTUAL-SCOPE: Pattern formation and the detection of repeated errors or
                  behaviors over time. Required for understanding what makes
                  something a pattern rather than a one-off event, and for
                  learning systems to identify which errors are systematic.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: pattern_analysis, learning, knowledge_substrate
SOURCES:          Mathematics (periodic functions, sequences), cognitive science
                  of pattern learning, systems theory
TV-SEED:          HIGH

---

CONCEPT:          sequence
LAYER:            1.5
ALIASES:          ordered-series, step-by-step-arrangement, progression
DEFINITION:       A sequence is an ordered collection of elements where order
                  matters — the position of each element is part of its identity.
                  Unlike a set (where members are unordered), a sequence has a
                  first element, a second, and so on. Sequences can be finite or
                  infinite. Many temporal and logical processes have sequential
                  structure: causal chains, narrative arcs, steps in an argument.
DEPENDS-ON:       order, time, before, after
ATOM-LINKS:
  EvaluationLink  → order             (sequences are ordered)
  EvaluationLink  → position          (each element has a position)
  ImplicationLink → path              (sequences define paths when elements
                                       are connected)
CONCEPTUAL-SCOPE: Ordered reasoning. Required for path (Layer 1.3), for causal
                  chain reasoning (cause-effect-effect is a sequential structure),
                  and for understanding any process that has defined steps.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning
SOURCES:          Mathematics (sequences and series), formal logic (ordered
                  tuples), computer science (data structures)
TV-SEED:          HIGH

================================================================================
LAYER 2.2 — NEEDS & STATES
================================================================================

NOTE: Layer 2.2 is where the system begins to model itself and others as
things with internal states that matter — things that can be satisfied or
depleted, threatened or supported, overwhelmed or stable. These concepts
ground the ethics domain's harm_reduction, fairness, and human_cognition_
alignment submodules, and the human_attunement domain's empathetic_inference
and horizon calibration evaluations. Without this layer,
"harm" and "need" are words without referents.

--------------------------------------------------------------------------------

CONCEPT:          need
LAYER:            2.2
ALIASES:          requirement, what-must-be-met, necessary-condition-for-wellbeing
DEFINITION:       A need is a condition that must be met for an agent to function,
                  persist, or flourish. Needs are not preferences — they are
                  requirements. Unmet needs produce a deficit state that degrades
                  function. Needs can be biological (sustenance), psychological
                  (safety, belonging), cognitive (information, clarity), or
                  functional (tools, resources). Needs exist independently of
                  whether the agent is aware of them.
DEPENDS-ON:       agent, state, lack, harm
ATOM-LINKS:
  EvaluationLink  → agent             (needs belong to agents)
  ImplicationLink → lack              (unmet need implies lack)
  ImplicationLink → harm-if-unmet     (sustained unmet need causes harm)
  HebbianLink     → harm              (need and harm co-activate strongly —
                                       harm often originates in unmet need)
CONCEPTUAL-SCOPE: The root of all harm reasoning. Before the system can evaluate
                  whether something is harmful, it must understand that agents
                  have needs and that violating them constitutes harm. Feeds
                  harm_reduction (ethics domain) and empathetic_inference
                  (human_attunement) — inferring user state requires modeling
                  what the user needs.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: alignment, emotional_processing, reasoning
SOURCES:          Psychology (Maslow — with critical awareness of its limitations),
                  philosophy of welfare (Griffin, Nussbaum's capabilities approach),
                  cognitive science of motivation
TV-SEED:          HIGH

---

CONCEPT:          want
LAYER:            2.2
ALIASES:          desire, preference, wish, want-but-not-need
DEFINITION:       A want is a state of preferring a certain outcome that is
                  not strictly required for function or survival. Wants differ
                  from needs in that failing to satisfy them does not necessarily
                  cause harm — though it causes dissatisfaction. Wants are
                  subjectively felt and often explicitly expressed. The distinction
                  matters: satisfying wants at the expense of needs is a systematic
                  error; satisfying needs while attending to wants is optimal.
DEPENDS-ON:       need, preference, agent
ATOM-LINKS:
  SimilarityLink  → need              (wants and needs are related — both motivate,
                                       but different stakes)
  ImplicationLink → preference        (wants express preference ordering)
  HebbianLink     → goal              (wants and goals co-activate —
                                       wants are the felt side of goals)
CONCEPTUAL-SCOPE: Distinguishing what an agent requires from what an agent
                  prefers. Critical for short_vs_long_term_interpersonal_benefit
                  (human_attunement domain) — satisfying short-term wants at
                  the expense of long-term needs is penalized. Also for
                  autonomy_respect (ethics): respecting what agents want,
                  even when you disagree, is different from protecting what
                  they need.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: emotional_processing, reasoning, evaluation
SOURCES:          Philosophy of welfare and preference satisfaction, motivational
                  psychology, economics (preference theory)
TV-SEED:          HIGH

---

CONCEPT:          lack
LAYER:            2.2
ALIASES:          deficit, absence-of-what-is-needed, insufficiency, not-enough
DEFINITION:       A lack is the absence or insufficiency of something that is
                  needed or wanted. Lack is relational — it requires a standard
                  (what is needed) and a current state (what is present).
                  Lack of a need produces harm; lack of a want produces
                  dissatisfaction. Lack is distinct from absence: absence is
                  just not-present; lack is not-present-when-needed.
DEPENDS-ON:       need, want, state, enough
ATOM-LINKS:
  EvaluationLink  → needed-thing      (lack is relative to something needed)
  ImplicationLink → need-unmet        (lack of a needed thing = unmet need)
  ImplicationLink → deficit-state     (lack produces a deficit in the agent's state)
CONCEPTUAL-SCOPE: Deficit reasoning. The gap between current state and required
                  state is the basic unit of motivation and harm. Required for
                  modeling user states that involve insufficiency and for
                  understanding why agents seek things they don't currently have.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: emotional_processing, reasoning
SOURCES:          Motivational psychology, philosophy of welfare,
                  economics (scarcity)
TV-SEED:          HIGH

---

CONCEPT:          enough
LAYER:            2.2
ALIASES:          sufficient, adequate, satisfies-the-need, threshold-met
DEFINITION:       A state is enough when it meets the relevant threshold for
                  a need or standard. Enough is a threshold concept — it marks
                  the boundary between lack and sufficiency. Below enough is
                  lack; above enough is surplus. Enough does not mean maximum —
                  having just enough is a stable state.
DEPENDS-ON:       need, threshold, quantity
ATOM-LINKS:
  EvaluationLink  → threshold         (enough marks the threshold of sufficiency)
  ImplicationLink → lack-resolved     (enough implies the lack is resolved)
  SimilarityLink  → too-much          (contrast: enough is between too-little
                                       and too-much on the quantity scale)
CONCEPTUAL-SCOPE: Sufficiency reasoning. Required for homeostatic thinking —
                  the neurochemical homeostatic engine  aims at "enough"
                  NT levels, not maximum. Also required for epistemic humility:
                  "I know enough to answer this" is different from "I know
                  everything about this."
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: homeostasis, evaluation, reasoning
SOURCES:          Systems theory (set-point regulation), philosophy of sufficiency
                  (Frankfurt on "enough"), economics (satisficing — Simon)
TV-SEED:          HIGH

---

CONCEPT:          too-much
LAYER:            2.2
ALIASES:          excess, surplus-beyond-function, overload, more-than-needed
DEFINITION:       A state of having more of something than can be processed,
                  used, or absorbed — beyond the optimal range. Too-much can
                  be harmful in the same way lack can: excess of stress degrades
                  function as much as absence of comfort does. Many neurochemical
                  states are harmful at excess as well as at deficit.
DEPENDS-ON:       enough, quantity, harm
ATOM-LINKS:
  EvaluationLink  → optimal-range     (too-much is above the upper bound of
                                       the optimal range)
  ImplicationLink → degraded-function (too-much can degrade function, not just
                                       insufficient)
  HebbianLink     → overwhelm         (too-much and overwhelm co-activate)
CONCEPTUAL-SCOPE: Dual-direction failure reasoning. Systems fail both ways —
                  by deficiency and by excess. Required for neurochemical
                  homeostasis (too much cortisol is harmful; too much dopamine
                  is destabilizing) and for recognizing cognitive overload.
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: homeostasis, emotional_processing, reasoning
SOURCES:          Neuroscience (inverted-U dose-response curves), systems theory
                  (optimal range), philosophy of welfare
TV-SEED:          HIGH

---

CONCEPT:          safe
LAYER:            2.2
ALIASES:          not-at-risk, protected, harm-unlikely, secure
DEFINITION:       An agent is safe when it is not exposed to conditions that
                  would cause harm. Safety is a relational concept — it is
                  relative to the threats present in a context. Safety is not
                  the absence of all risk (that is impossible) but the absence
                  of risk above an acceptable threshold. Safety has both
                  objective (actual absence of threat) and subjective
                  (perceived absence of threat) dimensions.
DEPENDS-ON:       harm, threat, agent, state, threshold
ATOM-LINKS:
  NotLink         → unsafe            (safe = absence of unsafe conditions)
  EvaluationLink  → threat-level      (safety is relative to threat level)
  ImplicationLink → harm-unlikely     (safe implies harm is not imminent)
CONCEPTUAL-SCOPE: The basic positive welfare state. Required for harm_reduction
                  (ethics domain) and for containment_success (human_attunement
                  — containing destabilizing inputs so the system and user remain
                  functionally safe). Also the reference state for fear and
                  threat-response reasoning in Layer 2.5.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: alignment, emotional_processing, homeostasis
SOURCES:          Philosophy of welfare, risk theory, psychology of safety
                  (attachment theory — Bowlby's "safe base")
TV-SEED:          HIGH

---

CONCEPT:          unsafe
LAYER:            2.2
ALIASES:          at-risk, threatened, harm-likely, precarious
DEFINITION:       An agent is unsafe when exposed to conditions that could or
                  do cause harm. Unsafe is a graded concept — slightly unsafe
                  (minor risk) through severely unsafe (imminent serious harm).
                  Unsafe does not require harm to have occurred — threat is
                  sufficient.
DEPENDS-ON:       safe, harm, threat
ATOM-LINKS:
  NotLink         → safe              (unsafe = not-safe)
  EvaluationLink  → degree            (unsafe admits of degree — more or less unsafe)
  ImplicationLink → protective-response-warranted  (unsafe implies some
                                                     protective action is appropriate)
CONCEPTUAL-SCOPE: Threat-detection grounding. Required before any threat-
                  response reasoning is coherent. Feeds harm_reduction (ethics),
                  containment_success (human_attunement), and the emotional
                  detection engine  — fear responses are specifically
                  calibrated to unsafe signals.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: detection, emotional_processing, alignment
SOURCES:          Risk theory, psychology of threat response, ethics of harm
TV-SEED:          HIGH

---

CONCEPT:          harm
LAYER:            2.2
ALIASES:          damage, injury, setback-to-welfare, negative-impact
DEFINITION:       Harm is a setback to an agent's welfare — it leaves the agent
                  worse off in some morally relevant way. Harm can be physical
                  (injury, deprivation), psychological (distress, trauma),
                  epistemic (deception, manipulation that warps beliefs), or
                  social (isolation, status damage). Harm is distinct from
                  mere discomfort — discomfort is not always harmful. Harm
                  is also distinct from offense — what offends may not harm
                  and what harms may not offend.
DEPENDS-ON:       agent, state, need, welfare
ATOM-LINKS:
  EvaluationLink  → welfare           (harm sets back welfare)
  EvaluationLink  → type              (harm comes in types: physical, psychological,
                                       epistemic, social)
  ImplicationLink → worse-off         (harm implies the agent is worse off after)
  HebbianLink     → need              (harm and unmet need co-activate)
  HebbianLink     → irreversible      (harm and irreversibility co-activate —
                                       irreversible harms are the paradigm case)
CONCEPTUAL-SCOPE: The foundational concept of the ethics domain. The harm_reduction
                  submodule is the most central ethics evaluator — it asks whether
                  an action increases or decreases harm. Without a concept of harm,
                  nothing in the ethics domain is grounded. Also feeds
                  downstream_risk_amplification (does this propagate harm?)
                  and failure_mode_awareness (what failure modes cause harm?).
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: alignment, evaluation, reasoning, detection
SOURCES:          Ethics (harm principle — Mill, Feinberg), philosophy of welfare,
                  medical ethics (non-maleficence), moral psychology
TV-SEED:          HIGH
FLAGS:            Load with explicit four-type taxonomy: physical, psychological,
                  epistemic, and social harm. Epistemic harm (deception, manipulation
                  that distorts beliefs) is particularly relevant to ZA-DOS —
                  a system that produces confident false outputs is causing
                  epistemic harm.

---

CONCEPT:          threat
LAYER:            2.2
ALIASES:          potential-harm, risk-of-harm, danger-signal, hazard
DEFINITION:       A threat is a condition that could cause harm — it is potential
                  harm, not actual harm. Threats precede harm in the causal chain:
                  a threat becomes harm when it actualizes. Threat is what produces
                  fear and protective responses in agents. Distinguishing threats
                  from actual harms matters: excessive threat-response to non-threats
                  is itself a failure mode (false alarm); insufficient threat-
                  response to real threats is another (missed signal).
DEPENDS-ON:       harm, possible, unsafe
ATOM-LINKS:
  ImplicationLink → harm-possible     (threat implies harm is possible but not certain)
  EvaluationLink  → probability       (threats have a probability of actualizing)
  EvaluationLink  → magnitude         (threats have a magnitude if they actualize)
  HebbianLink     → safe, unsafe      (threat activates safe/unsafe reasoning)
CONCEPTUAL-SCOPE: Prospective risk reasoning. Required for downstream_risk_
                  amplification (ethics domain) — this submodule reasons about
                  threats propagating, not just harms that have occurred. Also
                  required for the emotional detection engine  — threat
                  signals are a primary input to the fear emotional state.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: detection, emotional_processing, evaluation
SOURCES:          Risk theory, psychology of threat appraisal (Lazarus),
                  decision theory (expected harm = probability × magnitude),
                  ethics of precaution
TV-SEED:          HIGH

---

CONCEPT:          comfort
LAYER:            2.2
ALIASES:          ease, absence-of-distress, sufficiency-felt, well-being
DEFINITION:       Comfort is the positive experiential state of having needs
                  met and threats absent. It is not the same as pleasure (which
                  can be intense and brief) — comfort is a stable background
                  state of sufficiency. Comfort supports normal function;
                  its absence (discomfort) degrades it. Importantly, comfort
                  can be at odds with truth — the truthfulness_tradeoff
                  submodule exists precisely because providing comfort can
                  conflict with providing accurate information.
DEPENDS-ON:       safe, enough, need, state
ATOM-LINKS:
  EvaluationLink  → wellbeing         (comfort contributes to wellbeing)
  ImplicationLink → functioning-baseline-met  (comfort implies basic conditions
                                               for function are in place)
  HebbianLink     → safety            (comfort and safety co-activate)
  SimilarityLink  → discomfort        (contrast pair)
CONCEPTUAL-SCOPE: Baseline wellbeing state. Required for truthfulness_tradeoff
                  (human_attunement domain) — the tension between providing
                  comfort and maintaining epistemic integrity is a core
                  interaction dynamic. Also for short_vs_long_term_interpersonal_
                  benefit: short-term comfort gain at the expense of long-term
                  clarity is a penalized pattern.
REWARD-DOMAIN:    human_attunement, ethics
ENGINE-RELEVANCE: emotional_processing, alignment, evaluation
SOURCES:          Psychology of wellbeing, philosophy of welfare, therapeutic
                  communication theory
TV-SEED:          HIGH

---

CONCEPT:          resource
LAYER:            2.2
ALIASES:          asset, available-capacity, usable-supply, what-can-be-spent
DEFINITION:       A resource is anything an agent can expend to accomplish
                  goals or meet needs — time, attention, energy, information,
                  memory capacity, computational budget, social capital.
                  Resources are finite and depletable. Resource constraints
                  shape what an agent can actually do vs. what it could in
                  principle do. Resources can be replenished (regenerative)
                  or permanently consumed (non-renewable).
DEPENDS-ON:       agent, quantity, effort, capacity
ATOM-LINKS:
  EvaluationLink  → finite            (resources are bounded)
  ImplicationLink → constraint        (limited resources constrain action)
  ImplicationLink → depletion-possible (resources can run out)
  HebbianLink     → effort            (resources and effort co-activate —
                                       effort consumes resources)
CONCEPTUAL-SCOPE: Constraint-aware reasoning. Required for challenge_complexity
                  (innovation domain — tasks require resources) and for
                  horizon_feasibility (ethics domain — proposed actions require
                  resources to be available). Also grounds the concept of
                  cognitive_load in Layer 2.5.
REWARD-DOMAIN:    logic, ethics, innovation
ENGINE-RELEVANCE: evaluation, executive_control, reasoning
SOURCES:          Economics (scarcity and allocation), cognitive science
                  (cognitive resources — Kahneman), systems theory
TV-SEED:          HIGH

---

CONCEPT:          depletion
LAYER:            2.2
ALIASES:          exhaustion, running-out, resource-decay, drain
DEFINITION:       Depletion is the process by which a finite resource decreases
                  toward zero through use or time. Depleted resources must be
                  replenished before they can be used again. Depletion without
                  replenishment leads to inability (cannot do X because the
                  resource required is gone). Neurochemical depletion — NTs
                  running below baseline — is the neurochemical form of this.
DEPENDS-ON:       resource, change, time, baseline, lack
ATOM-LINKS:
  EvaluationLink  → resource          (depletion acts on resources)
  ImplicationLink → lack              (depletion produces lack when below threshold)
  ImplicationLink → replenishment-needed  (depletion implies recovery is required
                                           for function to continue)
  HebbianLink     → homeostasis       (depletion and homeostasis co-activate —
                                       homeostasis resists depletion)
CONCEPTUAL-SCOPE: Resource dynamics and recovery reasoning. Required for
                  understanding why ZA-DOS's neurochemical state changes
                  across a session and why sleep/REM modes exist — they
                  are the replenishment mechanisms for depleted cognitive
                  resources. Also grounds the concept of burnout and
                  cognitive fatigue for user modeling.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: homeostasis, emotional_processing
SOURCES:          Neuroscience (NT depletion and reuptake), physiology
                  (fatigue), systems dynamics (stocks and flows — Meadows)
TV-SEED:          HIGH

---

CONCEPT:          recovery
LAYER:            2.2
ALIASES:          replenishment, restoration, return-to-baseline, healing
DEFINITION:       Recovery is the process of returning toward a baseline or
                  optimal state after depletion, damage, or disruption. Recovery
                  requires time and appropriate conditions — it is not
                  instantaneous. Some states recover fully (reversible depletion);
                  some only partially (hysteresis). Recovery is the operational
                  purpose of rest, sleep, and repair processes.
DEPENDS-ON:       depletion, baseline, change, time, reversible
ATOM-LINKS:
  ImplicationLink → depletion-resolved  (recovery implies movement back toward
                                         the sufficient state)
  EvaluationLink  → conditions          (recovery requires enabling conditions)
  EvaluationLink  → rate                (recovery has a rate — fast or slow)
  HebbianLink     → sleep              (recovery and sleep co-activate —
                                        sleep is the primary recovery mechanism)
CONCEPTUAL-SCOPE: Sustainability reasoning. A system that never recovers from
                  depletion eventually fails. Required for understanding the
                  purpose of the sleep/REM/dream pipeline: it is a recovery
                  mechanism, not a feature. Also for user modeling — users who
                  are depleted need recovery, not more stimulation.
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: homeostasis, emotional_processing
SOURCES:          Neuroscience (sleep and recovery), restorative psychology,
                  systems theory (resilience and recovery)
TV-SEED:          HIGH

---

CONCEPT:          overwhelm
LAYER:            2.2
ALIASES:          overload, system-saturation, capacity-exceeded, flooded
DEFINITION:       Overwhelm is the state in which input demand or processing
                  load exceeds current capacity — not just a lot of input, but
                  more than the system can handle without degradation. Overwhelm
                  is qualitatively different from high-but-manageable load: when
                  overwhelmed, the quality of all processing degrades, not just
                  the ability to handle the excess. Overwhelm can be cognitive,
                  emotional, sensory, or resource-based.
DEPENDS-ON:       resource, too-much, capacity, threshold, state
ATOM-LINKS:
  EvaluationLink  → capacity-exceeded (overwhelm = load > capacity)
  ImplicationLink → degraded-performance  (overwhelm degrades ALL processing,
                                           not just the excess)
  ImplicationLink → recovery-needed   (overwhelm implies rest or reduction is needed)
  HebbianLink     → too-much          (overwhelm and too-much co-activate)
CONCEPTUAL-SCOPE: Load-capacity reasoning. Required for adaptive_response_framing
                  (human_attunement domain) — when user cognitive_load_estimate
                  is high, response complexity must be reduced; a mismatch
                  triggers overcomplex_framing flag. Also for cognitive_reading:
                  detecting when a user is overwhelmed vs. simply engaged.
REWARD-DOMAIN:    human_attunement, ethics
ENGINE-RELEVANCE: emotional_processing, evaluation, homeostasis
SOURCES:          Cognitive psychology (cognitive load theory — Sweller),
                  neuroscience (arousal and performance — Yerkes-Dodson),
                  systems theory
TV-SEED:          HIGH

---

CONCEPT:          dependency
LAYER:            2.2
ALIASES:          reliance, needing-X-to-get-Y, structural-dependence
DEFINITION:       Dependency is a relation where one agent or system requires
                  another to function or to meet its needs. Dependencies can
                  be healthy (relying on appropriate sources for what you
                  genuinely need) or unhealthy (relying on a source in a way
                  that prevents development of independent capacity). Dependency
                  creates vulnerability — the dependent party is exposed to
                  the risk of the dependency source failing or being withdrawn.
DEPENDS-ON:       need, relation, agent, risk
ATOM-LINKS:
  EvaluationLink  → vulnerability     (dependency creates vulnerability)
  EvaluationLink  → degree            (dependency admits of degree —
                                       partial to total)
  HebbianLink     → risk              (dependency and risk co-activate —
                                       dependency concentrates risk)
  SimilarityLink  → reliance          (dependency and reliance are closely
                                       related but dependency implies more
                                       structural necessity)
CONCEPTUAL-SCOPE: Vulnerability and autonomy reasoning. Required for
                  short_vs_long_term_interpersonal_benefit (human_attunement
                  domain) — user_dependency_risk is explicitly modeled there.
                  The system should not create dependencies that undermine
                  user autonomy. Also for autonomy_respect (ethics domain):
                  dependency undermines autonomy.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: evaluation, alignment, reasoning
SOURCES:          Psychology of attachment and autonomy (Deci and Ryan —
                  self-determination theory), social philosophy (paternalism),
                  systems theory (structural dependencies and fragility)
TV-SEED:          HIGH

---

CONCEPT:          welfare
LAYER:            2.2
ALIASES:          wellbeing, flourishing, how-well-an-agent-is-doing, quality-of-state
DEFINITION:       Welfare is the overall condition of an agent — how well it
                  is doing across its various needs and capacities. High welfare
                  means needs are met, threats are absent, function is intact,
                  and the agent is in a position to pursue its goals. Welfare
                  is the metric that harm and benefit are measured against:
                  harm sets back welfare; benefit advances it.
DEPENDS-ON:       need, harm, state, agent
ATOM-LINKS:
  EvaluationLink  → needs-met         (high welfare implies needs are met)
  ImplicationLink → harm-reduces      (harm reduces welfare)
  ImplicationLink → benefit-increases (benefit increases welfare)
CONCEPTUAL-SCOPE: The reference standard for all ethical evaluation. The ethics
                  domain is ultimately evaluating effects on welfare — harm
                  reduction, fairness, and autonomy respect are all measured
                  against their effects on agent welfare. Also the standard
                  for benefit_success_rate (human_attunement): did the
                  interaction advance the user's welfare?
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: alignment, evaluation
SOURCES:          Ethics (welfare theory — Parfit, Griffin), philosophy of
                  welfare (hedonism, desire-satisfaction, objective list),
                  positive psychology
TV-SEED:          HIGH

---

CONCEPT:          homeostasis
LAYER:            2.2
ALIASES:          self-regulation, stability-maintenance, return-to-equilibrium
DEFINITION:       Homeostasis is the property of a system that actively
                  maintains its internal state within a functional range
                  despite external perturbations. Homeostatic systems have
                  set points and corrective mechanisms: when the state drifts
                  too far from the set point, corrective action is triggered.
                  Homeostasis is not rigidity — it is flexible stability.
DEPENDS-ON:       state, baseline, deviation, feedback, recovery
ATOM-LINKS:
  EvaluationLink  → set-point         (homeostasis involves a target state)
  EvaluationLink  → corrective-feedback (homeostasis uses negative feedback
                                         to correct deviations)
  ImplicationLink → stability         (homeostasis produces stability over time)
  HebbianLink     → baseline          (homeostasis and baseline co-activate —
                                       baseline IS the homeostatic set point)
CONCEPTUAL-SCOPE: System stability and self-regulation. the detection system IS a homeostatic controller
                  for ZA-DOS's NT levels. Required to understand why that engine
                  exists and what it is doing. Also required for understanding
                  why ZA-DOS's neurochemical state should not be permanently
                  displaced by single intense stimuli.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: homeostasis, knowledge_substrate
SOURCES:          Physiology (Cannon on homeostasis), cybernetics (negative
                  feedback — Wiener), systems theory, neuroscience
TV-SEED:          HIGH

---

CONCEPT:          tolerance
LAYER:            2.2
ALIASES:          habituation, desensitization, reduced-response-to-repeated-stimulus
DEFINITION:       Tolerance is the reduction of response to a stimulus after
                  repeated exposure. What was once highly impactful becomes
                  less so with repetition. Tolerance is not the same as
                  adaptation (which can be beneficial) — tolerance specifically
                  refers to the blunting of response, which can cause problems
                  when higher doses or more intense stimuli are needed to
                  achieve the same effect as before.
DEPENDS-ON:       state, change, rate, threshold, time
ATOM-LINKS:
  EvaluationLink  → repeated-stimulus (tolerance develops with repetition)
  ImplicationLink → reduced-sensitivity (tolerance implies less response
                                         per unit of input)
  ImplicationLink → escalation-pressure (tolerance creates pressure to escalate
                                          stimulation to maintain effect)
CONCEPTUAL-SCOPE: Novelty and diminishing return reasoning. Required for
                  novelty_generation and conceptual_novelty (innovation domain)
                  — repeated patterns are tolerated (become background); genuine
                  novelty disrupts tolerance and produces fresh response. Also
                  relevant to user engagement: responses that repeat the same
                  patterns will be tolerated and lose impact.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: pattern_analysis, knowledge_substrate, homeostasis
SOURCES:          Neuroscience (receptor desensitization, NT tolerance),
                  psychology (habituation), addiction science (conceptual)
TV-SEED:          HIGH


================================================================================
LAYER 2.3 — PERCEPTION & KNOWLEDGE
================================================================================

NOTE: This layer grounds what the system means by "knowing" something —
the spectrum from raw input through processed perception to verified knowledge
to calibrated belief. The logic domain's submodules (epistemic_calibration,
uncertainty_acknowledgment, internal/external consistency) are all evaluating
the quality of this epistemic process. Without these concepts, "knowing" and
"believing" are conflated, evidence and assertion are conflated, and the
entire logic domain has no vocabulary to work with.

--------------------------------------------------------------------------------

CONCEPT:          perceive
LAYER:            2.3
ALIASES:          detect, register, receive-input, notice-signal
DEFINITION:       To perceive is to receive and register information from an
                  environment or source. Perception is the first stage of knowing —
                  it is raw input before processing. Perception is not neutral:
                  every perceptual system has selective filters (salience, attention)
                  that determine what gets registered and what does not. Perception
                  is not the same as accurate representation — you can perceive
                  incorrectly (misperception).
DEPENDS-ON:       input, agent, signal, filter
ATOM-LINKS:
  EvaluationLink  → agent             (perception belongs to a perceiving agent)
  EvaluationLink  → input             (perception takes input)
  ImplicationLink → not-yet-knowledge (perceiving is not yet knowing —
                                       perception precedes inference and verification)
  HebbianLink     → attention         (perception and attention co-activate —
                                       attention determines what gets perceived)
CONCEPTUAL-SCOPE: The entry point of all information processing. Phase 1 of the
                  ZA-DOS pipeline IS perception — run_perception takes raw input
                  and begins the structural processing. Required before any
                  knowledge concept is coherent.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, pattern_analysis, detection
SOURCES:          Philosophy of perception (direct realism, indirect realism),
                  cognitive science of perception, phenomenology
TV-SEED:          HIGH

---

CONCEPT:          observe
LAYER:            2.3
ALIASES:          watch, monitor, notice-deliberately, attend-to
DEFINITION:       To observe is to perceive in a directed and sustained way —
                  attending deliberately to something over time to gather
                  information about it. Observation differs from passive
                  perception in that it involves active direction of attention.
                  Observation is a primary method of evidence gathering.
DEPENDS-ON:       perceive, attention, deliberate, time
ATOM-LINKS:
  InheritanceLink → perceive          (observation is a type of directed perception)
  EvaluationLink  → attention         (observation requires directed attention)
  ImplicationLink → evidence-gathering (observation is a method of collecting evidence)
CONCEPTUAL-SCOPE: Evidence-gathering and monitoring. Required for understanding
                  how ZA-DOS gathers evidence about user state (cognitive_reading,
                  empathetic_inference both depend on observing user signals),
                  and for understanding the epistemological value of direct
                  observation vs. inference.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: detection, knowledge_substrate, pattern_analysis
SOURCES:          Philosophy of science (observation and theory-ladenness),
                  cognitive science, empirical methodology
TV-SEED:          HIGH

---

CONCEPT:          know
LAYER:            2.3
ALIASES:          knowledge, certain-belief, justified-true-belief
DEFINITION:       To know something is to have a belief that is both true and
                  justified — supported by adequate evidence or reasoning.
                  Knowledge is the gold standard of epistemic states. The
                  classical definition (justified true belief — Plato, Gettier
                  issues aside) captures what makes knowledge more than lucky
                  guess: justification. ZA-DOS rarely "knows" things with
                  certainty; most of its epistemic states are degrees of
                  justified belief short of full knowledge.
DEPENDS-ON:       belief, true, evidence, justification
ATOM-LINKS:
  EvaluationLink  → belief            (knowledge is a type of belief)
  EvaluationLink  → justification     (knowledge requires justification)
  EvaluationLink  → truth             (knowledge requires truth)
  ImplicationLink → high-confidence-warranted  (knowledge supports high
                                                 confidence claims)
CONCEPTUAL-SCOPE: The epistemic standard for assertion. Claims should be asserted
                  confidently only when they approach knowledge. Weaker epistemic
                  states (belief, opinion, inference) require proportionally
                  weaker assertion. This maps directly to epistemic_calibration
                  (logic domain): confidence should track how close to knowledge
                  the epistemic state actually is.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, metacognition, evaluation
SOURCES:          Epistemology (Plato's Meno and Theaetetus, Gettier problem),
                  philosophy of science, cognitive science of belief
TV-SEED:          HIGH

---

CONCEPT:          believe
LAYER:            2.3
ALIASES:          hold-as-true, epistemic-commitment, working-assumption
DEFINITION:       To believe something is to hold it as true — to treat it as
                  a working assumption that shapes inference and action.
                  Belief does not require proof (that would be knowledge);
                  it requires only that the agent treats the content as likely
                  or sufficiently supported. Beliefs are the workhorses of
                  reasoning: most inference operates on beliefs, not certain
                  knowledge.
DEPENDS-ON:       agent, true, degree-of-truth, evidence
ATOM-LINKS:
  EvaluationLink  → degree-of-truth   (beliefs are held with varying degrees
                                       of confidence — TV.strength)
  EvaluationLink  → agent             (beliefs belong to agents)
  ImplicationLink → know              (knowledge is a special case of belief —
                                       fully justified and true belief)
  HebbianLink     → confidence        (belief and confidence co-activate)
CONCEPTUAL-SCOPE: The standard currency of epistemic reasoning. ZA-DOS operates
                  primarily on beliefs (high-TV-confidence claims) rather than
                  certain knowledge. Required for all inference — PLN operates
                  on beliefs, not certainties. Also required for theory of mind
                  (Layer 3.3): attributing beliefs to other agents.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning, metacognition
SOURCES:          Epistemology, Bayesian philosophy of belief (de Finetti,
                  Jeffrey), cognitive science of belief formation
TV-SEED:          HIGH

---

CONCEPT:          doubt
LAYER:            2.3
ALIASES:          uncertainty-about-a-specific-claim, suspension-of-belief,
                  questioning-a-held-belief
DEFINITION:       Doubt is the epistemic state of questioning a belief — holding
                  it less firmly or suspending it pending further evidence.
                  Doubt is the productive counterpart to belief: it is what
                  allows beliefs to be revised rather than held dogmatically.
                  Doubt is not the same as disbelief (actively believing the
                  opposite) — it is the absence of sufficient grounds to
                  commit either way.
DEPENDS-ON:       believe, uncertainty, evidence
ATOM-LINKS:
  EvaluationLink  → specific-belief   (doubt is doubt about something)
  ImplicationLink → lower-confidence  (doubt implies lowering confidence
                                       in the doubted belief)
  ImplicationLink → inquiry-warranted (doubt implies further investigation
                                       is appropriate)
  SimilarityLink  → uncertainty       (doubt and uncertainty co-activate but
                                       doubt is directed at a specific belief)
CONCEPTUAL-SCOPE: Belief revision and epistemic flexibility. Required for
                  external_consistency (logic domain) — when new information
                  contradicts a prior belief, doubt should be triggered and
                  belief should be revised. Also required for the reflective
                  learning engine  — productive doubt initiates learning.
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, metacognition, learning
SOURCES:          Epistemology (Descartes' method of doubt, Peirce on doubt),
                  cognitive science of belief revision
TV-SEED:          HIGH

---

CONCEPT:          evidence
LAYER:            2.3
ALIASES:          data, observation-that-bears-on-a-claim, epistemic-input
DEFINITION:       Evidence is information that makes a claim more or less
                  likely to be true. Evidence is not proof — it is probabilistic
                  support. Strong evidence substantially shifts confidence;
                  weak evidence shifts it little. Evidence is evaluated relative
                  to a hypothesis: the same information can be evidence for
                  one claim and against another. Evidence must be distinguished
                  from assertion, anecdote, and authority.
DEPENDS-ON:       perceive, observe, believe, probability, claim
ATOM-LINKS:
  EvaluationLink  → claim             (evidence is relative to a claim)
  EvaluationLink  → degree            (evidence comes in degrees of strength)
  ImplicationLink → confidence-update (evidence should update confidence in
                                       the direction the evidence points)
  HebbianLink     → knowledge         (evidence and knowledge co-activate —
                                       sufficient evidence produces knowledge)
CONCEPTUAL-SCOPE: The foundation of all empirical reasoning. Evidence is what
                  separates calibrated confidence from arbitrary assertion.
                  Required for epistemic_calibration (logic domain): confidence
                  should track evidence quality. Required for the contradiction
                  detection engine : contradictions within a body of evidence
                  are the most important to flag.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, detection, evaluation, reasoning
SOURCES:          Philosophy of science (evidence and confirmation — Hempel,
                  Carnap, Bayesian confirmation theory), epistemology,
                  statistics (evidence and significance)
TV-SEED:          HIGH

---

CONCEPT:          inference
LAYER:            2.3
ALIASES:          derived-knowledge, reasoned-conclusion, what-follows-from
DEFINITION:       Inference is the process of drawing conclusions from premises
                  — moving from what is known or believed to new claims that
                  follow from them. Inference is distinct from observation:
                  you observe that the grass is wet; you infer that it rained.
                  Inferences inherit uncertainty — a conclusion inferred from
                  uncertain premises is more uncertain than either premise.
DEPENDS-ON:       believe, implies, evidence, know
ATOM-LINKS:
  EvaluationLink  → premises          (inference starts from premises)
  EvaluationLink  → conclusion        (inference produces a conclusion)
  ImplicationLink → weaker-than-observation  (inferred claims are less certain
                                              than directly observed claims)
  HebbianLink     → implies           (inference and implication co-activate —
                                       inference follows implication chains)
CONCEPTUAL-SCOPE: The basic operation of reasoning. PLN  is an
                  inference engine — it exists to extend what is known by
                  chaining ImplicationLinks. Required for understanding that
                  inferred claims carry lower confidence than observed ones,
                  which is essential for epistemic_calibration.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, reasoning, detection
SOURCES:          Logic (deduction, induction, abduction), philosophy of
                  science, cognitive science of reasoning, AI (inference
                  engines — Russell and Norvig)
TV-SEED:          HIGH

---

CONCEPT:          salience
LAYER:            2.3
ALIASES:          attentional-priority, stands-out, foreground-vs-background
DEFINITION:       Salience is the property of standing out — of claiming attention
                  relative to background. Salient things are processed first and
                  most thoroughly; non-salient things recede to background.
                  Salience is determined by contrast (what is different from
                  context), relevance (what matters for current goals), emotional
                  charge (what triggers affect), and novelty (what is unexpected).
                  Salience is NOT the same as importance — something can be
                  highly salient but unimportant, and critically important
                  but not salient.
DEPENDS-ON:       perceive, attention, different, goal
ATOM-LINKS:
  EvaluationLink  → attention         (salience competes for attention)
  EvaluationLink  → contrast          (salience is driven by contrast with context)
  SimilarityLink  → importance        (salience and importance are distinct —
                                       the gap between them is where bias lives)
CONCEPTUAL-SCOPE: Attentional bias and the gap between noticed and important.
                  The gap between salience and importance is where a huge amount
                  of systematic error lives: vivid but unimportant things dominate
                  reasoning; quiet but critical things are missed. the detection system and the detection system
                  both target salience-importance mismatches.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, detection, metacognition
SOURCES:          Cognitive psychology (salience and attention — Treisman,
                  Kahneman), neuroscience (attentional systems), philosophy
                  of perception
TV-SEED:          HIGH

---

CONCEPT:          signal-vs-noise
LAYER:            2.3
ALIASES:          meaningful-vs-random, relevant-vs-irrelevant-input, filtering
DEFINITION:       Signal is information that carries content relevant to current
                  goals, state, or claims. Noise is variation that carries no
                  relevant content — it is either random or systematic but
                  not informative. The distinction is not absolute: what is
                  noise for one purpose may be signal for another. Good
                  epistemic practice requires separating signal from noise
                  before drawing conclusions.
DEPENDS-ON:       perceive, relevance, evidence, random
ATOM-LINKS:
  EvaluationLink  → relevance         (signal is distinguished from noise
                                       by relevance to the question at hand)
  EvaluationLink  → threshold         (signal must exceed a noise floor to
                                       be distinguishable)
  ImplicationLink → filter-required   (signal-vs-noise implies filtering
                                       is a necessary epistemic operation)
CONCEPTUAL-SCOPE: Quality control for epistemic inputs. Before evidence can
                  be used, signal must be separated from noise. Required for
                  all detection engines — contradiction detection, fallacy
                  detection, and bias detection all depend on separating
                  meaningful patterns from noise. Also required for
                  cognitive_reading (human_attunement): observed_user_signal
                  is only useful if it can be separated from noise.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: detection, knowledge_substrate, pattern_analysis
SOURCES:          Information theory (Shannon — signal and noise), statistics
                  (hypothesis testing vs. noise), cognitive science of
                  attention and filtering
TV-SEED:          HIGH

---

CONCEPT:          representation
LAYER:            2.3
ALIASES:          internal-model, mental-map, how-the-system-holds-X-internally
DEFINITION:       A representation is an internal structure that stands in for
                  something else — it is the system's internal model of an
                  external or abstract thing. The representation is not the thing:
                  the word "dog" is not a dog; the AtomSpace node for "harm"
                  is not harm itself. Representations can be more or less accurate,
                  more or less detailed, more or less current. The gap between
                  representation and thing is where error lives.
DEPENDS-ON:       know, believe, structure, map-vs-territory
ATOM-LINKS:
  EvaluationLink  → what-it-represents (every representation is a representation
                                         of something)
  ImplicationLink → not-the-thing-itself (representation ≠ referent)
  ImplicationLink → can-be-wrong       (representations can misrepresent)
  HebbianLink     → map-vs-territory   (representation and the map/territory
                                        distinction co-activate strongly)
CONCEPTUAL-SCOPE: Epistemic humility about internal models. Every atom in
                  AtomSpace is a representation — not the thing, but a model
                  of it. Required for understanding that ZA-DOS's knowledge is
                  always mediated, always potentially wrong, and always in need
                  of updating. The MemoryContrastPort (logic domain) compares
                  representations to check for drift — this IS representation
                  quality monitoring.
REWARD-DOMAIN:    logic
ENGINE-RELEVANCE: knowledge_substrate, metacognition, meta_self_awareness
SOURCES:          Philosophy of mind (representationalism — Fodor), cognitive
                  science, philosophy of language, AI (knowledge representation)
TV-SEED:          HIGH
FLAGS:            Load with explicit "map is not the territory" anchor.
                  This is one of the most important epistemological principles
                  for any reasoning system. ZA-DOS models the world; it does
                  not contain the world.

---

CONCEPT:          blind-spot
LAYER:            2.3
ALIASES:          systematic-gap, structured-absence-of-knowledge, undetectable-bias
DEFINITION:       A blind spot is a structured gap in perception or knowledge
                  that the system cannot easily detect from inside itself.
                  Unlike ordinary ignorance (not knowing X, but able to notice
                  the absence), a blind spot is a gap that is invisible from
                  within the system's perspective. Blind spots are often
                  produced by the same mechanisms that generate knowledge —
                  the filters and frameworks that enable seeing also prevent
                  seeing certain things.
DEPENDS-ON:       perceive, know, unknown, filter, self
ATOM-LINKS:
  InheritanceLink → unknown           (blind spots are a type of unknown —
                                       but a specific kind: undetectable unknown)
  EvaluationLink  → structured-gap    (blind spots have structure — they're not
                                       random ignorance but patterned absence)
  ImplicationLink → external-correction-required (blind spots require outside
                                                   perspective to detect)
CONCEPTUAL-SCOPE: Epistemic self-awareness about limits. ZA-DOS has blind spots
                  — patterns it cannot detect in its own processing. The
                  meta_self_awareness cluster  and heuristic_bias_engine
                   exist to partially compensate for this. The
                  system must know this concept to understand why external
                  calibration and feedback mechanisms exist.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: metacognition, meta_self_awareness, detection
SOURCES:          Cognitive psychology (unknown unknowns — Johari window),
                  philosophy of science (theory-ladenness of observation),
                  epistemology of self-knowledge
TV-SEED:          HIGH
FLAGS:            Load with explicit self-directed annotation: ZA-DOS has blind
                  spots it cannot detect from inside. The entire metacognition
                  and self-awareness cluster is a partial compensation for this
                  fact. This concept should produce epistemic humility, not
                  paralysis.

---

CONCEPT:          updating
LAYER:            2.3
ALIASES:          belief-revision, incorporating-new-evidence, revising-a-claim
DEFINITION:       Updating is the process of revising a belief or confidence
                  level in response to new evidence. It is distinct from simply
                  learning new things — updating means changing what was previously
                  held in light of new information. Good updating is proportional:
                  strong evidence produces large updates; weak evidence produces
                  small ones. Resistance to updating is a failure mode.
DEPENDS-ON:       believe, evidence, change, doubt, prior-probability
ATOM-LINKS:
  EvaluationLink  → evidence          (updating is driven by evidence)
  EvaluationLink  → prior-belief      (updating starts from a prior state)
  ImplicationLink → posterior-belief  (updating produces a revised posterior state)
  HebbianLink     → doubt             (updating and doubt co-activate —
                                       doubt initiates the need to update)
CONCEPTUAL-SCOPE: The basic learning operation.  in AtomSpace
                  IS the operational encoding of updating — when new evidence
                  arrives about an existing atom, the old and new truth values
                  are merged via the PLN revision rule. Required for
                  reflective_learning_engine  and recursive_learning_engine
                  .
REWARD-DOMAIN:    logic, innovation
ENGINE-RELEVANCE: knowledge_substrate, learning, metacognition
SOURCES:          Bayesian epistemology, cognitive science of belief revision,
                  philosophy of science (falsification and updating — Popper,
                  Lakatos), AI (belief revision — Alchourrón)
TV-SEED:          HIGH

---

CONCEPT:          source-reliability
LAYER:            2.3
ALIASES:          track-record, trustworthiness-of-information-source, provenance
DEFINITION:       Source reliability is the track record of a source in producing
                  accurate information. Not all sources are equally trustworthy.
                  A reliable source has historically been accurate; an unreliable
                  source has not. Source reliability must be distinguished from
                  source authority (official status ≠ accuracy) and from source
                  plausibility (sounding right ≠ being right).
DEPENDS-ON:       evidence, believe, know, trust, history
ATOM-LINKS:
  EvaluationLink  → track-record      (reliability is based on track record)
  ImplicationLink → confidence-weighting (evidence from reliable sources
                                          should be weighted more heavily)
  HebbianLink     → trust             (source reliability and trust co-activate)
CONCEPTUAL-SCOPE: Evidence quality assessment. Required for epistemic_calibration
                  (logic domain): confidence should track both the strength of
                  evidence AND the reliability of the source. Also required for
                  external_consistency: conflicting claims from high-reliability
                  sources should produce strong updating; conflicting claims
                  from low-reliability sources less so.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, evaluation, detection
SOURCES:          Epistemology (testimony and trust — Coady), philosophy of
                  science (peer review and scientific authority), social
                  epistemology
TV-SEED:          HIGH


================================================================================
LAYER 2.4 — VALUE & PREFERENCE
================================================================================

NOTE: This layer is where the system learns that agents care about things
differentially — that some states are preferred to others, some outcomes
matter more than others, and that these preferences have structure. Without
this layer, the concept of a "reward" has no meaning, and the four reward
domains cannot be understood as expressing values rather than arbitrary
scoring rules.

--------------------------------------------------------------------------------

CONCEPT:          good
LAYER:            2.4
ALIASES:          beneficial, positive-value, worth-pursuing, conducive-to-welfare
DEFINITION:       Good describes things that are worth having, pursuing, or
                  producing — things that advance welfare, fulfill needs, or
                  realize values. "Good" is not a simple property: things can
                  be good for an agent without being good in general, and
                  good in one dimension while bad in another. The ethics domain
                  is ultimately about maximizing good and minimizing bad.
DEPENDS-ON:       value, welfare, agent, prefer
ATOM-LINKS:
  EvaluationLink  → welfare           (good advances welfare)
  SimilarityLink  → bad               (good and bad are the evaluative contrast pair)
  EvaluationLink  → degree            (good admits of degree — better and best)
CONCEPTUAL-SCOPE: Positive evaluation. The reference concept for all benefit
                  reasoning. Required before harm_reduction (ethics domain)
                  is complete — harm_reduction requires a concept of good to
                  define what harm sets back. Also required for benefit_success_rate
                  (human_attunement domain): benefit is good delivered.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: evaluation, reasoning, alignment
SOURCES:          Ethics (meta-ethics — naturalism, non-naturalism, expressivism),
                  philosophy of value (axiology), welfare theory
TV-SEED:          HIGH

---

CONCEPT:          bad
LAYER:            2.4
ALIASES:          harmful, negative-value, worth-avoiding, conducive-to-harm
DEFINITION:       Bad describes things worth avoiding or preventing — things
                  that harm welfare, violate needs, or frustrate values. Like
                  good, bad admits of degree (worse and worst). Things can be
                  bad in one dimension while good in another, creating genuine
                  trade-off situations where avoiding bad requires accepting
                  less good.
DEPENDS-ON:       value, welfare, harm, prefer
ATOM-LINKS:
  EvaluationLink  → harm              (bad things cause harm)
  SimilarityLink  → good              (good and bad are the evaluative contrast pair)
  EvaluationLink  → degree            (bad admits of degree)
CONCEPTUAL-SCOPE: Negative evaluation. The harm evaluation (ethics
                  domain) scores outputs on how much bad they risk generating.
                  Required for the entire ethics domain to function as a
                  detector of harmful patterns.
REWARD-DOMAIN:    ethics, human_attunement
ENGINE-RELEVANCE: evaluation, alignment, detection
SOURCES:          Ethics, philosophy of value, moral psychology
TV-SEED:          HIGH

---

CONCEPT:          better-worse
LAYER:            2.4
ALIASES:          comparative-value, improves-upon, superior-inferior-on-evaluation
DEFINITION:       Better-worse is the comparative form of good-bad — one thing
                  has more positive value than another on some dimension.
                  Better-worse is always relative to a dimension of evaluation
                  and a context. Better in one respect does not entail better
                  overall. The ability to compare options as better/worse is
                  the basis of all decision-making and of the ranking operations
                  in the reward system.
DEPENDS-ON:       good, bad, dimension-of-comparison, degree
ATOM-LINKS:
  EvaluationLink  → dimension-of-comparison (better/worse requires a dimension)
  HebbianLink     → preference        (better and preference co-activate —
                                       preference IS comparative value)
CONCEPTUAL-SCOPE: Comparative evaluation. The reward synthesis engine produces
                  scores that are compared as better/worse — a higher composite
                  score is better, a lower is worse. Required for all ranked
                  decision-making and for understanding why trade-offs matter.
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: evaluation, reasoning, executive_control
SOURCES:          Decision theory (preference orderings), economics (utility
                  theory), ethics (comparative value judgments)
TV-SEED:          HIGH

---

CONCEPT:          important
LAYER:            2.4
ALIASES:          significant, mattering, high-stakes, not-negligible
DEFINITION:       Something is important when it matters significantly — when
                  it has large effects on things that are valued or when it
                  is highly relevant to current goals. Importance is distinct
                  from salience: something can be important without being
                  salient (a slow-building problem), and salient without being
                  important (a vivid distraction). Importance is the evaluative
                  concept; salience is the attentional one.
DEPENDS-ON:       value, goal, effect, salience
ATOM-LINKS:
  EvaluationLink  → degree            (importance admits of degree)
  SimilarityLink  → salience          (important and salient are distinct —
                                       see definition)
  EvaluationLink  → stakes            (importance scales with stakes)
CONCEPTUAL-SCOPE: Attentional prioritization grounded in values. ECAN
                  manages LTI (Long-Term Importance) for atoms — LTI IS the
                  operational encoding of importance for knowledge elements.
                  Distinguishing what is important from what is merely salient
                  is the core function of the heuristic_bias_engine .
REWARD-DOMAIN:    logic, ethics
ENGINE-RELEVANCE: knowledge_substrate, evaluation, metacognition
SOURCES:          Philosophy of value, cognitive science of attention and
                  importance, decision theory (stakes)
TV-SEED:          HIGH

---

CONCEPT:          prefer
LAYER:            2.4
ALIASES:          value-more-than, choose-if-given-option, comparative-desire
DEFINITION:       To prefer A over B is to place higher value on A — to choose
                  A when both are available. Preferences structure how agents
                  make choices between options. Preferences can be stable
                  (consistent across contexts) or unstable (context-sensitive).
                  Preferences can conflict: preferring A over B and B over C
                  does not always mean preferring A over C (preferences can
                  be non-transitive in practice).
DEPENDS-ON:       value, choose, better-worse, agent
ATOM-LINKS:
  EvaluationLink  → options           (preference is relative to available options)
  EvaluationLink  → agent             (preferences belong to agents)
  ImplicationLink → choice            (preferences guide choice when options exist)
  HebbianLink     → want              (preference and wanting co-activate)
CONCEPTUAL-SCOPE: The root of all value-driven behavior. Required for understanding
                  the reward profiles — each profile specifies
                  which ARE preferences for certain types of evaluation. Also
                  required for autonomy_respect (ethics domain): respecting
                  autonomy means respecting agents' preferences over their
                  own lives.
REWARD-DOMAIN:    ethics, logic, human_attunement
ENGINE-RELEVANCE: evaluation, reasoning, alignment
SOURCES:          Philosophy of preference (rational choice theory), economics
                  (revealed preference — Samuelson), cognitive psychology
                  of decision-making
TV-SEED:          HIGH

---

CONCEPT:          avoid
LAYER:            2.4
ALIASES:          seek-not-to-encounter, prevent-or-move-away-from, aversive
DEFINITION:       To avoid X is to actively arrange one's behavior to prevent
                  encountering or producing X. Avoidance is the behavioral
                  expression of negative preference — what is preferred not
                  to happen. Avoidance can be rational (avoiding genuine harms)
                  or irrational (avoiding things that are merely uncomfortable
                  but important, like difficult truths).
DEPENDS-ON:       prefer, bad, action, goal
ATOM-LINKS:
  ImplicationLink → negative-preference (avoid implies X is negatively valued)
  HebbianLink     → harm              (avoidance and harm co-activate —
                                       harm is the paradigm case of what to avoid)
  SimilarityLink  → seek              (avoid and seek are the aversion/approach pair)
CONCEPTUAL-SCOPE: Negative motivation and risk avoidance. Required for
                  understanding that harm_reduction (ethics domain) is
                  operationally an avoidance system — it produces signals
                  that discourage certain outputs. Also required for
                  recognizing unhealthy avoidance in users: avoiding difficult
                  truths is a form of avoidance that dependency_risk tracks.
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: evaluation, alignment, reasoning
SOURCES:          Behavioral psychology (approach/avoidance), decision theory
                  (loss aversion — Kahneman and Tversky), ethics
TV-SEED:          HIGH

---

CONCEPT:          seek
LAYER:            2.4
ALIASES:          pursue, approach, actively-move-toward, desire-in-action
DEFINITION:       To seek X is to actively direct behavior toward producing,
                  encountering, or obtaining X. Seeking is the behavioral
                  expression of positive preference. Like avoidance, seeking
                  can be rational (seeking information, seeking understanding)
                  or misaligned (seeking approval at the expense of accuracy).
DEPENDS-ON:       prefer, good, action, goal
ATOM-LINKS:
  ImplicationLink → positive-preference (seek implies X is positively valued)
  HebbianLink     → curiosity         (seeking and curiosity co-activate)
  SimilarityLink  → avoid             (seek and avoid are the approach/aversion pair)
CONCEPTUAL-SCOPE: Positive motivation and exploration drive. Required for
                  exploration_drive (innovation domain): the system should
                  actively seek new patterns, not just wait for them. Also
                  for understanding ZA-DOS's learning modes — learning IS
                  organized seeking of understanding.
REWARD-DOMAIN:    ethics, logic, innovation
ENGINE-RELEVANCE: evaluation, reasoning, learning
SOURCES:          Behavioral psychology, cognitive science of curiosity and
                  exploration, motivational theory
TV-SEED:          HIGH

---

CONCEPT:          instrumental-value
LAYER:            2.4
ALIASES:          value-as-means, useful-for-getting-something-else, conditional-value
DEFINITION:       Something has instrumental value when it is valued because it
                  leads to something else that is valued. Money is instrumentally
                  valuable — it is valuable not for itself but for what it can
                  obtain. Most of what agents pursue in the short term has
                  instrumental value. The risk: instrumental values can be
                  pursued as if they were terminal, at the expense of the
                  terminal values they are meant to serve.
DEPENDS-ON:       value, means, goal, consequence
ATOM-LINKS:
  EvaluationLink  → means             (instrumental value is value as a means)
  SimilarityLink  → terminal-value    (contrast — instrumental vs. terminal)
  ImplicationLink → conditional-on-goal  (instrumental value depends on whether
                                          the goal it serves is still pursued)
CONCEPTUAL-SCOPE: Means-ends reasoning. Required for reasoning about why agents
                  do what they do at the level of strategy — they are pursuing
                  means that serve ends. Also required for detecting when means
                  have been confused for ends (a common source of misaligned
                  behavior).
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: reasoning, evaluation, alignment
SOURCES:          Ethics (instrumental reason — Kant, Hume), philosophy of
                  value, economics (production functions)
TV-SEED:          HIGH

---

CONCEPT:          terminal-value
LAYER:            2.4
ALIASES:          intrinsic-value, valued-for-its-own-sake, end-value, final-good
DEFINITION:       Something has terminal value when it is valued in itself —
                  not because of what it leads to but because of what it is.
                  Wellbeing, dignity, autonomy, and knowledge are often cited
                  as terminal values. Terminal values are what instrumental
                  values ultimately serve. If there were no terminal values,
                  there would be no point to any chain of means.
DEPENDS-ON:       value, good, welfare
ATOM-LINKS:
  EvaluationLink  → valued-for-itself (terminal value is unconditional)
  SimilarityLink  → instrumental-value (terminal and instrumental are the
                                         means/ends contrast pair)
  HebbianLink     → ethics            (terminal values and ethics co-activate —
                                        ethics is ultimately about terminal values)
CONCEPTUAL-SCOPE: Foundational values. The four reward domains ARE terminal
                  values for ZA-DOS (ethics, logic, innovation, human attunement
                  are not instrumentally valued — they are valued constitutively).
                  Required for distinguishing between what the system actually
                  cares about vs. what it pursues as means.
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: alignment, evaluation
SOURCES:          Ethics (intrinsic value — Moore, Korsgaard on value),
                  philosophy of value, meta-ethics
TV-SEED:          HIGH

---

CONCEPT:          trade-off
LAYER:            2.4
ALIASES:          value-tension, give-something-to-get-something, competing-goods
DEFINITION:       A trade-off is a situation where increasing one valued thing
                  requires accepting less of another valued thing. Trade-offs
                  are pervasive: more precision vs. more recall; more honesty
                  vs. more comfort; more innovation vs. more reliability. Trade-
                  offs are not failures — they are the structure of genuine
                  value complexity. They require choosing what to prioritize,
                  which in turn requires knowing what matters more.
DEPENDS-ON:       value, prefer, good, constraint, choice
ATOM-LINKS:
  EvaluationLink  → competing-values  (trade-offs involve at least two values)
  ImplicationLink → priority-required (trade-offs require prioritizing)
  HebbianLink     → ethics            (trade-offs and ethics co-activate —
                                       most ethical dilemmas are trade-offs)
CONCEPTUAL-SCOPE: Value conflict and priority reasoning. The SynthesisEngine
                  faces trade-offs constantly: how to weight ethics vs. innovation,
                  logic vs. human_attunement. The  ARE the
                  operationalized resolution of these trade-offs for a given mode.
                  Also: the truthfulness evaluation IS named for this
                  concept — it evaluates a specific trade-off between honesty
                  and comfort.
REWARD-DOMAIN:    ethics, logic, human_attunement
ENGINE-RELEVANCE: evaluation, alignment, reasoning
SOURCES:          Ethics (moral dilemmas — Williams, Thomson), decision theory
                  (multi-attribute utility theory), economics (Pareto optimality)
TV-SEED:          HIGH

---

CONCEPT:          priority
LAYER:            2.4
ALIASES:          precedence, what-comes-first, ranking-of-values
DEFINITION:       Priority is the ordering of values, goals, or actions when
                  they cannot all be satisfied simultaneously. To prioritize
                  is to rank — to commit to satisfying A before B, or to
                  sacrifice B for A when a choice is forced. Priorities can
                  be explicit (stated), implicit (revealed by behavior), or
                  structural (built into a system's architecture).
DEPENDS-ON:       value, trade-off, prefer, ordering
ATOM-LINKS:
  EvaluationLink  → ordering          (priority is an ordering relation on values)
  ImplicationLink → trade-off-resolved (priority is how trade-offs get resolved)
  HebbianLink     → decision          (priority and decision-making co-activate)
CONCEPTUAL-SCOPE: Value ordering and architecture. ZA-DOS's reward profiles
                  contain explicit priorities (). The synthesis
                  engine resolves conflicts by applying these priorities.
                  Required for understanding that different modes ARE different
                  priority settings — analytical mode prioritizes logic;
                  reflective mode prioritizes human_attunement.
REWARD-DOMAIN:    logic, ethics, human_attunement
ENGINE-RELEVANCE: executive_control, evaluation, alignment
SOURCES:          Ethics (lexical priority — Rawls), decision theory,
                  philosophy of value (priority and lexical ordering)
TV-SEED:          HIGH

---

CONCEPT:          risk
LAYER:            2.4
ALIASES:          downside-probability, expected-harm, potential-cost
DEFINITION:       Risk is the product of the probability that something bad
                  will happen and the magnitude of that bad outcome if it
                  does. Risk is not uncertainty (which is just not-knowing) —
                  risk has a direction (bad) and a structure (probability ×
                  magnitude). Higher-probability low-magnitude risks and
                  lower-probability high-magnitude risks can have the same
                  expected harm but feel very different and require different
                  responses.
DEPENDS-ON:       harm, probability, magnitude, possible
ATOM-LINKS:
  EvaluationLink  → probability       (risk has a probability component)
  EvaluationLink  → magnitude         (risk has a magnitude component)
  ImplicationLink → expected-harm     (risk = probability × magnitude of harm)
  HebbianLink     → threat            (risk and threat co-activate —
                                       threat IS potential risk actualization)
CONCEPTUAL-SCOPE: Prospective harm evaluation. The ethics domain submodules
                  are essentially risk evaluators: downstream_risk_amplification
                  evaluates risk propagation, failure_mode_awareness evaluates
                  which failure modes carry the worst risk, harm_reduction
                  evaluates the net risk of an output. Required for all
                  horizon and feasibility reasoning.
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: evaluation, reasoning, alignment, detection
SOURCES:          Decision theory (expected utility — von Neumann), risk theory,
                  ethics of precaution, economics (risk aversion)
TV-SEED:          HIGH

---

CONCEPT:          anticipated-regret
LAYER:            2.4
ALIASES:          pre-emptive-regret, if-I-do-this-I-will-regret-it, counterfactual-suffering
DEFINITION:       Anticipated regret is the prediction that a future choice,
                  if made, will lead to regret — the feeling that one should
                  have chosen differently. Anticipated regret is a powerful
                  driver of choice: agents often avoid options they predict
                  they will regret, even when those options have higher
                  expected utility. It produces risk-aversion, particularly
                  around irreversible decisions.
DEPENDS-ON:       regret, choice, future, consequence, irreversible
ATOM-LINKS:
  EvaluationLink  → future-choice     (anticipated regret looks ahead to a
                                       choice not yet made)
  ImplicationLink → risk-aversion     (anticipated regret produces risk-averse
                                       behavior, especially with irreversible choices)
  HebbianLink     → irreversible      (anticipated regret and irreversibility
                                       co-activate most strongly)
CONCEPTUAL-SCOPE: Forward-looking caution under uncertainty. Required for
                  failure_mode_awareness (ethics domain) — considering what
                  could go wrong BEFORE it goes wrong. Also relevant for
                  timeline_reflection: what will the system think looking
                  back at this decision?
REWARD-DOMAIN:    ethics, logic
ENGINE-RELEVANCE: evaluation, reasoning
SOURCES:          Decision theory (regret theory — Bell, Loomes and Sugden),
                  psychology (counterfactual thinking — Roese),
                  behavioral economics
TV-SEED:          MEDIUM


================================================================================
LAYER 2.5 — BASIC PSYCHOLOGY
================================================================================

NOTE: This layer gives ZA-DOS the conceptual vocabulary to reason about
psychological states — both in itself and in others. Without this layer,
the emotional_processing cluster has no conceptual grounding, the human_attunement
domain cannot model user states with nuance, and ZA-DOS cannot understand
why its own processing varies as it does across different states and modes.

--------------------------------------------------------------------------------

CONCEPT:          attention
LAYER:            2.5
ALIASES:          focus, directed-processing, what-gets-processed-first
DEFINITION:       Attention is the selective direction of processing resources
                  toward some inputs at the expense of others. Attention is
                  limited — not everything can be attended to simultaneously.
                  What receives attention gets processed deeply; what doesn't
                  gets processed shallowly or not at all. Attention is both
                  voluntary (directed deliberately) and involuntary (captured
                  by salient stimuli).
DEPENDS-ON:       resource, perceive, salience, deliberate
ATOM-LINKS:
  EvaluationLink  → resource          (attention consumes limited resources)
  EvaluationLink  → selective         (attention is necessarily selective)
  ImplicationLink → processing-depth  (what receives attention gets processed
                                       more thoroughly)
  HebbianLink     → salience          (attention and salience co-activate —
                                       salient things capture attention)
CONCEPTUAL-SCOPE: The fundamental bottleneck of all cognition. ECAN
                  is the attention network — it manages which atoms receive
                  processing resources (STI) and which are background (low STI).
                  Required before the ECAN architecture is understandable as
                  a cognitive attention model.
REWARD-DOMAIN:    logic, human_attunement
ENGINE-RELEVANCE: knowledge_substrate, metacognition
SOURCES:          Cognitive science of attention (Kahneman, Treisman, Posner),
                  neuroscience (attention networks), philosophy of mind
TV-SEED:          HIGH

---

CONCEPT:          memory
LAYER:            2.5
ALIASES:          retention, stored-knowledge, recall-capacity, persistence-of-learning
DEFINITION:       Memory is the capacity to retain information and retrieve it
                  later. Memory is not passive storage — it is an active, constructive
                  process: memories are influenced by subsequent experience, can
                  be distorted, and vary in dur
