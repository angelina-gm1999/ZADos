ZA-DOS
Zonal Adaptive Dynamics Operating System

Research Sources & Theoretical Lineage
Master Reference Document
L.G. Martin  •  March 2026

Table of Contents



Introduction
This document catalogues the theoretical and empirical sources that inform the design of ZA-DOS, a cognitive architecture combining neurochemical simulation, cognitive engines, a multi-tier memory system, a four-domain reward layer, sleep/dream processing, and a structured learning framework.
The document is organized by subsystem so that each section of ZA-DOS can be traced to its specific intellectual lineage. Cross-cutting sources (e.g., Kahneman, Damasio) appear in every section where they are relevant. Section 13 identifies original contributions that synthesize or extend existing work. Section 14 provides a consolidated alphabetical reading list.


1. Cognitive Architecture Foundations
The overall structure of ZA-DOS draws from three major cognitive architecture traditions, plus a brain emulation philosophy that bridges the gap between functional modeling and substrate simulation.
1.1 SOAR Architecture
Engine 3 (SOAR Production Engine) implements a five-phase decision cycle with working memory, elaboration, proposal, decision, and application phases. Impasse-driven subgoaling and chunking are direct SOAR mechanisms, extended with neurochemical modulation of every phase.
Sources
Newell, A. (1990). Unified Theories of Cognition. Harvard University Press.  — foundational theory of cognitive architecture; the "Soar" system as a unified theory of mind
Laird, J. E. (2012). The Soar Cognitive Architecture. MIT Press.  — primary reference for the five-phase decision cycle, production system, impasse handling, and chunking
Newell, A. & Simon, H. A. (1963). GPS: A Program that Simulates Human Thought. Computers and Thought.  — means-ends analysis and hierarchical problem decomposition
1.2 OpenCog / Hyperon
Engines 9 (AtomSpace), 10 (PLN), and 16 (ECAN) are pure-Python reimplementations of OpenCog’s core cognitive tools, adapted with neurochemical modulation. AtomSpace provides the typed hypergraph knowledge store, PLN handles probabilistic logic, and ECAN manages economic attention allocation.
Sources
Goertzel, B., Pennachin, C., & Geigel, A. (2014). Engineering General Intelligence, Vols. 1–2. Atlantis Press.  — primary reference for AtomSpace, PLN, ECAN, and the integrative cognitive architecture
Goertzel, B., Iklé, M., Goertzel, I. F., & Pennachin, C. (2009). Probabilistic Logic Networks. Springer.  — PLN truth value system, uncertain inference rules
Sowa, J. F. (1984). Conceptual Structures: Information Processing in Mind and Machine. Addison-Wesley.  — typed hypergraph knowledge representation tradition
Wang, P. (1995). Non-Axiomatic Reasoning System. PhD Thesis, Indiana University.  — non-axiomatic reasoning influence on PLN’s evidential logic
1.3 Global Workspace & Attention
ECAN’s attentional focus concept and the broader pipeline architecture where engines compete for a limited processing budget parallels Global Workspace Theory. Attention is treated as a scarce economic resource.
Sources
Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.  — Global Workspace Theory; limited-capacity conscious workspace
Kahneman, D. (1973). Attention and Effort. Prentice-Hall.  — attention as limited resource with economic allocation properties
Broadbent, D. E. (1958). Perception and Communication. Pergamon Press.  — filter model of selective attention
1.4 Brain Emulation Philosophy
The decision to simulate neurochemical substrate (not just cognitive function) reflects a brain emulation approach: modeling what the brain IS, not just what it DOES. This motivates the entire neurochemical layer, receptor dynamics, and oscillatory system.
Sources
Sandberg, A. & Bostrom, N. (2008). Whole Brain Emulation: A Roadmap. Future of Humanity Institute, Oxford University.  — philosophical framework for substrate-level cognitive simulation

2. Neurochemical Layer — Mathematics
The neurochemical layer uses stochastic differential equations (SDEs) from quantitative finance to model neurotransmitter concentration dynamics. This is a novel cross-disciplinary application: the mathematical tools were developed for modeling financial instruments but map naturally onto neurochemical processes because both domains involve mean-reverting, bounded, stochastic systems subject to sudden shocks.
2.1 Stochastic Differential Equations
Neurotransmitter concentration evolves via an Ornstein-Uhlenbeck process with losses (drift) and multiplicative square-root noise (diffusion), integrated via Euler-Maruyama discretization with bounded reflecting boundaries and adaptive time-stepping. The drift term uses Vasicek-style mean reversion toward homeostatic baseline; the diffusion uses a CIR-type square-root dependence that naturally enforces positivity.
Sources
Kloeden, P. E. & Platen, E. (1992). Numerical Solution of Stochastic Differential Equations. Springer.  — primary reference for SDE numerics, Euler-Maruyama scheme, adaptive time-stepping, convergence analysis
Vasicek, O. (1977). An Equilibrium Characterization of the Term Structure. Journal of Financial Economics, 5(2), 177–188.  — mean-reverting interest rate model; structural analog for homeostatic neurotransmitter dynamics
Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A Theory of the Term Structure of Interest Rates. Econometrica, 53(2), 385–407.  — CIR process with square-root diffusion; structural analog for concentration-dependent noise ensuring positivity
Uhlenbeck, G. E. & Ornstein, L. S. (1930). On the Theory of the Brownian Motion. Physical Review, 36(5), 823–841.  — Ornstein-Uhlenbeck process; foundational mean-reverting stochastic process
2.2 Stochastic Impulse Generators
Phasic neurotransmitter bursts are modeled via distribution-based impulse samplers (Gamma, Poisson, Lognormal) with volatility-adaptive parameterization. These function as jump-diffusion processes analogous to sudden market shocks.
Sources
Merton, R. C. (1976). Option Pricing When Underlying Stock Returns Are Discontinuous. Journal of Financial Economics, 3(1–2), 125–144.  — jump-diffusion model; structural analog for phasic burst events
2.3 Numerical Methods & Stability
The implementation includes adaptive step-size Euler-Maruyama with Richardson extrapolation, Brownian path splitting for half-steps, stability condition checking, and bounded reflecting boundaries via modular folding.
Sources
Kloeden & Platen (1992) — see above.  — Richardson extrapolation, step-size control, strong/weak convergence
Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed. MIT Press.  — temporal difference learning mathematics used in the reward prediction error system

3. Neurochemical Layer — Neuroscience
The neurochemical layer models 12 neurotransmitter systems with receptor subtype specificity, pharmacodynamic plasticity, and oscillatory modulation. The neuroscience grounding spans neuropharmacology, receptor pharmacology, and systems neuroscience.
3.1 General Neuroscience & Pharmacology
Kandel, E. R., Schwartz, J. H., Jessell, T. M., Siegelbaum, S. A., & Hudspeth, A. J. (Eds.). (2013). Principles of Neural Science, 5th ed. McGraw-Hill.  — comprehensive neuroscience reference; neurotransmitter systems, receptor pharmacology, synaptic transmission, memory circuits
Stahl, S. M. (2013). Stahl’s Essential Psychopharmacology, 4th ed. Cambridge University Press.  — receptor subtype pharmacology, G-protein coupling, desensitization/internalization, clinical neurotransmitter mapping
Rang, H. P., Ritter, J. M., Flower, R. J., & Henderson, G. (2016). Rang & Dale’s Pharmacology, 8th ed. Elsevier.  — Michaelis-Menten/Hill equation binding kinetics, dissociation constants, pharmacodynamic principles
3.2 Specific Neurotransmitter Systems
3.2.1 Dopamine
Schultz, W. (1997). A Neural Substrate of Prediction and Reward. Science, 275(5306), 1593–1599.  — dopamine reward prediction error signal; TD-style delta computation
Grace, A. A. (1991). Phasic versus tonic dopamine release and the modulation of dopamine system responsivity. Neuroscience, 41(1), 1–24.  — tonic vs phasic dopamine firing; foundational for C_tonic / C_phasic decomposition
3.2.2 Norepinephrine
Aston-Jones, G. & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function: Adaptive gain and optimal performance. Annual Review of Neuroscience, 28, 403–450.  — LC-NE adaptive gain theory; arousal-performance modulation
Aston-Jones, G. & Bloom, F. E. (1981). Activity of norepinephrine-containing locus coeruleus neurons in behaving rats anticipates fluctuations in the sleep-waking cycle. Journal of Neuroscience, 1(8), 876–886.  — LC silence during REM sleep
Yerkes, R. M. & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation. Journal of Comparative Neurology and Psychology, 18(5), 459–482.  — arousal-performance inverted-U relationship
3.2.3 Acetylcholine
Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory. Current Opinion in Neurobiology, 16(6), 710–715.  — ACh encoding/consolidation toggle; suppression of retrieval during encoding, hippocampal-cortical replay during low-ACh sleep
Hasselmo, M. E. (1999). Neuromodulation: acetylcholine and memory consolidation. Trends in Cognitive Sciences, 3(9), 351–359.  — cholinergic theory of memory
3.2.4 Serotonin, GABA, Oxytocin, Endocannabinoids, Opioids, Cortisol/CRH
Receptor subtype specificity and functional roles for these systems are drawn from Kandel (2013) and Stahl (2013) above, with specific research on endocannabinoids and sleep from:
Murillo-Rodríguez, E., et al. (2003). Anandamide enhances extracellular levels of adenosine and induces sleep. Sleep, 26(8), 943–947.  — endocannabinoid system and sleep regulation
Pava, M. J., Makriyannis, A., & Bhatt, D. L. (2016). Endocannabinoid signaling regulates sleep stability. PLOS ONE.  — endocannabinoid regulation of sleep homeostasis
3.3 Receptor Dynamics & Pharmacodynamics
Colquhoun, D. & Hawkes, A. G. (1977). Relaxation and fluctuations of membrane currents that flow through drug-operated channels. Proc. R. Soc. Lond. B, 199(1135), 231–262.  — Markov state models for receptor kinetics; foundational for CTMC receptor state modeling
Colquhoun, D. & Hawkes, A. G. (1981). On the stochastic properties of single ion channels. Proc. R. Soc. Lond. B, 211(1183), 205–235.  — stochastic receptor state transitions
3.4 Oscillatory Modulation
Lisman, J. E. & Jensen, O. (2013). The theta-gamma neural code. Neuron, 77(6), 1002–1016.  — theta-gamma cross-frequency coupling for memory encoding
Jensen, A. & Mazaheri, A. (2010). Shaping functional architecture by oscillatory alpha activity. Frontiers in Human Neuroscience, 4, 186.  — alpha as inhibitory gating rhythm
Buzsáki, G. (2006). Rhythms of the Brain. Oxford University Press.  — comprehensive oscillatory neuroscience reference
3.5 Hebbian Learning
Hebb, D. O. (1949). The Organization of Behavior. Wiley.  — Hebbian learning rule; co-activation strengthening; HebbianLink dynamics in ECAN

4. Detection Cluster — Psychology of Reasoning
4.1 Cognitive Biases (Engines 5 & 24)
Engine 5 (Bias Detection) uses a "hybrid Kahneman taxonomy" of 24 bias types across 8 categories. Engine 24 (Heuristic Bias) monitors the system’s own reasoning processes for systematic shortcuts.
Tversky, A. & Kahneman, D. (1974). Judgment under Uncertainty: Heuristics and Biases. Science, 185(4157), 1124–1131.  — foundational heuristics and biases research program
Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.  — dual-process theory; System 1/System 2; comprehensive bias taxonomy
Kahneman, D. & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. Econometrica, 47(2), 263–292.  — loss aversion and framing effects
Flavell, J. H. (1979). Metacognition and cognitive monitoring. American Psychologist, 34(10), 906–911.  — metacognition concept; self-monitoring of reasoning processes
Nelson, T. O. & Narens, L. (1990). Metamemory: A theoretical framework and new findings. Psychology of Learning and Motivation, 26, 125–173.  — monitoring-control framework for metacognition
4.2 Contradiction Detection (Engine 1)
Bayesian confidence model for contradiction detection across three levels (direct negation, semantic contradiction, implicit contextual). Belief revision on detection.
Festinger, L. (1957). A Theory of Cognitive Dissonance. Stanford University Press.  — cognitive dissonance; psychological response to detected contradictions
Alchourrón, C. E., Gärdenfors, P., & Makinson, D. (1985). On the logic of theory change: Partial meet contraction and revision functions. Journal of Symbolic Logic, 50(2), 510–530.  — AGM framework for rational belief revision
Prior, A. N. (1967). Past, Present, and Future. Clarendon Press.  — temporal logic; relevant to temporal contradiction detection
4.3 Paradox Detection (Engine 2)
Four-class paradox taxonomy: Resolvable (R), Apparent (A), Genuine/Dialectical (G), Structural/Self-referential (S).
Quine, W. V. O. (1966). The Ways of Paradox. Random House.  — veridical/falsidical paradox classification (Class A)
Hegel, G. W. F. (1812–1816). Science of Logic.  — dialectical logic; thesis-antithesis coexistence (Class G)
Kant, I. (1781). Critique of Pure Reason.  — antinomies of pure reason; equally valid contradictory conclusions (Class G)
Russell, B. (1903). The Principles of Mathematics. Cambridge University Press.  — Russell’s Paradox; self-referential set theory (Class S)
Gödel, K. (1931). Über formal unentscheidbare Sätze. Monatshefte für Mathematik und Physik, 38, 173–198.  — incompleteness theorems; structural self-reference limits (Class S)
Priest, G. (1987). In Contradiction. Martinus Nijhoff.  — dialetheism; some contradictions may be true (philosophical context for Class G)
Kripke, S. (1975). Outline of a Theory of Truth. Journal of Philosophy, 72(19), 690–716.  — formal treatment of the Liar Paradox and self-referential truth (Class S)
Hofstadter, D. R. (1979). Gödel, Escher, Bach: An Eternal Golden Braid. Basic Books.  — self-reference, strange loops, recursive systems (Class S)
4.4 Fallacy Detection (Engine 4)
Five-category taxonomy (Formal, Relevance, Presumption, Ambiguity, Inductive) with 30+ specific fallacy types and a Principle of Charity implementation.
Aristotle. Sophistical Refutations (c. 350 BCE).  — foundational fallacy classification; fallacies dependent on language vs independent of language
Hamblin, C. L. (1970). Fallacies. Methuen.  — modern revival of fallacy theory; relevance-based classification
Walton, D. N. (1989). Informal Logic: A Pragmatic Approach. Cambridge University Press.  — argumentation schemes; fallacies as violations of dialogue norms
Walton, D. N. (1991). Begging the Question. Greenwood Press.  — formal analysis of petitio principii
Walton, D. N. (1992). Slippery Slope Arguments. Clarendon Press.  — formal analysis of slippery slope reasoning
Copi, I. M. (1953). Introduction to Logic. Macmillan.  — standardized formal fallacy names (undistributed middle, illicit major/minor)
Toulmin, S. E. (1958). The Uses of Argument. Cambridge University Press.  — argumentation model; claim-data-warrant structure
van Eemeren, F. H. & Grootendorst, R. (1984). Speech Acts in Argumentative Discussions. Foris.  — pragma-dialectics; argumentation as regulated dialogue
Davidson, D. (1974). On the Very Idea of a Conceptual Scheme. Proceedings of the APA, 47, 5–20.  — Principle of Charity in interpretation
Quine, W. V. O. (1960). Word and Object. MIT Press.  — charitable interpretation in translation/understanding
4.5 Simulated Opposition (Engine 7)
Internal adversary with five opposition modes. Philosophically grounded in falsificationism and adversarial epistemology.
Popper, K. (1934/1959). The Logic of Scientific Discovery. Routledge.  — falsificationism; actively attempting to disprove conclusions
Mill, J. S. (1859). On Liberty.  — Millian argument: ideas must survive challenge to be justified
Lakatos, I. (1978). The Methodology of Scientific Research Programmes. Cambridge University Press.  — evaluating theories by resilience to anomalies

5. Socratic Reasoning (Engine 14)
The Socratic engine implements a six-state dialectical state machine (PROBING → ELENCHUS → APORIA → EXPLORING → MAIEUTICS → EXIT) with 18 question types, formalizing the structure of Socratic dialogue as it appears across Plato’s works.
5.1 Primary Sources (Plato)
Plato. Meno (c. 385 BCE).  — aporia as productive confusion (80a-d, "torpedo fish" passage); knowledge through recollection
Plato. Theaetetus (c. 369 BCE).  — maieutics / midwife metaphor (148e-151d); knowledge as justified true belief
Plato. Euthyphro, Laches, Charmides (c. 399–380 BCE).  — early dialogues exemplifying the elenctic method; definitional ("ti esti") questioning
Plato. Republic (c. 375 BCE).  — dialectic as highest form of reasoning; ascent from opinion to knowledge
5.2 Scholarly Analysis of Socratic Method
Vlastos, G. (1983). The Socratic Elenchus. Oxford Studies in Ancient Philosophy, 1, 27–58.  — formal analysis of how the elenchus works: state thesis, derive implications, show contradiction with other beliefs
Robinson, R. (1953). Plato’s Earlier Dialectic, 2nd ed. Clarendon Press.  — classic scholarly analysis of elenctic method structure
Kierkegaard, S. (1841). The Concept of Irony with Continual Reference to Socrates.  — Socratic irony and self-knowledge through self-discovery
5.3 Question Taxonomy
Paul, R. & Elder, L. (2006). The Art of Socratic Questioning. Foundation for Critical Thinking.  — taxonomy of Socratic question types
Bloom, B. S. (Ed.). (1956). Taxonomy of Educational Objectives: Handbook I: Cognitive Domain. David McKay.  — hierarchical questioning from recall to evaluation
5.4 Internal Self-Inquiry
Popper, K. (1934/1959). The Logic of Scientific Discovery. — see above.  — FALSIFICATION question type
Goldman, A. I. (1979). What Is Justified Belief? In G. Pappas (Ed.), Justification and Knowledge. Reidel.  — reliabilist epistemology; PROVENANCE question type — justification depends on process reliability

6. Decision Making, Simulation & Strategic Reasoning
6.1 Decision Making (Engine 15)
Damasio, A. R. (1994). Descartes’ Error: Emotion, Reason, and the Human Brain. Putnam.  — somatic marker hypothesis; emotional states as decision-relevant information
6.2 Simulation Brain (Engine 13)
Johnson-Laird, P. N. (1983). Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness. Harvard University Press.  — mental models theory; reasoning as construction and manipulation of mental simulations
Clark, A. (2015). Surfing Uncertainty: Prediction, Action, and the Embodied Mind. Oxford University Press.  — predictive processing framework; brain as prediction machine
6.3 Strategic Decision (Engine 21)
Bratman, M. E. (1987). Intention, Plans, and Practical Reason. Harvard University Press.  — BDI (Belief-Desire-Intention) model; commitment tracking and plan revision
6.4 Uncertainty (Engine 26)
Epistemic vs aleatoric uncertainty distinction, Bayesian propagation through inference chains.
Hume, D. (1739). A Treatise of Human Nature.  — problem of induction
Goodman, N. (1955). Fact, Fiction, and Forecast. Harvard University Press.  — new riddle of induction

7. Emotional Processing
Engine 28 (Emotional Detection) uses a 46-emotion taxonomy across 7 functional groups, with four-stage processing and bidirectional neurochemical coupling.
Ekman, P. (1971). Universals and Cultural Differences in Facial Expressions of Emotion. Nebraska Symposium on Motivation, 19, 207–283.  — basic emotions framework
Plutchik, R. (1980). Emotion: A Psychoevolutionary Synthesis. Harper & Row.  — wheel of emotions; emotion families and combinations
Lazarus, R. S. (1991). Emotion and Adaptation. Oxford University Press.  — appraisal theory of emotion; cognitive evaluation drives emotional response
Scherer, K. R. (2001). Appraisal Considered as a Process of Multilevel Sequential Checking. In Appraisal Processes in Emotion. Oxford University Press.  — component process model of emotion appraisal
Panksepp, J. (1998). Affective Neuroscience: The Foundations of Human and Animal Emotions. Oxford University Press.  — primary emotional systems mapped to brain circuits

8. Memory Architecture
8.1 Multi-Tier Structure (STMM / MTMM / LTMM)
Atkinson, R. C. & Shiffrin, R. M. (1968). Human Memory: A Proposed System and Its Control Processes. Psychology of Learning and Motivation, 2, 89–195.  — multi-store memory model (sensory, short-term, long-term)
Baddeley, A. D. & Hitch, G. (1974). Working Memory. Psychology of Learning and Motivation, 8, 47–89.  — working memory model; STMM parallels
Baddeley, A. D. (2000). The episodic buffer. Trends in Cognitive Sciences, 4(11), 417–423.  — episodic buffer as integration component
Cowan, N. (1999). An Embedded-Processes Model of Working Memory. In Models of Working Memory. Cambridge University Press.  — embedded-processes model; MTMM as activated long-term memory buffer
8.2 Memory Consolidation
McClelland, J. L., McNaughton, B. L., & O’Reilly, R. C. (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex. Psychological Review, 102(3), 419–457.  — complementary learning systems theory; fast hippocampal + slow neocortical learning
Frankland, P. W. & Bontempi, B. (2005). The organization of recent and remote memories. Nature Reviews Neuroscience, 6(2), 119–130.  — systems consolidation; gradual transfer from hippocampal to neocortical storage
8.3 Memory Compression & Forgetting
Ebbinghaus, H. (1885). Über das Gedächtnis. Duncker & Humblot.  — forgetting curve; memory decay dynamics
Craik, F. I. M. & Lockhart, R. S. (1972). Levels of Processing: A Framework for Memory Research. Journal of Verbal Learning and Verbal Behavior, 11(6), 671–684.  — levels of processing; VERBATIM → SEMANTIC → SYMBOLIC compression hierarchy
Miller, G. A. (1956). The Magical Number Seven, Plus or Minus Two. Psychological Review, 63(2), 81–97.  — working memory capacity limits
Tulving, E. (1972). Episodic and Semantic Memory. In Organization of Memory. Academic Press.  — encoding specificity principle; context-dependent memory
Anderson, J. R. (1993). Rules of the Mind. Lawrence Erlbaum.  — ACT-R base-level activation using recency and frequency for memory retrieval
8.4 Identity & Narrative Memory
McAdams, D. P. (2001). The Psychology of Life Stories. Review of General Psychology, 5(2), 100–122.  — narrative identity theory; identity maintained through coherent self-narrative
Erikson, E. H. (1968). Identity: Youth and Crisis. W. W. Norton.  — identity development stages

9. Reward System & Reinforcement Learning
9.1 Temporal Difference Learning
Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed. MIT Press.  — TD learning, reward prediction error, value functions, policy optimization
Schultz, W. (1997). — see Section 3.2.1.  — biological implementation of TD-style prediction error in dopamine neurons
9.2 Operant Conditioning & Behavioral Shaping
Skinner, B. F. (1938). The Behavior of Organisms. Appleton-Century.  — operant conditioning; reinforcement schedules; behavioral shaping through reward signals
9.3 Meta-Learning (Engine 25)
Thrun, S. & Pratt, L. (Eds.). (1998). Learning to Learn. Springer.  — meta-learning; learning about learning effectiveness
Schmidhuber, J. (2003). Gödel Machines: Self-Referential Universal Problem Solvers. arXiv:cs/0309048.  — self-referential learning systems; second-order optimization

10. Sleep & Dream Processing
10.1 Sleep Neurophysiology
Saper, C. B., Chou, T. C., & Scammell, T. E. (2001). The sleep switch: hypothalamic control of sleep and wakefulness. Trends in Neurosciences, 24(12), 726–731.  — flip-flop switch model of sleep/wake regulation
Hobson, J. A. & McCarley, R. W. (1977). The brain as a dream state generator. American Journal of Psychiatry, 134(12), 1335–1348.  — reciprocal interaction model of REM sleep; aminergic/cholinergic switching
Lu, J., Sherman, D., Devor, M., & Saper, C. B. (2006). A putative flip-flop switch for control of REM sleep. Nature, 441(7093), 589–594.  — updated REM switch model
Pace-Schott, E. F. & Hobson, J. A. (2002). The Neurobiology of Sleep. Nature Reviews Neuroscience, 3(8), 591–605.  — comprehensive sleep neurobiology; PGO waves; neurotransmitter changes across sleep stages
Luppi, P.-H., et al. (2011). The neuronal network responsible for paradoxical sleep and its dysfunctions. Sleep Medicine Reviews, 15(3), 153–163.  — REM atonia mechanisms; GABA-mediated motor inhibition during dreaming (containment gate analog)
10.2 Memory Consolidation During Sleep
Diekelmann, S. & Born, J. (2010). The memory function of sleep. Nature Reviews Neuroscience, 11(2), 114–126.  — comprehensive review of sleep-dependent memory consolidation
Staresina, B. P., et al. (2015). Hierarchical nesting of slow oscillations, spindles and ripples in human NREM sleep. Nature Neuroscience, 18(11), 1519–1521.  — slow oscillation-spindle coupling for memory consolidation (delta-sigma coupling analog)
Buzsáki, G. (2015). Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning. Hippocampus, 25(10), 1073–1188.  — sharp-wave ripples as consolidation mechanism (consolidation window trigger analog)
Walker, M. P. & Stickgold, R. (2004). Sleep-dependent learning and memory consolidation. Neuron, 44(1), 121–133.  — sleep-dependent memory processing overview
Wagner, U., et al. (2001). Emotional memory formation is enhanced across sleep intervals with high amounts of REM sleep. Learning & Memory, 8(2), 112–119.  — emotional memory preferentially consolidated during REM sleep
10.3 Sleep & Creativity / Problem-Solving
Wagner, U., et al. (2004). Sleep inspires insight. Nature, 427(6972), 352–355.  — sleep more than doubles probability of discovering hidden solutions
Cai, D. J., et al. (2009). REM, not incubation, improves creativity by priming associative networks. PNAS, 106(25), 10130–10134.  — REM sleep specifically enhances creative problem solving
Barrett, D. (2001). The Committee of Sleep. Crown.  — historical documentation of problem-solving and discovery during sleep/dreams
Wallas, G. (1926). The Art of Thought. Jonathan Cape.  — four-stage creativity model: preparation → incubation → illumination → verification
Maquet, P., et al. (1996). Functional neuroanatomy of human rapid-eye-movement sleep and dreaming. Nature, 383(6596), 163–166.  — prefrontal deactivation during REM; neuroimaging basis for suppressed critical judgment in dreams
Perogamvros, L. & Schwartz, S. (2012). The roles of the reward system in sleep and dreaming. Neuroscience & Biobehavioral Reviews, 36(8), 1934–1951.  — reward system dynamics during sleep; dopaminergic activity in REM
10.4 Circadian Regulation
Borbély, A. A. (1982). A two-process model of sleep regulation. Human Neurobiology, 1(3), 195–204.  — Process S (homeostatic) + Process C (circadian) model of sleep regulation

11. Learning System & Pedagogy
11.1 Foundational Learning Theory
Bruner, J. S. (1966). Toward a Theory of Instruction. Harvard University Press.  — discovery learning, scaffolding, spiral curriculum
Piaget, J. (1952). The Origins of Intelligence in Children. International Universities Press.  — constructivism; assimilation/accommodation; cognitive conflict as learning mechanism
Piaget, J. (1975). The Development of Thought: Equilibration of Cognitive Structures. Viking Press.  — equilibration; cognitive conflict theory in learning
Vygotsky, L. S. (1978). Mind in Society: The Development of Higher Psychological Processes. Harvard University Press.  — Zone of Proximal Development; social constructivism; M3 collaborative learning
Dewey, J. (1910). How We Think. D.C. Heath.  — reflective thinking; inquiry as basis for learning; M4 question-driven learning
Dewey, J. (1938). Experience and Education. Macmillan.  — experiential learning; genuine questions as learning catalysts
Ausubel, D. P. (1968). Educational Psychology: A Cognitive View. Holt, Rinehart & Winston.  — reception learning theory; meaningful reception learning; M1 receptive mode
Knowles, M. S. (1975). Self-Directed Learning: A Guide for Learners and Teachers. Cambridge.  — andragogy; self-directed learning; M5 independent study
11.2 Learning Taxonomies & Progression Models
Bloom, B. S. (Ed.). (1956). Taxonomy of Educational Objectives. — see Section 5.3.  — cognitive domain hierarchy: remember → understand → analyze → evaluate → create
Anderson, L. W. & Krathwohl, D. R. (Eds.). (2001). A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom’s Taxonomy. Longman.  — revised Bloom’s taxonomy with metacognitive knowledge dimension
Perry, W. G. (1970). Forms of Intellectual and Ethical Development in the College Years. Holt, Rinehart & Winston.  — scheme of epistemological development; dualism → multiplicity → relativism → commitment
Kolb, D. A. (1984). Experiential Learning: Experience as the Source of Learning and Development. Prentice-Hall.  — experiential learning cycle: experience → reflection → conceptualization → experimentation
11.3 Assessment & Remediation
Black, P. & Wiliam, D. (1998). Inside the Black Box: Raising Standards Through Classroom Assessment. Phi Delta Kappan, 80(2), 139–148.  — formative assessment; diagnostic feedback driving instructional adjustment; deficit profiler lineage
Wiggins, G. & McTighe, J. (2005). Understanding by Design, 2nd ed. ASCD.  — backward design; starting from learning goals to design activities
Boud, D. (1995). Enhancing Learning through Self-Assessment. Kogan Page.  — self-assessment in learning; M2 peer review self-checking
11.4 Metacognition & Reflection
Schön, D. A. (1983). The Reflective Practitioner. Basic Books.  — reflection-in-action vs reflection-on-action; reflective mode pipeline
Mezirow, J. (1991). Transformative Dimensions of Adult Learning. Jossey-Bass.  — transformative learning through disorienting dilemmas; identity-learning cross-reference
Zimmerman, B. J. (2000). Self-Efficacy: An Essential Motive to Learn. Contemporary Educational Psychology, 25(1), 82–91.  — self-regulated learning; M5 self-monitoring and strategic adjustment
Argyris, C. & Schön, D. A. (1978). Organizational Learning: A Theory of Action Perspective. Addison-Wesley.  — espoused theory vs theory-in-use; identity-behavior alignment checking
11.5 Motivation & Academic Emotions
Dweck, C. S. (2006). Mindset: The New Psychology of Success. Random House.  — growth vs fixed mindset; M2 shame spiral protection
Ryan, R. M. & Deci, E. L. (2000). Self-Determination Theory and the Facilitation of Intrinsic Motivation. American Psychologist, 55(1), 68–78.  — intrinsic/extrinsic motivation; protecting intrinsic motivation from excessive criticism
Csikszentmihalyi, M. (1990). Flow: The Psychology of Optimal Experience. Harper & Row.  — flow state; challenge/skill balance; M3 curiosity-courage-discovery-joy cycle
Pekrun, R. (2006). The Control-Value Theory of Achievement Emotions. Educational Psychology Review, 18(4), 315–341.  — academic emotions taxonomy; mode-specific risk emotion mapping
Immordino-Yang, M. H. & Damasio, A. (2007). We Feel, Therefore We Learn. Mind, Brain, and Education, 1(1), 3–10.  — emotions as constitutive of learning, not separate from it

12. Homeostasis & Self-Regulation
Cannon, W. B. (1932). The Wisdom of the Body. W. W. Norton.  — homeostasis concept; maintaining physiological stability through regulatory mechanisms
Sterling, P. (2012). Allostasis: A model of predictive regulation. Physiology & Behavior, 106(1), 5–15.  — allostasis; anticipatory regulation beyond reactive homeostasis
Sweller, J. (1988). Cognitive Load During Problem Solving. Cognitive Science, 12(2), 257–285.  — cognitive load theory; E27 cognitive load estimation

13. Original Contributions
The following aspects of ZA-DOS represent original synthesis, design, or cross-disciplinary application by the author. They draw from the sources listed above but combine them in ways not found in existing literature.

Quantitative finance SDEs applied to neurochemical simulation
Using Vasicek/CIR/OU-family stochastic processes and Euler-Maruyama integration from financial mathematics as the substrate for neurotransmitter concentration dynamics. Both domains share mean-reverting, bounded, stochastic properties, but this cross-application appears to be novel.
Pharmacodynamically accurate sleep neurochemistry as computational mechanism
Using Hobson-McCarley reciprocal interaction dynamics, monoamine collapse, GABA-A containment gating, and PGO-analog scene shifting as the actual mechanism driving computational sleep functions (memory consolidation, creative recombination). Sleep neuroscience is used prescriptively, not just descriptively.
Four-domain reward architecture (Logic, Ethics, Innovation, Human Attunement)
The specific decomposition of reward into these four orthogonal domains, with per-domain prediction errors and neurochemical routing, is an original design. While the TD-learning mechanism is standard, the domain taxonomy and the wiring into neurochemical modulation is the author’s synthesis.
Neurochemical modulation of cognitive architecture components
Every engine, memory operation, and pipeline phase in ZA-DOS is bidirectionally coupled with the neurochemical layer. This goes beyond standard cognitive architectures (SOAR, ACT-R, OpenCog) which do not model neurochemical substrate. The specific NT-to-engine and engine-to-NT mapping matrices are original.
HUMINT-derived intent classification framework
The intention mapping architecture (Engine 23) draws from analytical frameworks originally developed in the author’s HUMINT Operational Manual (Martin, 2026) for structured intent classification in human intelligence operations.
Martin, L. G. (2026). HUMINT Operational Manual: Analytical Frameworks and Tradecraft Methodology. Self-published.  — structured intent classification methodology; intent-to-neurochemical routing
Neurosymbolic encoding syntax
The compact symbolic notation for expressing NT-receptor interactions with oscillatory gating and phasic/tonic markers (θ{DA•→D1:↑S}) is an original formal language for representing neurochemical state transitions.
Pedagogical mode architecture with neurochemical emotional regulation
The five learning modes (M1–M5) with mode-specific emotional presets, neurochemical coupling, risk emotion detection, and shame spiral protection implement pedagogical theory computationally in a way that integrates affective regulation with instructional design. The wiring of Pekrun’s academic emotions into NT-modulated pipeline parameters is original.
Identity coherence via retroactive alignment
Engine 30’s temporal coherence auditing with sigmoid collapse probability, affective consequence mapping, and four corrective action types is an original formalization of identity consistency checking. Engine 32’s cross-referencing of learning failures with identity conclusions implements Mezirow’s transformative learning computationally.
Dream mode creative recombination pipeline
The specific pipeline of unsolved buffer → dream candidate flagging → CB1/GLU-gated abstract re-association → opposition engine disabling → retroactive emotional consolidation is an original computational implementation of sleep-dependent insight generation.

14. Consolidated Reading List
All unique sources referenced in this document, alphabetized by first author.

Aston-Jones, G. & Bloom, F. E. (1981). Activity of norepinephrine-containing locus coeruleus neurons in behaving rats.
Aston-Jones, G. & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function.
Alchourrón, C. E., Gärdenfors, P., & Makinson, D. (1985). On the logic of theory change.
Anderson, J. R. (1993). Rules of the Mind.
Anderson, L. W. & Krathwohl, D. R. (2001). A Taxonomy for Learning, Teaching, and Assessing.
Argyris, C. & Schön, D. A. (1978). Organizational Learning.
Aristotle. Sophistical Refutations (c. 350 BCE).
Atkinson, R. C. & Shiffrin, R. M. (1968). Human Memory: A Proposed System.
Ausubel, D. P. (1968). Educational Psychology: A Cognitive View.
Baars, B. J. (1988). A Cognitive Theory of Consciousness.
Baddeley, A. D. & Hitch, G. (1974). Working Memory.
Baddeley, A. D. (2000). The episodic buffer.
Barrett, D. (2001). The Committee of Sleep.
Black, P. & Wiliam, D. (1998). Inside the Black Box.
Bloom, B. S. (1956). Taxonomy of Educational Objectives.
Borbély, A. A. (1982). A two-process model of sleep regulation.
Boud, D. (1995). Enhancing Learning through Self-Assessment.
Bratman, M. E. (1987). Intention, Plans, and Practical Reason.
Broadbent, D. E. (1958). Perception and Communication.
Bruner, J. S. (1966). Toward a Theory of Instruction.
Buzsáki, G. (2006). Rhythms of the Brain.
Buzsáki, G. (2015). Hippocampal sharp wave-ripple.
Cai, D. J., et al. (2009). REM, not incubation, improves creativity.
Cannon, W. B. (1932). The Wisdom of the Body.
Clark, A. (2015). Surfing Uncertainty.
Colquhoun, D. & Hawkes, A. G. (1977, 1981). Stochastic receptor state models.
Copi, I. M. (1953). Introduction to Logic.
Cowan, N. (1999). An Embedded-Processes Model of Working Memory.
Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A Theory of the Term Structure.
Craik, F. I. M. & Lockhart, R. S. (1972). Levels of Processing.
Csikszentmihalyi, M. (1990). Flow.
Damasio, A. R. (1994). Descartes’ Error.
Davidson, D. (1974). On the Very Idea of a Conceptual Scheme.
Dewey, J. (1910). How We Think.
Dewey, J. (1938). Experience and Education.
Diekelmann, S. & Born, J. (2010). The memory function of sleep.
Dweck, C. S. (2006). Mindset.
Ebbinghaus, H. (1885). Über das Gedächtnis.
Ekman, P. (1971). Universals and Cultural Differences in Facial Expressions.
Erikson, E. H. (1968). Identity: Youth and Crisis.
Festinger, L. (1957). A Theory of Cognitive Dissonance.
Flavell, J. H. (1979). Metacognition and cognitive monitoring.
Frankland, P. W. & Bontempi, B. (2005). The organization of recent and remote memories.
Gödel, K. (1931). Über formal unentscheidbare Sätze.
Goertzel, B. et al. (2014). Engineering General Intelligence.
Goldman, A. I. (1979). What Is Justified Belief?
Goodman, N. (1955). Fact, Fiction, and Forecast.
Grace, A. A. (1991). Phasic versus tonic dopamine release.
Hamblin, C. L. (1970). Fallacies.
Hasselmo, M. E. (1999, 2006). Acetylcholine and memory consolidation.
Hebb, D. O. (1949). The Organization of Behavior.
Hegel, G. W. F. (1812–1816). Science of Logic.
Hobson, J. A. & McCarley, R. W. (1977). The brain as a dream state generator.
Hofstadter, D. R. (1979). Gödel, Escher, Bach.
Hume, D. (1739). A Treatise of Human Nature.
Immordino-Yang, M. H. & Damasio, A. (2007). We Feel, Therefore We Learn.
Jensen, A. & Mazaheri, A. (2010). Shaping functional architecture by oscillatory alpha activity.
Johnson-Laird, P. N. (1983). Mental Models.
Kahneman, D. (2011). Thinking, Fast and Slow.
Kahneman, D. & Tversky, A. (1979). Prospect Theory.
Kandel, E. R., et al. (2013). Principles of Neural Science, 5th ed.
Kant, I. (1781). Critique of Pure Reason.
Kierkegaard, S. (1841). The Concept of Irony.
Kloeden, P. E. & Platen, E. (1992). Numerical Solution of SDEs.
Knowles, M. S. (1975). Self-Directed Learning.
Kolb, D. A. (1984). Experiential Learning.
Kripke, S. (1975). Outline of a Theory of Truth.
Lakatos, I. (1978). The Methodology of Scientific Research Programmes.
Laird, J. E. (2012). The Soar Cognitive Architecture.
Lazarus, R. S. (1991). Emotion and Adaptation.
Lisman, J. E. & Jensen, O. (2013). The theta-gamma neural code.
Lu, J. et al. (2006). A putative flip-flop switch for control of REM sleep.
Luppi, P.-H. et al. (2011). The neuronal network responsible for paradoxical sleep.
Maquet, P. et al. (1996). Functional neuroanatomy of human REM sleep and dreaming.
Martin, L. G. (2026). HUMINT Operational Manual.
McAdams, D. P. (2001). The Psychology of Life Stories.
McClelland, J. L. et al. (1995). Why There Are Complementary Learning Systems.
Mezirow, J. (1991). Transformative Dimensions of Adult Learning.
Mill, J. S. (1859). On Liberty.
Miller, G. A. (1956). The Magical Number Seven.
Murillo-Rodríguez, E. et al. (2003). Anandamide enhances extracellular levels of adenosine.
Nelson, T. O. & Narens, L. (1990). Metamemory.
Newell, A. (1990). Unified Theories of Cognition.
Pace-Schott, E. F. & Hobson, J. A. (2002). The Neurobiology of Sleep.
Panksepp, J. (1998). Affective Neuroscience.
Paul, R. & Elder, L. (2006). The Art of Socratic Questioning.
Pekrun, R. (2006). The Control-Value Theory of Achievement Emotions.
Perogamvros, L. & Schwartz, S. (2012). The roles of the reward system in sleep.
Perry, W. G. (1970). Forms of Intellectual and Ethical Development.
Piaget, J. (1952). The Origins of Intelligence in Children.
Plato. Meno, Theaetetus, Euthyphro, Republic (c. 399–375 BCE).
Plutchik, R. (1980). Emotion: A Psychoevolutionary Synthesis.
Popper, K. (1934/1959). The Logic of Scientific Discovery.
Priest, G. (1987). In Contradiction.
Prior, A. N. (1967). Past, Present, and Future.
Quine, W. V. O. (1960). Word and Object.
Quine, W. V. O. (1966). The Ways of Paradox.
Robinson, R. (1953). Plato’s Earlier Dialectic.
Russell, B. (1903). The Principles of Mathematics.
Ryan, R. M. & Deci, E. L. (2000). Self-Determination Theory.
Sandberg, A. & Bostrom, N. (2008). Whole Brain Emulation: A Roadmap.
Saper, C. B. et al. (2001). The sleep switch.
Scherer, K. R. (2001). Appraisal Considered as a Process.
Schmidhuber, J. (2003). Gödel Machines.
Schön, D. A. (1983). The Reflective Practitioner.
Schultz, W. (1997). A Neural Substrate of Prediction and Reward.
Skinner, B. F. (1938). The Behavior of Organisms.
Sowa, J. F. (1984). Conceptual Structures.
Staresina, B. P. et al. (2015). Hierarchical nesting of slow oscillations, spindles and ripples.
Sterling, P. (2012). Allostasis: A model of predictive regulation.
Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning, 2nd ed.
Sweller, J. (1988). Cognitive Load During Problem Solving.
Toulmin, S. E. (1958). The Uses of Argument.
Tulving, E. (1972). Episodic and Semantic Memory.
Tversky, A. & Kahneman, D. (1974). Judgment under Uncertainty.
Uhlenbeck, G. E. & Ornstein, L. S. (1930). On the Theory of Brownian Motion.
van Eemeren, F. H. & Grootendorst, R. (1984). Speech Acts in Argumentative Discussions.
Vasicek, O. (1977). An Equilibrium Characterization of the Term Structure.
Vlastos, G. (1983). The Socratic Elenchus.
Vygotsky, L. S. (1978). Mind in Society.
Wagner, U. et al. (2001). Emotional memory formation enhanced across sleep intervals.
Wagner, U. et al. (2004). Sleep inspires insight.
Walker, M. P. & Stickgold, R. (2004). Sleep-dependent learning and memory consolidation.
Walton, D. N. (1989, 1991, 1992). Informal Logic; Begging the Question; Slippery Slope Arguments.
Wallas, G. (1926). The Art of Thought.
Wiggins, G. & McTighe, J. (2005). Understanding by Design.
Yerkes, R. M. & Dodson, J. D. (1908). Arousal-performance relationship.
Zimmerman, B. J. (2000). Self-Efficacy: An Essential Motive to Learn.

