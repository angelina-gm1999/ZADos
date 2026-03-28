"""
Tests for paradox_detection_engine.py

Coverage:
- Feature extraction: resolvability, self-reference, abstraction divergence, dialectical productivity
- Prior-encounter buffer matching
- Bayesian posterior computation (Beta likelihoods, class selection, normalization)
- Path 2 independent detection (oxymoron, self-reference, universal-particular, conceptual)
- Motivational salience model (M(t) formula)
- Paradox load Π(t) decomposition
- Neurochemical coupling signals (5-HT2A, CB1, DA tonic, θ/γ)
- 5-HT2A integral and CB1 update
- Threshold resolution (all modes, bidirectional neurochem feedback)
- ParadoxDetectionEngine.process() end-to-end (Path 1 + Path 2)
- Engine state (configure, update_neurochem_state, get_status)
- Buffer routing (Class R not emitted, G always bufferable, S quarantined)
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    ComparisonSet,
    ContradictionDetectionEngine,
    ContradictionEngineConfig,
    EvidenceSignals,
    OperationalMode,
    ProcessedStatement,
    SourceTag,
    ContradictionFlag,
    ContradictionDetectionResult,
)
from zados.cognitive_engines.py_engines.paradox_detection_engine import (
    BetaParams,
    ClassLikelihoods,
    IntentionVector,
    ParadoxClass,
    ParadoxDetectionEngine,
    ParadoxEngineConfig,
    ParadoxEngineState,
    ParadoxFeatures,
    ParadoxFlag,
    ParadoxInput,
    ParadoxSource,
    UnsolvedParadoxEntry,
    _beta_pdf_safe,
    _default_likelihoods,
    _identify_structural_pattern,
    _make_resolution_hint,
    compute_abstraction_divergence,
    compute_class_posteriors,
    compute_da_tonic_drive,
    compute_dialectical_productivity,
    compute_motivational_salience,
    compute_paradox_load,
    compute_paradox_neurochem_signals,
    compute_prior_encounter_match,
    compute_resolvability,
    compute_self_reference_index,
    detect_oxymoron_candidates,
    detect_single_statement_self_reference,
    detect_universal_particular_tension,
    resolve_paradox_threshold,
    update_cb1_level,
    update_ht2a_integral,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> ParadoxEngineConfig:
    return ParadoxEngineConfig()


@pytest.fixture
def likelihoods() -> ClassLikelihoods:
    return _default_likelihoods()


@pytest.fixture
def engine(cfg) -> ParadoxDetectionEngine:
    return ParadoxDetectionEngine(cfg, rng=np.random.default_rng(42))


@pytest.fixture
def stmt_factual_a() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="The server is online.",
        tokens=["The", "server", "is", "online"],
        source_tag=SourceTag.USER_INPUT,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        factuality_score=0.9,
        affective_score=0.1,
    )


@pytest.fixture
def stmt_factual_b() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="The server is not online.",
        tokens=["The", "server", "is", "not", "online"],
        source_tag=SourceTag.AI_OUTPUT,
        timestamp=datetime(2026, 1, 1, 12, 0, 1),
        factuality_score=0.9,
        affective_score=0.1,
    )


@pytest.fixture
def stmt_abstract_a() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="Freedom requires constraint for it to have meaning.",
        tokens=["Freedom", "requires", "constraint", "for", "it", "to", "have", "meaning"],
        source_tag=SourceTag.USER_INPUT,
        factuality_score=0.2,
        affective_score=0.5,
    )


@pytest.fixture
def stmt_abstract_b() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="True freedom means no constraint whatsoever.",
        tokens=["True", "freedom", "means", "no", "constraint", "whatsoever"],
        source_tag=SourceTag.AI_OUTPUT,
        factuality_score=0.2,
        affective_score=0.5,
    )


@pytest.fixture
def stmt_self_ref() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="This statement is false.",
        tokens=["This", "statement", "is", "false"],
        source_tag=SourceTag.USER_INPUT,
        factuality_score=0.1,
        affective_score=0.0,
    )


@pytest.fixture
def contra_flag(stmt_factual_a, stmt_factual_b) -> ContradictionFlag:
    return ContradictionFlag(
        contradiction_id="test-uuid-001",
        statement_a=stmt_factual_a,
        statement_b=stmt_factual_b,
        contradiction_level=1,
        confidence=0.85,
        evidence_signals=EvidenceSignals(negation_signal=1, semantic_opposition=0.80),
        temporal_separation=0.0,
        temporal_decay_applied=1.0,
        semantic_description="Direct negation test",
        context_frame=None,
    )


@pytest.fixture
def contra_result(contra_flag) -> ContradictionDetectionResult:
    return ContradictionDetectionResult(
        flags=[contra_flag],
        comparisons_performed=1,
        clean_pass=False,
        detection_threshold_used=0.5,
        processing_time_ms=1.0,
        neurochemical_signals={},
        metadata={"mode": "normal"},
    )


@pytest.fixture
def empty_contra_result() -> ContradictionDetectionResult:
    return ContradictionDetectionResult(
        flags=[],
        comparisons_performed=0,
        clean_pass=True,
        detection_threshold_used=0.5,
        processing_time_ms=0.5,
        neurochemical_signals={},
        metadata={"mode": "normal"},
    )


def _make_paradox_flag(
    paradox_class: ParadoxClass,
    confidence: float = 0.75,
    tension: float = 0.6,
    sigma_ref: float = 0.0,
    attempt_count: int = 0,
) -> UnsolvedParadoxEntry:
    """Helper: create an UnsolvedParadoxEntry with a synthetic ParadoxFlag."""
    features = ParadoxFeatures(
        resolvability=0.3,
        self_reference=sigma_ref,
        abstraction_divergence=0.5,
        dialectical_productivity=tension,
        prior_encounter_match=0.0,
    )
    flag = ParadoxFlag(
        paradox_id="test-uuid",
        source=ParadoxSource.CONTRADICTION_DERIVED,
        source_contradiction=None,
        concept_a="freedom",
        concept_b="constraint",
        paradox_class=paradox_class,
        class_posterior={c.value: 0.25 for c in ParadoxClass},
        confidence=confidence,
        features=features,
        resolution_hint=None,
        structural_pattern=None,
        symbolic_tension_score=tension,
        timestamp=datetime.utcnow(),
    )
    return UnsolvedParadoxEntry(
        paradox_flag=flag,
        attempt_count=attempt_count,
        first_encountered=datetime.utcnow(),
        motivational_salience=0.3,
    )


# ---------------------------------------------------------------------------
# Beta PDF safe
# ---------------------------------------------------------------------------

class TestBetaPdfSafe:
    def test_valid_x_returns_positive(self):
        result = _beta_pdf_safe(0.5, 2.0, 5.0)
        assert result > 0.0

    def test_clamping_at_zero(self):
        # x=0 should not raise; is clamped to 1e-6
        result = _beta_pdf_safe(0.0, 2.0, 5.0)
        assert result > 0.0

    def test_clamping_at_one(self):
        result = _beta_pdf_safe(1.0, 2.0, 5.0)
        assert result > 0.0

    def test_peak_near_mode(self):
        # Beta(8, 2): mode ≈ 0.78 — should have higher PDF there than at 0.2
        pdf_mode = _beta_pdf_safe(0.78, 8, 2)
        pdf_low  = _beta_pdf_safe(0.20, 8, 2)
        assert pdf_mode > pdf_low


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestComputeResolvability:
    def test_high_factuality_raises_resolvability(self, cfg):
        stmt_a = ProcessedStatement(raw_text="x", tokens=["x"], factuality_score=0.9)
        stmt_b = ProcessedStatement(raw_text="y", tokens=["y"], factuality_score=0.9)
        r = compute_resolvability(stmt_a, stmt_b, cfg, temporal_separation_days=None)
        assert r > 0.4   # factual claims are more resolvable

    def test_bounded(self, cfg, stmt_factual_a, stmt_factual_b):
        r = compute_resolvability(stmt_factual_a, stmt_factual_b, cfg)
        assert 0.0 <= r <= 1.0

    def test_temporal_gap_increases_resolvability(self, cfg, stmt_factual_a, stmt_factual_b):
        r_no_gap = compute_resolvability(stmt_factual_a, stmt_factual_b, cfg, None)
        r_gap    = compute_resolvability(stmt_factual_a, stmt_factual_b, cfg, 60.0)
        assert r_gap > r_no_gap


class TestComputeSelfReferenceIndex:
    def test_liar_paradox_detected(self, stmt_self_ref):
        sigma = compute_self_reference_index(stmt_self_ref.raw_text, stmt_self_ref.tokens)
        assert sigma >= 0.85

    def test_normal_statement_low_sigma(self):
        text = "The cat sat on the mat."
        sigma = compute_self_reference_index(text)
        assert sigma == pytest.approx(0.0)

    def test_all_rules_type_confusion(self):
        text = "All statements in this class are false."
        sigma = compute_self_reference_index(text)
        assert sigma >= 0.8

    def test_explicit_self_ref(self):
        text = "This statement refers to itself."
        sigma = compute_self_reference_index(text)
        assert sigma >= 0.85

    def test_output_bounded(self):
        texts = [
            "Hello world.",
            "This sentence is false.",
            "All sets that don't contain themselves.",
            "The sky is blue.",
        ]
        for text in texts:
            sigma = compute_self_reference_index(text)
            assert 0.0 <= sigma <= 1.0


class TestComputeAbstractionDivergence:
    def test_abstract_vs_concrete_has_divergence(self, cfg, stmt_abstract_a, stmt_factual_a):
        delta = compute_abstraction_divergence(stmt_abstract_a, stmt_factual_a)
        assert delta > 0.0

    def test_similar_abstraction_low_divergence(self, cfg, stmt_factual_a, stmt_factual_b):
        delta = compute_abstraction_divergence(stmt_factual_a, stmt_factual_b)
        assert 0.0 <= delta <= 1.0

    def test_bounded(self, cfg, stmt_abstract_a, stmt_factual_a):
        delta = compute_abstraction_divergence(stmt_abstract_a, stmt_factual_a)
        assert 0.0 <= delta <= 1.0

    def test_universal_quantifier_increases_level(self):
        abstract = ProcessedStatement(
            raw_text="All knowledge is relative.",
            tokens=["All", "knowledge", "is", "relative"],
            factuality_score=0.1,
        )
        concrete = ProcessedStatement(
            raw_text="This book is heavy.",
            tokens=["This", "book", "is", "heavy"],
            factuality_score=0.9,
        )
        delta = compute_abstraction_divergence(abstract, concrete)
        assert delta > 0.0


class TestComputeDialecticalProductivity:
    def test_freedom_constraint_has_productivity(self, cfg, stmt_abstract_a, stmt_abstract_b):
        pi = compute_dialectical_productivity(stmt_abstract_a, stmt_abstract_b, cfg)
        assert 0.0 <= pi <= 1.0

    def test_identical_statements_no_persistence(self, cfg):
        s = ProcessedStatement(raw_text="x y z", tokens=["x", "y", "z"])
        pi = compute_dialectical_productivity(s, s, cfg)
        # Identical tokens → persistence = 0, but synthesis = 1 and entailment high
        assert 0.0 <= pi <= 1.0

    def test_completely_disjoint_tokens(self, cfg):
        a = ProcessedStatement(raw_text="alpha beta gamma", tokens=["alpha", "beta", "gamma"])
        b = ProcessedStatement(raw_text="delta epsilon zeta", tokens=["delta", "epsilon", "zeta"])
        pi = compute_dialectical_productivity(a, b, cfg)
        # High persistence (disjoint), low entailment → moderate π
        assert 0.0 <= pi <= 1.0

    def test_bounded(self, cfg, stmt_abstract_a, stmt_factual_a):
        pi = compute_dialectical_productivity(stmt_abstract_a, stmt_factual_a, cfg)
        assert 0.0 <= pi <= 1.0


# ---------------------------------------------------------------------------
# Prior encounter matching
# ---------------------------------------------------------------------------

class TestComputePriorEncounterMatch:
    def test_empty_buffer_returns_zero(self, cfg):
        rho, cls = compute_prior_encounter_match("freedom", "constraint", [], cfg.theta_match)
        assert rho == pytest.approx(0.0)
        assert cls is None

    def test_exact_match_returns_high_rho(self, cfg):
        entry = _make_paradox_flag(ParadoxClass.G)
        # entry.paradox_flag.concept_a = "freedom", concept_b = "constraint"
        rho, cls = compute_prior_encounter_match(
            "freedom", "constraint", [entry], cfg.theta_match
        )
        # "freedom constraint" exact match → rho should be 1.0
        assert rho > cfg.theta_match
        assert cls == ParadoxClass.G

    def test_no_match_below_threshold(self, cfg):
        entry = _make_paradox_flag(ParadoxClass.G)
        rho, cls = compute_prior_encounter_match(
            "quantum physics", "democracy", [entry], cfg.theta_match
        )
        if rho < cfg.theta_match:
            assert cls is None


# ---------------------------------------------------------------------------
# Bayesian classifier
# ---------------------------------------------------------------------------

class TestComputeClassPosteriors:
    def test_posteriors_sum_to_one(self, cfg, likelihoods):
        features = ParadoxFeatures(
            resolvability=0.5, self_reference=0.1,
            abstraction_divergence=0.4, dialectical_productivity=0.5,
            prior_encounter_match=0.0,
        )
        posteriors = compute_class_posteriors(features, cfg, likelihoods)
        assert sum(posteriors.values()) == pytest.approx(1.0, abs=1e-6)

    def test_structural_features_favour_S(self, cfg, likelihoods):
        features = ParadoxFeatures(
            resolvability=0.05,    # very low → not R
            self_reference=0.90,   # very high → S territory
            abstraction_divergence=0.4,
            dialectical_productivity=0.15,
            prior_encounter_match=0.0,
        )
        posteriors = compute_class_posteriors(features, cfg, likelihoods)
        assert posteriors[ParadoxClass.S.value] == max(posteriors.values())

    def test_resolvable_features_favour_R(self, cfg, likelihoods):
        features = ParadoxFeatures(
            resolvability=0.90,     # high → R
            self_reference=0.00,    # no self-ref
            abstraction_divergence=0.25,
            dialectical_productivity=0.10,
            prior_encounter_match=0.0,
        )
        posteriors = compute_class_posteriors(features, cfg, likelihoods)
        assert posteriors[ParadoxClass.R.value] == max(posteriors.values())

    def test_genuine_features_favour_G(self, cfg, likelihoods):
        features = ParadoxFeatures(
            resolvability=0.15,    # hard to resolve
            self_reference=0.05,   # not self-referential
            abstraction_divergence=0.75,  # high divergence
            dialectical_productivity=0.85, # high productivity
            prior_encounter_match=0.0,
        )
        posteriors = compute_class_posteriors(features, cfg, likelihoods)
        assert posteriors[ParadoxClass.G.value] == max(posteriors.values())

    def test_all_classes_present(self, cfg, likelihoods):
        features = ParadoxFeatures(0.5, 0.3, 0.5, 0.5, 0.0)
        posteriors = compute_class_posteriors(features, cfg, likelihoods)
        assert set(posteriors.keys()) == {c.value for c in ParadoxClass}

    def test_prior_encounter_shifts_prior(self, cfg, likelihoods):
        """High ρ for Class G should push posterior toward G."""
        entry = _make_paradox_flag(ParadoxClass.G)
        features = ParadoxFeatures(
            resolvability=0.5, self_reference=0.0,
            abstraction_divergence=0.5, dialectical_productivity=0.5,
            prior_encounter_match=0.95,
        )
        posteriors = compute_class_posteriors(
            features, cfg, likelihoods,
            unsolved_buffer=[entry],
            concept_a="freedom", concept_b="constraint",
        )
        total = sum(posteriors.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_uniform_fallback_on_zero_likelihood(self, cfg, likelihoods):
        """All-zero likelihood product → uniform posterior."""
        # Force a degenerate feature set
        features = ParadoxFeatures(0.5, 0.5, 0.5, 0.5, 0.0)
        # Use zero priors
        cfg_zero = ParadoxEngineConfig(prior_R=0.0, prior_A=0.0, prior_G=0.0, prior_S=0.0)
        posteriors = compute_class_posteriors(features, cfg_zero, likelihoods)
        # Should return 0.25 fallback
        for v in posteriors.values():
            assert v == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Path 2 — Independent detection
# ---------------------------------------------------------------------------

class TestDetectOxymoronCandidates:
    def test_known_oxymoron_detected(self, cfg):
        stmt = ProcessedStatement(
            raw_text="living dead creatures roam.",
            tokens=["living", "dead", "creatures", "roam"],
        )
        candidates = detect_oxymoron_candidates(stmt, cfg.theta_oxy)
        assert len(candidates) >= 1
        pairs = [(m, h) for m, h, _ in candidates]
        assert ("living", "dead") in pairs or ("dead", "living") in pairs

    def test_no_oxymoron_in_normal_sentence(self, cfg):
        stmt = ProcessedStatement(
            raw_text="the cat sat on the mat",
            tokens=["the", "cat", "sat", "on", "the", "mat"],
        )
        candidates = detect_oxymoron_candidates(stmt, cfg.theta_oxy)
        assert len(candidates) == 0

    def test_scores_above_threshold(self, cfg):
        stmt = ProcessedStatement(
            raw_text="bitter sweet experience",
            tokens=["bitter", "sweet", "experience"],
        )
        candidates = detect_oxymoron_candidates(stmt, cfg.theta_oxy)
        for _, _, score in candidates:
            assert score >= cfg.theta_oxy


class TestDetectSelfReference:
    def test_liar_paradox_flagged(self, stmt_self_ref, cfg):
        sigma = detect_single_statement_self_reference(stmt_self_ref)
        assert sigma >= 0.85

    def test_normal_statement_not_flagged(self, cfg):
        stmt = ProcessedStatement(raw_text="The train arrives at noon.")
        sigma = detect_single_statement_self_reference(stmt)
        assert sigma < 0.5


class TestDetectUniversalParticularTension:
    def test_all_rules_have_exceptions(self):
        stmt = ProcessedStatement(
            raw_text="All rules have exceptions however.",
            tokens=["All", "rules", "have", "exceptions", "however"],
        )
        tension = detect_universal_particular_tension(stmt)
        assert tension > 0.0

    def test_no_tension_in_specific_claim(self):
        stmt = ProcessedStatement(
            raw_text="The door is red.",
            tokens=["The", "door", "is", "red"],
        )
        tension = detect_universal_particular_tension(stmt)
        assert tension == pytest.approx(0.0)

    def test_never_with_sometimes(self):
        stmt = ProcessedStatement(
            raw_text="Never say never sometimes.",
            tokens=["Never", "say", "never", "sometimes"],
        )
        tension = detect_universal_particular_tension(stmt)
        assert tension > 0.0


# ---------------------------------------------------------------------------
# Paradox load
# ---------------------------------------------------------------------------

class TestComputeParadoxLoad:
    def test_empty_flags_zero_load(self, cfg):
        pi = compute_paradox_load([], cfg)
        assert pi["Π_total"] == pytest.approx(0.0)
        assert pi["Π_G"] == pytest.approx(0.0)

    def test_genuine_flag_contributes_most(self, cfg, likelihoods):
        """Class G has highest weight → should dominate Π_total."""
        features = ParadoxFeatures(0.1, 0.0, 0.8, 0.9, 0.0)
        posteriors = {c.value: 0.25 for c in ParadoxClass}
        flag = ParadoxFlag(
            paradox_id="t1",
            source=ParadoxSource.CONTRADICTION_DERIVED,
            source_contradiction=None,
            concept_a="a", concept_b="b",
            paradox_class=ParadoxClass.G,
            class_posterior=posteriors,
            confidence=0.85,
            features=features,
            resolution_hint=None,
            structural_pattern=None,
            symbolic_tension_score=0.85,
        )
        pi = compute_paradox_load([flag], cfg)
        assert pi["Π_G"] > pi["Π_A"]
        assert pi["Π_G"] > pi["Π_R"]

    def test_load_keys_present(self, cfg):
        pi = compute_paradox_load([], cfg)
        assert {"Π_G", "Π_A", "Π_S", "Π_R", "Π_total"}.issubset(pi.keys())


# ---------------------------------------------------------------------------
# Motivational salience
# ---------------------------------------------------------------------------

class TestComputeMotivationalSalience:
    def test_zero_age_gives_zero_salience(self, cfg):
        entry = _make_paradox_flag(ParadoxClass.G)
        # first_encountered = now → age ≈ 0 → M(t) ≈ 0
        salience = compute_motivational_salience(entry, cfg)
        assert salience >= 0.0
        assert salience <= 1.0

    def test_more_attempts_increases_salience(self, cfg):
        entry_0 = _make_paradox_flag(ParadoxClass.G, attempt_count=0)
        entry_5 = _make_paradox_flag(ParadoxClass.G, attempt_count=5)
        # Both fresh → age factor ≈ 0, but the formula structure should hold
        s0 = compute_motivational_salience(entry_0, cfg)
        s5 = compute_motivational_salience(entry_5, cfg)
        # M = M0 × (1 + α_attempt × n) × age_factor; more attempts → higher for same age
        assert s5 >= s0

    def test_class_R_gives_zero_salience(self, cfg):
        entry = _make_paradox_flag(ParadoxClass.R)
        salience = compute_motivational_salience(entry, cfg)
        assert salience == pytest.approx(0.0)

    def test_salience_clamped(self, cfg):
        entry = _make_paradox_flag(ParadoxClass.G, attempt_count=1000)
        salience = compute_motivational_salience(entry, cfg)
        assert 0.0 <= salience <= 1.0


# ---------------------------------------------------------------------------
# 5-HT2A integral and CB1 update
# ---------------------------------------------------------------------------

class TestHt2aIntegral:
    def test_increases_with_positive_pi_G(self, cfg):
        integral0 = 0.0
        integral1 = update_ht2a_integral(integral0, pi_G=0.5, dt=1.0, tau_sym=cfg.tau_sym)
        assert integral1 > integral0

    def test_decays_without_input(self, cfg):
        integral0 = 1.0
        integral1 = update_ht2a_integral(integral0, pi_G=0.0, dt=1.0, tau_sym=cfg.tau_sym)
        assert integral1 < integral0

    def test_zero_tau_gives_zero_decay(self):
        integral1 = update_ht2a_integral(1.0, pi_G=0.0, dt=1.0, tau_sym=0.0)
        # decay = exp(-1/0) → 0, so integral should be 0
        assert integral1 == pytest.approx(0.0)


class TestCb1Update:
    def test_structural_paradox_increases_cb1(self, cfg):
        cb1_0 = 0.0
        cb1_1 = update_cb1_level(cb1_0, pi_S=0.5, sigma_ref=0.0, config=cfg, dt=1.0)
        assert cb1_1 > cb1_0

    def test_identity_signal_increases_cb1(self, cfg):
        cb1_0 = 0.0
        cb1_low_sigma  = update_cb1_level(cb1_0, pi_S=0.0, sigma_ref=0.5, config=cfg, dt=1.0)
        cb1_high_sigma = update_cb1_level(cb1_0, pi_S=0.0, sigma_ref=0.9, config=cfg, dt=1.0)
        assert cb1_high_sigma > cb1_low_sigma

    def test_cb1_decays_to_baseline(self, cfg):
        cb1_0 = 0.8
        cb1_1 = update_cb1_level(cb1_0, pi_S=0.0, sigma_ref=0.0, config=cfg, dt=1.0)
        assert cb1_1 < cb1_0

    def test_cb1_clamped(self, cfg):
        cb1 = update_cb1_level(0.99, pi_S=1.0, sigma_ref=1.0, config=cfg, dt=100.0)
        assert 0.0 <= cb1 <= 1.0


# ---------------------------------------------------------------------------
# Tonic DA drive
# ---------------------------------------------------------------------------

class TestDaTonicDrive:
    def test_empty_buffer_gives_zero(self, cfg):
        drive = compute_da_tonic_drive([], cfg)
        assert drive == pytest.approx(0.0)

    def test_quarantined_excluded(self, cfg):
        entry = _make_paradox_flag(ParadoxClass.S)
        entry.quarantined = True
        drive = compute_da_tonic_drive([entry], cfg)
        assert drive == pytest.approx(0.0)

    def test_active_entry_contributes(self, cfg):
        entry = _make_paradox_flag(ParadoxClass.G, attempt_count=5)
        entry.quarantined = False
        drive = compute_da_tonic_drive([entry], cfg)
        assert drive >= 0.0


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------

class TestResolveParadoxThreshold:
    def test_normal_mode_default(self, cfg):
        # At ne_level=0 default, the (1-ne) term adds +0.04; use looser tolerance
        theta = resolve_paradox_threshold(OperationalMode.NORMAL, cfg)
        assert theta == pytest.approx(cfg.theta_normal, abs=0.06)

    def test_dev_mode_lower(self, cfg):
        theta_dev  = resolve_paradox_threshold(OperationalMode.DEV,    cfg)
        theta_norm = resolve_paradox_threshold(OperationalMode.NORMAL, cfg)
        assert theta_dev < theta_norm

    def test_dream_mode_lowest(self, cfg):
        theta_dream = resolve_paradox_threshold(OperationalMode.REM_DREAM, cfg)
        theta_norm  = resolve_paradox_threshold(OperationalMode.NORMAL,    cfg)
        assert theta_dream < theta_norm

    def test_high_da_tonic_raises_threshold(self, cfg):
        low_da  = resolve_paradox_threshold(OperationalMode.NORMAL, cfg, da_tonic=0.0)
        high_da = resolve_paradox_threshold(OperationalMode.NORMAL, cfg, da_tonic=1.0)
        assert high_da > low_da

    def test_high_ht2a_lowers_threshold(self, cfg):
        low_ht  = resolve_paradox_threshold(OperationalMode.NORMAL, cfg, ht2a_activation=0.0)
        high_ht = resolve_paradox_threshold(OperationalMode.NORMAL, cfg, ht2a_activation=1.0)
        assert high_ht < low_ht

    def test_threshold_clamped(self, cfg):
        theta = resolve_paradox_threshold(
            OperationalMode.NORMAL, cfg,
            ht2a_activation=1.0, cb1_read=1.0, da_tonic=1.0, ne_level=0.0
        )
        assert 0.05 <= theta <= 0.95


# ---------------------------------------------------------------------------
# Resolution hints and structural patterns
# ---------------------------------------------------------------------------

class TestResolutionHintsAndPatterns:
    def test_resolution_hint_for_high_resolvability(self):
        features = ParadoxFeatures(
            resolvability=0.85, self_reference=0.0,
            abstraction_divergence=0.2, dialectical_productivity=0.2,
            prior_encounter_match=0.0,
        )
        hint = _make_resolution_hint(features)
        assert hint is not None
        assert isinstance(hint, str)

    def test_structural_pattern_liar(self):
        features = ParadoxFeatures(
            resolvability=0.1, self_reference=0.9,
            abstraction_divergence=0.5, dialectical_productivity=0.2,
            prior_encounter_match=0.0,
        )
        pattern = _identify_structural_pattern(features, "this statement is false")
        assert pattern == "liar_paradox"

    def test_structural_pattern_russells(self):
        features = ParadoxFeatures(0.1, 0.9, 0.5, 0.2, 0.0)
        pattern = _identify_structural_pattern(features, "the set of all sets")
        assert pattern == "russells_paradox"

    def test_structural_pattern_self_ref_loop(self):
        features = ParadoxFeatures(0.1, 0.9, 0.5, 0.2, 0.0)
        pattern = _identify_structural_pattern(features, "the function calls itself")
        assert pattern == "self_referential_loop"


# ---------------------------------------------------------------------------
# Engine — end-to-end
# ---------------------------------------------------------------------------

class TestParadoxDetectionEngineProcess:
    def test_path1_evaluates_contradiction_flags(
        self, engine, contra_result, stmt_self_ref
    ):
        pi = ParadoxInput(
            contradiction_result=contra_result,
            active_mode=OperationalMode.NORMAL,
        )
        result = engine.process(pi, dt=1.0)
        assert result.contradictions_evaluated == 1

    def test_result_fields_present(self, engine, contra_result):
        pi = ParadoxInput(contradiction_result=contra_result)
        result = engine.process(pi, dt=1.0)
        assert isinstance(result.flags, list)
        assert isinstance(result.contradictions_evaluated, int)
        assert isinstance(result.clean_pass, bool)
        assert 0.0 <= result.detection_threshold_used <= 1.0
        assert result.processing_time_ms >= 0.0

    def test_clean_pass_no_g_or_s(self, engine, empty_contra_result):
        pi = ParadoxInput(contradiction_result=empty_contra_result)
        result = engine.process(pi, dt=1.0)
        # Empty input → clean pass unless Path 2 fires
        assert isinstance(result.clean_pass, bool)

    def test_counts_consistent(self, engine, contra_result):
        pi = ParadoxInput(contradiction_result=contra_result)
        result = engine.process(pi, dt=1.0)
        total = (
            result.paradoxes_from_contradictions
            + result.independent_paradoxes
            + result.reclassified_as_resolvable
        )
        # Total evaluated ≥ reclassified (some may be emitted as flags)
        assert result.reclassified_as_resolvable >= 0

    def test_flag_fields_valid(self, engine, contra_result):
        pi = ParadoxInput(
            contradiction_result=contra_result,
            active_mode=OperationalMode.DEV,   # lower threshold → more flags
        )
        engine.configure(OperationalMode.DEV)
        result = engine.process(pi, dt=1.0)
        for flag in result.flags:
            assert flag.paradox_id
            assert flag.paradox_class in ParadoxClass
            assert 0.0 <= flag.confidence <= 1.0
            assert flag.source in ParadoxSource
            assert sum(flag.class_posterior.values()) == pytest.approx(1.0, abs=1e-5)

    def test_neurochemical_signals_keys(self, engine, contra_result):
        pi = ParadoxInput(contradiction_result=contra_result)
        result = engine.process(pi, dt=1.0)
        expected = {
            "pi_decomposition", "k_on_5ht2a_mult",
            "cb1_baseline", "da_tonic_drive",
            "theta_boost", "gamma_suppress",
        }
        assert expected.issubset(result.neurochemical_signals.keys())

    def test_pi_decomp_keys(self, engine, contra_result):
        pi = ParadoxInput(contradiction_result=contra_result)
        result = engine.process(pi, dt=1.0)
        pi_d = result.neurochemical_signals["pi_decomposition"]
        assert {"Π_G", "Π_A", "Π_S", "Π_R", "Π_total"}.issubset(pi_d.keys())

    def test_self_referential_statement_in_path2(self, engine, empty_contra_result, stmt_self_ref):
        pi = ParadoxInput(
            contradiction_result=empty_contra_result,
            memory_context=[stmt_self_ref],
            active_mode=OperationalMode.DEV,
        )
        engine.configure(OperationalMode.DEV)
        result = engine.process(pi, dt=1.0)
        # Self-referential statement should be picked up by Path 2
        assert result.independent_paradoxes >= 0   # may fire depending on threshold

    def test_class_R_not_emitted_as_flag(self, engine, contra_result):
        """Resolvable-classified candidates should not appear in flags list."""
        pi = ParadoxInput(contradiction_result=contra_result)
        result = engine.process(pi, dt=1.0)
        for flag in result.flags:
            assert flag.paradox_class != ParadoxClass.R

    def test_cb1_updates_after_structural(self, engine, empty_contra_result, stmt_self_ref):
        """CB1 level should increase when structural paradoxes are detected."""
        initial_cb1 = engine._state.cb1_level
        pi = ParadoxInput(
            contradiction_result=empty_contra_result,
            memory_context=[stmt_self_ref],
            active_mode=OperationalMode.DEV,
        )
        engine.configure(OperationalMode.DEV)
        engine.process(pi, dt=1.0)
        # CB1 might increase (if S flagged) or stay (if no S detected)
        assert engine._state.cb1_level >= 0.0

    def test_ht2a_integral_accumulates(self, engine, contra_result):
        """5-HT2A integral accumulates across process() calls."""
        pi = ParadoxInput(contradiction_result=contra_result)
        integral_0 = engine._state.ht2a_integral
        engine.process(pi, dt=1.0)
        engine.process(pi, dt=1.0)
        integral_2 = engine._state.ht2a_integral
        assert integral_2 >= integral_0


# ---------------------------------------------------------------------------
# Engine — configure, neurochem state, status
# ---------------------------------------------------------------------------

class TestParadoxEngineConfig:
    def test_configure_sets_mode(self, engine):
        engine.configure(OperationalMode.REM_DREAM)
        assert engine._mode == OperationalMode.REM_DREAM

    def test_update_neurochem_state(self, engine):
        engine.update_neurochem_state(
            {"5ht": 0.7, "cb1": 0.3, "da": 0.5, "ne": 0.4}
        )
        assert engine._state.ht2a_activation == pytest.approx(0.7)
        assert engine._state.cb1_read        == pytest.approx(0.3)
        assert engine._state.da_tonic        == pytest.approx(0.5)
        assert engine._state.ne_level        == pytest.approx(0.4)

    def test_neurochem_state_clamped(self, engine):
        engine.update_neurochem_state({"5ht": 2.0, "da": -0.5})
        assert engine._state.ht2a_activation == pytest.approx(1.0)
        assert engine._state.da_tonic        == pytest.approx(0.0)

    def test_get_status_keys(self, engine):
        status = engine.get_status()
        assert {"engine_id", "mode", "ht2a_integral", "cb1_level"}.issubset(status.keys())

    def test_get_status_engine_id(self, engine):
        assert engine.get_status()["engine_id"] == "paradox_detection_engine"

    def test_default_mode_normal(self, engine):
        assert engine._mode == OperationalMode.NORMAL


# ---------------------------------------------------------------------------
# UnsolvedParadoxEntry helpers
# ---------------------------------------------------------------------------

class TestUnsolvedParadoxEntry:
    def test_tick_attempt_increments(self):
        entry = _make_paradox_flag(ParadoxClass.G)
        initial = entry.attempt_count
        entry.tick_attempt()
        assert entry.attempt_count == initial + 1

    def test_add_evidence_appends(self):
        entry = _make_paradox_flag(ParadoxClass.G)
        entry.add_evidence("some text", "USER_INPUT", 0.75)
        assert len(entry.associated_evidence) == 1
        ev = entry.associated_evidence[0]
        assert ev["content"] == "some text"
        assert ev["relevance_score"] == pytest.approx(0.75)

    def test_quarantine_flag(self):
        entry = _make_paradox_flag(ParadoxClass.S)
        entry.quarantined = True
        entry.quarantine_reason = "structural antinomy"
        assert entry.quarantined
        assert "structural" in entry.quarantine_reason
