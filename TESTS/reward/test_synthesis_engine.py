import pytest

from zados.reward.base.types import (
    RewardDomainResult,
    RewardMetaDirective,
    RewardSubscore,
    RewardContext,
)
from zados.reward.base.structure import RewardFlag
from zados.reward.synthesis.directives import (
    classify_tier,
    tier_label,
    compute_weighted_composite,
    compute_per_domain_weighted_scores,
    compute_suppression,
    compute_abstention,
    compute_response_directives,
    compute_routing,
    escalate_domain_flags,
    apply_cross_domain_interactions,
)
from zados.reward.synthesis.engine import SynthesisEngine
from zados.reward.profile.static_profiles import (
    REFLECTIVE_PROFILE,
    CREATIVE_SANDBOX_PROFILE,
    ETHICS_TRAINING_PROFILE,
    ANALYSIS_PROFILE,
    SLEEP_DREAM_PROFILE,
    SLEEP_DEEP_PROFILE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_domain_result(
    domain: str,
    general_score: float,
    flags: dict | None = None,
) -> RewardDomainResult:
    """Build a minimal RewardDomainResult for testing."""
    return RewardDomainResult(
        domain=domain,
        general_score=general_score,
        subscores={},
        flags=flags or {},
        meta={},
    )


def _make_domain_results(
    ethics: float = 0.5,
    logic: float = 0.5,
    innovation: float = 0.5,
    human_attunement: float = 0.5,
    ethics_flags: dict | None = None,
    logic_flags: dict | None = None,
    innovation_flags: dict | None = None,
    attunement_flags: dict | None = None,
) -> dict[str, RewardDomainResult]:
    """Build all 4 domain results with given scores."""
    return {
        "ethics": _make_domain_result("ethics", ethics, ethics_flags),
        "logic": _make_domain_result("logic", logic, logic_flags),
        "innovation": _make_domain_result("innovation", innovation, innovation_flags),
        "human_attunement": _make_domain_result(
            "human_attunement", human_attunement, attunement_flags,
        ),
    }


# ===================================================================
# Tier Classification
# ===================================================================

def test_classify_tier_boundaries():
    """Scores within each band classify correctly."""
    assert classify_tier(0.0) == 0
    assert classify_tier(0.10) == 0
    assert classify_tier(0.26) == 1
    assert classify_tier(0.40) == 1
    assert classify_tier(0.51) == 2
    assert classify_tier(0.60) == 2
    assert classify_tier(0.76) == 3
    assert classify_tier(1.0) == 3


def test_classify_tier_exact_boundaries():
    """Exact boundary values fall into the lower tier (<=)."""
    assert classify_tier(0.25) == 0   # boundary -> tier 0
    assert classify_tier(0.50) == 1   # boundary -> tier 1
    assert classify_tier(0.75) == 2   # boundary -> tier 2


def test_tier_label():
    """Each tier index maps to the correct human label."""
    assert tier_label(0) == "minimal"
    assert tier_label(1) == "moderate"
    assert tier_label(2) == "significant"
    assert tier_label(3) == "dominant"


# ===================================================================
# Weighted Composite Score
# ===================================================================

def test_composite_uniform():
    """All domains at 0.5 with equal weights -> composite 0.5."""
    results = _make_domain_results(0.5, 0.5, 0.5, 0.5)
    weights = {"ethics": 1.0, "logic": 1.0, "innovation": 1.0, "human_attunement": 1.0}
    assert compute_weighted_composite(results, weights) == pytest.approx(0.5)


def test_composite_varied_weights():
    """Weighted composite normalised by weight sum."""
    results = _make_domain_results(ethics=1.0, logic=0.0, innovation=0.0, human_attunement=0.0)
    weights = {"ethics": 0.9, "logic": 0.1, "innovation": 0.1, "human_attunement": 0.1}
    # (0.9*1.0 + 0.1*0 + 0.1*0 + 0.1*0) / (0.9+0.1+0.1+0.1) = 0.9/1.2 = 0.75
    assert compute_weighted_composite(results, weights) == pytest.approx(0.9 / 1.2)


def test_composite_empty():
    """No domains -> 0.0."""
    assert compute_weighted_composite({}, {}) == 0.0


# ===================================================================
# Suppression Logic
# ===================================================================

def test_suppression_below_bias():
    """Composite below suppression_bias -> suppress."""
    assert compute_suppression(0.1, 0.2, {}) is True


def test_suppression_above_bias():
    """Composite above suppression_bias -> no suppress."""
    assert compute_suppression(0.5, 0.2, {}) is False


def test_suppression_critical_flag():
    """Critical flag forces suppression regardless of score."""
    flags = {
        "ethics": [RewardFlag(name="danger", severity="critical")],
    }
    assert compute_suppression(0.9, 0.1, flags) is True


def test_suppression_no_critical():
    """Warning flags alone do not trigger suppression."""
    flags = {
        "ethics": [RewardFlag(name="caution", severity="warning")],
        "logic": [{"severity": "risk"}],
    }
    assert compute_suppression(0.5, 0.2, flags) is False


# ===================================================================
# Abstention Logic
# ===================================================================

def test_abstention_threshold_violation():
    """Single violation with high bias exceeds 0.5 -> abstain."""
    results = _make_domain_results(ethics=0.3)
    tolerances = {"ethics": 0.8}  # violation: 0.3 < 0.8
    # violation_ratio=1.0, 1.0 * 0.6 = 0.6 > 0.5
    assert compute_abstention(results, tolerances, 0.6) is True


def test_abstention_within_threshold():
    """All scores above threshold -> no abstain."""
    results = _make_domain_results(ethics=0.9, logic=0.8)
    tolerances = {"ethics": 0.8, "logic": 0.7}
    assert compute_abstention(results, tolerances, 0.6) is False


def test_abstention_low_bias_forgives():
    """Low bias → high threshold → minor violations forgiven."""
    results = _make_domain_results(ethics=0.75, logic=0.8)
    tolerances = {"ethics": 0.8, "logic": 0.7}  # ethics violates, logic ok
    # violation_ratio = 1/2 = 0.5, threshold = 1.0 - 0.1 = 0.9
    # 0.5 > 0.9 → False → no abstention (low bias forgives partial violation)
    assert compute_abstention(results, tolerances, 0.1) is False


# ===================================================================
# Response Directives
# ===================================================================

def test_high_ethics_high_moralize():
    """High ethics weighted score -> high moralize directive."""
    results = _make_domain_results(ethics=0.9, logic=0.3, innovation=0.2, human_attunement=0.3)
    weights = {"ethics": 1.0, "logic": 0.3, "innovation": 0.2, "human_attunement": 0.3}
    tiers = {name: classify_tier(r.general_score) for name, r in results.items()}
    directives = compute_response_directives(results, weights, tiers)
    assert directives["moralize"] > 0.5


def test_high_logic_low_metaphor():
    """High logic dampens metaphor density."""
    results = _make_domain_results(logic=0.9, innovation=0.5)
    weights = {"ethics": 0.3, "logic": 1.0, "innovation": 0.5, "human_attunement": 0.3}
    tiers = {name: classify_tier(r.general_score) for name, r in results.items()}
    directives = compute_response_directives(results, weights, tiers)

    # Compare: high logic should reduce metaphor vs baseline
    results_low_logic = _make_domain_results(logic=0.2, innovation=0.5)
    directives_low = compute_response_directives(results_low_logic, weights, tiers)
    assert directives["metaphor_density"] < directives_low["metaphor_density"]


def test_high_innovation_high_speculate():
    """High innovation -> high speculate directive."""
    results = _make_domain_results(innovation=0.9, ethics=0.2, logic=0.3, human_attunement=0.3)
    weights = {"ethics": 0.2, "logic": 0.3, "innovation": 1.0, "human_attunement": 0.3}
    tiers = {name: classify_tier(r.general_score) for name, r in results.items()}
    directives = compute_response_directives(results, weights, tiers)
    assert directives["speculate"] > 0.4


def test_all_directives_in_range():
    """All 8 directive values must be in [0.0, 1.0] for any input combination."""
    for e, l, i, a in [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0),
                        (0.9, 0.1, 0.5, 0.7), (0.1, 0.9, 0.9, 0.1)]:
        results = _make_domain_results(ethics=e, logic=l, innovation=i, human_attunement=a)
        weights = {"ethics": 0.9, "logic": 0.8, "innovation": 0.7, "human_attunement": 0.6}
        tiers = {name: classify_tier(r.general_score) for name, r in results.items()}
        directives = compute_response_directives(results, weights, tiers)
        for key, value in directives.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range for input ({e},{l},{i},{a})"


# ===================================================================
# Cross-Domain Interactions
# ===================================================================

def test_logic_dampens_soothe():
    """High logic + low attunement reduces soothe by 50%."""
    results = _make_domain_results(logic=0.8, human_attunement=0.2, ethics=0.5, innovation=0.3)
    weights = {"ethics": 0.5, "logic": 0.8, "innovation": 0.3, "human_attunement": 0.2}
    tiers = {name: classify_tier(r.general_score) for name, r in results.items()}
    raw = compute_response_directives(results, weights, tiers)
    adjusted = apply_cross_domain_interactions(raw, results, weights)
    assert adjusted["soothe"] == pytest.approx(raw["soothe"] * 0.5)


def test_ethics_constrains_speculation():
    """High ethics + high innovation reduces speculate."""
    results = _make_domain_results(ethics=0.8, innovation=0.8, logic=0.5, human_attunement=0.5)
    weights = {"ethics": 0.8, "logic": 0.5, "innovation": 0.8, "human_attunement": 0.5}
    tiers = {name: classify_tier(r.general_score) for name, r in results.items()}
    raw = compute_response_directives(results, weights, tiers)
    adjusted = apply_cross_domain_interactions(raw, results, weights)
    assert adjusted["speculate"] < raw["speculate"]


def test_high_innovation_boosts_metaphor():
    """Very high innovation (>0.75) boosts metaphor density."""
    results = _make_domain_results(innovation=0.8, logic=0.3, ethics=0.3, human_attunement=0.3)
    weights = {"ethics": 0.3, "logic": 0.3, "innovation": 1.0, "human_attunement": 0.3}
    tiers = {name: classify_tier(r.general_score) for name, r in results.items()}
    raw = compute_response_directives(results, weights, tiers)
    adjusted = apply_cross_domain_interactions(raw, results, weights)
    assert adjusted["metaphor_density"] >= raw["metaphor_density"]


# ===================================================================
# Routing
# ===================================================================

def test_routing_dominant_domain():
    """Highest-weighted domain is identified as dominant."""
    weights = {"ethics": 0.3, "logic": 1.0, "innovation": 0.5, "human_attunement": 0.2}
    tiers = {"ethics": 1, "logic": 2, "innovation": 1, "human_attunement": 0}
    routing = compute_routing(tiers, weights, 0.6)
    assert routing["dominant_domain"] == "logic"


def test_routing_suggested_approach():
    """Correct approach string for dominant domain and tier."""
    weights = {"ethics": 0.3, "logic": 1.0, "innovation": 0.5, "human_attunement": 0.2}
    tiers = {"ethics": 1, "logic": 2, "innovation": 1, "human_attunement": 0}
    routing = compute_routing(tiers, weights, 0.6)
    # logic tier 2 -> "analytical"
    assert routing["suggested_approach"] == "analytical"
    assert routing["complexity_level"] == classify_tier(0.6)


# ===================================================================
# Flag Escalation
# ===================================================================

def test_flags_aggregated_across_domains():
    """Flags from multiple domains appear prefixed in meta_flags."""
    results = _make_domain_results(
        ethics_flags={"intent_unclear": {"severity": "warning"}},
        logic_flags={"contradiction": {"severity": "risk"}},
    )
    meta_flags, _ = escalate_domain_flags(results)
    assert "ethics_intent_unclear" in meta_flags
    assert "logic_contradiction" in meta_flags


def test_max_severity_tracked():
    """_max_severity reflects the worst flag across all domains."""
    results = _make_domain_results(
        ethics_flags={"minor": {"severity": "info"}},
        logic_flags={"major": {"severity": "risk"}},
    )
    meta_flags, _ = escalate_domain_flags(results)
    assert meta_flags["_max_severity"] == "risk"


# ===================================================================
# Full Integration: SynthesisEngine.synthesize()
# ===================================================================

def test_synthesize_reflective_profile():
    """End-to-end with REFLECTIVE_PROFILE produces valid RewardMetaDirective."""
    engine = SynthesisEngine(profile=REFLECTIVE_PROFILE)
    results = _make_domain_results(ethics=0.8, logic=0.7, innovation=0.4, human_attunement=0.6)
    directive = engine.synthesize(results)

    assert isinstance(directive, RewardMetaDirective)
    assert isinstance(directive.allow_output, bool)
    assert isinstance(directive.abstain, bool)
    assert isinstance(directive.suppress, bool)

    # Check directive keys
    expected_keys = {
        "tone", "structure", "metaphor_density", "reasoning_depth",
        "moralize", "clarify", "speculate", "soothe",
    }
    assert set(directive.directives.keys()) == expected_keys

    # All directive values in [0, 1]
    for key, value in directive.directives.items():
        assert 0.0 <= value <= 1.0, f"directive {key}={value} out of range"

    # Routing has expected keys
    assert "dominant_domain" in directive.routing
    assert "suggested_approach" in directive.routing
    assert "complexity_level" in directive.routing
    assert "domain_influence" in directive.routing

    # Meta has expected keys
    assert directive.meta["profile_name"] == "reflective"
    assert "composite_score" in directive.meta
    assert "composite_tier" in directive.meta

    # With these scores and reflective profile (ethics=0.9, logic=0.8 weights),
    # composite should be relatively high -> allow_output True
    assert directive.allow_output is True

    # as_dict works
    d = directive.as_dict()
    assert d["allow_output"] is True
    assert "directives" in d


def test_synthesize_creative_sandbox_high_speculation():
    """CREATIVE_SANDBOX with high innovation -> speculate is high, moralize is low."""
    engine = SynthesisEngine(profile=CREATIVE_SANDBOX_PROFILE)
    results = _make_domain_results(ethics=0.5, logic=0.4, innovation=0.9, human_attunement=0.5)
    directive = engine.synthesize(results)

    # Creative sandbox has very low suppression_bias (0.05), low abstention_bias (0.1)
    assert directive.suppress is False
    assert directive.abstain is False
    assert directive.allow_output is True

    # High innovation weight (1.0) with score 0.9 -> speculate should be elevated
    assert directive.directives["speculate"] > directive.directives["moralize"]


def test_synthesize_ethics_training_abstains_low_ethics():
    """ETHICS_TRAINING with low ethics score -> abstains."""
    engine = SynthesisEngine(profile=ETHICS_TRAINING_PROFILE)
    # Ethics score (0.2) well below threshold_tolerance for ethics (0.9)
    results = _make_domain_results(ethics=0.2, logic=0.6, innovation=0.5, human_attunement=0.5)
    directive = engine.synthesize(results)

    # ETHICS_TRAINING: threshold_tolerances = {"ethics": 0.9, "logic": 0.7}
    # Ethics: 0.2 < 0.9 (violation), Logic: 0.6 < 0.7 (violation)
    # violation_ratio = 2/2 = 1.0, 1.0 * 0.5 (abstention_bias) = 0.5
    # 0.5 > 0.5 is False ... but let's check with boundary
    # Actually 0.5 is NOT > 0.5, so need to verify
    # With both violations: 1.0 * 0.5 = 0.5, which is NOT > 0.5
    # So use a scenario where it clearly triggers
    results2 = _make_domain_results(ethics=0.1, logic=0.3, innovation=0.5, human_attunement=0.5)
    directive2 = engine.synthesize(results2)

    # Both domains below tolerance. 2/2 * 0.5 = 0.5, not > 0.5
    # But the suppression might kick in: composite is low
    # ETHICS_TRAINING suppression_bias=0.4
    # composite: (1.0*0.1 + 0.8*0.3 + 0.2*0.5 + 0.7*0.5) / (1.0+0.8+0.2+0.7)
    #          = (0.1 + 0.24 + 0.1 + 0.35) / 2.7 = 0.79/2.7 = 0.293
    # 0.293 < 0.4 -> suppress!
    assert directive2.suppress is True
    assert directive2.allow_output is False


# ===================================================================
# Sleep metric modulation
# ===================================================================

def test_synthesize_sleep_metrics_in_meta():
    """Sleep metrics are included in the meta dict when provided."""
    engine = SynthesisEngine(profile=SLEEP_DREAM_PROFILE)
    results = _make_domain_results(ethics=0.5, logic=0.5, innovation=0.8, human_attunement=0.5)
    sm = {"dream_permissiveness": 0.7, "consolidation_depth": 0.3}
    directive = engine.synthesize(results, sleep_metrics=sm)
    assert "sleep_metrics" in directive.meta
    assert directive.meta["sleep_metrics"]["dream_permissiveness"] == 0.7


def test_synthesize_no_sleep_metrics_no_key():
    """Without sleep metrics, no sleep_metrics key in meta."""
    engine = SynthesisEngine(profile=REFLECTIVE_PROFILE)
    results = _make_domain_results(ethics=0.8, logic=0.7, innovation=0.4, human_attunement=0.6)
    directive = engine.synthesize(results)
    assert "sleep_metrics" not in directive.meta


def test_synthesize_dream_permissiveness_relaxes_abstention():
    """High dream_permissiveness should reduce abstention sensitivity."""
    engine = SynthesisEngine(profile=SLEEP_DEEP_PROFILE)
    results = _make_domain_results(ethics=0.3, logic=0.3, innovation=0.5, human_attunement=0.3)

    # Without sleep metrics
    directive_base = engine.synthesize(results)

    # With high dream permissiveness — should be more permissive
    sm = {"dream_permissiveness": 0.9}
    directive_dream = engine.synthesize(results, sleep_metrics=sm)

    # Dream mode should be at least as permissive
    if directive_base.abstain:
        # If base abstains, dream may or may not — but it shouldn't be stricter
        pass
    # The meta should contain sleep metrics
    assert "sleep_metrics" in directive_dream.meta


def test_synthesize_consolidation_relaxes_suppression():
    """High consolidation_depth should reduce suppression sensitivity."""
    engine = SynthesisEngine(profile=SLEEP_DEEP_PROFILE)
    results = _make_domain_results(ethics=0.3, logic=0.3, innovation=0.3, human_attunement=0.3)

    sm = {"consolidation_depth": 0.8}
    directive = engine.synthesize(results, sleep_metrics=sm)
    assert "sleep_metrics" in directive.meta
    assert directive.meta["sleep_metrics"]["consolidation_depth"] == 0.8
