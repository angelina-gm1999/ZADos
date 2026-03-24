import pytest

from zados.reward.base.types import (
    RewardDomainResult,
    RewardMetaDirective,
    RewardSubscore,
)
from zados.reward.feedback.modulator import (
    DEFAULT_FEEDBACK_GAINS,
    extract_contradiction_load,
    extract_timeline_mismatch,
    compute_baseline_feedback,
    compute_reuptake_feedback,
    compute_affinity_feedback,
    compute_reward_feedback,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_domain_result(
    domain: str,
    general_score: float = 0.5,
    subscores: dict | None = None,
    flags: dict | None = None,
) -> RewardDomainResult:
    """Build a RewardDomainResult with optional subscores."""
    ss = {}
    if subscores:
        for name, score in subscores.items():
            ss[name] = RewardSubscore(name=name, score=score)
    return RewardDomainResult(
        domain=domain,
        general_score=general_score,
        subscores=ss,
        flags=flags or {},
    )


def _make_meta_directive(per_domain_weighted_scores: dict) -> RewardMetaDirective:
    """Build a minimal RewardMetaDirective with per_domain_weighted_scores."""
    return RewardMetaDirective(
        meta={"per_domain_weighted_scores": per_domain_weighted_scores},
    )


# =====================================================================
# extract_contradiction_load
# =====================================================================


class TestExtractContradictionLoad:

    def test_from_subscores(self):
        """Logic result with internal_consistency=0.3 → load=0.7."""
        logic = _make_domain_result(
            "logic", subscores={"internal_consistency": 0.3},
        )
        assert extract_contradiction_load(logic) == pytest.approx(0.7)

    def test_missing_subscore(self):
        """No internal_consistency subscore → load=0.0."""
        logic = _make_domain_result(
            "logic", subscores={"epistemic_calibration": 0.8},
        )
        assert extract_contradiction_load(logic) == 0.0

    def test_none_result(self):
        """None input → 0.0."""
        assert extract_contradiction_load(None) == 0.0

    def test_high_consistency_low_load(self):
        """internal_consistency=0.95 → load=0.05."""
        logic = _make_domain_result(
            "logic", subscores={"internal_consistency": 0.95},
        )
        assert extract_contradiction_load(logic) == pytest.approx(0.05)

    def test_zero_consistency_max_load(self):
        """internal_consistency=0.0 → load=1.0."""
        logic = _make_domain_result(
            "logic", subscores={"internal_consistency": 0.0},
        )
        assert extract_contradiction_load(logic) == pytest.approx(1.0)


# =====================================================================
# extract_timeline_mismatch
# =====================================================================


class TestExtractTimelineMismatch:

    def test_from_subscores(self):
        """Ethics result with timeline_reflection=0.4 → mismatch=0.6."""
        ethics = _make_domain_result(
            "ethics", subscores={"timeline_reflection": 0.4},
        )
        assert extract_timeline_mismatch(ethics) == pytest.approx(0.6)

    def test_missing_subscore(self):
        """No timeline_reflection subscore → mismatch=0.0."""
        ethics = _make_domain_result(
            "ethics", subscores={"failure_mode_awareness": 0.8},
        )
        assert extract_timeline_mismatch(ethics) == 0.0

    def test_none_result(self):
        """None input → 0.0."""
        assert extract_timeline_mismatch(None) == 0.0


# =====================================================================
# compute_baseline_feedback
# =====================================================================


class TestBaselineFeedback:

    def test_above_center(self):
        """score=0.8, center=0.5 → positive delta."""
        delta = compute_baseline_feedback(0.8, gain=0.05, center=0.5)
        assert delta > 0.0
        # (0.8-0.5)*0.05*2 = 0.03
        assert delta == pytest.approx(0.03)

    def test_below_center(self):
        """score=0.2, center=0.5 → negative delta."""
        delta = compute_baseline_feedback(0.2, gain=0.05, center=0.5)
        assert delta < 0.0
        # (0.2-0.5)*0.05*2 = -0.03
        assert delta == pytest.approx(-0.03)

    def test_at_center(self):
        """score=center → delta=0.0 exactly."""
        delta = compute_baseline_feedback(0.5, gain=0.05, center=0.5)
        assert delta == pytest.approx(0.0)

    def test_clamp_high(self):
        """Extreme score → clamped to +gain."""
        delta = compute_baseline_feedback(1.0, gain=0.05, center=0.0)
        assert delta == pytest.approx(0.05)

    def test_clamp_low(self):
        """Extreme low score → clamped to -gain."""
        delta = compute_baseline_feedback(0.0, gain=0.05, center=1.0)
        assert delta == pytest.approx(-0.05)


# =====================================================================
# compute_reuptake_feedback
# =====================================================================


class TestReuptakeFeedback:

    def test_high_load(self):
        """R_Logic=0.8, contradiction=0.9 → multiplier > 1.0."""
        mult = compute_reuptake_feedback(0.8, load=0.9, gain=0.3)
        assert mult > 1.0
        # 1.0 + 0.8*0.9*0.3 = 1.216
        assert mult == pytest.approx(1.216)

    def test_zero_load(self):
        """contradiction=0.0 → multiplier = 1.0 (no effect)."""
        mult = compute_reuptake_feedback(0.8, load=0.0, gain=0.3)
        assert mult == pytest.approx(1.0)

    def test_clamped_high(self):
        """max score and max load → clamped to 1+gain."""
        mult = compute_reuptake_feedback(1.0, load=1.0, gain=0.3)
        assert mult == pytest.approx(1.3)

    def test_zero_score(self):
        """Zero score → multiplier = 1.0 regardless of load."""
        mult = compute_reuptake_feedback(0.0, load=1.0, gain=0.3)
        assert mult == pytest.approx(1.0)


# =====================================================================
# compute_affinity_feedback
# =====================================================================


class TestAffinityFeedback:

    def test_high_mismatch(self):
        """R_Ethics=0.8, mismatch=0.7 → multiplier < 1.0."""
        mult = compute_affinity_feedback(0.8, mismatch=0.7, gain=0.2)
        assert mult < 1.0
        # 1.0 - 0.8*0.7*0.2 = 0.888
        assert mult == pytest.approx(0.888)

    def test_zero_mismatch(self):
        """mismatch=0.0 → multiplier = 1.0 (no effect)."""
        mult = compute_affinity_feedback(0.8, mismatch=0.0, gain=0.2)
        assert mult == pytest.approx(1.0)

    def test_clamped_low(self):
        """max score and max mismatch → clamped to 1-gain."""
        mult = compute_affinity_feedback(1.0, mismatch=1.0, gain=0.2)
        assert mult == pytest.approx(0.8)

    def test_zero_score(self):
        """Zero score → multiplier = 1.0 regardless of mismatch."""
        mult = compute_affinity_feedback(0.0, mismatch=1.0, gain=0.2)
        assert mult == pytest.approx(1.0)


# =====================================================================
# compute_reward_feedback (orchestrator)
# =====================================================================


class TestComputeRewardFeedback:

    def test_all_pathways(self):
        """Verify all 4 pathways produce correct structure and values."""
        meta = _make_meta_directive({
            "human_attunement": 0.7,
            "innovation": 0.3,
            "logic": 0.6,
            "ethics": 0.8,
        })
        domain_results = {
            "logic": _make_domain_result(
                "logic",
                general_score=0.6,
                subscores={"internal_consistency": 0.3},
            ),
            "ethics": _make_domain_result(
                "ethics",
                general_score=0.8,
                subscores={"timeline_reflection": 0.4},
            ),
            "human_attunement": _make_domain_result(
                "human_attunement", general_score=0.7,
            ),
            "innovation": _make_domain_result(
                "innovation", general_score=0.5,
            ),
        }

        result = compute_reward_feedback(meta, domain_results)

        # Structure check
        assert "neurotransmitters" in result
        assert "receptors" in result
        assert "OXT" in result["neurotransmitters"]
        assert "CB1" in result["neurotransmitters"]
        assert "NE" in result["neurotransmitters"]
        assert "GABA_B" in result["receptors"]

        # OXT: baseline delta from R_Attunement=0.7
        # (0.7-0.5)*0.05*2 = 0.02
        oxt = result["neurotransmitters"]["OXT"]
        assert "C_baseline_delta" in oxt
        assert oxt["C_baseline_delta"] == pytest.approx(0.02)

        # CB1: baseline delta from R_Innovation=0.3
        # (0.3-0.5)*0.05*2 = -0.02
        cb1 = result["neurotransmitters"]["CB1"]
        assert cb1["C_baseline_delta"] == pytest.approx(-0.02)

        # NE: reuptake from R_Logic=0.6, ContradictionLoad=0.7
        # 1.0 + 0.6*0.7*0.3 = 1.126
        ne = result["neurotransmitters"]["NE"]
        assert "u_base_multiplier" in ne
        assert ne["u_base_multiplier"] == pytest.approx(1.126)

        # GABA_B: affinity from R_Ethics=0.8, TimelineMismatch=0.6
        # 1.0 - 0.8*0.6*0.2 = 0.904
        gaba = result["receptors"]["GABA_B"]
        assert "K_d_multiplier" in gaba
        assert gaba["K_d_multiplier"] == pytest.approx(0.904)

    def test_missing_domains(self):
        """Gracefully handles absent domains → default-from-zero signals."""
        meta = _make_meta_directive({})
        domain_results = {}

        result = compute_reward_feedback(meta, domain_results)

        # Missing scores default to 0.0, not center.
        # (0.0 - 0.5) * 0.05 * 2 = -0.05
        assert result["neurotransmitters"]["OXT"]["C_baseline_delta"] == pytest.approx(-0.05)
        assert result["neurotransmitters"]["CB1"]["C_baseline_delta"] == pytest.approx(-0.05)

        # No load/mismatch (no domain results to extract from) → multipliers=1.0
        assert result["neurotransmitters"]["NE"]["u_base_multiplier"] == pytest.approx(1.0)
        assert result["receptors"]["GABA_B"]["K_d_multiplier"] == pytest.approx(1.0)

    def test_custom_gains(self):
        """Custom gain parameters override defaults."""
        meta = _make_meta_directive({
            "human_attunement": 0.8,
            "innovation": 0.8,
            "logic": 0.8,
            "ethics": 0.8,
        })
        domain_results = {
            "logic": _make_domain_result(
                "logic", subscores={"internal_consistency": 0.0},
            ),
            "ethics": _make_domain_result(
                "ethics", subscores={"timeline_reflection": 0.0},
            ),
        }

        custom_gains = {
            "baseline_gain": 0.1,
            "baseline_center": 0.5,
            "reuptake_gain": 0.5,
            "affinity_gain": 0.4,
        }

        result = compute_reward_feedback(meta, domain_results, gains=custom_gains)

        # OXT: (0.8-0.5)*0.1*2 = 0.06
        assert result["neurotransmitters"]["OXT"]["C_baseline_delta"] == pytest.approx(0.06)

        # NE: 1.0 + 0.8*1.0*0.5 = 1.4, clamped to 1+gain=1.5
        assert result["neurotransmitters"]["NE"]["u_base_multiplier"] == pytest.approx(1.4)

        # GABA_B: 1.0 - 0.8*1.0*0.4 = 0.68, clamped to 1-gain=0.6
        assert result["receptors"]["GABA_B"]["K_d_multiplier"] == pytest.approx(0.68)

    def test_partial_domains(self):
        """Only logic domain present; others default to zero."""
        meta = _make_meta_directive({"logic": 0.5})
        domain_results = {
            "logic": _make_domain_result(
                "logic",
                general_score=0.5,
                subscores={"internal_consistency": 0.5},
            ),
        }

        result = compute_reward_feedback(meta, domain_results)

        # No attunement → OXT delta = (0.0-0.5)*0.05*2 = -0.05
        assert result["neurotransmitters"]["OXT"]["C_baseline_delta"] == pytest.approx(-0.05)

        # Logic=0.5, contradiction=0.5 → 1+0.5*0.5*0.3=1.075
        assert result["neurotransmitters"]["NE"]["u_base_multiplier"] == pytest.approx(1.075)

        # No ethics → mismatch=0 → multiplier=1.0
        assert result["receptors"]["GABA_B"]["K_d_multiplier"] == pytest.approx(1.0)


# =====================================================================
# Consolidation gate (sleep metric modulation)
# =====================================================================


class TestConsolidationGate:

    def test_no_sleep_metrics_no_gating(self):
        """Without sleep metrics, feedback is unchanged."""
        meta = _make_meta_directive({"human_attunement": 0.8})
        domain_results = {
            "human_attunement": _make_domain_result("human_attunement", 0.8),
        }
        result_base = compute_reward_feedback(meta, domain_results)
        result_sm = compute_reward_feedback(meta, domain_results, sleep_metrics={})
        assert result_base == result_sm

    def test_consolidation_below_threshold_no_gating(self):
        """consolidation_depth <= 0.5 does NOT trigger gating."""
        meta = _make_meta_directive({"human_attunement": 0.8})
        domain_results = {
            "human_attunement": _make_domain_result("human_attunement", 0.8),
        }
        sm_low = {"consolidation_depth": 0.4}
        result_base = compute_reward_feedback(meta, domain_results)
        result_low = compute_reward_feedback(meta, domain_results, sleep_metrics=sm_low)
        # Below threshold — no gating applied
        assert result_base["neurotransmitters"]["OXT"]["C_baseline_delta"] == pytest.approx(
            result_low["neurotransmitters"]["OXT"]["C_baseline_delta"]
        )

    def test_consolidation_above_threshold_gates_feedback(self):
        """consolidation_depth > 0.5 reduces feedback magnitude."""
        meta = _make_meta_directive({"human_attunement": 0.8})
        domain_results = {
            "human_attunement": _make_domain_result("human_attunement", 0.8),
        }
        result_base = compute_reward_feedback(meta, domain_results)
        sm_high = {"consolidation_depth": 0.8}
        result_gated = compute_reward_feedback(meta, domain_results, sleep_metrics=sm_high)

        base_delta = result_base["neurotransmitters"]["OXT"]["C_baseline_delta"]
        gated_delta = result_gated["neurotransmitters"]["OXT"]["C_baseline_delta"]

        # Gated delta should be smaller in magnitude
        assert abs(gated_delta) < abs(base_delta)

    def test_consolidation_gates_multipliers_toward_neutral(self):
        """Reuptake/affinity multipliers pulled toward 1.0 during consolidation."""
        meta = _make_meta_directive({"logic": 0.8})
        domain_results = {
            "logic": _make_domain_result(
                "logic", 0.8,
                subscores={"internal_consistency": 0.3},
            ),
        }
        result_base = compute_reward_feedback(meta, domain_results)
        sm = {"consolidation_depth": 0.9}
        result_gated = compute_reward_feedback(meta, domain_results, sleep_metrics=sm)

        base_ne = result_base["neurotransmitters"]["NE"]["u_base_multiplier"]
        gated_ne = result_gated["neurotransmitters"]["NE"]["u_base_multiplier"]

        # Gated NE multiplier should be closer to 1.0 than base
        assert abs(gated_ne - 1.0) < abs(base_ne - 1.0)
