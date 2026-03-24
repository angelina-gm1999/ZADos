"""
Integration tests for the feedback loop: SynthesisEngine → compute_reward_feedback → engine.apply_feedback.

Verifies that reward-conditioned secondary gradients correctly modulate
neurochemical baselines, reuptake rates, and receptor affinities.
"""

import pytest

from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState, ReceptorState, OscillationState

from zados.reward.base.types import (
    RewardDomainResult,
    RewardMetaDirective,
    RewardSubscore,
)
from zados.reward.feedback.modulator import compute_reward_feedback
from zados.reward.synthesis.engine import SynthesisEngine
from zados.reward.profile.static_profiles import REFLECTIVE_PROFILE


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


def _build_engine_with_nt(
    nt_name: str = "OXT",
    c_baseline: float = 0.5,
    u_base: float = 0.1,
) -> NeurochemicalEngine:
    """Create an engine with a single NT registered."""
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter(
        nt_name,
        config={"C_baseline": c_baseline, "u_base": u_base},
    )
    return engine


def _build_engine_with_receptor(
    receptor_id: str = "GABA_B",
    k_d: float = 0.5,
    parent_nt: str = "GABA",
) -> NeurochemicalEngine:
    """Create an engine with a parent NT and receptor registered."""
    engine = NeurochemicalEngine(dt=0.01, seed=42)
    engine.add_neurotransmitter(parent_nt, config={"C_baseline": 0.5})
    engine.add_receptor(
        receptor_id,
        config={"K_d": k_d, "parent_nt": parent_nt},
    )
    return engine


# =====================================================================
# Engine feedback application tests
# =====================================================================


class TestApplyFeedbackNT:

    def test_modulates_baseline(self):
        """OXT C_baseline shifts after apply_feedback."""
        engine = _build_engine_with_nt("OXT", c_baseline=0.5)

        feedback = {
            "neurotransmitters": {
                "OXT": {"C_baseline_delta": 0.03},
            },
            "receptors": {},
        }

        engine.apply_feedback(feedback)

        config = engine.registry.get_config("OXT")
        assert config["C_baseline"] == pytest.approx(0.53)

    def test_modulates_reuptake(self):
        """NE u_base changes after apply_feedback."""
        engine = _build_engine_with_nt("NE", c_baseline=0.5, u_base=0.1)

        feedback = {
            "neurotransmitters": {
                "NE": {"u_base_multiplier": 1.2},
            },
            "receptors": {},
        }

        engine.apply_feedback(feedback)

        config = engine.registry.get_config("NE")
        assert config["u_base"] == pytest.approx(0.12)

    def test_skips_unregistered(self):
        """No crash when feedback targets NTs not in the engine."""
        engine = _build_engine_with_nt("DA", c_baseline=0.5)

        feedback = {
            "neurotransmitters": {
                "OXT": {"C_baseline_delta": 0.03},
                "CB1": {"C_baseline_delta": -0.02},
                "NE": {"u_base_multiplier": 1.1},
            },
            "receptors": {},
        }

        # Should not raise
        engine.apply_feedback(feedback)

        # DA should be untouched
        config = engine.registry.get_config("DA")
        assert config["C_baseline"] == pytest.approx(0.5)

    def test_clamps_baseline_high(self):
        """C_baseline clamped to 1.0 on large positive delta."""
        engine = _build_engine_with_nt("OXT", c_baseline=0.98)

        feedback = {
            "neurotransmitters": {
                "OXT": {"C_baseline_delta": 0.05},
            },
            "receptors": {},
        }

        engine.apply_feedback(feedback)

        config = engine.registry.get_config("OXT")
        assert config["C_baseline"] == pytest.approx(1.0)

    def test_clamps_baseline_low(self):
        """C_baseline clamped to 0.0 on large negative delta."""
        engine = _build_engine_with_nt("OXT", c_baseline=0.02)

        feedback = {
            "neurotransmitters": {
                "OXT": {"C_baseline_delta": -0.05},
            },
            "receptors": {},
        }

        engine.apply_feedback(feedback)

        config = engine.registry.get_config("OXT")
        assert config["C_baseline"] == pytest.approx(0.0)

    def test_clamps_u_base_floor(self):
        """u_base clamped to 0.01 minimum on very small multiplier."""
        engine = _build_engine_with_nt("NE", c_baseline=0.5, u_base=0.02)

        feedback = {
            "neurotransmitters": {
                "NE": {"u_base_multiplier": 0.1},
            },
            "receptors": {},
        }

        engine.apply_feedback(feedback)

        config = engine.registry.get_config("NE")
        # 0.02 * 0.1 = 0.002 → clamped to 0.01
        assert config["u_base"] == pytest.approx(0.01)


class TestApplyFeedbackReceptor:

    def test_modulates_receptor_affinity(self):
        """GABA_B K_d changes after apply_feedback."""
        engine = _build_engine_with_receptor("GABA_B", k_d=0.5)

        feedback = {
            "neurotransmitters": {},
            "receptors": {
                "GABA_B": {"K_d_multiplier": 0.9},
            },
        }

        engine.apply_feedback(feedback)

        config = engine.registry.get_config("GABA_B")
        assert config["K_d"] == pytest.approx(0.45)

    def test_clamps_kd_high(self):
        """K_d clamped to 10.0 maximum."""
        engine = _build_engine_with_receptor("GABA_B", k_d=8.0)

        feedback = {
            "neurotransmitters": {},
            "receptors": {
                "GABA_B": {"K_d_multiplier": 2.0},
            },
        }

        engine.apply_feedback(feedback)

        config = engine.registry.get_config("GABA_B")
        # 8.0 * 2.0 = 16.0 → clamped to 10.0
        assert config["K_d"] == pytest.approx(10.0)

    def test_skips_unregistered_receptor(self):
        """No crash when receptor not in engine."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        engine.add_neurotransmitter("DA", config={"C_baseline": 0.5})

        feedback = {
            "neurotransmitters": {},
            "receptors": {
                "GABA_B": {"K_d_multiplier": 0.9},
            },
        }

        # Should not raise
        engine.apply_feedback(feedback)


# =====================================================================
# End-to-end loop tests
# =====================================================================


class TestEndToEndLoop:

    def test_full_loop_synthesis_to_engine(self):
        """SynthesisEngine → compute_reward_feedback → engine.apply_feedback → config changed."""
        # 1. Domain results
        domain_results = {
            "ethics": _make_domain_result(
                "ethics", general_score=0.85,
                subscores={"timeline_reflection": 0.3},
            ),
            "logic": _make_domain_result(
                "logic", general_score=0.75,
                subscores={"internal_consistency": 0.4},
            ),
            "innovation": _make_domain_result(
                "innovation", general_score=0.6,
            ),
            "human_attunement": _make_domain_result(
                "human_attunement", general_score=0.7,
            ),
        }

        # 2. Synthesis
        synth = SynthesisEngine(profile=REFLECTIVE_PROFILE)
        directive = synth.synthesize(domain_results)

        # 3. Compute feedback
        feedback = compute_reward_feedback(directive, domain_results)

        # 4. Build engine with relevant NTs and receptors
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        engine.add_neurotransmitter("OXT", config={"C_baseline": 0.5, "u_base": 0.1})
        engine.add_neurotransmitter("NE", config={"C_baseline": 0.5, "u_base": 0.1})
        engine.add_neurotransmitter("GABA", config={"C_baseline": 0.5})
        engine.add_receptor("GABA_B", config={"K_d": 0.5, "parent_nt": "GABA"})

        # Record pre-feedback values
        oxt_before = engine.registry.get_config("OXT")["C_baseline"]
        ne_u_before = engine.registry.get_config("NE")["u_base"]
        gaba_kd_before = engine.registry.get_config("GABA_B")["K_d"]

        # 5. Apply feedback
        engine.apply_feedback(feedback)

        # 6. Verify changes
        oxt_after = engine.registry.get_config("OXT")["C_baseline"]
        ne_u_after = engine.registry.get_config("NE")["u_base"]
        gaba_kd_after = engine.registry.get_config("GABA_B")["K_d"]

        # OXT baseline should shift (attunement > center → positive delta)
        assert oxt_after != oxt_before

        # NE u_base should change (logic has contradiction_load > 0)
        assert ne_u_after != ne_u_before

        # GABA_B K_d should change (ethics has timeline_mismatch > 0)
        assert gaba_kd_after != gaba_kd_before

    def test_feedback_affects_next_step(self):
        """After feedback shifts C_baseline, next step() drifts toward new baseline."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        engine.add_neurotransmitter(
            "OXT",
            initial_state=NeurotransmitterState(C_tonic=0.5),
            config={
                "C_baseline": 0.5,
                "u_base": 0.0,      # Zero removal terms for clean drift test
                "d_base": 0.0,
                "c_base": 0.0,
                "theta_tonic": 0.5,  # Faster reversion for test
                "sigma_tonic": 0.0,  # No noise for determinism
                "sigma_phasic": 0.0,
            },
        )

        # Step a few times at current baseline to let it settle
        for _ in range(50):
            engine.step()

        c_before = engine.registry.get_neurotransmitter("OXT").C_tonic

        # Apply feedback: shift baseline UP
        feedback = {
            "neurotransmitters": {
                "OXT": {"C_baseline_delta": 0.05},
            },
            "receptors": {},
        }
        engine.apply_feedback(feedback)

        new_baseline = engine.registry.get_config("OXT")["C_baseline"]
        assert new_baseline == pytest.approx(0.55)

        # Step many more times → tonic should drift toward new baseline
        for _ in range(200):
            engine.step()

        c_after = engine.registry.get_neurotransmitter("OXT").C_tonic

        # After many steps with higher baseline, tonic should have increased
        assert c_after > c_before

    def test_feedback_with_adapter_and_synthesis(self):
        """
        Full pipeline: adapter.transform() provides step signals,
        compute_reward_feedback() provides feedback, both applied.
        """
        # Setup engine with DA and OXT
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        engine.add_neurotransmitter("DA", config={
            "C_baseline": 0.5, "u_base": 0.1,
            "sigma_tonic": 0.0, "sigma_phasic": 0.0,
        })
        engine.add_neurotransmitter("OXT", config={
            "C_baseline": 0.5, "u_base": 0.1,
            "sigma_tonic": 0.0, "sigma_phasic": 0.0,
        })
        engine.add_neurotransmitter("NE", config={
            "C_baseline": 0.5, "u_base": 0.1,
            "sigma_tonic": 0.0, "sigma_phasic": 0.0,
        })

        # Step with adapter-like signals (simulate adapter output)
        adapter_signals = {
            "DA": {"novelty": 0.7, "rpe": 0.3, "effort": 0.1},
        }
        engine.step(adapter_signals)

        # Now apply feedback from synthesis
        domain_results = {
            "human_attunement": _make_domain_result(
                "human_attunement", general_score=0.8,
            ),
            "logic": _make_domain_result(
                "logic", general_score=0.6,
                subscores={"internal_consistency": 0.5},
            ),
            "ethics": _make_domain_result("ethics", general_score=0.7),
            "innovation": _make_domain_result("innovation", general_score=0.5),
        }

        synth = SynthesisEngine(profile=REFLECTIVE_PROFILE)
        directive = synth.synthesize(domain_results)
        feedback = compute_reward_feedback(directive, domain_results)

        oxt_before = engine.registry.get_config("OXT")["C_baseline"]
        engine.apply_feedback(feedback)
        oxt_after = engine.registry.get_config("OXT")["C_baseline"]

        # Attunement score is high → OXT baseline should increase
        assert oxt_after > oxt_before

        # Engine continues stepping normally after feedback
        for _ in range(10):
            engine.step(adapter_signals)

        # Should still be valid state
        da_state = engine.registry.get_neurotransmitter("DA")
        assert 0.0 <= da_state.C_tonic <= 1.0
        oxt_state = engine.registry.get_neurotransmitter("OXT")
        assert 0.0 <= oxt_state.C_tonic <= 1.0
