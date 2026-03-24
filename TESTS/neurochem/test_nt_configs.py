"""
Phase 9 tests: Default neurotransmitter configurations, registration helpers,
engine integration with all 12 NT systems and their receptors.
"""

import pytest

from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import (
    NeurotransmitterState,
    ReceptorState,
    OscillationState,
)
from zados.neurochem.neurotransmitters.configs import (
    DEFAULT_NT_CONFIGS,
    DEFAULT_RECEPTOR_CONFIGS,
    NT_RECEPTOR_MAP,
    register_neurotransmitter,
    register_all_neurotransmitters,
)
from zados.reward.feedback.modulator import compute_reward_feedback
from zados.reward.base.types import (
    RewardDomainResult,
    RewardMetaDirective,
    RewardSubscore,
)


# =====================================================================
# Helpers
# =====================================================================

REQUIRED_NT_KEYS = {
    "C_baseline", "theta_tonic", "theta_phasic",
    "sigma_tonic", "sigma_phasic", "u_base", "d_base", "c_base",
}

REQUIRED_RECEPTOR_KEYS = {"K_d", "parent_nt", "exposure_tau"}

ALL_NT_NAMES = [
    "DA", "5HT", "NE", "ACh", "OXT", "MOR", "CB1",
    "cortisol", "CRH", "GABA", "GLU", "histamine",
]


def _make_full_engine(seed: int = 42) -> NeurochemicalEngine:
    """Build an engine with all NTs and receptors registered."""
    engine = NeurochemicalEngine(dt=0.01, seed=seed)
    register_all_neurotransmitters(engine)
    engine.set_oscillation_state(OscillationState())
    return engine


def _make_domain_result(domain, score=0.5, subscores=None):
    ss = {}
    if subscores:
        for name, val in subscores.items():
            ss[name] = RewardSubscore(name=name, score=val)
    return RewardDomainResult(domain=domain, general_score=score, subscores=ss)


# =====================================================================
# A. Config Completeness
# =====================================================================


class TestConfigCompleteness:

    def test_all_12_nts_present(self):
        """DEFAULT_NT_CONFIGS has exactly 12 NT entries."""
        assert len(DEFAULT_NT_CONFIGS) == 12
        for name in ALL_NT_NAMES:
            assert name in DEFAULT_NT_CONFIGS, f"Missing NT: {name}"

    def test_nt_configs_have_required_keys(self):
        """Every NT config dict contains the 8 engine keys."""
        for name, config in DEFAULT_NT_CONFIGS.items():
            for key in REQUIRED_NT_KEYS:
                assert key in config, f"NT {name!r} missing key {key!r}"

    def test_nt_values_positive(self):
        """All NT kinetic parameter values are positive."""
        for name, config in DEFAULT_NT_CONFIGS.items():
            for key in REQUIRED_NT_KEYS:
                val = config[key]
                assert val >= 0.0, f"NT {name!r} key {key!r} = {val} (negative)"

    def test_nt_baselines_in_range(self):
        """C_baseline is in (0, 1] for all NTs."""
        for name, config in DEFAULT_NT_CONFIGS.items():
            bl = config["C_baseline"]
            assert 0.0 < bl <= 1.0, f"NT {name!r} C_baseline={bl}"

    def test_all_receptors_in_map_have_configs(self):
        """Every receptor_id in NT_RECEPTOR_MAP has a DEFAULT_RECEPTOR_CONFIGS entry."""
        for nt_name, receptor_ids in NT_RECEPTOR_MAP.items():
            for rid in receptor_ids:
                assert rid in DEFAULT_RECEPTOR_CONFIGS, (
                    f"Receptor {rid!r} (NT={nt_name}) missing from DEFAULT_RECEPTOR_CONFIGS"
                )

    def test_receptor_configs_have_required_keys(self):
        """Every receptor config has K_d, parent_nt, exposure_tau."""
        for rid, config in DEFAULT_RECEPTOR_CONFIGS.items():
            for key in REQUIRED_RECEPTOR_KEYS:
                assert key in config, f"Receptor {rid!r} missing key {key!r}"

    def test_receptor_parent_nt_valid(self):
        """Every receptor's parent_nt points to a valid NT name."""
        for rid, config in DEFAULT_RECEPTOR_CONFIGS.items():
            parent = config["parent_nt"]
            assert parent in DEFAULT_NT_CONFIGS, (
                f"Receptor {rid!r} has parent_nt={parent!r} "
                f"which is not in DEFAULT_NT_CONFIGS"
            )

    def test_receptor_kd_positive(self):
        """All receptor K_d values are positive."""
        for rid, config in DEFAULT_RECEPTOR_CONFIGS.items():
            assert config["K_d"] > 0.0, f"Receptor {rid!r} K_d={config['K_d']}"

    def test_nt_receptor_map_covers_all_nts(self):
        """NT_RECEPTOR_MAP has an entry for every NT in DEFAULT_NT_CONFIGS."""
        for nt_name in DEFAULT_NT_CONFIGS:
            assert nt_name in NT_RECEPTOR_MAP, (
                f"NT {nt_name!r} missing from NT_RECEPTOR_MAP"
            )


# =====================================================================
# B. Registration
# =====================================================================


class TestRegistration:

    def test_single_nt_no_receptors(self):
        """register_neurotransmitter with include_receptors=False."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA", include_receptors=False)

        assert "DA" in engine.registry.neurotransmitter_names()
        assert len(engine.registry.receptor_ids()) == 0

    def test_single_nt_with_receptors(self):
        """register_neurotransmitter registers NT + correct receptor count."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")

        assert "DA" in engine.registry.neurotransmitter_names()
        expected_receptors = NT_RECEPTOR_MAP["DA"]
        for rid in expected_receptors:
            assert rid in engine.registry.receptor_ids()
        assert len(engine.registry.receptor_ids()) == len(expected_receptors)

    def test_register_all(self):
        """register_all_neurotransmitters populates all 12 NTs + all receptors."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)

        assert len(engine.registry.neurotransmitter_names()) == 12

        total_receptors = sum(len(v) for v in NT_RECEPTOR_MAP.values())
        assert len(engine.registry.receptor_ids()) == total_receptors

    def test_config_override(self):
        """nt_config_overrides are applied correctly."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(
            engine, "DA",
            nt_config_overrides={"C_baseline": 0.8},
            include_receptors=False,
        )

        config = engine.registry.get_config("DA")
        assert config["C_baseline"] == 0.8
        # Other keys should remain at defaults
        assert config["u_base"] == DEFAULT_NT_CONFIGS["DA"]["u_base"]

    def test_receptor_config_override(self):
        """receptor_config_overrides are applied correctly."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(
            engine, "DA",
            receptor_config_overrides={"DA_D1": {"K_d": 0.99}},
        )

        config = engine.registry.get_config("DA_D1")
        assert config["K_d"] == 0.99
        assert config["parent_nt"] == "DA"  # not overridden

    def test_unknown_nt_raises(self):
        """KeyError for nonexistent NT name."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        with pytest.raises(KeyError, match="Unknown NT"):
            register_neurotransmitter(engine, "FAKE_NT")

    def test_cb1_name_collision_handled(self):
        """CB1 NT config and CB1 receptor config both accessible after registration."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "CB1")

        # NT should be registered
        assert "CB1" in engine.registry.neurotransmitter_names()
        # Receptor should be registered
        assert "CB1" in engine.registry.receptor_ids()

        # Merged config should have BOTH NT and receptor keys
        config = engine.registry.get_config("CB1")
        assert "C_baseline" in config  # NT key
        assert "K_d" in config         # Receptor key
        assert "parent_nt" in config   # Receptor key


# =====================================================================
# C. Engine Stepping
# =====================================================================


class TestEngineStepping:

    def test_step_no_signals(self):
        """Engine with all NTs can step() without crash."""
        engine = _make_full_engine()
        engine.step()
        assert engine.current_time == pytest.approx(0.01)

    def test_step_with_da_signals(self):
        """Engine with all NTs handles DA modulation signals."""
        engine = _make_full_engine()
        signals = {"DA": {"novelty": 0.8, "rpe": 0.5, "effort": 0.2}}
        engine.step(signals)

        da = engine.registry.get_neurotransmitter("DA")
        assert 0.0 <= da.C_tonic <= 1.0
        assert 0.0 <= da.C_phasic <= 1.0

    def test_step_with_non_da_signals_no_crash(self):
        """Non-DA signal keys (e.g. NE precision) don't crash the generic stepper."""
        engine = _make_full_engine()
        signals = {
            "NE": {"precision": 0.7, "uncertainty": 0.3},
            "OXT": {"empathy": 0.8, "social_engagement": 0.6},
        }
        # These signal keys are ignored by the generic stepper (it only
        # reads novelty/rpe/effort), but should not crash.
        engine.step(signals)

        ne = engine.registry.get_neurotransmitter("NE")
        assert 0.0 <= ne.C_tonic <= 1.0

    def test_100_steps_all_bounded(self):
        """After 100 steps, all NT states remain in [0, 1]."""
        engine = _make_full_engine()

        for _ in range(100):
            engine.step()

        for nt_name in engine.registry.neurotransmitter_names():
            state = engine.registry.get_neurotransmitter(nt_name)
            assert 0.0 <= state.C_tonic <= 1.0, f"{nt_name} C_tonic out of bounds"
            assert 0.0 <= state.C_phasic <= 1.0, f"{nt_name} C_phasic out of bounds"
            assert 0.0 <= state.F <= 1.0, f"{nt_name} F out of bounds"

    def test_100_steps_receptor_states_valid(self):
        """After 100 steps, receptor states remain valid."""
        engine = _make_full_engine()

        for _ in range(100):
            engine.step()

        for rid in engine.registry.receptor_ids():
            state = engine.registry.get_receptor(rid)
            assert 0.0 <= state.rho <= 1.0, f"{rid} rho out of bounds"
            assert 0.0 <= state.sigma <= 1.0, f"{rid} sigma out of bounds"

    def test_multiple_steps_accumulate_time(self):
        """Time increments correctly with all NTs."""
        engine = _make_full_engine()
        for _ in range(50):
            engine.step()
        assert engine.current_time == pytest.approx(0.5)


# =====================================================================
# D. Neurosymbolic Readout Integration
# =====================================================================


class TestNeurosymbolicReadout:

    def test_readout_returns_all_8_metrics(self):
        """With all NTs registered, readout returns all 8 metrics."""
        engine = _make_full_engine()
        engine.step()

        readout = engine.get_neurosymbolic_readout()

        expected = [
            "motivation", "empathy", "cognitive_rigidity", "fatigue",
            "precision", "openness", "anxiety", "social_engagement",
        ]
        for key in expected:
            assert key in readout, f"Missing metric: {key}"

    def test_all_metrics_in_range(self):
        """All 8 metrics are in [0, 1]."""
        engine = _make_full_engine()
        for _ in range(10):
            engine.step()

        readout = engine.get_neurosymbolic_readout()

        for key, val in readout.items():
            assert 0.0 <= val <= 1.0, f"Metric {key}={val} out of [0,1]"

    def test_oxtr_resolves_to_oxt(self):
        """OXTR receptor saturation uses OXT NT concentration (readout bug fix)."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        # Register OXT with high concentration
        engine.add_neurotransmitter(
            "OXT",
            initial_state=NeurotransmitterState(C_tonic=0.9),
            config=dict(DEFAULT_NT_CONFIGS["OXT"]),
        )
        # Register OXTR with parent_nt override
        engine.add_receptor(
            "OXTR",
            config=dict(DEFAULT_RECEPTOR_CONFIGS["OXTR"]),
        )
        engine.set_oscillation_state(OscillationState())
        engine.step()

        readout = engine.get_neurosymbolic_readout()

        # OXTR drives motivation and social_engagement;
        # with high OXT concentration, these should be non-trivially above 0.
        # motivation = (S_DA_D3 + S_OXT - S_GABA_B + 1) / 3
        # S_OXT = 0.9 / (0.9 + 0.35) ≈ 0.72
        # Without the fix, S_OXT would be 0.0 (NT "OXTR" not found).
        assert readout["motivation"] > 0.33  # baseline with no DA/GABA

    def test_high_da_elevates_motivation(self):
        """High DA → elevated motivation metric."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        engine.add_neurotransmitter(
            "DA",
            initial_state=NeurotransmitterState(C_tonic=0.9),
            config=dict(DEFAULT_NT_CONFIGS["DA"]),
        )
        for rid in ["DA_D3"]:
            engine.add_receptor(rid, config=dict(DEFAULT_RECEPTOR_CONFIGS[rid]))
        engine.set_oscillation_state(OscillationState())
        engine.step()

        readout = engine.get_neurosymbolic_readout()
        # S_DA_D3 = 0.9/(0.9+0.2) ≈ 0.82 → high motivation
        assert readout["motivation"] > 0.5

    def test_high_ne_crh_elevates_anxiety(self):
        """High NE + CRH → elevated anxiety metric."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        engine.add_neurotransmitter(
            "NE",
            initial_state=NeurotransmitterState(C_tonic=0.9),
            config=dict(DEFAULT_NT_CONFIGS["NE"]),
        )
        engine.add_neurotransmitter(
            "CRH",
            initial_state=NeurotransmitterState(C_tonic=0.8),
            config=dict(DEFAULT_NT_CONFIGS["CRH"]),
        )
        engine.add_neurotransmitter(
            "cortisol",
            initial_state=NeurotransmitterState(C_tonic=0.7),
            config=dict(DEFAULT_NT_CONFIGS["cortisol"]),
        )
        engine.set_oscillation_state(OscillationState())
        engine.step()

        readout = engine.get_neurosymbolic_readout()
        # anxiety = (C_NE + C_CRH + C_cortisol)/3 - S_GABA_A
        # ≈ (0.9 + 0.8 + 0.7)/3 - 0.0 = 0.8 → normalized ≈ 0.9
        assert readout["anxiety"] > 0.7

    def test_readout_with_partial_nts(self):
        """Readout works with only some NTs registered (others default to 0)."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.set_oscillation_state(OscillationState())
        engine.step()

        readout = engine.get_neurosymbolic_readout()
        # Should not crash, all 11 keys present (8 base + 3 sleep metrics)
        assert len(readout) == 11


# =====================================================================
# E. Feedback Loop Integration
# =====================================================================


class TestFeedbackIntegration:

    def test_feedback_modifies_registered_nts(self):
        """Feedback targets (OXT, CB1, NE, GABA_B) get modified when registered."""
        engine = _make_full_engine()

        meta = RewardMetaDirective(
            meta={"per_domain_weighted_scores": {
                "human_attunement": 0.8,
                "innovation": 0.7,
                "logic": 0.6,
                "ethics": 0.9,
            }},
        )
        domain_results = {
            "logic": _make_domain_result(
                "logic", subscores={"internal_consistency": 0.3},
            ),
            "ethics": _make_domain_result(
                "ethics", subscores={"timeline_reflection": 0.4},
            ),
        }

        oxt_before = engine.registry.get_config("OXT")["C_baseline"]
        ne_u_before = engine.registry.get_config("NE")["u_base"]

        feedback = compute_reward_feedback(meta, domain_results)
        engine.apply_feedback(feedback)

        oxt_after = engine.registry.get_config("OXT")["C_baseline"]
        ne_u_after = engine.registry.get_config("NE")["u_base"]

        assert oxt_after != oxt_before
        assert ne_u_after != ne_u_before

    def test_feedback_then_steps_stable(self):
        """System remains stable after feedback + many steps."""
        engine = _make_full_engine()

        meta = RewardMetaDirective(
            meta={"per_domain_weighted_scores": {
                "human_attunement": 0.9,
                "innovation": 0.1,
                "logic": 0.8,
                "ethics": 0.3,
            }},
        )
        domain_results = {
            "logic": _make_domain_result("logic", subscores={"internal_consistency": 0.2}),
            "ethics": _make_domain_result("ethics", subscores={"timeline_reflection": 0.3}),
        }

        feedback = compute_reward_feedback(meta, domain_results)
        engine.apply_feedback(feedback)

        for _ in range(200):
            engine.step()

        for nt_name in engine.registry.neurotransmitter_names():
            state = engine.registry.get_neurotransmitter(nt_name)
            assert 0.0 <= state.C_tonic <= 1.0, f"{nt_name} unstable after feedback"

    def test_cb1_feedback_and_config_integrity(self):
        """CB1 NT config survives the receptor config merge."""
        engine = _make_full_engine()

        # Verify merged config has both NT and receptor keys
        config = engine.registry.get_config("CB1")
        assert "C_baseline" in config  # NT key
        assert "K_d" in config         # Receptor key

        # Apply feedback targeting CB1
        meta = RewardMetaDirective(
            meta={"per_domain_weighted_scores": {
                "innovation": 0.8,
                "human_attunement": 0.5,
                "logic": 0.5,
                "ethics": 0.5,
            }},
        )
        domain_results = {}
        feedback = compute_reward_feedback(meta, domain_results)
        engine.apply_feedback(feedback)

        # CB1 baseline should have shifted (innovation 0.8 > center 0.5)
        config_after = engine.registry.get_config("CB1")
        assert config_after["C_baseline"] != DEFAULT_NT_CONFIGS["CB1"]["C_baseline"]

        # Engine can still step (CB1 NT + receptor both functional)
        engine.step()


# =====================================================================
# F. Adapter Integration
# =====================================================================


class TestAdapterIntegration:

    def test_adapter_output_consumed_by_engine(self):
        """Adapter-like signal dicts for all mapped NTs don't crash engine."""
        engine = _make_full_engine()

        # Simulate adapter output structure
        adapter_signals = {
            "DA": {"novelty": 0.7, "rpe": 0.3, "effort": 0.1},
            "NE": {"precision": 0.6, "uncertainty": 0.4},
            "OXT": {"empathy": 0.8, "social_engagement": 0.5},
            "cortisol": {"level": 0.3},
            "CRH": {"level": 0.2},
            "GABA": {"inhibition": 0.4},
        }

        # Should not crash — generic stepper ignores unknown signal keys
        engine.step(adapter_signals)

        # All states valid
        for nt_name in engine.registry.neurotransmitter_names():
            state = engine.registry.get_neurotransmitter(nt_name)
            assert 0.0 <= state.C_tonic <= 1.0

    def test_full_pipeline_adapter_readout_feedback(self):
        """Full pipeline: register all → step with signals → readout → feedback → step."""
        engine = _make_full_engine()

        # Step with adapter signals
        signals = {"DA": {"novelty": 0.5, "rpe": 0.2, "effort": 0.1}}
        for _ in range(10):
            engine.step(signals)

        # Readout
        readout = engine.get_neurosymbolic_readout()
        assert all(0.0 <= v <= 1.0 for v in readout.values())

        # Feedback
        meta = RewardMetaDirective(
            meta={"per_domain_weighted_scores": {
                "human_attunement": 0.6,
                "innovation": 0.5,
                "logic": 0.7,
                "ethics": 0.6,
            }},
        )
        feedback = compute_reward_feedback(meta, {})
        engine.apply_feedback(feedback)

        # Continue stepping
        for _ in range(10):
            engine.step(signals)

        # Final readout still valid
        readout2 = engine.get_neurosymbolic_readout()
        assert all(0.0 <= v <= 1.0 for v in readout2.values())

    def test_signals_for_unregistered_nt_ignored(self):
        """Signals for NTs not in the engine are silently ignored."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")

        # FAKE_NT not registered — should not crash
        signals = {
            "DA": {"novelty": 0.5},
            "FAKE_NT": {"something": 1.0},
        }
        engine.step(signals)
        assert engine.current_time == pytest.approx(0.01)
