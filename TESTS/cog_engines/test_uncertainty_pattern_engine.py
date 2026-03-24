"""Tests for Engine 26 — Uncertainty Pattern Engine."""
import math

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.uncertainty_pattern_engine import (
    UncertaintyPatternEngine, UPConfig, UPState,
    UncertaintyType, PatternType,
    ClaimWithConfidence, InferenceStep, InferenceChain, CalibrationData,
    UncertaintyEstimate, PropagationResult, UncertaintyPattern,
    UncertaintyNeurochem, UncertaintyPatternInput, UncertaintyPatternResult,
    beta_decay, classify_uncertainty_type, refine_uncertainty,
    emotion_modulate_uncertainty, extract_uncertainty_map,
    compute_system_entropy,
    propagate_chain, compute_propagation_depth,
    compute_calibration_error,
    detect_cascade_pattern, detect_island_pattern,
    detect_divergence_pattern, detect_stagnation_pattern,
    compute_epistemic_fraction, compute_aleatoric_fraction,
    find_reducible_claims, compute_uncertainty_neurochem,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Helpers
# =====================================================================

def _claim(cid="c1", conf=0.8, ev=0, chain_len=0, pred_hor=0, ambig=0.0, src="e1"):
    return ClaimWithConfidence(
        claim_id=cid, confidence=conf, evidence_count=ev,
        reasoning_chain_length=chain_len, prediction_horizon=pred_hor,
        ambiguity_score=ambig, source_engine=src,
    )

def _chain(cid, steps):
    return InferenceChain(chain_id=cid, steps=tuple(steps))

def _step(premise, conclusion, conf=0.95):
    return InferenceStep(premise_id=premise, conclusion_id=conclusion, inference_confidence=conf)


# =====================================================================
# 1. Enums
# =====================================================================

class TestEnums:
    def test_uncertainty_types(self):
        assert UncertaintyType.EPISTEMIC.value == "epistemic"
        assert UncertaintyType.ALEATORIC.value == "aleatoric"
        assert UncertaintyType.MODEL.value == "model"
        assert UncertaintyType.LINGUISTIC.value == "linguistic"

    def test_pattern_types(self):
        assert PatternType.CASCADE.value == "cascade"
        assert PatternType.ISLAND.value == "island"
        assert PatternType.DIVERGENCE.value == "divergence"
        assert PatternType.STAGNATION.value == "stagnation"


# =====================================================================
# 2. Config
# =====================================================================

class TestConfig:
    def test_defaults(self):
        cfg = UPConfig()
        assert cfg.theta_aleatoric == 3
        assert cfg.theta_model == 5
        assert cfg.theta_linguistic == 0.40
        assert cfg.u_inference_default == 0.05
        assert cfg.gamma_ne == 0.20

    def test_mode_params(self):
        cfg = UPConfig()
        assert cfg.lambda_evidence["normal"] == 0.15
        assert cfg.lambda_evidence["rem_dream"] == 0.05
        assert cfg.D_base["dev"] == 7


# =====================================================================
# 3. Beta Decay
# =====================================================================

class TestBetaDecay:
    def test_zero_evidence(self):
        assert beta_decay(0) == 1.0

    def test_positive_evidence(self):
        d = beta_decay(5, 0.15)
        assert 0 < d < 1
        assert abs(d - math.exp(-0.75)) < 1e-9

    def test_more_evidence_less_uncertainty(self):
        assert beta_decay(10) < beta_decay(5)

    def test_negative_evidence(self):
        assert beta_decay(-1) == 1.0


# =====================================================================
# 4. Uncertainty Type Classification
# =====================================================================

class TestClassification:
    def test_linguistic(self):
        c = _claim(ambig=0.5)
        assert classify_uncertainty_type(c) == UncertaintyType.LINGUISTIC

    def test_aleatoric(self):
        c = _claim(pred_hor=5)
        assert classify_uncertainty_type(c) == UncertaintyType.ALEATORIC

    def test_model(self):
        c = _claim(chain_len=7)
        assert classify_uncertainty_type(c) == UncertaintyType.MODEL

    def test_epistemic_default(self):
        c = _claim()
        assert classify_uncertainty_type(c) == UncertaintyType.EPISTEMIC

    def test_linguistic_priority(self):
        c = _claim(ambig=0.5, pred_hor=5, chain_len=7)
        assert classify_uncertainty_type(c) == UncertaintyType.LINGUISTIC


# =====================================================================
# 5. Refine Uncertainty
# =====================================================================

class TestRefine:
    def test_no_evidence(self):
        assert refine_uncertainty(0.5, 0) == 0.5

    def test_evidence_reduces(self):
        r = refine_uncertainty(0.5, 5, 0.15)
        assert r < 0.5

    def test_clamped(self):
        assert refine_uncertainty(1.5, 0) == 1.0
        assert refine_uncertainty(-0.5, 0) == 0.0


# =====================================================================
# 6. Emotion Modulation
# =====================================================================

class TestEmotionModulate:
    def test_no_emotions(self):
        assert emotion_modulate_uncertainty(0.5, None) == 0.5

    def test_anxiety_increases(self):
        u = emotion_modulate_uncertainty(0.5, {"anxiety": 1.0})
        assert u > 0.5

    def test_confident_decreases(self):
        u = emotion_modulate_uncertainty(0.5, {"confident": 1.0})
        assert u < 0.5

    def test_combined(self):
        u = emotion_modulate_uncertainty(0.5, {"anxiety": 0.5, "confident": 0.5})
        # Anxiety kappa (0.20) > confident kappa (0.10), so net increase
        assert u > 0.5


# =====================================================================
# 7. System Entropy
# =====================================================================

class TestSystemEntropy:
    def test_empty_map(self):
        assert compute_system_entropy({}) == 0.0

    def test_certain_claims(self):
        umap = {
            "c1": UncertaintyEstimate("c1", 0.0, 0.0, "epistemic", "e1", 5, 0.01),
        }
        # Near-zero uncertainty → low entropy
        h = compute_system_entropy(umap)
        assert h < 0.2

    def test_maximum_uncertainty(self):
        umap = {
            "c1": UncertaintyEstimate("c1", 0.5, 0.5, "epistemic", "e1", 0, 0.5),
            "c2": UncertaintyEstimate("c2", 0.5, 0.5, "epistemic", "e1", 0, 0.5),
        }
        h = compute_system_entropy(umap)
        assert h > 0.9  # Near max entropy


# =====================================================================
# 8. Propagation
# =====================================================================

class TestPropagation:
    def test_empty_chain(self):
        chain = _chain("ch1", [])
        pr = propagate_chain(chain, {})
        assert pr.chain_uncertainty == 0.0
        assert pr.depth_reached == 0

    def test_single_step(self):
        umap = {
            "p1": UncertaintyEstimate("p1", 0.3, 0.3, "epistemic", "e1", 1, 0.3),
        }
        chain = _chain("ch1", [_step("p1", "c1", 0.95)])
        pr = propagate_chain(chain, umap)
        # u = 1 - (1-0.3)*(1-0.05) = 1 - 0.7*0.95 = 1 - 0.665 = 0.335
        assert abs(pr.chain_uncertainty - 0.335) < 0.01

    def test_multi_step_amplifies(self):
        umap = {
            "p1": UncertaintyEstimate("p1", 0.3, 0.3, "epistemic", "e1", 1, 0.3),
            "p2": UncertaintyEstimate("p2", 0.3, 0.3, "epistemic", "e1", 1, 0.3),
        }
        chain = _chain("ch1", [
            _step("p1", "mid", 0.90),
            _step("p2", "c1", 0.90),
        ])
        pr = propagate_chain(chain, umap)
        assert pr.chain_uncertainty > 0.3  # Amplified
        assert pr.depth_reached == 2

    def test_bottleneck_detection(self):
        umap = {
            "p1": UncertaintyEstimate("p1", 0.8, 0.8, "epistemic", "e1", 0, 0.8),
            "p2": UncertaintyEstimate("p2", 0.1, 0.1, "epistemic", "e1", 5, 0.1),
        }
        chain = _chain("ch1", [
            _step("p1", "mid", 0.95),
            _step("p2", "c1", 0.95),
        ])
        pr = propagate_chain(chain, umap)
        assert pr.bottleneck_premise == "p1"
        assert pr.bottleneck_contribution > 0.5


class TestPropagationDepth:
    def test_basic(self):
        d = compute_propagation_depth(5, 3, 4, 0.5, 0.5)
        # 5 + 3*0.5 - 4*0.5 = 5 + 1.5 - 2.0 = 4
        assert d == 4

    def test_high_theta_gamma(self):
        d = compute_propagation_depth(5, 3, 4, 1.0, 0.0)
        assert d == 8  # 5 + 3 - 0

    def test_minimum(self):
        d = compute_propagation_depth(1, 0, 10, 0.0, 1.0)
        assert d >= 1


# =====================================================================
# 9. Calibration Error
# =====================================================================

class TestCalibration:
    def test_none(self):
        assert compute_calibration_error(None) == 0.0

    def test_empty(self):
        assert compute_calibration_error(CalibrationData({})) == 0.0

    def test_perfect_calibration(self):
        # Bin 2 (midpoint 0.5): 50% correct out of 100
        cal = CalibrationData(bins={2: (50, 100)})
        ece = compute_calibration_error(cal)
        assert ece < 0.01  # Near perfect

    def test_overconfident(self):
        # Bin 4 (midpoint 0.9): only 40% correct
        cal = CalibrationData(bins={4: (40, 100)})
        ece = compute_calibration_error(cal)
        assert ece > 0.3  # Badly calibrated


# =====================================================================
# 10. Pattern Detection
# =====================================================================

class TestCascade:
    def test_no_cascade_short_chain(self):
        chain = _chain("ch1", [_step("p1", "c1")])
        umap = {"p1": UncertaintyEstimate("p1", 0.6, 0.6, "epistemic", "e1", 0, 0.6)}
        assert detect_cascade_pattern(chain, umap) is None

    def test_cascade_detected(self):
        umap = {
            f"p{i}": UncertaintyEstimate(f"p{i}", 0.6, 0.6, "epistemic", "e1", 0, 0.6)
            for i in range(4)
        }
        steps = [_step(f"p{i}", f"p{i+1}") for i in range(3)]
        chain = _chain("ch1", steps)
        pat = detect_cascade_pattern(chain, umap, min_length=3, threshold=0.50)
        assert pat is not None
        assert pat.pattern_type == PatternType.CASCADE.value


class TestIsland:
    def test_island_detected(self):
        umap = {
            "p0": UncertaintyEstimate("p0", 0.7, 0.7, "epistemic", "e1", 0, 0.7),
            "p1": UncertaintyEstimate("p1", 0.1, 0.1, "epistemic", "e1", 5, 0.1),
            "p2": UncertaintyEstimate("p2", 0.7, 0.7, "epistemic", "e1", 0, 0.7),
        }
        chain = _chain("ch1", [_step("p0", "p1"), _step("p1", "p2")])
        patterns = detect_island_pattern(umap, chain, delta_threshold=0.3)
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == PatternType.ISLAND.value


class TestDivergence:
    def test_no_divergence_single_estimate(self):
        umap = {"c1": UncertaintyEstimate("c1", 0.5, 0.5, "epistemic", "e1", 0, 0.5)}
        assert detect_divergence_pattern(umap) == []


class TestStagnation:
    def test_stagnation_detected(self):
        umap = {"c1": UncertaintyEstimate("c1", 0.5, 0.5, "epistemic", "e1", 0, 0.5)}
        history = {"c1": [0.50, 0.50, 0.50, 0.50, 0.50]}
        patterns = detect_stagnation_pattern(umap, history, min_cycles=5, delta=0.05)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.STAGNATION.value

    def test_no_stagnation_insufficient_history(self):
        umap = {"c1": UncertaintyEstimate("c1", 0.5, 0.5, "epistemic", "e1", 0, 0.5)}
        history = {"c1": [0.50, 0.50]}
        assert detect_stagnation_pattern(umap, history) == []


# =====================================================================
# 11. Epistemic/Aleatoric Fractions
# =====================================================================

class TestFractions:
    def test_all_epistemic(self):
        umap = {
            "c1": UncertaintyEstimate("c1", 0.5, 0.5, "epistemic", "e1", 0, 0.5),
            "c2": UncertaintyEstimate("c2", 0.5, 0.5, "epistemic", "e1", 0, 0.5),
        }
        assert compute_epistemic_fraction(umap) == 1.0
        assert compute_aleatoric_fraction(umap) == 0.0

    def test_all_aleatoric(self):
        umap = {
            "c1": UncertaintyEstimate("c1", 0.5, 0.5, "aleatoric", "e1", 0, 0.5),
        }
        assert compute_aleatoric_fraction(umap) == 1.0
        assert compute_epistemic_fraction(umap) == 0.0

    def test_mixed(self):
        umap = {
            "c1": UncertaintyEstimate("c1", 0.5, 0.5, "epistemic", "e1", 0, 0.5),
            "c2": UncertaintyEstimate("c2", 0.5, 0.5, "aleatoric", "e1", 0, 0.5),
        }
        ep = compute_epistemic_fraction(umap)
        al = compute_aleatoric_fraction(umap)
        assert abs(ep - 0.5) < 0.01
        assert abs(al - 0.5) < 0.01


class TestReducible:
    def test_finds_top(self):
        umap = {
            "c1": UncertaintyEstimate("c1", 0.8, 0.8, "epistemic", "e1", 0, 0.8),
            "c2": UncertaintyEstimate("c2", 0.3, 0.3, "aleatoric", "e1", 0, 0.3),
            "c3": UncertaintyEstimate("c3", 0.5, 0.5, "linguistic", "e1", 0, 0.5),
        }
        r = find_reducible_claims(umap, top_n=2)
        assert "c1" in r
        assert "c3" in r
        assert "c2" not in r


# =====================================================================
# 12. Neurochemical Signals
# =====================================================================

class TestNeurochem:
    def test_high_entropy(self):
        nc = compute_uncertainty_neurochem(
            system_entropy=0.8, theta_alert=0.6,
            cascade_count=1, max_cascades=3,
            has_bottleneck=True, bottleneck_contribution=0.8,
            delta_h=0.1, ece=0.0, cfg=UPConfig(),
        )
        assert nc.delta_ne > 0   # High entropy → NE alert
        assert nc.delta_da < 0   # High entropy → DA dampened
        assert nc.delta_ach > 0  # Bottleneck → ACh focus

    def test_improving_entropy(self):
        nc = compute_uncertainty_neurochem(
            system_entropy=0.4, theta_alert=0.6,
            cascade_count=0, max_cascades=3,
            has_bottleneck=False, bottleneck_contribution=0.0,
            delta_h=-0.2, ece=0.0, cfg=UPConfig(),
        )
        assert nc.delta_gaba > 0  # Improving → GABA stabilize

    def test_overconfidence(self):
        nc = compute_uncertainty_neurochem(
            system_entropy=0.3, theta_alert=0.6,
            cascade_count=0, max_cascades=3,
            has_bottleneck=False, bottleneck_contribution=0.0,
            delta_h=0.0, ece=0.20, cfg=UPConfig(),
        )
        assert nc.beta_suppress > 0  # Overconfidence → suppress beta


# =====================================================================
# 13. Full Pipeline
# =====================================================================

class TestFullPipeline:
    def _make_engine(self, seed=42):
        return UncertaintyPatternEngine(rng=np.random.default_rng(seed))

    def test_basic_run(self):
        engine = self._make_engine()
        inp = UncertaintyPatternInput(
            engine_claims={
                "e1": (
                    _claim("c1", 0.8, ev=3),
                    _claim("c2", 0.4, ev=0),
                ),
            },
        )
        result = engine.process(inp)
        assert result.engine_id == "uncertainty_pattern_engine"
        assert len(result.uncertainty_map) == 2
        assert result.system_entropy > 0
        assert result.processing_time_ms > 0

    def test_with_chains(self):
        engine = self._make_engine()
        inp = UncertaintyPatternInput(
            engine_claims={
                "e1": (
                    _claim("p1", 0.6, ev=1),
                    _claim("p2", 0.6, ev=1),
                    _claim("p3", 0.6, ev=1),
                ),
            },
            inference_chains=(
                _chain("ch1", [
                    _step("p1", "p2", 0.90),
                    _step("p2", "p3", 0.90),
                ]),
            ),
        )
        result = engine.process(inp)
        assert len(result.propagation_results) == 1
        assert result.propagation_results[0].chain_uncertainty > 0

    def test_calibration_alert(self):
        engine = self._make_engine()
        # Bad calibration: bin 4 (high confidence) but only 30% correct
        cal = CalibrationData(bins={4: (30, 100)})
        inp = UncertaintyPatternInput(
            engine_claims={"e1": (_claim("c1", 0.8),)},
            historical_calibration=cal,
        )
        result = engine.process(inp)
        assert result.calibration_error > 0.1
        assert result.overconfidence_alert is True

    def test_epistemic_aleatoric_split(self):
        engine = self._make_engine()
        inp = UncertaintyPatternInput(
            engine_claims={
                "e1": (
                    _claim("c1", 0.5, pred_hor=0),    # Epistemic
                    _claim("c2", 0.5, pred_hor=5),    # Aleatoric
                ),
            },
        )
        result = engine.process(inp)
        assert result.epistemic_fraction > 0
        assert result.aleatoric_fraction > 0

    def test_empty_input(self):
        engine = self._make_engine()
        result = engine.process(UncertaintyPatternInput())
        assert len(result.uncertainty_map) == 0
        assert result.system_entropy == 0.0


# =====================================================================
# 14. Mode Configuration
# =====================================================================

class TestModes:
    def test_configure(self):
        engine = UncertaintyPatternEngine()
        engine.configure(OperationalMode.DEV)
        status = engine.get_status()
        assert status["mode"] == "dev"

    def test_reflective_stricter(self):
        cfg = UPConfig()
        assert cfg.theta_alert["reflective"] < cfg.theta_alert["normal"]
        assert cfg.theta_calibration_alarm["reflective"] < cfg.theta_calibration_alarm["normal"]

    def test_dream_permissive(self):
        cfg = UPConfig()
        assert cfg.theta_alert["rem_dream"] > cfg.theta_alert["normal"]
        assert cfg.lambda_evidence["rem_dream"] < cfg.lambda_evidence["normal"]


# =====================================================================
# 15. NT Feedback
# =====================================================================

class TestNTFeedback:
    def test_update_state(self):
        engine = UncertaintyPatternEngine()
        engine.update_neurochem_state({"ne": 0.8, "da": 0.3, "cor": 0.5})
        status = engine.get_status()
        assert status["nt_levels"]["ne"] == 0.8
        assert status["nt_levels"]["da"] == 0.3
        assert status["nt_levels"]["cor"] == 0.5

    def test_clamps(self):
        engine = UncertaintyPatternEngine()
        engine.update_neurochem_state({"ne": 1.5, "da": -0.5})
        status = engine.get_status()
        assert status["nt_levels"]["ne"] == 1.0
        assert status["nt_levels"]["da"] == 0.0


# =====================================================================
# 16. Introspection
# =====================================================================

class TestIntrospection:
    def test_status_keys(self):
        engine = UncertaintyPatternEngine()
        status = engine.get_status()
        assert "engine_id" in status
        assert "cluster" in status
        assert "mode" in status
        assert "cycle_count" in status
        assert "total_analyses" in status
        assert "nt_levels" in status

    def test_cycle_increments(self):
        engine = UncertaintyPatternEngine()
        engine.process(UncertaintyPatternInput(
            engine_claims={"e1": (_claim("c1", 0.8),)},
        ))
        engine.process(UncertaintyPatternInput(
            engine_claims={"e1": (_claim("c1", 0.8),)},
        ))
        assert engine.get_status()["cycle_count"] == 2


# =====================================================================
# 17. Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_all_perfect_confidence(self):
        engine = UncertaintyPatternEngine(rng=np.random.default_rng(42))
        inp = UncertaintyPatternInput(
            engine_claims={
                "e1": (_claim("c1", 1.0, ev=10), _claim("c2", 1.0, ev=10)),
            },
        )
        result = engine.process(inp)
        assert result.system_entropy < 0.1

    def test_all_zero_confidence(self):
        engine = UncertaintyPatternEngine(rng=np.random.default_rng(42))
        inp = UncertaintyPatternInput(
            engine_claims={
                "e1": (_claim("c1", 0.0), _claim("c2", 0.0)),
            },
        )
        result = engine.process(inp)
        assert result.system_entropy < 0.1  # All max uncertainty → low entropy (all same)

    def test_single_claim(self):
        engine = UncertaintyPatternEngine(rng=np.random.default_rng(42))
        result = engine.process(UncertaintyPatternInput(
            engine_claims={"e1": (_claim("c1", 0.5),)},
        ))
        assert len(result.uncertainty_map) == 1

    def test_claim_history_bounded(self):
        engine = UncertaintyPatternEngine(rng=np.random.default_rng(42))
        for _ in range(25):
            engine.process(UncertaintyPatternInput(
                engine_claims={"e1": (_claim("c1", 0.5),)},
            ))
        assert len(engine._state.claim_history["c1"]) <= 20
