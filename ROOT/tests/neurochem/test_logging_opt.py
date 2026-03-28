"""Tests for hierarchical logging (Appendix N.6)."""

import os
import tempfile

import numpy as np
import pytest

from zados.neurochem.optimization.logging import (
    LogTierConfig,
    HierarchicalLogger,
    DEFAULT_LOG_TIERS,
)
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.neurotransmitters.configs import (
    register_all_neurotransmitters,
    DEFAULT_NT_CONFIGS,
    NT_RECEPTOR_MAP,
)
from zados.neurochem.state import OscillationState


def _make_engine(seed=42):
    engine = NeurochemicalEngine(dt=0.01, seed=seed)
    register_all_neurotransmitters(engine)
    engine.set_oscillation_state(OscillationState(
        delta=0.3, theta=0.4, alpha=0.5, beta=0.6, gamma=0.7,
    ))
    return engine


class TestLogTierConfig:
    def test_creation(self):
        tier = LogTierConfig("test", 10, ["concentrations"])
        assert tier.name == "test"
        assert tier.sample_interval == 10
        assert tier.variables == ["concentrations"]

    def test_default_tiers(self):
        assert len(DEFAULT_LOG_TIERS) == 3
        names = [t.name for t in DEFAULT_LOG_TIERS]
        assert "high_res" in names
        assert "med_res" in names
        assert "low_res" in names


class TestHierarchicalLogger:
    def test_should_log_interval(self):
        logger = HierarchicalLogger([
            LogTierConfig("fast", 5, ["concentrations"]),
        ])
        fired = [n for n in range(20) if logger.should_log("fast", n)]
        assert fired == [0, 5, 10, 15]

    def test_should_log_unknown_tier(self):
        logger = HierarchicalLogger([])
        assert not logger.should_log("unknown", 0)

    def test_log_concentrations(self):
        engine = _make_engine()
        logger = HierarchicalLogger([
            LogTierConfig("conc", 1, ["concentrations"]),
        ])

        engine.step()
        logger.log_concentrations(0, engine)

        data = logger.get_tier_data("conc")
        assert "steps" in data
        assert len(data["steps"]) == 1

        # Should have 12 NTs × 3 components = 36 columns
        n_nt = len(DEFAULT_NT_CONFIGS)
        expected_keys = n_nt * 3 + 1  # +1 for "steps"
        assert len(data) == expected_keys

    def test_log_receptors(self):
        engine = _make_engine()
        logger = HierarchicalLogger([
            LogTierConfig("rec", 1, ["receptors"]),
        ])

        engine.step()
        logger.log_receptors(0, engine)

        data = logger.get_tier_data("rec")
        total_receptors = sum(len(recs) for recs in NT_RECEPTOR_MAP.values())
        expected_keys = total_receptors * 2 + 1  # rho + sigma per receptor + steps
        assert len(data) == expected_keys

    def test_log_oscillations(self):
        engine = _make_engine()
        logger = HierarchicalLogger([
            LogTierConfig("osc", 1, ["oscillations"]),
        ])

        logger.log_oscillations(0, engine)
        data = logger.get_tier_data("osc")
        assert len(data) == 6  # 5 bands + steps
        assert "delta" in data
        assert "gamma" in data
        assert data["gamma"][0] == pytest.approx(0.7)

    def test_multi_tier_different_rates(self):
        engine = _make_engine()
        logger = HierarchicalLogger([
            LogTierConfig("fast", 1, ["concentrations"]),
            LogTierConfig("slow", 5, ["oscillations"]),
        ])

        for step in range(20):
            engine.step()
            if logger.should_log("fast", step):
                logger.log_concentrations(step, engine)
            if logger.should_log("slow", step):
                logger.log_oscillations(step, engine)

        fast_data = logger.get_tier_data("fast")
        slow_data = logger.get_tier_data("slow")

        assert len(fast_data["steps"]) == 20  # every step
        assert len(slow_data["steps"]) == 4   # steps 0, 5, 10, 15

    def test_save_load_npz(self):
        engine = _make_engine()
        logger = HierarchicalLogger([
            LogTierConfig("conc", 1, ["concentrations"]),
            LogTierConfig("osc", 1, ["oscillations"]),
        ])

        for step in range(5):
            engine.step()
            logger.log_concentrations(step, engine)
            logger.log_oscillations(step, engine)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            logger.save_npz(path)
            assert os.path.exists(path)

            loaded = HierarchicalLogger.load_npz(path)
            orig_conc = logger.get_tier_data("conc")
            load_conc = loaded.get_tier_data("conc")

            np.testing.assert_array_equal(orig_conc["steps"], load_conc["steps"])

            # Check a specific column round-trips
            key = f"DA_C_tonic"
            np.testing.assert_allclose(
                orig_conc[key], load_conc[key], atol=1e-5,
            )
        finally:
            os.unlink(path)

    def test_empty_tier_data(self):
        logger = HierarchicalLogger([LogTierConfig("empty", 1, [])])
        data = logger.get_tier_data("empty")
        assert "steps" in data
        assert len(data["steps"]) == 0
