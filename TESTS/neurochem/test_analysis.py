"""Tests for post-processing analysis utilities (Appendix N.9)."""

import numpy as np
import pytest

from zados.neurochem.optimization.analysis import (
    temporal_mean,
    temporal_variance,
    temporal_std,
    cross_run_statistics,
)
from zados.neurochem.optimization.batch_runner import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from zados.neurochem.optimization.logging import LogTierConfig, HierarchicalLogger


class TestTemporalMean:
    def test_1d(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        assert temporal_mean(data) == pytest.approx(2.5)

    def test_2d(self):
        data = np.array([[1.0, 10.0], [3.0, 30.0]])
        result = temporal_mean(data)
        np.testing.assert_allclose(result, [2.0, 20.0])

    def test_axis(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = temporal_mean(data, axis=1)
        np.testing.assert_allclose(result, [1.5, 3.5])


class TestTemporalVariance:
    def test_constant(self):
        data = np.array([5.0, 5.0, 5.0])
        assert temporal_variance(data) == pytest.approx(0.0)

    def test_known_variance(self):
        data = np.array([1.0, 3.0])
        # mean=2, var = ((1-2)^2 + (3-2)^2)/2 = 1.0
        assert temporal_variance(data) == pytest.approx(1.0)

    def test_2d(self):
        data = np.array([[0.0, 0.0], [2.0, 4.0]])
        result = temporal_variance(data)
        np.testing.assert_allclose(result, [1.0, 4.0])


class TestTemporalStd:
    def test_known_std(self):
        data = np.array([1.0, 3.0])
        assert temporal_std(data) == pytest.approx(1.0)


class TestCrossRunStatistics:
    def test_basic(self):
        """Cross-run stats from real simulation runs."""
        tiers = (LogTierConfig("conc", 10, ["concentrations"]),)
        configs = [
            SimulationConfig(n_steps=50, seed=i, log_tiers=tiers)
            for i in range(3)
        ]
        results = [run_simulation(c) for c in configs]

        stats = cross_run_statistics(results, "conc", "DA_C_tonic")
        assert stats["n_runs"] == 3
        assert len(stats["mean"]) == 5  # 50 steps / interval 10
        assert len(stats["std"]) == 5
        assert len(stats["min"]) == 5
        assert len(stats["max"]) == 5

        # min <= mean <= max at every step
        for i in range(5):
            assert stats["min"][i] <= stats["mean"][i] <= stats["max"][i]

    def test_single_run(self):
        """Single run: std should be 0."""
        tiers = (LogTierConfig("conc", 1, ["concentrations"]),)
        cfg = SimulationConfig(n_steps=10, seed=0, log_tiers=tiers)
        results = [run_simulation(cfg)]

        stats = cross_run_statistics(results, "conc", "DA_C_tonic")
        assert stats["n_runs"] == 1
        np.testing.assert_allclose(stats["std"], 0.0)
        np.testing.assert_array_equal(stats["mean"], stats["min"])
        np.testing.assert_array_equal(stats["mean"], stats["max"])

    def test_no_logger_raises(self):
        """Results without loggers raise ValueError."""
        cfg = SimulationConfig(n_steps=10, seed=0)
        results = [run_simulation(cfg)]
        # No log_tiers → logger is None
        with pytest.raises(ValueError, match="No valid traces"):
            cross_run_statistics(results, "conc", "DA_C_tonic")

    def test_missing_variable_raises(self):
        """Missing variable in tier raises ValueError."""
        tiers = (LogTierConfig("osc", 1, ["oscillations"]),)
        cfg = SimulationConfig(n_steps=10, seed=0, log_tiers=tiers)
        results = [run_simulation(cfg)]
        with pytest.raises(ValueError, match="No valid traces"):
            cross_run_statistics(results, "osc", "NONEXISTENT_VAR")
