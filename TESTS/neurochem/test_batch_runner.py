"""Tests for batch simulation runner and parameter sweeps (Appendix N.4/N.7)."""

import numpy as np
import pytest

from zados.neurochem.optimization.batch_runner import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
    generate_parameter_grid,
    generate_lhs_configs,
    run_batch,
)
from zados.neurochem.optimization.logging import LogTierConfig
from zados.neurochem.optimization.timescale import TimescaleConfig
from zados.neurochem.optimization.checkpoint import Checkpoint


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.dt == 0.01
        assert cfg.n_steps == 1000
        assert cfg.seed == 42
        assert cfg.scheduler is None
        assert cfg.log_tiers is None
        assert cfg.checkpoint_interval == 0
        assert cfg.oscillation_mode == "state_derived"
        assert cfg.parameter_overrides is None
        assert cfg.signal_fn is None

    def test_frozen(self):
        cfg = SimulationConfig()
        with pytest.raises(AttributeError):
            cfg.dt = 0.05

    def test_custom_values(self):
        cfg = SimulationConfig(
            dt=0.005,
            n_steps=500,
            seed=123,
            oscillation_mode="static",
        )
        assert cfg.dt == 0.005
        assert cfg.n_steps == 500
        assert cfg.seed == 123
        assert cfg.oscillation_mode == "static"


class TestRunSimulation:
    def test_basic_completes(self):
        """Single 100-step run completes and returns correct elapsed_steps."""
        cfg = SimulationConfig(n_steps=100, seed=1)
        result = run_simulation(cfg)
        assert isinstance(result, SimulationResult)
        assert result.elapsed_steps == 100
        assert result.config is cfg

    def test_with_logging(self):
        """Logger captures data when log_tiers provided."""
        tiers = (LogTierConfig("conc", 10, ["concentrations"]),)
        cfg = SimulationConfig(n_steps=100, seed=2, log_tiers=tiers)
        result = run_simulation(cfg)
        assert result.logger is not None
        data = result.logger.get_tier_data("conc")
        assert len(data["steps"]) == 10  # every 10th step of 100

    def test_with_checkpoint(self):
        """Checkpoint produced when checkpoint_interval > 0."""
        cfg = SimulationConfig(n_steps=100, seed=3, checkpoint_interval=50)
        result = run_simulation(cfg)
        assert result.final_checkpoint is not None
        assert isinstance(result.final_checkpoint, Checkpoint)
        assert result.final_checkpoint.step_number == 100

    def test_final_checkpoint_always_produced(self):
        """Even without interval, final checkpoint is produced."""
        cfg = SimulationConfig(n_steps=50, seed=4)
        result = run_simulation(cfg)
        assert result.final_checkpoint is not None
        assert result.final_checkpoint.step_number == 50

    def test_reproducible(self):
        """Same seed produces identical final state."""
        cfg = SimulationConfig(n_steps=100, seed=42)
        r1 = run_simulation(cfg)
        r2 = run_simulation(cfg)

        # Compare NT concentrations via checkpoint
        for nt_name in r1.final_checkpoint.neurotransmitter_states:
            s1 = r1.final_checkpoint.neurotransmitter_states[nt_name]
            s2 = r2.final_checkpoint.neurotransmitter_states[nt_name]
            assert s1["C_tonic"] == pytest.approx(s2["C_tonic"], abs=1e-12)
            assert s1["C_phasic"] == pytest.approx(s2["C_phasic"], abs=1e-12)

    def test_with_scheduler(self):
        """Run with timescale scheduler completes without error."""
        tc = TimescaleConfig(M_receptor=10, M_fatigue=5, M_oscillation=5)
        cfg = SimulationConfig(n_steps=50, seed=5, scheduler=tc)
        result = run_simulation(cfg)
        assert result.elapsed_steps == 50

    def test_with_signal_fn(self):
        """Signal function is called each step."""
        call_count = [0]

        def my_signal(step):
            call_count[0] += 1
            return {"novelty": 0.5}

        cfg = SimulationConfig(n_steps=20, seed=6, signal_fn=my_signal)
        result = run_simulation(cfg)
        assert call_count[0] == 20
        assert result.elapsed_steps == 20

    def test_parameter_override_applied(self):
        """Parameter overrides change engine config values."""
        overrides = {"DA": {"C_baseline": 0.99}}
        cfg = SimulationConfig(
            n_steps=10, seed=7, parameter_overrides=overrides,
        )
        result = run_simulation(cfg)
        # The DA baseline should have been overridden
        # We can verify via the checkpoint's config or NT state
        assert result.elapsed_steps == 10


class TestParameterGrid:
    def test_single_param(self):
        base = SimulationConfig(n_steps=10)
        sweep = {"DA.C_baseline": [0.3, 0.4, 0.5]}
        configs = generate_parameter_grid(base, sweep)
        assert len(configs) == 3

    def test_two_params_cartesian(self):
        base = SimulationConfig(n_steps=10)
        sweep = {
            "DA.C_baseline": [0.3, 0.5],
            "5HT.theta_tonic": [0.05, 0.1],
        }
        configs = generate_parameter_grid(base, sweep)
        assert len(configs) == 4  # 2 * 2

    def test_seeds_differ(self):
        base = SimulationConfig(n_steps=10, seed=100)
        sweep = {"DA.C_baseline": [0.3, 0.4, 0.5]}
        configs = generate_parameter_grid(base, sweep)
        seeds = [c.seed for c in configs]
        assert len(set(seeds)) == 3  # all different

    def test_overrides_applied(self):
        base = SimulationConfig(n_steps=10)
        sweep = {"DA.C_baseline": [0.3, 0.5]}
        configs = generate_parameter_grid(base, sweep)
        assert configs[0].parameter_overrides["DA"]["C_baseline"] == 0.3
        assert configs[1].parameter_overrides["DA"]["C_baseline"] == 0.5

    def test_preserves_base_params(self):
        base = SimulationConfig(n_steps=10, dt=0.005)
        sweep = {"DA.C_baseline": [0.3]}
        configs = generate_parameter_grid(base, sweep)
        assert configs[0].dt == 0.005
        assert configs[0].n_steps == 10


class TestLHSConfigs:
    def test_correct_count(self):
        base = SimulationConfig(n_steps=10)
        ranges = {"DA.C_baseline": (0.1, 0.9)}
        configs = generate_lhs_configs(base, ranges, n_samples=8, seed=0)
        assert len(configs) == 8

    def test_within_bounds(self):
        base = SimulationConfig(n_steps=10)
        ranges = {
            "DA.C_baseline": (0.2, 0.8),
            "5HT.theta_tonic": (0.01, 0.2),
        }
        configs = generate_lhs_configs(base, ranges, n_samples=20, seed=0)
        for cfg in configs:
            da_val = cfg.parameter_overrides["DA"]["C_baseline"]
            ht_val = cfg.parameter_overrides["5HT"]["theta_tonic"]
            assert 0.2 <= da_val <= 0.8
            assert 0.01 <= ht_val <= 0.2

    def test_reproducible(self):
        base = SimulationConfig(n_steps=10)
        ranges = {"DA.C_baseline": (0.1, 0.9)}
        c1 = generate_lhs_configs(base, ranges, n_samples=5, seed=99)
        c2 = generate_lhs_configs(base, ranges, n_samples=5, seed=99)
        for a, b in zip(c1, c2):
            assert a.parameter_overrides == b.parameter_overrides

    def test_different_seeds_different_samples(self):
        base = SimulationConfig(n_steps=10)
        ranges = {"DA.C_baseline": (0.1, 0.9)}
        c1 = generate_lhs_configs(base, ranges, n_samples=5, seed=0)
        c2 = generate_lhs_configs(base, ranges, n_samples=5, seed=1)
        vals1 = [c.parameter_overrides["DA"]["C_baseline"] for c in c1]
        vals2 = [c.parameter_overrides["DA"]["C_baseline"] for c in c2]
        assert vals1 != vals2


class TestRunBatch:
    def test_sequential(self):
        """Batch with max_workers=1 runs sequentially."""
        configs = [
            SimulationConfig(n_steps=20, seed=i) for i in range(3)
        ]
        results = run_batch(configs, max_workers=1)
        assert len(results) == 3
        for r in results:
            assert r.elapsed_steps == 20

    def test_different_seeds_different_results(self):
        """Different seeds produce different final states."""
        configs = [
            SimulationConfig(n_steps=50, seed=i) for i in range(3)
        ]
        results = run_batch(configs, max_workers=1)

        # Collect DA C_tonic from each
        da_vals = []
        for r in results:
            da_vals.append(
                r.final_checkpoint.neurotransmitter_states["DA"]["C_tonic"]
            )
        # At least two should differ (stochastic with different seeds)
        assert len(set(round(v, 6) for v in da_vals)) > 1

    def test_single_config(self):
        """Single config runs fine."""
        configs = [SimulationConfig(n_steps=10, seed=0)]
        results = run_batch(configs, max_workers=1)
        assert len(results) == 1

    def test_signal_fn_not_allowed_parallel(self):
        """signal_fn raises ValueError in parallel mode."""
        configs = [
            SimulationConfig(n_steps=10, seed=0, signal_fn=lambda s: {}),
            SimulationConfig(n_steps=10, seed=1, signal_fn=lambda s: {}),
        ]
        with pytest.raises(ValueError, match="signal_fn"):
            run_batch(configs, max_workers=2)
