"""Tests for timescale separation (Appendix N.2)."""

import pytest

from zados.neurochem.optimization.timescale import (
    TimescaleConfig,
    SparseUpdateScheduler,
    DEFAULT_TIMESCALE_CONFIG,
)
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.neurotransmitters.configs import register_all_neurotransmitters


class TestTimescaleConfig:
    def test_defaults(self):
        config = TimescaleConfig()
        assert config.M_receptor == 100
        assert config.M_fatigue == 50
        assert config.M_oscillation == 10

    def test_frozen(self):
        config = TimescaleConfig()
        with pytest.raises(AttributeError):
            config.M_receptor = 200

    def test_default_timescale_config(self):
        assert DEFAULT_TIMESCALE_CONFIG.M_receptor == 100


class TestSparseUpdateScheduler:
    def test_should_update_receptor(self):
        scheduler = SparseUpdateScheduler(TimescaleConfig(M_receptor=10))
        fired = [n for n in range(100) if scheduler.should_update("receptor", n)]
        assert fired == list(range(0, 100, 10))

    def test_should_update_fatigue(self):
        scheduler = SparseUpdateScheduler(TimescaleConfig(M_fatigue=5))
        fired = [n for n in range(20) if scheduler.should_update("fatigue", n)]
        assert fired == [0, 5, 10, 15]

    def test_should_update_oscillation(self):
        scheduler = SparseUpdateScheduler(TimescaleConfig(M_oscillation=3))
        fired = [n for n in range(12) if scheduler.should_update("oscillation", n)]
        assert fired == [0, 3, 6, 9]

    def test_every_tick_when_interval_1(self):
        scheduler = SparseUpdateScheduler(TimescaleConfig(M_receptor=1))
        assert all(scheduler.should_update("receptor", n) for n in range(10))

    def test_get_interval(self):
        scheduler = SparseUpdateScheduler(TimescaleConfig(M_receptor=50))
        assert scheduler.get_interval("receptor") == 50

    def test_get_interval_unknown_group(self):
        scheduler = SparseUpdateScheduler()
        with pytest.raises(KeyError):
            scheduler.get_interval("unknown")

    def test_scaled_dt(self):
        scheduler = SparseUpdateScheduler(TimescaleConfig(M_receptor=100))
        assert scheduler.get_scaled_dt("receptor", 0.01) == pytest.approx(1.0)

    def test_scaled_dt_fatigue(self):
        scheduler = SparseUpdateScheduler(TimescaleConfig(M_fatigue=50))
        assert scheduler.get_scaled_dt("fatigue", 0.01) == pytest.approx(0.5)


class TestEngineWithScheduler:
    def test_engine_with_scheduler_receptor_sparse(self):
        """Receptors should update less frequently with scheduler."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        engine.scheduler = SparseUpdateScheduler(
            TimescaleConfig(M_receptor=5, M_oscillation=1)
        )

        # Record DA_D1 rho after each step
        rho_values = []
        for _ in range(10):
            engine.step()
            da_d1 = engine.registry.get_receptor("DA_D1")
            rho_values.append(da_d1.rho)

        # Without scheduler, rho would potentially change every tick.
        # With M_receptor=5, receptors only update at steps 0, 5, 10...
        # So rho should be constant between update ticks.
        # Steps 1-4 should have same rho as step 0 update
        assert rho_values[0] == rho_values[1]
        assert rho_values[0] == rho_values[2]
        assert rho_values[0] == rho_values[3]

    def test_engine_without_scheduler_unchanged(self):
        """Without scheduler, everything updates every tick (backward compat)."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        assert engine.scheduler is None

        # Should complete without errors
        for _ in range(20):
            engine.step()

        # Verify state changed
        da = engine.registry.get_neurotransmitter("DA")
        assert da.C_tonic != 0.5 or da.F > 0.0  # something changed
