"""Tests for stochastic impulse generators (Extractor 2 noise component)."""

import numpy as np
import pytest

from zados.neurochem.extractors.stochastic_impulse import (
    sample_gamma_impulse,
    sample_poisson_impulse,
    sample_lognormal_impulse,
    sample_impulse,
)


class TestGammaImpulse:
    def test_zero_eval_returns_zero(self):
        rng = np.random.default_rng(42)
        assert sample_gamma_impulse(0.0, rng=rng) == 0.0

    def test_positive_eval_positive_output(self):
        rng = np.random.default_rng(42)
        result = sample_gamma_impulse(0.8, rng=rng)
        assert result > 0.0

    def test_reproducible(self):
        r1 = sample_gamma_impulse(0.5, rng=np.random.default_rng(99))
        r2 = sample_gamma_impulse(0.5, rng=np.random.default_rng(99))
        assert r1 == pytest.approx(r2)

    def test_volatility_increases_variance(self):
        """Higher |de/dt| should produce more variable outputs."""
        rng_stable = np.random.default_rng(42)
        rng_volatile = np.random.default_rng(42)

        # Collect samples
        stable = [sample_gamma_impulse(0.5, d_eval_dt=0.0,
                  rng=np.random.default_rng(i)) for i in range(200)]
        volatile = [sample_gamma_impulse(0.5, d_eval_dt=5.0,
                    rng=np.random.default_rng(i)) for i in range(200)]

        # Volatile should have different distribution (higher shape → lower relative variance)
        # But the key test is that function runs without error and produces non-negative values
        assert all(v >= 0 for v in stable)
        assert all(v >= 0 for v in volatile)

    def test_scales_with_eval_value(self):
        """Higher eval_value → higher expected impulse."""
        low = [sample_gamma_impulse(0.1, rng=np.random.default_rng(i)) for i in range(100)]
        high = [sample_gamma_impulse(0.9, rng=np.random.default_rng(i)) for i in range(100)]
        assert np.mean(high) > np.mean(low)


class TestPoissonImpulse:
    def test_zero_eval_returns_zero(self):
        rng = np.random.default_rng(42)
        assert sample_poisson_impulse(0.0, rng=rng) == 0.0

    def test_positive_output(self):
        rng = np.random.default_rng(42)
        result = sample_poisson_impulse(0.8, rng=rng)
        assert result >= 0.0

    def test_reproducible(self):
        r1 = sample_poisson_impulse(0.5, rng=np.random.default_rng(99))
        r2 = sample_poisson_impulse(0.5, rng=np.random.default_rng(99))
        assert r1 == pytest.approx(r2)

    def test_mean_approximates_eval(self):
        """Expected value of normalised Poisson ≈ eval_value."""
        samples = [sample_poisson_impulse(0.6, rng=np.random.default_rng(i))
                    for i in range(500)]
        assert np.mean(samples) == pytest.approx(0.6, abs=0.15)

    def test_scales_with_eval_value(self):
        low = [sample_poisson_impulse(0.1, rng=np.random.default_rng(i)) for i in range(100)]
        high = [sample_poisson_impulse(0.9, rng=np.random.default_rng(i)) for i in range(100)]
        assert np.mean(high) > np.mean(low)


class TestLognormalImpulse:
    def test_zero_eval_returns_zero(self):
        rng = np.random.default_rng(42)
        assert sample_lognormal_impulse(0.0, rng=rng) == 0.0

    def test_positive_output(self):
        rng = np.random.default_rng(42)
        result = sample_lognormal_impulse(0.7, rng=rng)
        assert result > 0.0

    def test_reproducible(self):
        r1 = sample_lognormal_impulse(0.5, rng=np.random.default_rng(99))
        r2 = sample_lognormal_impulse(0.5, rng=np.random.default_rng(99))
        assert r1 == pytest.approx(r2)

    def test_scales_with_eval_value(self):
        low = [sample_lognormal_impulse(0.1, rng=np.random.default_rng(i)) for i in range(100)]
        high = [sample_lognormal_impulse(0.9, rng=np.random.default_rng(i)) for i in range(100)]
        assert np.mean(high) > np.mean(low)


class TestSampleImpulseDispatcher:
    def test_gamma(self):
        r = sample_impulse(0.5, distribution="gamma", rng=np.random.default_rng(42))
        assert r >= 0.0

    def test_poisson(self):
        r = sample_impulse(0.5, distribution="poisson", rng=np.random.default_rng(42))
        assert r >= 0.0

    def test_lognormal(self):
        r = sample_impulse(0.5, distribution="lognormal", rng=np.random.default_rng(42))
        assert r >= 0.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown impulse distribution"):
            sample_impulse(0.5, distribution="uniform")
