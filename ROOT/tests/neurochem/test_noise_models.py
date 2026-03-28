import numpy as np
from zados.neurochem.stochastic_modulation.noise_models import concentration_scaled_noise


def test_noise_mean_is_zeroish():
    rng = np.random.default_rng(seed=42)
    C = 1.0
    alpha = 0.1

    samples = [concentration_scaled_noise(C, alpha, rng) for _ in range(10000)]
    mean = np.mean(samples)

    assert abs(mean) < 0.01  # Expect near-zero mean for Gaussian noise


def test_noise_scales_with_concentration():
    rng = np.random.default_rng(seed=123)
    low_C = 0.5
    high_C = 2.0
    alpha = 0.2

    low_noise = [concentration_scaled_noise(low_C, alpha, rng) for _ in range(1000)]
    high_noise = [concentration_scaled_noise(high_C, alpha, rng) for _ in range(1000)]

    assert np.std(high_noise) > np.std(low_noise)
