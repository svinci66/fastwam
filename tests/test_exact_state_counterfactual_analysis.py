import numpy as np

from experiments.libero.analyze_exact_state_counterfactuals import (
    bootstrap_mean_ci,
    spearman_correlation,
)


def test_spearman_tracks_monotonic_quality():
    assert spearman_correlation([0.0, -0.075, -0.15, -0.30], [0.8, 0.6, 0.2, -0.1]) == 1.0
    assert spearman_correlation([0.0, -0.075, -0.15, -0.30], [-0.1, 0.2, 0.6, 0.8]) == -1.0


def test_spearman_uses_average_tie_ranks():
    value = spearman_correlation([0.0, -0.075, -0.15, -0.30], [0.8, 0.8, 0.2, -0.1])
    assert 0.9 < value < 1.0


def test_bootstrap_is_reproducible_and_contains_constant_mean():
    first = bootstrap_mean_ci([2.0] * 10, samples=100, seed=2042)
    second = bootstrap_mean_ci([2.0] * 10, samples=100, seed=2042)
    assert first == second
    np.testing.assert_equal(first["mean"], 2.0)
    np.testing.assert_equal(first["ci95_lower"], 2.0)
    np.testing.assert_equal(first["ci95_upper"], 2.0)
