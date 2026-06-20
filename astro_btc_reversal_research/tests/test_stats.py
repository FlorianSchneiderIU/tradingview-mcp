"""Statistical helper sanity checks."""

import numpy as np

from astro_reversal import stats


def test_lift_and_hit_rate():
    assert stats.hit_rate([1, 1, 0, 0]) == 0.5
    assert stats.lift(0.6, 0.3) == 2.0
    assert np.isnan(stats.lift(0.5, 0.0))


def test_binomial_one_sided():
    # 90/100 successes vs baseline 0.5 -> tiny p; vs 0.95 -> large p.
    assert stats.binomial_test(90, 100, 0.5) < 1e-9
    assert stats.binomial_test(90, 100, 0.95) > 0.5


def test_benjamini_hochberg_monotone():
    p = [0.001, 0.002, 0.2, 0.8]
    reject = stats.benjamini_hochberg(p, alpha=0.05)
    assert reject[0] and reject[1]
    assert not reject[3]


def test_empirical_pvalue_bounds():
    null = np.zeros(99)
    assert stats.empirical_pvalue(1.0, null) < 0.02   # observed beats all null
    assert stats.empirical_pvalue(-1.0, null) > 0.98  # observed below all null


def test_bootstrap_ci_contains_mean():
    data = [1, 0, 1, 0, 1, 1, 0, 1]
    lo, hi = stats.bootstrap_rate_ci(data, n_boot=2000, seed=1)
    assert lo <= np.mean(data) <= hi
