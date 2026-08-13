"""Tests for celldetect.stats — Wilson intervals, bootstrap, permutation (W13)."""

import math

import numpy as np
import pytest

from celldetect.stats import (
    bonferroni,
    bootstrap_ci,
    permutation_test,
    standard_error_of_proportion,
    wilson_interval,
)


def test_wilson_interval_contains_the_estimate():
    for k, n in [(5, 20), (17, 40), (350, 500)]:
        lo, hi = wilson_interval(k, n)
        assert lo < k / n < hi


def test_wilson_interval_stays_inside_zero_one():
    """The naive interval escapes [0,1] here; Wilson must not (W13)."""
    for k, n in [(0, 10), (10, 10), (1, 8), (19, 20)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_has_width_even_at_the_extremes():
    """0 out of 20 does NOT mean 'the probability is exactly 0' (W13)."""
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and hi > 0.1
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and lo < 0.9


def test_wilson_interval_narrows_as_the_square_root_of_n():
    """Four times the trials, half the width — the SAME square-root law that
    the project is about, now applied to its own error bars (W13)."""
    w1 = np.diff(wilson_interval(50, 100))[0]
    w2 = np.diff(wilson_interval(200, 400))[0]
    assert 1.9 < w1 / w2 < 2.1


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_bootstrap_ci_covers_the_true_mean():
    rng = np.random.default_rng(61)
    data = rng.normal(5.0, 2.0, size=400)
    lo, hi = bootstrap_ci(data, np.mean, 800, rng)
    assert lo < 5.0 < hi
    # ~4 standard errors wide at 95%: 2*1.96*2/sqrt(400) = 0.392
    assert 0.2 < hi - lo < 0.7


def test_bootstrap_ci_works_for_a_nonlinear_statistic():
    """The point of the bootstrap: it needs no formula for the statistic."""
    rng = np.random.default_rng(62)
    data = rng.normal(0.0, 3.0, size=500)
    lo, hi = bootstrap_ci(data, np.median, 600, rng)
    assert lo < 0.0 < hi


def test_bootstrap_ci_rejects_empty_sample():
    with pytest.raises(ValueError):
        bootstrap_ci([], np.mean, 10, np.random.default_rng(0))


def test_permutation_test_finds_a_real_difference():
    rng = np.random.default_rng(63)
    a = rng.normal(0.0, 1.0, size=120)
    b = rng.normal(1.2, 1.0, size=120)
    assert permutation_test(a, b, 500, rng) < 0.01


def test_permutation_test_is_not_fooled_by_noise():
    rng = np.random.default_rng(64)
    a = rng.normal(0.0, 1.0, size=300)
    b = rng.normal(0.0, 1.0, size=300)
    assert permutation_test(a, b, 500, rng) > 0.05


def test_permutation_test_never_returns_exactly_zero():
    """With 100 shuffles you cannot resolve p below 1/101 — and must not
    claim to (W13)."""
    rng = np.random.default_rng(65)
    a = np.full(30, 0.0)
    b = np.full(30, 100.0)
    p = permutation_test(a, b, 100, rng)
    assert p > 0.0
    assert math.isclose(p, 1 / 101, rel_tol=1e-9)


def test_provided_helpers():
    assert math.isclose(standard_error_of_proportion(50, 100),
                        math.sqrt(0.25 / 100), rel_tol=1e-12)
    assert math.isclose(bonferroni(0.05, 8), 0.00625, rel_tol=1e-12)
    with pytest.raises(ValueError):
        bonferroni(0.05, 0)
