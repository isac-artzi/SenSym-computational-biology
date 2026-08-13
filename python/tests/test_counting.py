"""Tests for celldetect.counting — Poisson counts, the Fano factor, thinning (W1-W2).

These tests import the STUDENT package `celldetect`; they fail with
NotImplementedError until the scaffold functions are implemented — that is the
intended workflow (the tests define "done"). All randomness is seeded.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from celldetect.counting import (
    downsample_counts,
    empirical_moments,
    fano_factor,
    poisson_logpmf,
    poisson_pmf,
    sample_counts,
    total_counts,
)


def test_sample_counts_shape_and_dtype():
    """sample_counts returns the requested number of non-negative integers (W1)."""
    rng = np.random.default_rng(1)
    x = sample_counts(4.0, 500, rng)
    assert x.shape == (500,)
    assert np.issubdtype(np.asarray(x).dtype, np.integer)
    assert (np.asarray(x) >= 0).all()


def test_sample_counts_is_reproducible_and_seed_dependent():
    """Same seed, same numbers. Different seed, different numbers (W1)."""
    a = sample_counts(3.0, 200, np.random.default_rng(7))
    b = sample_counts(3.0, 200, np.random.default_rng(7))
    c = sample_counts(3.0, 200, np.random.default_rng(8))
    npt.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_counts_mean_matches_lambda():
    """The sample mean converges to lambda (W1)."""
    rng = np.random.default_rng(2)
    for lam in (0.5, 5.0, 50.0):
        x = sample_counts(lam, 40_000, rng)
        # Standard error of the mean is sqrt(lam/n); allow 5 of them.
        assert abs(x.mean() - lam) < 5 * math.sqrt(lam / 40_000)


def test_sample_counts_rejects_negative_lambda():
    with pytest.raises(ValueError):
        sample_counts(-1.0, 10, np.random.default_rng(0))


def test_fano_factor_is_one_for_poisson():
    """Variance equals mean: THE fact the whole project rests on (W1)."""
    rng = np.random.default_rng(3)
    for lam in (2.0, 20.0, 200.0):
        x = sample_counts(lam, 60_000, rng)
        assert abs(fano_factor(x) - 1.0) < 0.05


def test_fano_factor_detects_overdispersion():
    """A negative-binomial sample has Fano > 1 — the Week 11 diagnostic (W1)."""
    rng = np.random.default_rng(4)
    lam, phi = 20.0, 0.5
    g = rng.gamma(shape=1 / phi, scale=phi * lam, size=60_000)
    x = rng.poisson(g)
    # Var = lam + phi*lam^2 = 20 + 200 = 220, so Fano ~ 11.
    assert fano_factor(x) > 5.0


def test_fano_factor_rejects_all_zero_sample():
    with pytest.raises(ValueError):
        fano_factor(np.zeros(10))


def test_poisson_pmf_sums_to_one():
    """The pmf is a probability distribution (W2)."""
    for lam in (0.3, 4.0, 40.0):
        k = np.arange(0, int(lam + 12 * math.sqrt(lam) + 40))
        assert abs(poisson_pmf(k, lam).sum() - 1.0) < 1e-9


def test_poisson_pmf_known_values():
    """Hand-checkable values (W2)."""
    assert math.isclose(float(poisson_pmf(0, 1.0)), math.exp(-1.0), rel_tol=1e-12)
    assert math.isclose(float(poisson_pmf(1, 1.0)), math.exp(-1.0), rel_tol=1e-12)
    assert math.isclose(float(poisson_pmf(2, 2.0)), 2 * math.exp(-2.0), rel_tol=1e-12)


def test_poisson_logpmf_survives_large_lambda():
    """Computing in log space is why depth 5000 does not overflow (W2)."""
    lp = poisson_logpmf(np.array([4800, 5000, 5200]), 5000.0)
    assert np.all(np.isfinite(lp))
    assert lp[1] > lp[0] and lp[1] > lp[2]      # the mode sits at the mean


def test_poisson_logpmf_matches_log_of_pmf_where_both_are_safe():
    k = np.arange(0, 30)
    npt.assert_allclose(poisson_logpmf(k, 8.0), np.log(poisson_pmf(k, 8.0)), atol=1e-10)


def test_downsample_counts_thins_by_the_right_factor():
    """Keeping half the molecules halves the mean (W2)."""
    rng = np.random.default_rng(5)
    x = sample_counts(100.0, 20_000, rng)
    y = downsample_counts(x, 0.5, rng)
    assert abs(y.mean() - 50.0) < 1.0
    assert (np.asarray(y) <= np.asarray(x)).all()


def test_downsample_counts_stays_poisson():
    """Thinned Poisson is Poisson — the theorem, checked (W2).

    This is the load-bearing fact of the whole downsampling protocol: if it
    failed, every real-data result in Week 11 would be meaningless.
    """
    rng = np.random.default_rng(6)
    x = sample_counts(80.0, 60_000, rng)
    y = downsample_counts(x, 0.25, rng)
    assert abs(y.mean() - 20.0) < 0.5
    assert abs(fano_factor(y) - 1.0) < 0.05


def test_downsample_counts_endpoints():
    rng = np.random.default_rng(7)
    x = sample_counts(10.0, 100, rng)
    npt.assert_array_equal(downsample_counts(x, 1.0, rng), x)
    npt.assert_array_equal(downsample_counts(x, 0.0, rng), np.zeros_like(x))


def test_downsample_counts_rejects_bad_probability():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError):
        downsample_counts(np.array([1, 2]), 1.5, rng)


def test_downsample_counts_works_on_matrices():
    """The same function thins a whole cells x genes matrix (W2)."""
    rng = np.random.default_rng(9)
    X = rng.poisson(30.0, size=(200, 40))
    Y = downsample_counts(X, 0.5, rng)
    assert Y.shape == X.shape
    assert abs(Y.mean() - 15.0) < 0.5


def test_provided_helpers():
    """The [PROVIDED] helpers behave as the experiments assume."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    npt.assert_array_equal(total_counts(X), np.array([6, 15]))
    m, v = empirical_moments([1.0, 2.0, 3.0])
    assert math.isclose(m, 2.0)
    assert math.isclose(v, 1.0)
