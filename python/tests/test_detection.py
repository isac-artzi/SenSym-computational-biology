"""Tests for celldetect.detection — d', Phi(d'/2), the log-ratio, ROC (W3-W5).

The square-root laws are not checked by "it looks right on a plot": the
scaling tests below assert the exponent to two decimal places.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from celldetect.counting import sample_counts
from celldetect.detection import (
    accuracy_from_d_prime,
    bayes_error_exact,
    combine_d_prime,
    d_prime,
    d_prime_from_auc,
    d_prime_total,
    log_likelihood_ratio,
    normal_cdf,
    posterior_log_odds,
    roc_auc,
    roc_curve,
)


def test_normal_cdf_known_values():
    """Phi(0) = 1/2, Phi(1.96) ~ 0.975, symmetry Phi(-z) = 1 - Phi(z) (W4)."""
    assert math.isclose(float(normal_cdf(0.0)), 0.5, abs_tol=1e-12)
    assert abs(float(normal_cdf(1.959964)) - 0.975) < 1e-5
    for z in (0.3, 1.0, 2.5):
        assert abs(float(normal_cdf(-z)) + float(normal_cdf(z)) - 1.0) < 1e-12


def test_d_prime_known_value():
    """A hand-checkable case (W3): lam = 100 vs 121 gives |21|/sqrt(110.5)."""
    assert math.isclose(float(d_prime(100.0, 121.0)), 21.0 / math.sqrt(110.5),
                        rel_tol=1e-12)


def test_d_prime_is_symmetric_and_non_negative():
    rng = np.random.default_rng(11)
    a = rng.uniform(0.1, 50, size=200)
    b = rng.uniform(0.1, 50, size=200)
    npt.assert_allclose(d_prime(a, b), d_prime(b, a), atol=1e-12)
    assert (np.asarray(d_prime(a, b)) >= 0).all()


def test_d_prime_zero_for_identical_rates_and_for_silent_genes():
    """No difference, no information; and 0/0 must not be NaN (W3)."""
    assert float(d_prime(7.0, 7.0)) == 0.0
    assert float(d_prime(0.0, 0.0)) == 0.0
    assert np.isfinite(np.asarray(d_prime(np.array([0.0, 3.0]),
                                          np.array([0.0, 5.0])))).all()


def test_d_prime_scales_as_sqrt_depth():
    """THE headline law of Aim 1: quadruple the depth, double d' (W3).

    Fit log d' against log depth; the slope must be 1/2 to numerical
    precision, because the relationship is exact, not asymptotic.
    """
    p1, p2 = 0.004, 0.0055
    depths = np.array([100.0, 400.0, 1600.0, 6400.0, 25600.0])
    dps = np.array([float(d_prime(D * p1, D * p2)) for D in depths])
    slope = np.polyfit(np.log(depths), np.log(dps), 1)[0]
    assert abs(slope - 0.5) < 1e-9
    # And the concrete statement: 4x depth is exactly 2x d'.
    assert math.isclose(dps[1] / dps[0], 2.0, rel_tol=1e-12)


def test_combine_d_prime_quadrature():
    """d' values add in squares (W5)."""
    assert math.isclose(combine_d_prime([3.0, 4.0]), 5.0, rel_tol=1e-12)
    assert math.isclose(combine_d_prime([]), 0.0, abs_tol=1e-12)


def test_combine_d_prime_scales_as_sqrt_k():
    """k identical genes give sqrt(k) times the single-gene d' (W5)."""
    single = 0.37
    for k in (1, 4, 9, 100):
        assert math.isclose(combine_d_prime([single] * k),
                            math.sqrt(k) * single, rel_tol=1e-12)


def test_d_prime_total_combines_the_sqrt_laws():
    """d'_total ~ sqrt(depth * k): the full Aim 1 statement (W5)."""
    base = 0.002
    def total(depth, k):
        # Total marker mass held fixed as k grows, as in simulate.marker_profiles.
        lam1 = np.full(k, depth * base * 1.2 / k)
        lam2 = np.full(k, depth * base * 0.8 / k)
        return d_prime_total(lam1, lam2)
    # Doubling depth at fixed k multiplies d' by sqrt(2).
    assert math.isclose(total(2000, 10) / total(1000, 10), math.sqrt(2), rel_tol=1e-9)
    # With total mass fixed, splitting it over more genes is exactly neutral:
    # per-gene d' falls as 1/sqrt(k) and quadrature restores it. This is the
    # subtle point of Week 5 and the reason the chapter insists on the
    # fixed-mass convention.
    assert math.isclose(total(1000, 40), total(1000, 10), rel_tol=1e-9)


def test_accuracy_from_d_prime_endpoints_and_monotonicity():
    """d' = 0 is chance; large d' saturates at 1; always increasing (W4)."""
    assert math.isclose(float(accuracy_from_d_prime(0.0)), 0.5, abs_tol=1e-12)
    assert float(accuracy_from_d_prime(12.0)) > 0.999
    dps = np.linspace(0, 6, 50)
    accs = np.asarray(accuracy_from_d_prime(dps))
    assert (np.diff(accs) > 0).all()


def test_accuracy_from_d_prime_matches_exact_bayes_at_high_rate():
    """The Gaussian approximation is good once the counts are big (W4)."""
    lam1, lam2 = 200.0, 260.0
    approx = float(accuracy_from_d_prime(float(d_prime(lam1, lam2))))
    exact = 1.0 - bayes_error_exact(lam1, lam2)
    assert abs(approx - exact) < 0.01


def test_gaussian_approximation_degrades_at_low_rate():
    """...and is visibly wrong when the counts are small — a Week 4 result,
    not a bug. The exact calculation is the one to trust at low depth."""
    lam1, lam2 = 0.2, 1.0
    approx = float(accuracy_from_d_prime(float(d_prime(lam1, lam2))))
    exact = 1.0 - bayes_error_exact(lam1, lam2)
    assert abs(approx - exact) > 0.02


def test_bayes_error_exact_bounds():
    """Error is 1/2 for identical rates and falls toward 0 as they separate (W4)."""
    assert math.isclose(bayes_error_exact(5.0, 5.0), 0.5, abs_tol=1e-9)
    assert bayes_error_exact(5.0, 50.0) < 1e-3
    errs = [bayes_error_exact(20.0, 20.0 + s) for s in (0, 2, 5, 10, 20)]
    assert all(a >= b - 1e-12 for a, b in zip(errs, errs[1:]))


def test_log_likelihood_ratio_is_linear_in_counts():
    """The log(x!) terms cancel, so the rule is linear — the key step of W5."""
    rng = np.random.default_rng(12)
    lam1 = rng.uniform(1, 10, size=6)
    lam2 = rng.uniform(1, 10, size=6)
    x = rng.poisson(lam1)
    z = rng.poisson(lam2)
    llr = log_likelihood_ratio
    # LLR(x) + LLR(z) - LLR(0) must equal LLR(x + z): additivity in the counts.
    lhs = (llr([x], lam1, lam2)[0] + llr([z], lam1, lam2)[0]
           - llr([np.zeros_like(x)], lam1, lam2)[0])
    rhs = llr([x + z], lam1, lam2)[0]
    assert abs(lhs - rhs) < 1e-9


def test_log_likelihood_ratio_sign_favours_the_true_type():
    """On average, the LLR is positive for type-1 cells and negative for type 2 (W5)."""
    rng = np.random.default_rng(13)
    lam1 = np.full(20, 6.0)
    lam2 = np.full(20, 3.0)
    X1 = rng.poisson(lam1, size=(400, 20))
    X2 = rng.poisson(lam2, size=(400, 20))
    assert log_likelihood_ratio(X1, lam1, lam2).mean() > 0
    assert log_likelihood_ratio(X2, lam1, lam2).mean() < 0


def test_log_likelihood_ratio_shape():
    rng = np.random.default_rng(14)
    X = rng.poisson(4.0, size=(37, 9))
    out = np.asarray(log_likelihood_ratio(X, np.full(9, 4.0), np.full(9, 5.0)))
    assert out.shape == (37,)


def test_roc_curve_perfect_and_chance():
    """A perfect separator has AUC 1; identical scores give AUC 1/2 (W4)."""
    labels = np.array([0] * 50 + [1] * 50)
    perfect = np.concatenate([np.zeros(50), np.ones(50)])
    assert math.isclose(roc_auc(perfect, labels), 1.0, abs_tol=1e-9)
    assert math.isclose(roc_auc(np.zeros(100), labels), 0.5, abs_tol=1e-9)


def test_roc_curve_starts_at_origin_and_is_monotone():
    rng = np.random.default_rng(15)
    labels = rng.integers(0, 2, size=300)
    scores = rng.normal(size=300) + labels
    fpr, tpr = roc_curve(scores, labels)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert math.isclose(fpr[-1], 1.0, abs_tol=1e-12)
    assert math.isclose(tpr[-1], 1.0, abs_tol=1e-12)
    assert (np.diff(fpr) >= -1e-12).all() and (np.diff(tpr) >= -1e-12).all()


def test_roc_curve_needs_both_classes():
    with pytest.raises(ValueError):
        roc_curve(np.array([1.0, 2.0]), np.array([1, 1]))


def test_auc_recovers_d_prime():
    """AUC = Phi(d'/sqrt(2)), so d' can be read off measured performance (W4)."""
    rng = np.random.default_rng(16)
    true_dp = 1.6
    n = 40_000
    scores = np.concatenate([rng.normal(0.0, 1.0, n),
                             rng.normal(true_dp, 1.0, n)])
    labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    assert abs(d_prime_from_auc(roc_auc(scores, labels)) - true_dp) < 0.05


def test_measured_accuracy_matches_the_prediction():
    """End-to-end Aim 1 check: simulate two Poisson genes, classify with the
    optimal rule, and compare the measured accuracy to Phi(d'/2) (W5).

    This single test is the project's hypothesis in miniature.
    """
    rng = np.random.default_rng(17)
    lam1 = np.full(8, 12.0)
    lam2 = np.full(8, 9.0)
    n = 20_000
    X1 = rng.poisson(lam1, size=(n, 8))
    X2 = rng.poisson(lam2, size=(n, 8))
    correct = (log_likelihood_ratio(X1, lam1, lam2) > 0).sum() \
        + (log_likelihood_ratio(X2, lam1, lam2) <= 0).sum()
    measured = correct / (2 * n)
    predicted = float(accuracy_from_d_prime(d_prime_total(lam1, lam2)))
    assert abs(measured - predicted) < 0.02


def test_posterior_log_odds_provided():
    """[PROVIDED] multi-class posteriors normalize and pick the right type."""
    rng = np.random.default_rng(18)
    rates = np.array([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0]])
    X = rng.poisson(rates[1], size=(200, 3))
    lp = posterior_log_odds(X, rates)
    npt.assert_allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-9)
    assert (lp.argmax(axis=1) == 1).mean() > 0.9
