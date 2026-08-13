"""Tests for celldetect.simulate — the synthetic-cell laboratory (W7, W9)."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from celldetect.counting import fano_factor
from celldetect.detection import accuracy_from_d_prime, d_prime_total
from celldetect.simulate import (
    apply_dropout,
    expected_rates,
    marker_profiles,
    multi_type_profiles,
    simulate_cells,
    simulate_dataset,
)


def test_marker_profiles_mass_and_fold_change():
    """Each profile carries the requested total mass at the requested ratio (W7)."""
    for k in (1, 5, 50):
        p1, p2 = marker_profiles(k, fold_change=1.5, marker_mass=0.02)
        assert p1.shape == (k,) and p2.shape == (k,)
        # Convention: the two types put an AVERAGE of marker_mass on the
        # marker set. They cannot each put exactly marker_mass and also
        # differ by a fold change — see the docstring, and exercise 7.2.
        assert math.isclose(0.5 * (p1.sum() + p2.sum()), 0.02, rel_tol=1e-12)
        assert math.isclose(0.5 * (p1[0] + p2[0]), 0.02 / k, rel_tol=1e-12)
        npt.assert_allclose(p1 / p2, 1.5, rtol=1e-12)


def test_marker_profiles_fold_change_one_is_indistinguishable():
    p1, p2 = marker_profiles(10, fold_change=1.0)
    npt.assert_allclose(p1, p2, atol=1e-15)
    assert math.isclose(d_prime_total(1000 * p1, 1000 * p2), 0.0, abs_tol=1e-12)


def test_marker_profiles_rejects_bad_arguments():
    with pytest.raises(ValueError):
        marker_profiles(0, 2.0)
    with pytest.raises(ValueError):
        marker_profiles(5, -1.0)


def test_simulate_cells_shape_mean_and_variance():
    """Counts are Poisson(depth * p): mean and variance both equal that (W7)."""
    rng = np.random.default_rng(21)
    p = np.full(4, 0.005)
    X = simulate_cells(p, depth=2000.0, n_cells=20_000, rng=rng)
    assert X.shape == (20_000, 4)
    assert abs(X.mean() - 10.0) < 0.1
    assert abs(fano_factor(X[:, 0]) - 1.0) < 0.06


def test_simulate_cells_is_seed_reproducible():
    p = np.full(3, 0.01)
    a = simulate_cells(p, 500.0, 50, np.random.default_rng(22))
    b = simulate_cells(p, 500.0, 50, np.random.default_rng(22))
    npt.assert_array_equal(a, b)


def test_simulate_cells_zero_depth_gives_zero_counts():
    X = simulate_cells(np.full(5, 0.01), 0.0, 10, np.random.default_rng(23))
    npt.assert_array_equal(X, np.zeros((10, 5), dtype=X.dtype))


def test_dropout_equals_reduced_depth():
    """Dropout q at depth D is the same distribution as depth (1-q)D (W9).

    The chapter proves it from the thinning theorem; this checks it, and it
    is the reason the dropout arm of the bead experiment tests the SAME law
    rather than a new one.
    """
    rng = np.random.default_rng(24)
    p = np.full(6, 0.004)
    dropped = simulate_cells(p, depth=3000.0, n_cells=30_000, rng=rng, dropout=0.4)
    shallow = simulate_cells(p, depth=1800.0, n_cells=30_000, rng=rng)
    assert abs(dropped.mean() - shallow.mean()) < 0.05
    assert abs(dropped.var() - shallow.var()) < 0.15


def test_apply_dropout_never_increases_counts():
    rng = np.random.default_rng(25)
    X = simulate_cells(np.full(5, 0.01), 1000.0, 200, rng)
    Y = apply_dropout(X, 0.3, rng)
    assert (Y <= X).all()
    assert abs(Y.mean() / X.mean() - 0.7) < 0.03


def test_simulate_dataset_labels_and_balance():
    rng = np.random.default_rng(26)
    p1, p2 = marker_profiles(8, 2.0)
    X, y = simulate_dataset([p1, p2], depth=1000.0, n_cells_per_type=150, rng=rng)
    assert X.shape == (300, 8)
    assert y.shape == (300,)
    assert (y == 0).sum() == 150 and (y == 1).sum() == 150
    # Type 0 should carry more counts, since p1 > p2 gene by gene.
    assert X[y == 0].mean() > X[y == 1].mean()


def test_simulate_dataset_handles_more_than_two_types():
    rng = np.random.default_rng(27)
    P = multi_type_profiles(5, 50, fold_change=3.0, rng=rng)
    X, y = simulate_dataset(P, depth=800.0, n_cells_per_type=40, rng=rng)
    assert X.shape == (200, 50)
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}


def test_expected_rates_provided():
    """[PROVIDED] expected_rates is depth * (1-dropout) * profile (W7)."""
    p1, p2 = marker_profiles(4, 2.0, marker_mass=0.04)
    R = expected_rates([p1, p2], depth=1000.0, dropout=0.5)
    assert R.shape == (2, 4)
    npt.assert_allclose(R[0], 500.0 * p1, rtol=1e-12)


def test_simulation_accuracy_matches_theory():
    """The whole of Aim 2's first claim, in one test (W9).

    Simulate, classify optimally, measure — and compare to Phi(d'/2)
    computed from the true rates. Agreement here is what licenses the
    chapters to use the theory curve as a reference line everywhere else.
    """
    from celldetect.classify import accuracy, naive_bayes_predict
    rng = np.random.default_rng(28)
    p1, p2 = marker_profiles(12, fold_change=1.4, marker_mass=0.03)
    depth = 4000.0
    X, y = simulate_dataset([p1, p2], depth, 6000, rng)
    lam1, lam2 = depth * p1, depth * p2
    # naive_bayes_predict returns 1 when type-1 is favoured; our labels call
    # type 1 "class 0", so compare against (1 - y).
    measured = accuracy(1 - y, naive_bayes_predict(X, lam1, lam2))
    predicted = float(accuracy_from_d_prime(d_prime_total(lam1, lam2)))
    assert abs(measured - predicted) < 0.02


def test_multi_type_profiles_provided():
    P = multi_type_profiles(4, 40, 5.0, marker_mass=0.05)
    assert P.shape == (4, 40)
    # multi_type_profiles DOES renormalize each type to exactly marker_mass,
    # because each type over-expresses a disjoint block and the blocks can
    # be traded off against each other. The two-type case cannot.
    npt.assert_allclose(P.sum(axis=1), 0.05, rtol=1e-12)
    # Each type is strongest on its own block.
    for t in range(4):
        assert P[t, t * 10:(t + 1) * 10].mean() > P[t, :10].mean() or t == 0
    with pytest.raises(ValueError):
        multi_type_profiles(3, 10, 2.0)
