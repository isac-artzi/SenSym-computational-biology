"""Tests for celldetect.classify — split, logistic regression, kNN (W8)."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from celldetect.classify import (
    LogisticRegression,
    accuracy,
    cross_validated_accuracy,
    knn_predict,
    naive_bayes_predict,
    train_test_split,
)
from celldetect.detection import accuracy_from_d_prime, d_prime_total
from celldetect.simulate import marker_profiles, simulate_dataset


def test_train_test_split_partitions_exactly():
    """No row is lost, none is duplicated, and the split is the requested size (W8)."""
    rng = np.random.default_rng(31)
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100) % 2
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, rng)
    assert Xte.shape[0] == 25 and Xtr.shape[0] == 75
    assert ytr.shape[0] == 75 and yte.shape[0] == 25
    all_rows = np.vstack([Xtr, Xte])
    npt.assert_array_equal(np.sort(all_rows[:, 0]), np.sort(X[:, 0]))


def test_train_test_split_labels_travel_with_rows():
    """The commonest catastrophic bug: shuffling X and y independently (W8)."""
    rng = np.random.default_rng(32)
    X = np.arange(300).reshape(100, 3).astype(float)
    y = X[:, 0].astype(int)            # label is recoverable from the row
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng)
    npt.assert_array_equal(Xtr[:, 0].astype(int), ytr)
    npt.assert_array_equal(Xte[:, 0].astype(int), yte)


def test_train_test_split_is_seed_reproducible():
    X = np.arange(80).reshape(40, 2)
    y = np.arange(40) % 2
    a = train_test_split(X, y, 0.5, np.random.default_rng(33))
    b = train_test_split(X, y, 0.5, np.random.default_rng(33))
    for u, v in zip(a, b):
        npt.assert_array_equal(u, v)


def test_train_test_split_rejects_bad_fraction():
    rng = np.random.default_rng(34)
    with pytest.raises(ValueError):
        train_test_split(np.zeros((10, 2)), np.zeros(10), 0.0, rng)
    with pytest.raises(ValueError):
        train_test_split(np.zeros((10, 2)), np.zeros(10), 1.0, rng)


def test_accuracy_basic():
    assert math.isclose(accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)
    assert math.isclose(accuracy([1, 1], [1, 1]), 1.0)
    with pytest.raises(ValueError):
        accuracy([1, 0], [1, 0, 1])


def test_naive_bayes_predict_beats_chance_and_hits_the_ceiling():
    """The optimal rule attains the predicted accuracy (W8)."""
    rng = np.random.default_rng(35)
    p1, p2 = marker_profiles(10, 1.5, 0.03)
    depth = 3000.0
    X, y = simulate_dataset([p1, p2], depth, 4000, rng)
    pred = naive_bayes_predict(X, depth * p1, depth * p2)
    measured = accuracy(1 - y, pred)
    predicted = float(accuracy_from_d_prime(d_prime_total(depth * p1, depth * p2)))
    assert measured > 0.6
    assert abs(measured - predicted) < 0.02


def test_logistic_regression_separable_case():
    """On cleanly separated blobs, logistic regression is near-perfect (W8)."""
    rng = np.random.default_rng(36)
    X = np.vstack([rng.normal(-2.0, 0.5, size=(200, 2)),
                   rng.normal(2.0, 0.5, size=(200, 2))])
    y = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    clf = LogisticRegression(lr=0.5, n_iter=800).fit(X, y)
    assert accuracy(y, clf.predict(X)) > 0.98


def test_logistic_regression_probabilities_are_calibrated_in_range():
    rng = np.random.default_rng(37)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (300,)
    assert (p >= 0).all() and (p <= 1).all()
    # The learned direction should point along the informative coordinate.
    assert abs(clf.w[0]) > 2 * np.abs(clf.w[1:]).max()


def test_logistic_regression_is_stable_at_extreme_scores():
    """A naive sigmoid overflows here; a stable one does not (W8)."""
    clf = LogisticRegression(standardize=False)
    clf.w = np.array([1000.0])
    clf.b = 0.0
    p = clf.predict_proba(np.array([[-5.0], [5.0]]))
    assert np.isfinite(p).all()
    assert p[0] < 1e-12 and p[1] > 1 - 1e-12


def test_logistic_regression_requires_fit_first():
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba(np.zeros((2, 2)))


def test_logistic_regression_approaches_the_bayes_ceiling():
    """With enough training cells, the learned rule nearly matches the optimal
    one — the point of Week 8: the classifier is not the bottleneck."""
    rng = np.random.default_rng(38)
    p1, p2 = marker_profiles(6, 1.6, 0.03)
    depth = 3000.0
    X, y = simulate_dataset([p1, p2], depth, 3000, rng)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng)
    clf = LogisticRegression(lr=0.5, n_iter=1500).fit(Xtr, 1 - ytr)
    learned = accuracy(1 - yte, clf.predict(Xte))
    optimal = accuracy(1 - yte, naive_bayes_predict(Xte, depth * p1, depth * p2))
    assert learned > optimal - 0.03


def test_knn_predict_shapes_and_trivial_case():
    Xtr = np.array([[0.0], [0.1], [5.0], [5.1]])
    ytr = np.array([0, 0, 1, 1])
    pred = knn_predict(Xtr, ytr, np.array([[0.05], [5.05]]), k=1)
    npt.assert_array_equal(pred, np.array([0, 1]))
    assert knn_predict(Xtr, ytr, np.array([[0.0]]), k=3).shape == (1,)


def test_knn_predict_rejects_bad_k():
    Xtr = np.zeros((4, 2))
    ytr = np.zeros(4, int)
    with pytest.raises(ValueError):
        knn_predict(Xtr, ytr, np.zeros((1, 2)), k=0)
    with pytest.raises(ValueError):
        knn_predict(Xtr, ytr, np.zeros((1, 2)), k=5)


def test_knn_is_beaten_by_the_linear_rule_in_high_dimensions():
    """Week 8's punchline: kNN degrades as the number of genes grows, because
    Euclidean distance stops meaning anything when every coordinate is noise.
    This is a result to report, not a reason to tune kNN."""
    rng = np.random.default_rng(39)
    p1, p2 = marker_profiles(60, 1.3, 0.03)
    depth = 3000.0
    X, y = simulate_dataset([p1, p2], depth, 500, rng)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng)
    knn_acc = accuracy(1 - yte, knn_predict(Xtr, 1 - ytr, Xte, k=15))
    nb_acc = accuracy(1 - yte, naive_bayes_predict(Xte, depth * p1, depth * p2))
    assert nb_acc > knn_acc


def test_cross_validated_accuracy_provided():
    """[PROVIDED] k-fold CV returns a mean and a standard error."""
    rng = np.random.default_rng(40)
    X = np.vstack([rng.normal(-1.5, 1.0, size=(150, 3)),
                   rng.normal(1.5, 1.0, size=(150, 3))])
    y = np.concatenate([np.zeros(150, int), np.ones(150, int)])
    mean, se = cross_validated_accuracy(X, y, lambda: LogisticRegression(), 5, rng)
    assert 0.85 < mean <= 1.0
    assert 0.0 <= se < 0.1
