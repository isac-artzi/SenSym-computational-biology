"""Classifiers, written from scratch. [SOLUTIONS]

Reference implementation of `celldetect.classify`.
Used by: Week 8 (and every accuracy measurement afterwards).

Why from scratch: the project's claim is that accuracy is limited by
*counting noise*, not by the classifier. That claim is only testable if you
know exactly what the classifier does. Three are enough:

  * naive Bayes with the true rates — the optimal rule, the ceiling;
  * logistic regression — the same linear decision rule, but with the
    coefficients LEARNED from data rather than given;
  * k-nearest neighbours — a non-linear rule that needs no model at all,
    and pays for it in high dimensions.

Nothing here is regularized or tuned. A gap between naive Bayes and logistic
regression is a finite-sample effect and is a result, not a bug to fix.
"""

from typing import Tuple

import numpy as np

from .detection import log_likelihood_ratio


def train_test_split(X, y, test_fraction: float, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into (X_train, X_test, y_train, y_test).

    Every accuracy in this project is a TEST accuracy. Reporting training
    accuracy is the single most common way student projects overstate a
    result; the split is not a formality.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    n = X.shape[0]
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true, y_pred) -> float:
    """Fraction of predictions that are correct."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(y_true == y_pred))


def naive_bayes_predict(X, lam1, lam2) -> np.ndarray:
    """Predict with the optimal Poisson rule: label 1 iff LLR(x) > 0.

    Requires the TRUE rates, so it is only available in simulation (or with
    estimated rates, which is Week 11's whole difficulty). Its accuracy is
    the ceiling every other classifier is measured against.
    """
    return (log_likelihood_ratio(X, lam1, lam2) > 0).astype(int)


class LogisticRegression:
    """Binary logistic regression by full-batch gradient descent.

    The model: P(y=1 | x) = sigma(w.x + b), sigma(z) = 1/(1+e^-z).
    The loss: mean binary cross-entropy. The gradient of that loss with
    respect to w is X^T (p - y) / n — one of the tidiest results in machine
    learning, and Week 8's exercise 8.3.

    Note the decision boundary is LINEAR in the counts, exactly like the
    naive-Bayes rule of Week 5. Logistic regression is not a more powerful
    model here; it is the same model with the coefficients estimated.
    """

    def __init__(self, lr: float = 0.5, n_iter: int = 800,
                 standardize: bool = True):
        self.lr = lr
        self.n_iter = n_iter
        self.standardize = standardize
        self.w = None
        self.b = 0.0
        self._mu = None
        self._sd = None

    def _prep(self, X, fit: bool):
        X = np.asarray(X, dtype=float)
        if not self.standardize:
            return X
        if fit:
            self._mu = X.mean(axis=0)
            # A gene with zero variance in the training set carries no
            # information; dividing by its sd would produce NaN, so pin it
            # to 1 and let the coefficient go to zero on its own.
            sd = X.std(axis=0)
            self._sd = np.where(sd > 0, sd, 1.0)
        return (X - self._mu) / self._sd

    def fit(self, X, y):
        """Fit by gradient descent. Returns self."""
        Z = self._prep(X, fit=True)
        y = np.asarray(y, dtype=float)
        n, d = Z.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = _sigmoid(Z @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (Z.T @ err) / n
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("call fit() before predict_proba()")
        Z = self._prep(X, fit=False)
        return _sigmoid(Z @ self.w + self.b)

    def decision_function(self, X) -> np.ndarray:
        """The raw score w.x + b — the right input to an ROC curve."""
        if self.w is None:
            raise RuntimeError("call fit() before decision_function()")
        Z = self._prep(X, fit=False)
        return Z @ self.w + self.b

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)


def _sigmoid(z):
    """[PROVIDED] Numerically stable logistic function.

    exp(z) overflows for z > ~710. The two-branch form below is algebraically
    identical and never exponentiates a positive number.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def knn_predict(X_train, y_train, X_query, k: int = 5) -> np.ndarray:
    """k-nearest-neighbour classification by Euclidean distance, majority vote.

    Ties in the vote are broken toward class 0 by the strict `>` below; with
    odd k and two classes ties cannot occur, which is why the chapters use
    odd k throughout.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_query = np.asarray(X_query, dtype=float)
    if k < 1 or k > X_train.shape[0]:
        raise ValueError(f"k must be in 1..{X_train.shape[0]}, got {k}")
    # (a-b)^2 = a^2 - 2ab + b^2, computed for all pairs at once. Faster than
    # a Python loop by ~100x on the sizes used here, and it is worth being
    # able to read it: each term is a matrix of one piece of the expansion.
    d2 = (np.sum(X_query ** 2, axis=1)[:, None]
          - 2.0 * X_query @ X_train.T
          + np.sum(X_train ** 2, axis=1)[None, :])
    nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    votes = y_train[nn].sum(axis=1)
    return (votes * 2 > k).astype(int)


def cross_validated_accuracy(X, y, classifier_factory, n_folds: int,
                             rng: np.random.Generator) -> Tuple[float, float]:
    """[PROVIDED] Mean and standard error of k-fold accuracy.

    classifier_factory() must return an object with .fit(X, y) and
    .predict(X). Used from Week 9 on, where a single split is too noisy to
    resolve the theory-vs-measurement gap being argued about.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    folds = np.array_split(rng.permutation(n), n_folds)
    accs = []
    for i in range(n_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        clf = classifier_factory().fit(X[train_idx], y[train_idx])
        accs.append(accuracy(y[test_idx], clf.predict(X[test_idx])))
    accs = np.asarray(accs, dtype=float)
    return float(accs.mean()), float(accs.std(ddof=1) / np.sqrt(n_folds))
