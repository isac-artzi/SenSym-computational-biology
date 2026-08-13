"""Classifiers, written from scratch.

Used by: Week 8 (and every accuracy measurement afterwards).

Build order:
    1. train_test_split      (W8)
    2. accuracy              (W8)
    3. naive_bayes_predict   (W8)
    4. LogisticRegression    (W8)
    5. knn_predict           (W8)

Why from scratch: the project's claim is that accuracy is limited by
*counting noise*, not by the classifier. That claim is only testable if you
know exactly what the classifier does. Three are enough:

  * naive Bayes with the true rates — the optimal rule, the ceiling;
  * logistic regression — the same linear decision rule, but with the
    coefficients LEARNED from data rather than given;
  * k-nearest neighbours — a non-linear rule that needs no model at all,
    and pays for it in high dimensions.

Nothing here is regularized or tuned, on purpose. A gap between naive Bayes
and logistic regression is a finite-sample effect and is a RESULT, not a bug
to fix by adding hyperparameters.
"""

from typing import Tuple

import numpy as np

from .detection import log_likelihood_ratio


def train_test_split(X, y, test_fraction: float, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into (X_train, X_test, y_train, y_test).

    Every accuracy in this project is a TEST accuracy. Reporting training
    accuracy is the single most common way student projects overstate a
    result, and a reviewer will ask. The split is not a formality.

    Raises
    ------
    ValueError if test_fraction is not strictly between 0 and 1.
    """
    # --- YOUR CODE HERE ---
    # 1. Convert X and y with np.asarray; validate test_fraction.
    # 2. perm = rng.permutation(n) — ONE permutation, applied to both X and
    #    y. Shuffling them separately destroys the labels silently; that is
    #    what test_train_test_split_labels_travel_with_rows checks.
    # 3. n_test = max(1, int(round(test_fraction * n))).
    # 4. test_idx, train_idx = perm[:n_test], perm[n_test:].
    # 5. Return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    #    — note the ORDER: X's first, then y's, matching the signature.
    raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")


def accuracy(y_true, y_pred) -> float:
    """Fraction of predictions that are correct.

    Raises
    ------
    ValueError if the shapes differ (which almost always means a split bug).
    """
    # --- YOUR CODE HERE ---
    # 1. Convert both to arrays; raise ValueError if the shapes disagree.
    # 2. Return float(np.mean(y_true == y_pred)).
    raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")


def naive_bayes_predict(X, lam1, lam2) -> np.ndarray:
    """Predict with the optimal Poisson rule: label 1 iff LLR(x) > 0.

    Requires the TRUE rates, so it is only available in simulation (or with
    estimated rates, which is Week 11's whole difficulty). Its accuracy is
    the ceiling every other classifier is measured against — and Week 5
    proved that ceiling equals Phi(d'/2).
    """
    # --- YOUR CODE HERE ---
    # 1. One line: threshold log_likelihood_ratio(X, lam1, lam2) at 0 and
    #    cast the booleans to int with .astype(int).
    raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")


class LogisticRegression:
    """Binary logistic regression by full-batch gradient descent.

    The model:  P(y=1 | x) = sigma(w.x + b),  sigma(z) = 1/(1+e^-z).
    The loss:   mean binary cross-entropy.
    The gradient of that loss with respect to w is  X^T (p - y) / n  — one of
    the tidiest results in machine learning, and Week 8's exercise 8.3. Derive
    it before you code it.

    Note that the decision boundary is LINEAR in the counts, exactly like the
    naive-Bayes rule of Week 5. Logistic regression is not a more powerful
    model here; it is the same model with the coefficients estimated from
    data instead of handed over. That is precisely why comparing them
    isolates the cost of not knowing the rates.
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
        """Centre and scale the columns (fit=True learns mu and sd)."""
        # --- YOUR CODE HERE ---
        # 1. X = np.asarray(X, dtype=float); if not self.standardize, return X.
        # 2. If fit: store self._mu = X.mean(axis=0) and
        #    self._sd = X.std(axis=0), but replace any zero sd with 1.0 —
        #    a gene with no variance carries no information, and dividing by
        #    its sd would fill the matrix with NaN.
        # 3. Return (X - self._mu) / self._sd. On the TEST set you must reuse
        #    the training mu and sd (fit=False) — recomputing them is a
        #    subtle form of leakage.
        raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")

    def fit(self, X, y):
        """Fit by gradient descent. Returns self."""
        # --- YOUR CODE HERE ---
        # 1. Z = self._prep(X, fit=True); y = np.asarray(y, dtype=float).
        # 2. Initialize self.w = np.zeros(d) and self.b = 0.0.
        # 3. Repeat n_iter times:
        #       p   = _sigmoid(Z @ self.w + self.b)
        #       err = p - y
        #       self.w -= self.lr * (Z.T @ err) / n
        #       self.b -= self.lr * err.mean()
        # 4. Return self (so you can write clf = LogisticRegression().fit(X, y)).
        raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")

    def predict_proba(self, X) -> np.ndarray:
        """P(y = 1 | x) for each row. Raises RuntimeError if not yet fitted."""
        # --- YOUR CODE HERE ---
        # 1. Raise RuntimeError if self.w is None — a clear message beats an
        #    obscure TypeError three frames deeper.
        # 2. Return _sigmoid(self._prep(X, fit=False) @ self.w + self.b).
        raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")

    def decision_function(self, X) -> np.ndarray:
        """The raw score w.x + b — the right input to an ROC curve.

        Feeding predict_proba into an ROC gives the identical curve (sigma is
        monotone), but the raw score is what the theory talks about.
        """
        # --- YOUR CODE HERE ---
        # 1. Same guard as predict_proba, then return the linear score
        #    without the sigmoid.
        raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")

    def predict(self, X) -> np.ndarray:
        """Hard 0/1 predictions at the 0.5 threshold."""
        # --- YOUR CODE HERE ---
        # 1. One line from predict_proba.
        raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")


def _sigmoid(z):
    """[PROVIDED] Numerically stable logistic function.

    exp(z) overflows for z > ~710, and a naive 1/(1+exp(-z)) then returns nan
    for confidently-negative scores. The two-branch form below is
    algebraically identical and never exponentiates a positive number. Read
    it — Week 8 asks you to explain why the two branches agree.
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

    Raises
    ------
    ValueError if k is outside 1..n_train.
    """
    # --- YOUR CODE HERE ---
    # 1. Convert the three inputs; validate k.
    # 2. All pairwise squared distances at once, via (a-b)^2 = a^2 - 2ab + b^2:
    #       d2 = (np.sum(X_query**2, axis=1)[:, None]
    #             - 2 * X_query @ X_train.T
    #             + np.sum(X_train**2, axis=1)[None, :])
    #    Write the double loop first if that is clearer, confirm the two
    #    agree on a small example, then keep the fast one.
    # 3. The k smallest per row, unsorted, are enough for a vote:
    #       nn = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
    # 4. votes = y_train[nn].sum(axis=1); return (votes * 2 > k).astype(int).
    #    Use ODD k with two classes and ties cannot happen — which is why
    #    every chapter uses odd k.
    raise NotImplementedError("Implement me! See docs/s/week08.html — and try before peeking at python/solutions/.")


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
