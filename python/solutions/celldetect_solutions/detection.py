"""Detection theory: turning counting noise into an error probability. [SOLUTIONS]

Reference implementation of `celldetect.detection`.
Used by: Weeks 3-5 (Aim 1), and as the prediction curve everywhere after.
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.special import erf, logsumexp

from .counting import poisson_logpmf


def normal_cdf(z) -> np.ndarray:
    """Phi(z): the probability a standard normal falls below z.

    Phi(z) = (1/2)(1 + erf(z / sqrt(2))) — the identity is Week 4's
    exercise 4.2; erf is just the Gaussian integral with a different
    normalization, so this is bookkeeping, not new mathematics.
    """
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def d_prime(lam1, lam2) -> np.ndarray:
    """Per-gene separation index d' = |lam1 - lam2| / sqrt((lam1+lam2)/2).

    THE central quantity of the project (Week 3). Numerator: how far apart
    the two expected counts are. Denominator: the standard deviation of the
    noise at the pooled mean — and it is sqrt(lam) *because* a Poisson
    variable's variance equals its mean. That single coincidence is what
    makes the square-root law appear.

    Scaling in depth: writing lam_i = depth * p_i gives
        d' = depth|p1-p2| / sqrt(depth * pbar) = sqrt(depth) * |p1-p2|/sqrt(pbar),
    i.e. d' is proportional to sqrt(depth). Four times the reads, twice the
    separation. Accepts scalars or arrays (elementwise over genes).
    """
    lam1 = np.asarray(lam1, dtype=float)
    lam2 = np.asarray(lam2, dtype=float)
    lbar = 0.5 * (lam1 + lam2)
    # A gene that is silent in both types carries no information; define its
    # d' as 0 rather than 0/0. np.where alone would still evaluate the
    # division, so guard the denominator first.
    safe = np.where(lbar > 0, lbar, 1.0)
    return np.where(lbar > 0, np.abs(lam1 - lam2) / np.sqrt(safe), 0.0)


def combine_d_prime(dprimes: Sequence[float]) -> float:
    """Combine independent genes: d'_total = sqrt(sum_g d'_g^2).

    Week 5. Independent evidence adds in *squares* — the log-likelihood
    ratio is a sum over genes, so its mean and variance both add, and d'
    (a mean over a standard deviation) therefore adds in quadrature. For k
    equally informative genes this gives d'_total = sqrt(k) * d'_single:
    the square-root law in the number of genes, with the same exponent as
    the one in depth. Hence d' ~ sqrt(depth * k).
    """
    d = np.asarray(dprimes, dtype=float)
    return float(np.sqrt(np.sum(d ** 2)))


def d_prime_total(lam1, lam2) -> float:
    """Total separation over all genes: combine_d_prime(d_prime(lam1, lam2))."""
    return combine_d_prime(d_prime(lam1, lam2))


def accuracy_from_d_prime(dp) -> np.ndarray:
    """Predicted accuracy of the optimal rule with equal priors: Phi(d'/2).

    Week 4. Under the Gaussian approximation the decision statistic is
    normal with means +/- d'/2 and unit variance; the optimal threshold sits
    at 0, so the error probability is Phi(-d'/2) and the accuracy Phi(d'/2).
    This is the closed-form prediction Aim 1 exists to produce; every later
    figure plots measurements against this curve.
    """
    return normal_cdf(np.asarray(dp, dtype=float) / 2.0)


def log_likelihood_ratio(X, lam1, lam2) -> np.ndarray:
    """Poisson naive-Bayes log-likelihood ratio, one value per cell.

    For a cell with counts x over genes g:

        LLR(x) = sum_g [ x_g * log(lam1_g / lam2_g) - (lam1_g - lam2_g) ]

    Derivation (Week 5): write the log Poisson pmf for each hypothesis and
    subtract; the log(x_g!) terms are identical under both hypotheses and
    cancel, which is why the rule is *linear* in the counts. Positive LLR
    means type 1 is more likely.

    Parameters
    ----------
    X    : (n_cells, n_genes) count matrix.
    lam1, lam2 : (n_genes,) expected counts under each type.
    """
    X = np.asarray(X, dtype=float)
    lam1 = np.asarray(lam1, dtype=float)
    lam2 = np.asarray(lam2, dtype=float)
    # Guard against a zero rate: a gene with lam = 0 under one type makes the
    # log ratio infinite. Real data always has a pseudocount for exactly this
    # reason; we use the smallest positive rate present, floored at 1e-12.
    eps = 1e-12
    a = np.maximum(lam1, eps)
    b = np.maximum(lam2, eps)
    return X @ np.log(a / b) - np.sum(a - b)


def bayes_error_exact(lam1: float, lam2: float, max_k: int = None) -> float:
    """Exact single-gene error probability of the optimal rule, by summation.

    No Gaussian approximation: enumerate every count k, decide it in favour
    of whichever Poisson gives it higher probability, and add up the mass
    that lands on the wrong side (with equal priors, halving ties).

    This is the honest yardstick for Week 4's Phi(d'/2) formula: the two
    agree well once both means exceed roughly 10, and visibly disagree at
    low depth — which is the first place the theory bends, and a result in
    its own right.
    """
    if max_k is None:
        # Go far enough out that the neglected tail is below 1e-12: ten
        # standard deviations past the larger mean, with a floor for small
        # means where the Poisson tail is still fat relative to its mean.
        big = max(lam1, lam2)
        max_k = int(big + 10.0 * np.sqrt(big) + 20)
    k = np.arange(max_k + 1)
    p1 = np.exp(poisson_logpmf(k, lam1))
    p2 = np.exp(poisson_logpmf(k, lam2))
    # With equal priors the optimal rule picks the larger pmf; the error is
    # the smaller pmf, averaged over the two equally likely truths.
    return float(0.5 * np.sum(np.minimum(p1, p2)))


def roc_curve(scores, labels) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical ROC: return (fpr, tpr) arrays, sorted by decreasing score.

    Sweep the decision threshold from +inf down to -inf; at each DISTINCT
    observed score, record the false-positive and true-positive rates.

    "Distinct" is the whole subtlety (Week 4, exercise 4.6). A threshold
    cannot separate two cells that received the same score, so tied scores
    must be consumed all at once. Break them apart instead and a classifier
    that assigns every cell the same score comes out with an AUC of 0 or 1
    depending on how the input happened to be ordered — a spectacular and
    entirely fictional result.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    order = np.argsort(-scores, kind="mergesort")   # stable: ties keep input order
    y = labels[order]
    s = scores[order]
    P = int(y.sum())
    N = int(len(y) - P)
    if P == 0 or N == 0:
        raise ValueError("ROC needs at least one example of each class")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    # Keep only the LAST index of each run of equal scores.
    last_of_tie = np.r_[np.diff(s) != 0, True]
    tpr = np.concatenate([[0.0], tp[last_of_tie] / P])
    fpr = np.concatenate([[0.0], fp[last_of_tie] / N])
    return fpr, tpr


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve, by the trapezoid rule on roc_curve output.

    Interpretation worth memorizing: AUC is the probability that a randomly
    chosen positive scores above a randomly chosen negative. Under the
    Gaussian model it equals Phi(d'/sqrt(2)) — a second, threshold-free way
    to read d' off data, and the check used in Week 4's lab.
    """
    fpr, tpr = roc_curve(scores, labels)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def d_prime_from_auc(auc: float) -> float:
    """Invert AUC = Phi(d'/sqrt(2)) to recover d' from measured performance."""
    from scipy.special import erfinv
    auc = float(np.clip(auc, 1e-12, 1 - 1e-12))
    # Phi^{-1}(a) = sqrt(2) * erfinv(2a - 1)
    z = np.sqrt(2.0) * erfinv(2.0 * auc - 1.0)
    return float(np.sqrt(2.0) * z)


def posterior_log_odds(X, rates, priors=None) -> np.ndarray:
    """[PROVIDED] Multi-class Poisson naive-Bayes log posteriors.

    rates: (n_types, n_genes) expected counts. Returns (n_cells, n_types)
    log posteriors, normalized. Used from Week 9 onward for the five-type
    extension; the two-class case reduces to log_likelihood_ratio.
    """
    X = np.asarray(X, dtype=float)
    rates = np.maximum(np.asarray(rates, dtype=float), 1e-12)
    n_types = rates.shape[0]
    if priors is None:
        priors = np.full(n_types, 1.0 / n_types)
    logp = X @ np.log(rates).T - rates.sum(axis=1) + np.log(priors)
    return logp - logsumexp(logp, axis=1, keepdims=True)
