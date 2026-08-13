"""Detection theory: turning counting noise into an error probability.

Used by: Weeks 3-5 (Aim 1), and as the prediction curve everywhere after.

Build order (each function may use the previous ones):
    1. normal_cdf            (W4)
    2. d_prime               (W3)  <- the centre of the whole project
    3. combine_d_prime       (W5)
    4. d_prime_total         (W5)
    5. accuracy_from_d_prime (W4)
    6. bayes_error_exact     (W4)
    7. log_likelihood_ratio  (W5)
    8. roc_curve, roc_auc    (W4)
    9. d_prime_from_auc      (W4)

By the end of Week 5 this module IS Aim 1: a closed-form prediction of
classification accuracy from depth and gene count, with no free parameters.
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.special import erf, erfinv, logsumexp

from .counting import poisson_logpmf


def normal_cdf(z) -> np.ndarray:
    """Phi(z): the probability a standard normal falls below z.

    Phi(z) = (1/2)(1 + erf(z / sqrt(2))) — Week 4, exercise 4.2. `erf` is
    imported at the top of this file; it is the Gaussian integral with a
    different normalization, so this identity is bookkeeping, not new
    mathematics.

    Returns an array (or 0-d array) so it composes with the vectorized
    functions below.
    """
    # --- YOUR CODE HERE ---
    # 1. z = np.asarray(z, dtype=float).
    # 2. Return 0.5 * (1 + erf(z / np.sqrt(2))).
    # Sanity checks to run yourself: Phi(0) = 0.5, Phi(1.96) ~ 0.975,
    # Phi(-z) = 1 - Phi(z).
    raise NotImplementedError("Implement me! See docs/s/week04.html — and try before peeking at python/solutions/.")


def d_prime(lam1, lam2) -> np.ndarray:
    """Per-gene separation index d' = |lam1 - lam2| / sqrt((lam1+lam2)/2).

    THE central quantity of the project (Week 3). Numerator: how far apart
    the two expected counts are. Denominator: the standard deviation of the
    noise at the pooled mean — and it is sqrt(lam) *because* a Poisson
    variable's variance equals its mean. That single coincidence is what
    makes the square-root law appear.

    Write lam_i = depth * p_i and watch the law fall out:
        d' = depth|p1-p2| / sqrt(depth * pbar) = sqrt(depth) * |p1-p2|/sqrt(pbar).
    Four times the reads, twice the separation. That is the hypothesis this
    whole project tests.

    Parameters
    ----------
    lam1, lam2 : floats or arrays of expected counts (elementwise over genes).

    Returns
    -------
    numpy array of d' values, same shape as the inputs, all >= 0.
    """
    # --- YOUR CODE HERE ---
    # 1. Convert both inputs with np.asarray(..., dtype=float).
    # 2. Compute the pooled mean lbar = (lam1 + lam2) / 2.
    # 3. A gene silent in both types has lbar = 0, and 0/0 is NaN — which
    #    would then poison every sum downstream. Define its d' to be 0:
    #       safe = np.where(lbar > 0, lbar, 1.0)      # a harmless denominator
    #       return np.where(lbar > 0, |lam1-lam2| / sqrt(safe), 0.0)
    #    Note WHY the two-step form is needed: np.where evaluates BOTH
    #    branches, so dividing by lbar directly still emits the warning and
    #    still produces the NaN before np.where discards it.
    raise NotImplementedError("Implement me! See docs/s/week03.html — and try before peeking at python/solutions/.")


def combine_d_prime(dprimes: Sequence[float]) -> float:
    """Combine independent genes: d'_total = sqrt(sum_g d'_g^2).

    Week 5. Independent evidence adds in *squares*: the log-likelihood ratio
    is a sum over genes, so its mean and its variance both add, and d' (a
    mean divided by a standard deviation) therefore adds in quadrature.

    For k equally informative genes this gives d'_total = sqrt(k) * d'_single
    — the same exponent as the depth law, from a completely different
    argument. Together: d' ~ sqrt(depth * k).
    """
    # --- YOUR CODE HERE ---
    # 1. d = np.asarray(dprimes, dtype=float).
    # 2. Return float(np.sqrt(np.sum(d ** 2))).
    #    (Check: [3, 4] must give exactly 5. If your answer is 7 you added
    #     the d' values instead of their squares — the single most common
    #     error in Week 5.)
    raise NotImplementedError("Implement me! See docs/s/week05.html — and try before peeking at python/solutions/.")


def d_prime_total(lam1, lam2) -> float:
    """Total separation over all genes: combine_d_prime(d_prime(lam1, lam2))."""
    # --- YOUR CODE HERE ---
    # 1. One line, composing the two functions above.
    raise NotImplementedError("Implement me! See docs/s/week05.html — and try before peeking at python/solutions/.")


def accuracy_from_d_prime(dp) -> np.ndarray:
    """Predicted accuracy of the optimal rule with equal priors: Phi(d'/2).

    Week 4, and the closed-form prediction that Aim 1 exists to produce.
    Under the Gaussian approximation the decision statistic is normal with
    means +/- d'/2 and unit variance; the optimal threshold sits at 0, so the
    error probability is Phi(-d'/2) and the accuracy Phi(d'/2).

    Every later figure in the project plots a measurement against this curve.
    """
    # --- YOUR CODE HERE ---
    # 1. Return normal_cdf(np.asarray(dp, dtype=float) / 2).
    #    Check the endpoints yourself: d' = 0 gives 0.5 (chance, as it must),
    #    and large d' saturates at 1.
    raise NotImplementedError("Implement me! See docs/s/week04.html — and try before peeking at python/solutions/.")


def bayes_error_exact(lam1: float, lam2: float, max_k: int = None) -> float:
    """Exact single-gene error probability of the optimal rule, by summation.

    No Gaussian approximation. Enumerate every possible count k, decide it in
    favour of whichever Poisson gives it the higher probability, and add up
    the mass that lands on the wrong side. With equal priors the error is
    (1/2) * sum_k min(P1(k), P2(k)).

    This is the honest yardstick for Week 4's Phi(d'/2) formula: the two
    agree well once both means exceed roughly 10, and visibly disagree at low
    depth — which is the first place the theory bends, and a result in its
    own right rather than a bug.

    Parameters
    ----------
    lam1, lam2 : the two rates.
    max_k      : where to truncate the sum. None picks a safe default.
    """
    # --- YOUR CODE HERE ---
    # 1. If max_k is None, choose it so the neglected tail is negligible:
    #    ten standard deviations past the larger mean, plus a constant floor
    #    of ~20 for the small-mean case where the tail is fat relative to
    #    the mean:  int(big + 10*sqrt(big) + 20)  with big = max(lam1, lam2).
    # 2. k = np.arange(max_k + 1).
    # 3. p1 = np.exp(poisson_logpmf(k, lam1)), same for p2.
    # 4. Return float(0.5 * np.sum(np.minimum(p1, p2))).
    #    Why the minimum: at each k the optimal rule keeps the larger pmf,
    #    so the smaller one is exactly the mass it gets wrong.
    #    Check: lam1 == lam2 must give exactly 0.5.
    raise NotImplementedError("Implement me! See docs/s/week04.html — and try before peeking at python/solutions/.")


def log_likelihood_ratio(X, lam1, lam2) -> np.ndarray:
    """Poisson naive-Bayes log-likelihood ratio, one value per cell.

    For a cell with counts x over genes g (Week 5):

        LLR(x) = sum_g [ x_g * log(lam1_g / lam2_g) - (lam1_g - lam2_g) ]

    Derivation: write the log Poisson pmf under each hypothesis and subtract.
    The log(x_g!) terms are IDENTICAL under both hypotheses and cancel —
    which is why the optimal rule is *linear* in the counts. That fact is the
    bridge to Week 8: logistic regression fits a linear rule, so it is
    fitting the right shape, and can at best match this.

    Positive LLR means type 1 is more likely.

    Parameters
    ----------
    X          : (n_cells, n_genes) count matrix.
    lam1, lam2 : (n_genes,) expected counts under each type.

    Returns
    -------
    numpy array of shape (n_cells,).
    """
    # --- YOUR CODE HERE ---
    # 1. Convert X, lam1, lam2 to float arrays.
    # 2. A rate of exactly 0 under one type makes log(lam1/lam2) infinite.
    #    Floor both rate vectors at a tiny eps = 1e-12 with np.maximum.
    #    (Real pipelines add a pseudocount for exactly this reason; Week 10
    #     asks you to think about what that pseudocount is really assuming.)
    # 3. The sum over genes is a matrix product. Write it as
    #       X @ np.log(a / b) - np.sum(a - b)
    #    and convince yourself the second term is a single scalar that does
    #    not depend on the cell at all — it only shifts the threshold.
    raise NotImplementedError("Implement me! See docs/s/week05.html — and try before peeking at python/solutions/.")


def roc_curve(scores, labels) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical ROC: return (fpr, tpr) arrays.

    Sweep the decision threshold from +inf down to -inf; at each DISTINCT
    observed score, record the false-positive and true-positive rates.
    Week 4.

    "Distinct" is the whole subtlety (exercise 4.6). A threshold cannot
    separate two cells that received the same score, so tied scores must be
    consumed all at once. Break them apart instead and a classifier that
    gives every cell the same score comes out with an AUC of 0 or 1
    depending on the input ordering — a spectacular and entirely fictional
    result.

    Parameters
    ----------
    scores : (n,) array — higher means "more likely class 1".
    labels : (n,) array of 0/1.

    Returns
    -------
    (fpr, tpr), both starting at 0.0 and ending at 1.0.

    Raises
    ------
    ValueError if either class is absent.
    """
    # --- YOUR CODE HERE ---
    # 1. order = np.argsort(-scores, kind="mergesort")  — descending, stable.
    #    Sort BOTH labels and scores by it.
    # 2. Count P (positives) and N (negatives); raise ValueError if either
    #    is 0 — an ROC needs both classes to have rates at all.
    # 3. tp = np.cumsum(y == 1), fp = np.cumsum(y == 0): after consuming the
    #    first i predictions, these are the counts above the threshold.
    # 4. Collapse ties: keep only the LAST index of each run of equal scores.
    #       last_of_tie = np.r_[np.diff(s) != 0, True]
    # 5. Prepend the origin: tpr = concat([[0.0], tp[last_of_tie] / P]) and
    #    likewise for fpr. Return (fpr, tpr).
    raise NotImplementedError("Implement me! See docs/s/week04.html — and try before peeking at python/solutions/.")


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve, by the trapezoid rule on roc_curve output.

    Interpretation worth memorizing: AUC is the probability that a randomly
    chosen positive scores above a randomly chosen negative. Under the
    Gaussian model it equals Phi(d'/sqrt(2)) — a second, threshold-free way
    to read d' off data, which is what makes d_prime_from_auc possible.
    """
    # --- YOUR CODE HERE ---
    # 1. fpr, tpr = roc_curve(scores, labels).
    # 2. Return float(np.trapezoid(tpr, fpr)). (On numpy < 2 the function is
    #    spelled np.trapz; the solutions file shows the compatible form.)
    raise NotImplementedError("Implement me! See docs/s/week04.html — and try before peeking at python/solutions/.")


def d_prime_from_auc(auc: float) -> float:
    """Invert AUC = Phi(d'/sqrt(2)) to recover d' from measured performance.

    Week 4. This is how a measured number becomes comparable to a predicted
    one: accuracy saturates near 1 and squashes differences, while d' keeps
    growing, so all the project's comparisons are made in d' space.
    """
    # --- YOUR CODE HERE ---
    # 1. Clip auc into (0, 1) so the inverse is finite: np.clip(auc, 1e-12,
    #    1 - 1e-12).
    # 2. Phi^{-1}(a) = sqrt(2) * erfinv(2a - 1)  (erfinv is imported above).
    # 3. AUC = Phi(d'/sqrt(2))  =>  d' = sqrt(2) * Phi^{-1}(AUC). Return it.
    raise NotImplementedError("Implement me! See docs/s/week04.html — and try before peeking at python/solutions/.")


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
