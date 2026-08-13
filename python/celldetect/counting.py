"""Counting statistics: what a sequencer actually hands you.

Used by: Weeks 1-2 (and by every module downstream).

Build order (each function may use the previous ones):
    1. sample_counts        (W1)
    2. fano_factor          (W1)
    3. poisson_logpmf       (W2)
    4. poisson_pmf          (W2)
    5. downsample_counts    (W2)

The whole project rests on one coincidence: for a Poisson variable, the
variance equals the mean. Everything you write here is a way of measuring,
using, or breaking that fact.
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.special import gammaln


def sample_counts(lam: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_samples` independent Poisson counts with mean `lam`.

    This is the model of a sequencer: a gene expressed at rate `lam`
    molecules per cell yields a count that is Poisson(lam), and a different
    cell of the same type yields a different count. See Week 1.

    Parameters
    ----------
    lam       : float, the expected count (non-negative).
    n_samples : int, how many independent counts to draw.
    rng       : a numpy Generator, e.g. np.random.default_rng(101).
                It is a REQUIRED argument, not an optional one, so that no
                figure in this project can be produced without a seed.

    Returns
    -------
    numpy array of shape (n_samples,), integer dtype, all entries >= 0.

    Raises
    ------
    ValueError if lam < 0.

    Example
    -------
    >>> sample_counts(3.0, 5, np.random.default_rng(0))
    array([...])
    """
    # --- YOUR CODE HERE ---
    # 1. Reject a negative lam with a ValueError whose message includes the
    #    offending value (a good error message names the bad input).
    # 2. Return rng.poisson(lam, size=n_samples). One line — the interesting
    #    part is the experiment you run with it, not the call.
    raise NotImplementedError("Implement me! See docs/s/week01.html — and try before peeking at python/solutions/.")


def fano_factor(samples: Sequence[float]) -> float:
    """Return variance / mean of `samples` — 1.0 for a Poisson process.

    The Fano factor is the single most useful diagnostic in the project:
    exactly 1 for Poisson, above 1 for anything with extra (biological)
    variability. In Week 11 it is what tells you that real scRNA-seq counts
    are NOT Poisson, which is the project's most important negative result.

    Parameters
    ----------
    samples : 1-D sequence of counts.

    Returns
    -------
    float : var / mean, using the UNBIASED (ddof=1) variance.

    Raises
    ------
    ValueError if the sample mean is zero (the ratio is undefined).
    """
    # --- YOUR CODE HERE ---
    # 1. Convert to a float numpy array (np.asarray(..., dtype=float)).
    # 2. Compute the mean. If it is 0, raise ValueError — you cannot divide.
    # 3. Compute the variance with ddof=1 (x.var(ddof=1)). Why ddof=1: the
    #    default ddof=0 divides by n instead of n-1 and is biased LOW by a
    #    factor (n-1)/n. At the n = 50 used in Week 1's lab that is a 2%
    #    bias — visible, and exactly the size of the effect you are hunting.
    # 4. Return the ratio as a plain float.
    raise NotImplementedError("Implement me! See docs/s/week01.html — and try before peeking at python/solutions/.")


def poisson_logpmf(k, lam: float) -> np.ndarray:
    """log P(X = k) for X ~ Poisson(lam), computed in log space.

    The formula (Week 2):
        log P(X = k) = k*log(lam) - lam - log(k!)
    and log(k!) = gammaln(k+1), already imported at the top of this file.

    Working in logs is not fussiness: at depth 5000 the factorial k!
    overflows a float64 (which caps out around 1e308) long before the
    probability itself becomes small. Compute the log, then exponentiate —
    never the other way round.

    Parameters
    ----------
    k   : int or array of ints (the counts to evaluate).
    lam : float, the Poisson mean (non-negative).

    Returns
    -------
    numpy array of log-probabilities, same shape as k.
    """
    # --- YOUR CODE HERE ---
    # 1. k = np.asarray(k), so the function works on scalars and arrays alike.
    # 2. Reject lam < 0 with a ValueError.
    # 3. Handle lam == 0 separately: all the mass sits on k = 0, so return
    #    0.0 (that is log 1) where k == 0 and -inf elsewhere. np.where does
    #    this in one line. Without this branch, log(0) warns and poisons the
    #    array with NaN.
    # 4. Otherwise return  k*np.log(lam) - lam - gammaln(k + 1).
    raise NotImplementedError("Implement me! See docs/s/week02.html — and try before peeking at python/solutions/.")


def poisson_pmf(k, lam: float) -> np.ndarray:
    """P(X = k) for X ~ Poisson(lam).

    One line once poisson_logpmf works. Kept as its own function because
    the exact-error calculation in Week 4 wants probabilities, while the
    classifier in Week 5 wants logs.
    """
    # --- YOUR CODE HERE ---
    # 1. Return np.exp(poisson_logpmf(k, lam)).
    raise NotImplementedError("Implement me! See docs/s/week02.html — and try before peeking at python/solutions/.")


def downsample_counts(counts, keep_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Thin counts: keep each molecule independently with probability `keep_prob`.

    This one function models BOTH of the project's "less data" knobs:
      * sequencing a library less deeply, and
      * the bead experiment's dropout rule.

    The key fact (Week 2, and you will prove it): if X ~ Poisson(lam) and
    each of the X items survives independently with probability q, the
    survivor count is Poisson(q*lam) — not approximately, exactly. Thinned
    Poisson data is still Poisson data, which is why the entire theory
    survives downsampling and why Week 11's real-data protocol is legitimate.

    Parameters
    ----------
    counts    : int or array of ints (works on a whole cells x genes matrix).
    keep_prob : float in [0, 1].
    rng       : numpy Generator.

    Returns
    -------
    numpy array of the same shape as `counts`, with every entry <= the input.

    Raises
    ------
    ValueError if keep_prob is outside [0, 1], or any count is negative.
    """
    # --- YOUR CODE HERE ---
    # 1. Validate keep_prob is in [0, 1]; raise ValueError if not.
    # 2. counts = np.asarray(counts); raise ValueError if any entry < 0.
    # 3. The thinning IS a binomial draw: each of the `count` molecules is an
    #    independent coin flip. So return
    #       rng.binomial(counts.astype(np.int64), keep_prob)
    #    numpy broadcasts this elementwise over an array of any shape, which
    #    is why the same function thins one gene or a whole matrix.
    raise NotImplementedError("Implement me! See docs/s/week02.html — and try before peeking at python/solutions/.")


def total_counts(matrix) -> np.ndarray:
    """[PROVIDED] Library size per cell: the row sums of a cells x genes matrix."""
    return np.asarray(matrix).sum(axis=1)


def empirical_moments(samples: Sequence[float]) -> Tuple[float, float]:
    """[PROVIDED] Return (mean, unbiased variance) of a 1-D sample."""
    x = np.asarray(samples, dtype=float)
    return float(x.mean()), float(x.var(ddof=1))
