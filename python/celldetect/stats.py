"""Honest error bars.

Used by: Week 13 (and retro-fitted to every earlier figure in Week 14).

Build order:
    1. wilson_interval    (W13)
    2. bootstrap_ci       (W13)
    3. permutation_test   (W13)

A measured accuracy is a proportion from a finite number of trials. Reported
without an interval it is not a measurement, it is a rumour. This module is
short on purpose: three tools, used everywhere.
"""

from typing import Callable, Sequence, Tuple

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    The naive interval p +/- z*sqrt(p(1-p)/n) fails exactly where this
    project lives: at n = 20 it can reach past 1, and at p = 1 it has zero
    width, claiming certainty from 20 trials. The Wilson interval is the set
    of p_0 not rejected by the score test, which fixes both:

        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z * sqrt( p(1-p)/n + z^2/(4n^2) ) / (1 + z^2/n)

    Use it for every accuracy in the report, including the bead
    experiment's, where n per point is only ~40.

    Returns
    -------
    (lo, hi), both clipped into [0, 1]. n = 0 returns (0, 1) — no data, no
    information, and the interval should say so rather than crash.

    Raises
    ------
    ValueError if k is outside 0..n.
    """
    # --- YOUR CODE HERE ---
    # 1. If n <= 0, return (0.0, 1.0).
    # 2. Validate 0 <= k <= n.
    # 3. p = k / n; z2 = z*z; denom = 1 + z2/n.
    # 4. centre = (p + z2/(2n)) / denom.
    # 5. half = z * sqrt(p(1-p)/n + z2/(4 n^2)) / denom.
    # 6. Return (max(0, centre - half), min(1, centre + half)) as floats.
    # Check yourself: (0, 20) must give lo = 0 and hi well above 0.1 —
    # zero successes in twenty trials does NOT mean the probability is zero.
    raise NotImplementedError("Implement me! See docs/s/week13.html — and try before peeking at python/solutions/.")


def bootstrap_ci(values: Sequence[float], statistic: Callable, n_boot: int,
                 rng: np.random.Generator, alpha: float = 0.05
                 ) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for any statistic.

    Resample the data WITH replacement n_boot times, recompute the statistic
    on each resample, and take the alpha/2 and 1-alpha/2 percentiles of the
    resulting distribution.

    What it does: quantifies how much the statistic would wobble under a
    fresh draw from the same population.
    What it cannot do: fix a biased estimator, or account for the fact that
    the population you sampled is not the one you care about. In Week 13 the
    second caveat is the one that matters — a bootstrap over cells says
    nothing about variation between donors, and the report must say so.

    Raises
    ------
    ValueError on an empty sample.
    """
    # --- YOUR CODE HERE ---
    # 1. values = np.asarray(values); raise ValueError if empty.
    # 2. Allocate stats = np.empty(n_boot).
    # 3. For each b: idx = rng.integers(0, n, size=n) — WITH replacement, so
    #    the same index may appear several times; that is the whole idea.
    #    stats[b] = statistic(values[idx]).
    # 4. lo, hi = np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)]).
    # 5. Return them as floats.
    raise NotImplementedError("Implement me! See docs/s/week13.html — and try before peeking at python/solutions/.")


def permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int,
                     rng: np.random.Generator) -> float:
    """Two-sided permutation p-value for a difference in means.

    Null hypothesis: the two samples come from the same distribution, so the
    group labels are exchangeable. Pool, reshuffle the labels n_perm times,
    and count how often the shuffled difference is at least as extreme as
    the observed one.

    Return (count + 1) / (n_perm + 1) — the "add-one" correction. It keeps
    the p-value from ever being exactly 0, which matters: with n_perm
    shuffles you cannot resolve anything below 1/(n_perm+1), and printing
    p = 0 claims a precision you did not buy.
    """
    # --- YOUR CODE HERE ---
    # 1. observed = abs(a.mean() - b.mean()).
    # 2. pooled = np.concatenate([a, b]); na = len(a).
    # 3. Loop n_perm times: perm = rng.permutation(pooled); compare
    #    abs(perm[:na].mean() - perm[na:].mean()) >= observed and count.
    # 4. Return (count + 1) / (n_perm + 1).
    raise NotImplementedError("Implement me! See docs/s/week13.html — and try before peeking at python/solutions/.")


def standard_error_of_proportion(k: int, n: int) -> float:
    """[PROVIDED] sqrt(p(1-p)/n) — for annotation only, never for an interval."""
    if n <= 0:
        return float("nan")
    p = k / n
    return float(np.sqrt(p * (1 - p) / n))


def bonferroni(alpha: float, n_tests: int) -> float:
    """[PROVIDED] The per-test level that holds the family-wise error at alpha.

    Week 13: the depth sweep tests the same hypothesis at eight depths. Eight
    chances at p < 0.05 is a ~34% chance of at least one false positive if
    nothing is going on. Either correct, or say plainly that the sweep is
    exploratory and no single point is being claimed.
    """
    if n_tests < 1:
        raise ValueError("n_tests must be at least 1")
    return alpha / n_tests
