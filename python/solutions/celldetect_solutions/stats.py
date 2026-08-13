"""Honest error bars. [SOLUTIONS]

Reference implementation of `celldetect.stats`.
Used by: Week 13 (and retro-fitted to every earlier figure in Week 14).

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
        half   = z * sqrt( p(1-p)/n + z^2/4n^2 ) / (1 + z^2/n)

    Week 13. Use it for every accuracy in the report, including the bead
    experiment's, where n per point is only ~40.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"k must be in 0..n, got k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_ci(values: Sequence[float], statistic: Callable, n_boot: int,
                 rng: np.random.Generator, alpha: float = 0.05
                 ) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for any statistic.

    Resample the data WITH replacement n_boot times, recompute the
    statistic on each resample, and take the alpha/2 and 1-alpha/2
    percentiles of the resulting distribution.

    What it does: quantifies how much the statistic would wobble under a
    fresh draw from the same population.
    What it cannot do: fix a biased estimator, or account for the fact that
    the population you sampled is not the one you care about. In Week 13 the
    second caveat is the one that matters, because a bootstrap over cells
    says nothing about variation between donors.
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic(values[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int,
                     rng: np.random.Generator) -> float:
    """Two-sided permutation p-value for a difference in means.

    Null hypothesis: the two samples come from the same distribution, so the
    group labels are exchangeable. Pool, reshuffle the labels n_perm times,
    and count how often the shuffled difference is at least as extreme as
    the observed one.

    The +1 in numerator and denominator (the "add-one" correction) keeps the
    p-value from ever being exactly 0 — with n_perm shuffles you cannot
    resolve anything below 1/(n_perm+1), and reporting p = 0 claims you can.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:na].mean() - perm[na:].mean()) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


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
