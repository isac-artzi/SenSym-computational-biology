"""Counting statistics: what a sequencer actually hands you. [SOLUTIONS]

Reference implementation of `celldetect.counting`.
Used by: Weeks 1-2 (and by every module downstream).
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.special import gammaln


def sample_counts(lam: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_samples` independent Poisson counts with mean `lam`."""
    # numpy's Generator.poisson does the sampling; we only insist that the
    # caller passes the Generator in, so that every figure is seeded.
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    return rng.poisson(lam, size=n_samples)


def fano_factor(samples: Sequence[float]) -> float:
    """Return variance / mean of `samples` — 1.0 for a Poisson process.

    The Fano factor is the single most useful diagnostic in the project:
    it is exactly 1 for Poisson, above 1 for anything with extra
    (biological) variability, and it is what tells us in Week 11 that real
    scRNA-seq counts are NOT Poisson.
    """
    x = np.asarray(samples, dtype=float)
    m = x.mean()
    if m == 0:
        raise ValueError("Fano factor is undefined for an all-zero sample")
    # ddof=1: the unbiased (sample) variance. With ddof=0 the factor is
    # biased low by a factor (n-1)/n — visible at the n = 50 used in Week 1.
    return float(x.var(ddof=1) / m)


def poisson_logpmf(k, lam: float) -> np.ndarray:
    """log P(X = k) for X ~ Poisson(lam), computed in log space.

    log pmf = k*log(lam) - lam - log(k!)
    and log(k!) = lgamma(k+1), which numpy provides as a vectorized ufunc.
    Working in logs is not fussiness: at depth 5000 the factorials overflow
    a float64 long before the probabilities themselves get small.
    """
    k = np.asarray(k)
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if lam == 0:
        # All the mass sits on k = 0.
        return np.where(k == 0, 0.0, -np.inf)
    # gammaln(k+1) == log(k!) for integer k, vectorized and overflow-free.
    return k * np.log(lam) - lam - gammaln(k + 1)


def poisson_pmf(k, lam: float) -> np.ndarray:
    """P(X = k) for X ~ Poisson(lam)."""
    return np.exp(poisson_logpmf(k, lam))


def downsample_counts(counts, keep_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Thin counts: keep each molecule independently with probability `keep_prob`.

    This one function models BOTH of the project's "less data" knobs:
      * sequencing a library less deeply, and
      * the bead experiment's dropout rule.

    The key fact (Week 2, proved): if X ~ Poisson(lam) and each of the X
    items survives independently with probability q, the survivor count is
    Poisson(q*lam) — not merely approximately, exactly. Thinned Poisson data
    is still Poisson data, which is why the whole theory survives
    downsampling.
    """
    if not 0.0 <= keep_prob <= 1.0:
        raise ValueError(f"keep_prob must be in [0, 1], got {keep_prob}")
    counts = np.asarray(counts)
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    # Binomial(n=count, p=keep_prob) applied elementwise IS the thinning.
    return rng.binomial(counts.astype(np.int64), keep_prob)


def total_counts(matrix) -> np.ndarray:
    """[PROVIDED] Library size per cell: the row sums of a cells x genes matrix."""
    return np.asarray(matrix).sum(axis=1)


def empirical_moments(samples: Sequence[float]) -> Tuple[float, float]:
    """[PROVIDED] Return (mean, unbiased variance) of a 1-D sample."""
    x = np.asarray(samples, dtype=float)
    return float(x.mean()), float(x.var(ddof=1))
