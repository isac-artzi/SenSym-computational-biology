"""The synthetic-cell simulator: Aim 2's laboratory. [SOLUTIONS]

Reference implementation of `celldetect.simulate`.
Used by: Weeks 7-9.

Model, stated once so it can be attacked later:
  * a cell type is a probability vector p over genes (p_g = the chance that
    any given transcript in a cell of this type is gene g);
  * sequencing a cell to depth D yields counts x_g ~ Poisson(D * p_g),
    independently across genes;
  * dropout deletes each captured molecule independently with probability q.
Every one of those three assumptions is false about real cells in a way the
project measures in Week 11. Writing them down is what makes that possible.
"""

from typing import Tuple

import numpy as np

from .counting import downsample_counts


def marker_profiles(n_genes: int, fold_change: float,
                    marker_mass: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Two expression profiles over `n_genes` marker genes.

    The convention, stated exactly because it is easy to get wrong: the two
    types put an AVERAGE of `marker_mass` on the marker set — that is,
    (sum(p1) + sum(p2)) / 2 = marker_mass — split equally among the genes,
    with type 1 expressing each marker `fold_change` times more strongly
    than type 2. Both types cannot each carry exactly marker_mass AND differ
    by a fold change; something has to give, and pinning the average is the
    choice that keeps d' symmetric in the two types. (The leftover mass
    difference is absorbed by the thousands of non-marker genes, which this
    simulation does not track.)

    Holding the pooled mass fixed as n_genes grows is also deliberate: it
    keeps the *per-gene* signal shrinking as 1/k, so the sqrt(k) law is
    tested against a genuine trade-off rather than against a free lunch.

    Returns (p1, p2), each of shape (n_genes,).
    """
    if n_genes < 1:
        raise ValueError("n_genes must be at least 1")
    if fold_change <= 0:
        raise ValueError("fold_change must be positive")
    base = marker_mass / n_genes
    # Solve  a/b = fold_change  and  (a+b)/2 = base  =>  a = 2*base*f/(1+f).
    a = 2.0 * base * fold_change / (1.0 + fold_change)
    b = 2.0 * base / (1.0 + fold_change)
    return np.full(n_genes, a), np.full(n_genes, b)


def simulate_cells(profile, depth: float, n_cells: int,
                   rng: np.random.Generator, dropout: float = 0.0) -> np.ndarray:
    """Sample `n_cells` cells of one type: Poisson counts at the given depth.

    Returns an (n_cells, n_genes) integer matrix. With dropout > 0 each
    molecule is discarded independently with that probability, which (by the
    thinning theorem of Week 2) is the same as sequencing at depth
    (1 - dropout) * depth — a fact the Week 9 lab checks numerically before
    the chapter proves it.
    """
    profile = np.asarray(profile, dtype=float)
    if depth < 0:
        raise ValueError("depth must be non-negative")
    lam = depth * profile
    X = rng.poisson(np.broadcast_to(lam, (n_cells, profile.size)))
    if dropout > 0:
        X = downsample_counts(X, 1.0 - dropout, rng)
    return X


def simulate_dataset(profiles, depth: float, n_cells_per_type: int,
                     rng: np.random.Generator,
                     dropout: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Stack several types into one labelled dataset.

    profiles : sequence of (n_genes,) profiles, one per cell type.
    Returns (X, y) with X of shape (n_types * n_cells_per_type, n_genes)
    and y the integer type label of each row.
    """
    profiles = [np.asarray(p, dtype=float) for p in profiles]
    blocks, labels = [], []
    for t, p in enumerate(profiles):
        blocks.append(simulate_cells(p, depth, n_cells_per_type, rng, dropout))
        labels.append(np.full(n_cells_per_type, t, dtype=int))
    return np.vstack(blocks), np.concatenate(labels)


def apply_dropout(X, dropout: float, rng: np.random.Generator) -> np.ndarray:
    """Delete each observed molecule independently with probability `dropout`."""
    return downsample_counts(X, 1.0 - dropout, rng)


def expected_rates(profiles, depth: float, dropout: float = 0.0) -> np.ndarray:
    """[PROVIDED] The (n_types, n_genes) matrix of expected counts.

    This is the ground truth the theory uses: pass it to
    detection.d_prime_total or detection.log_likelihood_ratio. Available
    only in simulation — Week 11's real-data work has to *estimate* it,
    which is where a second source of error enters.
    """
    profiles = np.atleast_2d(np.asarray(profiles, dtype=float))
    return depth * (1.0 - dropout) * profiles


def multi_type_profiles(n_types: int, n_genes: int, fold_change: float,
                        marker_mass: float = 0.02,
                        rng: np.random.Generator = None) -> np.ndarray:
    """[PROVIDED] `n_types` profiles, each over-expressing a disjoint block of markers.

    Type t expresses its own block `fold_change`-fold above baseline and the
    other blocks at baseline. Used for the five-type extension in Week 9;
    the two-type case of marker_profiles is NOT a special case of this one
    (there both types differ on every gene), and the chapters keep them
    distinct on purpose.
    """
    if n_genes % n_types:
        raise ValueError("n_genes must be divisible by n_types")
    block = n_genes // n_types
    base = marker_mass / n_genes
    P = np.full((n_types, n_genes), base, dtype=float)
    for t in range(n_types):
        P[t, t * block:(t + 1) * block] *= fold_change
    # Renormalize so every type still spends exactly marker_mass on markers.
    P *= marker_mass / P.sum(axis=1, keepdims=True)
    return P
