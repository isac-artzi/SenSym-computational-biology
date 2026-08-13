"""The synthetic-cell simulator: Aim 2's laboratory.

Used by: Weeks 7-9.

Build order:
    1. marker_profiles   (W7)
    2. simulate_cells    (W7)
    3. simulate_dataset  (W7)
    4. apply_dropout     (W9)

Model, stated once so it can be attacked later:
  * a cell type is a probability vector p over genes (p_g = the chance that
    any given transcript in a cell of this type is gene g);
  * sequencing a cell to depth D yields counts x_g ~ Poisson(D * p_g),
    independently across genes;
  * dropout deletes each captured molecule independently with probability q.

Every one of those three assumptions is false about real cells in a way this
project measures in Week 11. Writing them down explicitly is what makes that
measurement possible — an unstated assumption cannot be tested.
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
    than type 2.

    Both types cannot each carry exactly marker_mass AND differ by a fold
    change; something has to give, and pinning the average is the choice
    that keeps d' symmetric in the two types (exercise 7.2 asks you to
    check what the other choices do). The leftover mass difference is
    absorbed by the thousands of non-marker genes this simulation ignores.

    Holding the pooled mass fixed as n_genes grows is also deliberate: the
    per-gene signal then shrinks as 1/k, so the sqrt(k) law is tested
    against a genuine trade-off rather than against a free lunch.

    Returns (p1, p2), each of shape (n_genes,).

    Raises
    ------
    ValueError if n_genes < 1 or fold_change <= 0.
    """
    # --- YOUR CODE HERE ---
    # 1. Validate n_genes >= 1 and fold_change > 0.
    # 2. base = marker_mass / n_genes — the average rate per marker gene.
    # 3. Solve the two conditions  a/b = fold_change  and  (a+b)/2 = base:
    #       a = 2*base*fold_change / (1 + fold_change)
    #       b = 2*base            / (1 + fold_change)
    #    Do this algebra on paper first; it is exercise 7.1.
    # 4. Return (np.full(n_genes, a), np.full(n_genes, b)).
    raise NotImplementedError("Implement me! See docs/s/week07.html — and try before peeking at python/solutions/.")


def simulate_cells(profile, depth: float, n_cells: int,
                   rng: np.random.Generator, dropout: float = 0.0) -> np.ndarray:
    """Sample `n_cells` cells of one type: Poisson counts at the given depth.

    Returns an (n_cells, n_genes) integer matrix. With dropout > 0 each
    molecule is discarded independently with that probability — which, by
    the thinning theorem of Week 2, is the same as sequencing at depth
    (1 - dropout) * depth. The Week 9 lab checks that numerically before the
    chapter proves it.

    Raises
    ------
    ValueError if depth < 0.
    """
    # --- YOUR CODE HERE ---
    # 1. profile = np.asarray(profile, dtype=float); validate depth >= 0.
    # 2. lam = depth * profile — the (n_genes,) vector of expected counts.
    # 3. Sample the whole matrix in one call. numpy broadcasts a rate vector
    #    over rows if you give it the right shape:
    #       X = rng.poisson(np.broadcast_to(lam, (n_cells, profile.size)))
    #    (A Python loop over cells gives the same answer and is 50x slower;
    #     write the loop first if it helps you see it, then replace it.)
    # 4. If dropout > 0, pass X through downsample_counts(X, 1 - dropout, rng).
    # 5. Return X.
    raise NotImplementedError("Implement me! See docs/s/week07.html — and try before peeking at python/solutions/.")


def simulate_dataset(profiles, depth: float, n_cells_per_type: int,
                     rng: np.random.Generator,
                     dropout: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Stack several types into one labelled dataset.

    Parameters
    ----------
    profiles : sequence of (n_genes,) profiles, one per cell type.

    Returns
    -------
    (X, y) with X of shape (n_types * n_cells_per_type, n_genes) and y the
    integer type label of each row (0, 1, 2, ...).
    """
    # --- YOUR CODE HERE ---
    # 1. Loop over the profiles with enumerate() so you have both the index
    #    t (the label) and the profile p.
    # 2. For each: call simulate_cells(...) and collect the block, and
    #    collect np.full(n_cells_per_type, t, dtype=int) as its labels.
    # 3. Return np.vstack(blocks), np.concatenate(labels).
    #    Keep the rows and their labels in the SAME order — the single most
    #    destructive bug available in this whole file is a label that no
    #    longer matches its row, and it will not raise, it will just quietly
    #    produce chance accuracy forever.
    raise NotImplementedError("Implement me! See docs/s/week07.html — and try before peeking at python/solutions/.")


def apply_dropout(X, dropout: float, rng: np.random.Generator) -> np.ndarray:
    """Delete each observed molecule independently with probability `dropout`."""
    # --- YOUR CODE HERE ---
    # 1. One line: downsample_counts with keep probability (1 - dropout).
    #    Note the inversion — dropout is the probability of LOSS, keep_prob
    #    the probability of survival. Getting this backwards produces a
    #    figure that looks fine and is wrong.
    raise NotImplementedError("Implement me! See docs/s/week09.html — and try before peeking at python/solutions/.")


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
