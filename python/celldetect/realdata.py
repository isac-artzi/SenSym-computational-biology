"""Real scRNA-seq data: the 10x PBMC 3k dataset.

Used by: Weeks 10-11.

Build order:
    1. select_markers               (W10)
    2. downsample_matrix            (W10)
    3. estimate_rates               (W11)
    4. gene_fano                    (W11)
    5. negative_binomial_dispersion (W11)
    6. d_prime_overdispersed        (W11)

The dataset is not committed to this repository (it is ~35 MB of sparse
matrix). Instead:

  * `experiments/fetch_pbmc3k.py` downloads it once with scanpy and writes a
    compact cache to python/data/pbmc3k_cache.npz;
  * `load_pbmc` reads that cache;
  * if the cache is absent, the Week 10 and 11 experiments fall back to a
    clearly-labelled SURROGATE dataset, print a loud warning, and stamp the
    figure. A surrogate figure is never a result — it exists so the pipeline
    can be built and tested before the download.

Everything here works on a plain (n_cells, n_genes) integer array plus a
label vector, so the same code runs on real and surrogate data.
"""

import os
import warnings
from typing import Tuple

import numpy as np


def _find_data_dir() -> str:
    """[PROVIDED] Locate python/data/ by walking up from this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        here = os.path.dirname(here)
        candidate = os.path.join(here, "data")
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, "data")


CACHE_PATH = os.path.join(_find_data_dir(), "pbmc3k_cache.npz")


def load_pbmc(path: str = None, allow_missing: bool = False):
    """[PROVIDED] Load the cached PBMC 3k counts.

    Returns (X, y, gene_names, type_names) where X is (n_cells, n_genes)
    integer counts and y integer cell-type labels.

    Raises FileNotFoundError with a pointer to the fetch script if the cache
    is missing, unless allow_missing=True, in which case it returns None.
    """
    path = path or CACHE_PATH
    if not os.path.exists(path):
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"No PBMC cache at {path}.\n"
            "Run:  python experiments/fetch_pbmc3k.py\n"
            "(needs `pip install -r requirements-optional.txt` for scanpy; "
            "one download of ~35 MB, then never again)."
        )
    z = np.load(path, allow_pickle=True)
    return (z["X"], z["y"], list(z["genes"]), list(z["types"]))


def select_markers(X, y, n_markers: int, min_cells: int = 10) -> np.ndarray:
    """Return the indices of the `n_markers` most discriminative genes.

    Ranking rule: absolute log2 fold change between the two class means,
    with a pseudocount of 1 to keep silent genes finite, restricted to genes
    detected in at least `min_cells` cells.

    Two warnings the chapter makes a point of (Week 10):
      1. This uses the labels, so selecting markers on the SAME data you
         then evaluate on leaks information and inflates accuracy. The
         honest protocol selects on the training split only — which
         experiments/week10_markers.py does, and quantifies the difference.
      2. Fold change is not the same ranking as d'. A gene can have a huge
         fold change and negligible d' because both rates are tiny.
         Exercise 10.4 asks you to rank both ways and compare.

    Raises
    ------
    ValueError if y does not contain exactly two classes.
    """
    # --- YOUR CODE HERE ---
    # 1. Convert X and y; classes = np.unique(y); raise ValueError unless
    #    there are exactly 2.
    # 2. detected = (X > 0).sum(axis=0) >= min_cells — a boolean per gene.
    # 3. Class means: m1 = X[y == classes[0]].mean(axis=0), likewise m2.
    # 4. lfc = np.abs(np.log2((m1 + 1) / (m2 + 1))). The +1 is the
    #    pseudocount; without it a gene silent in one class gives inf and
    #    wins every ranking on the strength of one stray count.
    # 5. Disqualify the undetected genes: lfc = np.where(detected, lfc, -inf).
    # 6. Return the indices of the top n_markers:
    #       np.argsort(-lfc, kind="mergesort")[:n_markers]
    #    (clip n_markers to the number of finite entries first).
    raise NotImplementedError("Implement me! See docs/s/week10.html — and try before peeking at python/solutions/.")


def downsample_matrix(X, keep_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Thin every entry of a count matrix — 'sequence this library less deeply'.

    This is the experimental knob of Aim 2's real-data half. Because thinned
    Poisson data is Poisson (Week 2), downsampling a real dataset produces
    exactly the dataset a shallower run would have produced — up to the
    (real, measurable) fact that real counts are not Poisson to begin with,
    which is what Week 11 is about.

    Raises
    ------
    ValueError if keep_prob is outside [0, 1].
    """
    # --- YOUR CODE HERE ---
    # 1. Validate keep_prob.
    # 2. Return rng.binomial(np.asarray(X).astype(np.int64), keep_prob).
    #    (Yes, this is counting.downsample_counts again. It is repeated here
    #     so the real-data module reads standalone; if you prefer, import and
    #     delegate — but then say so in your progress log, because a reader
    #     of the report will want to know these are the same operation.)
    raise NotImplementedError("Implement me! See docs/s/week10.html — and try before peeking at python/solutions/.")


def estimate_rates(X, y) -> np.ndarray:
    """Per-class mean counts: the (n_types, n_genes) rate matrix, estimated.

    In simulation the rates are handed to you. Here they must be estimated
    from a finite sample, and that estimate is itself noisy — the extra error
    term Week 11 isolates by comparing "theory with true rates" against
    "theory with estimated rates".
    """
    # --- YOUR CODE HERE ---
    # 1. classes = np.unique(y).
    # 2. Return np.vstack([X[y == c].mean(axis=0) for c in classes]).
    raise NotImplementedError("Implement me! See docs/s/week11.html — and try before peeking at python/solutions/.")


def gene_fano(X) -> np.ndarray:
    """Per-gene Fano factor (variance / mean) across cells.

    The Poisson model predicts 1.0 for every gene. Real scRNA-seq gives
    values well above 1 for most expressed genes — the counts are
    overdispersed, because cells of the "same type" genuinely differ. This
    array is the evidence for Week 11's central negative result.

    Genes with zero mean get np.nan (not an exception, not a zero — an
    undefined value should look undefined downstream).
    """
    # --- YOUR CODE HERE ---
    # 1. m = X.mean(axis=0), v = X.var(axis=0, ddof=1).
    # 2. Return np.where(m > 0, v / m, np.nan) — but guard the division the
    #    same way you did in d_prime, since np.where evaluates both branches.
    # 3. np.nanmedian is how you summarize the result; plain np.median would
    #    return nan the moment one gene is undetected.
    raise NotImplementedError("Implement me! See docs/s/week11.html — and try before peeking at python/solutions/.")


def negative_binomial_dispersion(X) -> np.ndarray:
    """Method-of-moments dispersion phi per gene, for the NB model.

    Under the negative binomial with mean m and dispersion phi:
        Var = m + phi * m^2.
    Solving for phi:
        phi = (Var - m) / m^2,
    clipped at 0 — a negative estimate means the gene is, within noise,
    Poisson. Week 11.
    """
    # --- YOUR CODE HERE ---
    # 1. Same m and v as gene_fano.
    # 2. phi = (v - m) / m**2 where m > 0, else nan.
    # 3. Return np.clip(phi, 0.0, None).
    raise NotImplementedError("Implement me! See docs/s/week11.html — and try before peeking at python/solutions/.")


def d_prime_overdispersed(lam1, lam2, phi) -> np.ndarray:
    """d' when the counts are negative binomial rather than Poisson.

    Same numerator, larger denominator: the pooled variance becomes
    lbar + phi*lbar^2 instead of lbar, so

        d' = |lam1 - lam2| / sqrt(lbar + phi * lbar^2).

    Consequence, and the sharpest prediction of Week 11: as depth grows, lam
    grows, the phi*lam^2 term comes to dominate, and d' STOPS growing like
    sqrt(depth) — it saturates at |p1-p2| / (pbar * sqrt(phi)). Biological
    variability, not sequencing depth, becomes the limit. Deeper sequencing
    then buys nothing, which is a testable claim with real budget
    consequences for anyone designing an experiment.
    """
    # --- YOUR CODE HERE ---
    # 1. Convert all three inputs to float arrays.
    # 2. lbar = (lam1 + lam2) / 2;  var = lbar + phi * lbar**2.
    # 3. Return |lam1 - lam2| / sqrt(var), with the same guard against
    #    var == 0 that d_prime uses.
    # 4. Check yourself: phi = 0 must reproduce detection.d_prime exactly.
    raise NotImplementedError("Implement me! See docs/s/week11.html — and try before peeking at python/solutions/.")


def surrogate_pbmc(n_cells: int = 1200, n_genes: int = 500,
                   depth: float = 2500.0, rng: np.random.Generator = None
                   ) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """[PROVIDED] A clearly-labelled stand-in for PBMC 3k.

    Two "types" with a long-tailed expression profile and gene-specific
    overdispersion, so the pipeline meets something messier than the clean
    simulator — but this is NOT data. Any figure built from it must say so;
    the experiment scripts stamp 'SURROGATE DATA' across it and refuse to
    write into figures/report/.
    """
    rng = rng or np.random.default_rng(0)
    # Zipf-ish expression: a few genes carry most of the transcriptome.
    base = 1.0 / np.arange(1, n_genes + 1) ** 1.1
    base /= base.sum()
    lfc = rng.normal(0.0, 0.35, size=n_genes)      # most genes barely differ
    p1 = base * np.exp(lfc / 2)
    p2 = base * np.exp(-lfc / 2)
    p1 /= p1.sum(); p2 /= p2.sum()
    phi = rng.gamma(shape=2.0, scale=0.5, size=n_genes)   # overdispersion
    half = n_cells // 2
    blocks = []
    for p in (p1, p2):
        lam = depth * p
        # Gamma-Poisson mixture = negative binomial.
        g = rng.gamma(shape=1.0 / np.maximum(phi, 1e-6),
                      scale=np.maximum(phi, 1e-6) * lam,
                      size=(half, n_genes))
        blocks.append(rng.poisson(g))
    X = np.vstack(blocks)
    y = np.concatenate([np.zeros(half, int), np.ones(half, int)])
    genes = [f"SURROGATE{g:04d}" for g in range(n_genes)]
    return X, y, genes, ["surrogate_A", "surrogate_B"]
