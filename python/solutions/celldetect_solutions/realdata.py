"""Real scRNA-seq data: the 10x PBMC 3k dataset. [SOLUTIONS]

Reference implementation of `celldetect.realdata`.
Used by: Weeks 10-11.

The dataset is not committed to this repository (it is ~35 MB of sparse
matrix). Instead:

  * `fetch_pbmc3k.py` (in experiments/) downloads it once with scanpy and
    writes a compact cache to python/data/pbmc3k_cache.npz;
  * `load_pbmc` reads that cache;
  * if the cache is absent, the week-10 and week-11 experiments fall back to
    a clearly-labelled SURROGATE dataset from the simulator, print a loud
    warning, and stamp the figure. A surrogate figure is never a result —
    it exists so the pipeline can be built and tested before the download.

Everything in this module works on a plain (n_cells, n_genes) integer array
plus a label vector, so the same code runs on real and surrogate data.
"""

import os
import warnings
from typing import Tuple

import numpy as np

def _find_data_dir() -> str:
    """Locate python/data/ by walking up from this file.

    Written this way so that the same source works from the student package
    (python/celldetect/) and from the solutions copy — and from the
    temporary directory run_solution_check.py builds, where it simply
    falls back to a path that will not exist.
    """
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
    integer counts, y integer cell-type labels.

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
         fold change and negligible d' because both rates are tiny. Week 10
         exercise 10.4 asks you to rank both ways and compare.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"select_markers expects 2 classes, got {classes.size}")
    detected = (X > 0).sum(axis=0) >= min_cells
    m1 = X[y == classes[0]].mean(axis=0)
    m2 = X[y == classes[1]].mean(axis=0)
    lfc = np.abs(np.log2((m1 + 1.0) / (m2 + 1.0)))
    lfc = np.where(detected, lfc, -np.inf)
    n_markers = min(n_markers, int(np.isfinite(lfc).sum()))
    return np.argsort(-lfc, kind="mergesort")[:n_markers]


def downsample_matrix(X, keep_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Thin every entry of a count matrix — 'sequence this library less deeply'.

    This is the experimental knob of Aim 2's real-data half. Because thinned
    Poisson data is Poisson (Week 2), downsampling a real dataset produces
    exactly the dataset a shallower run would have produced, up to the
    (real, measurable) fact that real counts are not Poisson to begin with.
    """
    if not 0.0 <= keep_prob <= 1.0:
        raise ValueError(f"keep_prob must be in [0, 1], got {keep_prob}")
    return rng.binomial(np.asarray(X).astype(np.int64), keep_prob)


def estimate_rates(X, y) -> np.ndarray:
    """Per-class mean counts: the (n_types, n_genes) rate matrix, estimated.

    In simulation the rates are handed to you. Here they must be estimated
    from a finite sample, and that estimate is itself noisy — the extra
    error term that Week 11 isolates by comparing "theory with true rates"
    against "theory with estimated rates".
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    return np.vstack([X[y == c].mean(axis=0) for c in classes])


def gene_fano(X) -> np.ndarray:
    """Per-gene Fano factor (variance / mean) across cells.

    The Poisson model predicts 1.0 for every gene. Real scRNA-seq gives
    values well above 1 for most expressed genes — the counts are
    overdispersed, because cells of the "same type" genuinely differ. This
    array is the evidence for Week 11's central negative result.
    """
    X = np.asarray(X, dtype=float)
    m = X.mean(axis=0)
    v = X.var(axis=0, ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 0/0 for undetected genes
        f = np.where(m > 0, v / np.where(m > 0, m, 1.0), np.nan)
    return f


def negative_binomial_dispersion(X) -> np.ndarray:
    """Method-of-moments dispersion phi per gene, for the NB model.

    Under NB with mean m and dispersion phi: Var = m + phi * m^2. Solving,
    phi = (Var - m) / m^2, clipped at 0 (a negative estimate means the gene
    is, within noise, Poisson). Week 11.
    """
    X = np.asarray(X, dtype=float)
    m = X.mean(axis=0)
    v = X.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(m > 0, (v - m) / np.where(m > 0, m ** 2, 1.0), np.nan)
    return np.clip(phi, 0.0, None)


def d_prime_overdispersed(lam1, lam2, phi) -> np.ndarray:
    """d' when the counts are negative binomial rather than Poisson.

    Same numerator, larger denominator: the pooled variance becomes
    lbar + phi*lbar^2 instead of lbar, so

        d' = |lam1 - lam2| / sqrt(lbar + phi * lbar^2).

    Consequence, and the sharpest prediction of Week 11: as depth grows,
    lam grows, the phi*lam^2 term dominates, and d' STOPS growing like
    sqrt(depth) — it saturates at |p1-p2|/(pbar*sqrt(phi)). Biological
    variability, not sequencing depth, becomes the limit. Deeper sequencing
    then buys nothing, which is a testable and economically real claim.
    """
    lam1 = np.asarray(lam1, dtype=float)
    lam2 = np.asarray(lam2, dtype=float)
    phi = np.asarray(phi, dtype=float)
    lbar = 0.5 * (lam1 + lam2)
    var = lbar + phi * lbar ** 2
    return np.where(var > 0, np.abs(lam1 - lam2) / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)


def surrogate_pbmc(n_cells: int = 1200, n_genes: int = 500,
                   depth: float = 2500.0, rng: np.random.Generator = None
                   ) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """[PROVIDED] A clearly-labelled stand-in for PBMC 3k.

    Two "types" with a long-tailed expression profile and gene-specific
    overdispersion, so the pipeline meets something messier than the clean
    simulator — but this is NOT data. Any figure built from it must say so;
    the experiment scripts stamp 'SURROGATE DATA' on the figure and refuse
    to write into figures/report/.
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
