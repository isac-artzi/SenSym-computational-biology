"""Week 10 — real data I: choosing markers without fooling yourself.

Chapter: docs/s/week10.html (Lab 1, Lab 2). MILESTONE M4 begins here.
Figures:
  figures/week10_markers.png  — accuracy vs number of markers, selected two
      ways: on the full dataset (leaky) and on the training split only
      (honest). The gap between them is the size of the lie.
  figures/week10_ranking.png  — fold-change ranking against d' ranking:
      the same genes in a different order, and why.

Runs on the cached PBMC 3k if present (see fetch_pbmc3k.py), otherwise on
the SURROGATE dataset with a loud stamp. A stamped figure is not a result.

Needs implemented: realdata.select_markers, downsample_matrix,
                   estimate_rates; the Week 8 classifiers.
Expected runtime: ~30 s.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import (BLUE, GRAY, ORANGE, RED, TEAL, accuracy_axes, save_figure,
                     stamp_not_real, stamp_seed)

from celldetect.classify import LogisticRegression, accuracy, train_test_split
from celldetect.detection import d_prime
from celldetect.realdata import (
    estimate_rates,
    load_pbmc,
    select_markers,
    surrogate_pbmc,
)

SEED = 110
MARKER_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]

# Set to True by main() when the script had to fall back to stand-in data.
# week14_report_figures.py reads this and refuses to put the figure in the
# report. Do not set it by hand.
USED_STAND_IN = False


def load_data(rng, fast=False):
    """Return (X, y, genes, types, is_real)."""
    got = load_pbmc(allow_missing=True)
    if got is not None:
        X, y, genes, types = got
        return X, y, genes, types, True
    print("[week10] !! no PBMC cache — falling back to SURROGATE data.")
    print("[week10] !! run `python experiments/fetch_pbmc3k.py` for the real thing.")
    X, y, genes, types = surrogate_pbmc(n_cells=600 if fast else 1200,
                                        n_genes=300 if fast else 500, rng=rng)
    return X, y, genes, types, False


def main(fast: bool = False):
    global USED_STAND_IN
    rng = np.random.default_rng(SEED)
    paths = []
    X, y, genes, types, is_real = load_data(rng, fast)
    USED_STAND_IN = not is_real
    print(f"[week10] dataset: {X.shape[0]} cells x {X.shape[1]} genes, "
          f"types={types}, real={is_real}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng)

    leaky, honest = [], []
    for m in MARKER_COUNTS:
        # LEAKY: pick markers using every cell, including the test cells.
        idx_all = select_markers(X, y, m)
        clf = LogisticRegression(lr=0.5, n_iter=800).fit(Xtr[:, idx_all], ytr)
        leaky.append(accuracy(yte, clf.predict(Xte[:, idx_all])))
        # HONEST: pick markers using the training split only.
        idx_tr = select_markers(Xtr, ytr, m)
        clf = LogisticRegression(lr=0.5, n_iter=800).fit(Xtr[:, idx_tr], ytr)
        honest.append(accuracy(yte, clf.predict(Xte[:, idx_tr])))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.semilogx(MARKER_COUNTS, leaky, "o--", color=RED, base=2,
                label="markers chosen on ALL cells (leaky)")
    ax.semilogx(MARKER_COUNTS, honest, "o-", color=TEAL, base=2,
                label="markers chosen on the training split")
    accuracy_axes(ax, "number of marker genes")
    ax.set_title("Selection bias, measured rather than assumed")
    ax.legend(fontsize=9, loc="lower right")
    stamp_seed(ax, SEED, "real data" if is_real else "SURROGATE")
    if not is_real:
        stamp_not_real(fig)
    paths.append(save_figure(fig, "week10_markers"))

    # --- Two rankings of the same genes ------------------------------------
    rates = estimate_rates(X, y)
    dp = np.asarray(d_prime(rates[0], rates[1]))
    lfc = np.abs(np.log2((rates[0] + 1.0) / (rates[1] + 1.0)))
    top_fc = select_markers(X, y, 40)
    top_dp = np.argsort(-dp)[:40]
    overlap = len(set(top_fc.tolist()) & set(top_dp.tolist()))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(lfc, dp, s=14, color=GRAY, alpha=0.5, label="all genes")
    ax.scatter(lfc[top_fc], dp[top_fc], s=34, color=ORANGE,
               label="top 40 by fold change")
    ax.scatter(lfc[top_dp], dp[top_dp], s=18, color=BLUE,
               label="top 40 by $d'$")
    ax.set_xlabel(r"$|\log_2$ fold change$|$")
    ax.set_ylabel("per-gene $d'$ at the observed depth")
    ax.set_title(f"Fold change is not $d'$ — the top-40 lists overlap in {overlap}/40")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    stamp_seed(ax, SEED, "real data" if is_real else "SURROGATE")
    if not is_real:
        stamp_not_real(fig)
    paths.append(save_figure(fig, "week10_ranking"))

    print(f"[week10] seed={SEED}")
    print("[week10]  markers   leaky   honest    inflation")
    for m, a, b in zip(MARKER_COUNTS, leaky, honest):
        print(f"[week10] {m:8d}  {a:.4f}  {b:.4f}    {a - b:+.4f}")
    print(f"[week10] top-40 overlap between fold-change and d' rankings: {overlap}/40")
    print(f"[week10] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
