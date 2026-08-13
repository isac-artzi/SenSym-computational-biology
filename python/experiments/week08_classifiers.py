"""Week 8 — three classifiers, one ceiling.

Chapter: docs/s/week08.html (Lab 1, Lab 2).
Figures:
  figures/week08_classifiers.png — test accuracy of naive Bayes (with true
      rates), logistic regression, and kNN across depth. The Bayes curve is
      the ceiling; the gaps are what NOT knowing the rates costs.
  figures/week08_curse.png       — the same three against the number of
      genes at fixed total signal: kNN falls apart, the linear rules do not.
Needs implemented: classify.train_test_split, accuracy, naive_bayes_predict,
                   LogisticRegression, knn_predict; the Week 5-7 functions.
Expected runtime: ~40 s (about 6 s in fast mode).

The result to report is the ORDERING and the size of the gaps, not the
absolute numbers. Tuning kNN's k until it wins would be answering a
different, less interesting question.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, PURPLE, TEAL, accuracy_axes, save_figure, stamp_seed

from celldetect.classify import (
    LogisticRegression,
    accuracy,
    knn_predict,
    naive_bayes_predict,
    train_test_split,
)
from celldetect.detection import accuracy_from_d_prime, d_prime_total
from celldetect.simulate import marker_profiles, simulate_dataset

SEED = 108
DEPTHS = np.array([250, 500, 1000, 2000, 4000, 8000], float)
GENE_COUNTS = [2, 4, 8, 16, 32, 64, 128]
FOLD = 1.5
K_NN = 15


def _evaluate(X, y, lam1, lam2, rng):
    """Return (bayes, logistic, knn) test accuracies. Labels: type 1 -> 1."""
    z = 1 - y                      # so "class 1" means "cell type 1"
    Xtr, Xte, ztr, zte = train_test_split(X, z, 0.3, rng)
    bayes = accuracy(zte, naive_bayes_predict(Xte, lam1, lam2))
    clf = LogisticRegression(lr=0.5, n_iter=1200).fit(Xtr, ztr)
    logi = accuracy(zte, clf.predict(Xte))
    knn = accuracy(zte, knn_predict(Xtr, ztr, Xte, k=K_NN))
    return bayes, logi, knn


def main(fast: bool = False):
    n_cells = 300 if fast else 1_500
    rng = np.random.default_rng(SEED)
    paths = []

    # --- Figure 1: against depth, 8 genes ----------------------------------
    k = 8
    p1, p2 = marker_profiles(k, FOLD, 0.004 * k)
    res = {"bayes": [], "logistic": [], "knn": [], "theory": []}
    for D in DEPTHS:
        X, y = simulate_dataset([p1, p2], D, n_cells, rng)
        b, l, n_ = _evaluate(X, y, D * p1, D * p2, rng)
        res["bayes"].append(b); res["logistic"].append(l); res["knn"].append(n_)
        res["theory"].append(float(accuracy_from_d_prime(d_prime_total(D * p1, D * p2))))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.semilogx(DEPTHS, res["theory"], "-", color=ORANGE, lw=2, label=r"theory $\Phi(d'/2)$")
    ax.semilogx(DEPTHS, res["bayes"], "o-", color=BLUE, label="naive Bayes (true rates)")
    ax.semilogx(DEPTHS, res["logistic"], "s-", color=TEAL, label="logistic regression")
    ax.semilogx(DEPTHS, res["knn"], "^-", color=PURPLE, label=f"kNN (k={K_NN})")
    accuracy_axes(ax, "sequencing depth D")
    ax.set_title(f"Three classifiers, {k} marker genes, {n_cells} cells/type")
    ax.legend(fontsize=9, loc="lower right")
    stamp_seed(ax, SEED, "30% test split")
    paths.append(save_figure(fig, "week08_classifiers"))

    # --- Figure 2: against the number of genes, total signal held fixed ----
    depth = 3000.0
    res2 = {"bayes": [], "logistic": [], "knn": []}
    for kk in GENE_COUNTS:
        q1, q2 = marker_profiles(kk, FOLD, 0.032)   # FIXED total mass
        X, y = simulate_dataset([q1, q2], depth, n_cells, rng)
        b, l, n_ = _evaluate(X, y, depth * q1, depth * q2, rng)
        res2["bayes"].append(b); res2["logistic"].append(l); res2["knn"].append(n_)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.semilogx(GENE_COUNTS, res2["bayes"], "o-", color=BLUE, base=2,
                label="naive Bayes (true rates)")
    ax.semilogx(GENE_COUNTS, res2["logistic"], "s-", color=TEAL, base=2,
                label="logistic regression")
    ax.semilogx(GENE_COUNTS, res2["knn"], "^-", color=PURPLE, base=2,
                label=f"kNN (k={K_NN})")
    accuracy_axes(ax, "number of genes (total signal held fixed)")
    ax.set_title("Adding uninformative dimensions: who survives?")
    ax.legend(fontsize=9, loc="lower left")
    stamp_seed(ax, SEED, f"depth {depth:.0f}")
    paths.append(save_figure(fig, "week08_curse"))

    print(f"[week08] seed={SEED}, {n_cells} cells/type, k={K_NN} for kNN")
    print("[week08]   depth   theory   bayes  logistic     knn")
    for i, D in enumerate(DEPTHS):
        print(f"[week08] {D:7.0f}   {res['theory'][i]:.4f}  {res['bayes'][i]:.4f}   "
              f"{res['logistic'][i]:.4f}  {res['knn'][i]:.4f}")
    print("[week08]   genes   bayes  logistic     knn   (total signal fixed)")
    for i, kk in enumerate(GENE_COUNTS):
        print(f"[week08] {kk:7d}  {res2['bayes'][i]:.4f}   "
              f"{res2['logistic'][i]:.4f}  {res2['knn'][i]:.4f}")
    print(f"[week08] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
