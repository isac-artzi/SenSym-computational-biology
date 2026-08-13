"""Week 4 — from d' to an error probability, and where the formula breaks.

Chapter: docs/s/week04.html (Lab 1, Lab 2, Lab 3).
Figures:
  figures/week04_accuracy_curve.png — Phi(d'/2) against the EXACT Bayes
      accuracy computed by summation, over a range of rates. They agree at
      high rate and part company at low rate — the first honest limitation
      of Aim 1.
  figures/week04_roc.png            — ROC curves at four depths, with AUC
      printed, and the d' recovered from each AUC compared to the true d'.
Needs implemented: detection.normal_cdf, d_prime, accuracy_from_d_prime,
                   bayes_error_exact, roc_curve, roc_auc, d_prime_from_auc;
                   counting.sample_counts.
Expected runtime: ~10 s.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, TEAL, PURPLE, save_figure, stamp_seed

from celldetect.counting import sample_counts
from celldetect.detection import (
    accuracy_from_d_prime,
    bayes_error_exact,
    d_prime,
    d_prime_from_auc,
    roc_auc,
    roc_curve,
)

SEED = 104
FOLD = 1.4                        # lam1 = FOLD * lam2
BASE_RATES = np.geomspace(0.2, 300.0, 26)
ROC_RATES = [(1.0, 1.4), (5.0, 7.0), (25.0, 35.0), (100.0, 140.0)]


def main(fast: bool = False):
    n = 4_000 if fast else 40_000
    rng = np.random.default_rng(SEED)
    paths = []

    # --- Figure 1: Gaussian approximation vs exact -------------------------
    approx, exact = [], []
    for lam2 in BASE_RATES:
        lam1 = FOLD * lam2
        approx.append(float(accuracy_from_d_prime(float(d_prime(lam1, lam2)))))
        exact.append(1.0 - bayes_error_exact(lam1, lam2))
    approx = np.asarray(approx)
    exact = np.asarray(exact)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.semilogx(BASE_RATES, exact, "-", color=TEAL, lw=2,
                 label="exact (sum over all counts)")
    ax1.semilogx(BASE_RATES, approx, "--", color=ORANGE, lw=2,
                 label=r"Gaussian: $\Phi(d'/2)$")
    ax1.axhline(0.5, color=GRAY, ls=":", lw=1)
    ax1.set_xlabel(r"expected count of the weaker type, $\lambda_2$")
    ax1.set_ylabel("accuracy of the optimal rule")
    ax1.set_title(f"Fold change {FOLD}: two ways to compute the same number")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=9)

    ax2.semilogx(BASE_RATES, approx - exact, "o-", color=PURPLE, ms=4)
    ax2.axhline(0.0, color=GRAY, lw=1)
    ax2.axhspan(-0.01, 0.01, color=GRAY, alpha=0.15)
    ax2.set_xlabel(r"$\lambda_2$")
    ax2.set_ylabel("Gaussian $-$ exact")
    ax2.set_title("The approximation error, and where it stops mattering")
    ax2.grid(alpha=0.3, which="both")
    stamp_seed(ax2, SEED, "deterministic")
    fig.tight_layout()
    paths.append(save_figure(fig, "week04_accuracy_curve"))

    # --- Figure 2: ROC curves and d' recovered from AUC --------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    rows = []
    colors = [BLUE, TEAL, PURPLE, ORANGE]
    for (lam1, lam2), c in zip(ROC_RATES, colors):
        x1 = sample_counts(lam1, n, rng)
        x2 = sample_counts(lam2, n, rng)
        # Score: higher count favours the higher-rate type.
        scores = np.concatenate([x1, x2]).astype(float)
        labels = np.concatenate([np.ones(n, int), np.zeros(n, int)])
        fpr, tpr = roc_curve(scores, labels)
        auc = roc_auc(scores, labels)
        true_dp = float(d_prime(lam1, lam2))
        rec_dp = d_prime_from_auc(auc)
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f"$\\lambda$=({lam1:g},{lam2:g})  AUC={auc:.3f}")
        rows.append((lam1, lam2, auc, true_dp, rec_dp))
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("One gene, four depths")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    stamp_seed(ax, SEED, f"{n} cells/class")
    paths.append(save_figure(fig, "week04_roc"))

    print(f"[week04] seed={SEED}, {n} cells per class")
    print("[week04]  lam1   lam2     AUC   d'(true)  d'(from AUC)")
    for lam1, lam2, auc, td, rd in rows:
        print(f"[week04] {lam1:6.1f} {lam2:6.1f}  {auc:6.4f}   {td:6.3f}      {rd:6.3f}")
    worst = np.max(np.abs(approx - exact))
    print(f"[week04] max |Gaussian - exact| over the sweep: {worst:.4f}")
    print(f"[week04] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
