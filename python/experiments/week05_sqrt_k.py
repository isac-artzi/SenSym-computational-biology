"""Week 5 — many genes: the sqrt(k) law, and the closed-form prediction.

Chapter: docs/s/week05.html (Lab 1, Lab 2). MILESTONE M2 figure.
Figures:
  figures/week05_sqrt_k.png       — d' vs number of genes under the two
      conventions (fixed per-gene signal, and fixed total marker mass). One
      of them gives sqrt(k); the other gives a flat line. Knowing WHICH is
      the whole content of the week.
  figures/week05_prediction.png   — the Aim 1 deliverable: measured accuracy
      of the optimal rule against the predicted Phi(d'/2), over a grid of
      (depth, k). If Aim 1 is right, every point sits on the diagonal.
Needs implemented: detection.d_prime, combine_d_prime, d_prime_total,
                   accuracy_from_d_prime, log_likelihood_ratio;
                   simulate.marker_profiles, simulate_dataset.
Expected runtime: ~25 s (a few seconds in fast mode).
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, TEAL, save_figure, stamp_seed

from celldetect.detection import (
    accuracy_from_d_prime,
    d_prime_total,
    log_likelihood_ratio,
)
from celldetect.simulate import marker_profiles, simulate_dataset

SEED = 105
KS = np.array([1, 2, 4, 8, 16, 32, 64, 128])
DEPTH = 4000.0
FOLD = 1.5
MASS = 0.02
GRID_DEPTHS = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
GRID_KS = [1, 4, 16, 64]


def main(fast: bool = False):
    n_cells = 800 if fast else 8_000
    rng = np.random.default_rng(SEED)
    paths = []

    # --- Figure 1: two conventions, two very different answers -------------
    # (a) fixed TOTAL marker mass: per-gene signal falls as 1/k, quadrature
    #     restores exactly what was lost -> flat. Splitting the same evidence
    #     into more pieces gains nothing, which is obvious in hindsight and
    #     not at all obvious in advance.
    # (b) fixed PER-GENE mass: each new gene is genuinely new evidence
    #     -> sqrt(k).
    fixed_total, fixed_per_gene = [], []
    for k in KS:
        p1, p2 = marker_profiles(int(k), FOLD, MASS)
        fixed_total.append(d_prime_total(DEPTH * p1, DEPTH * p2))
        q1, q2 = marker_profiles(int(k), FOLD, MASS * k)   # mass grows with k
        fixed_per_gene.append(d_prime_total(DEPTH * q1, DEPTH * q2))
    fixed_total = np.asarray(fixed_total)
    fixed_per_gene = np.asarray(fixed_per_gene)
    slope = np.polyfit(np.log(KS), np.log(fixed_per_gene), 1)[0]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.loglog(KS, fixed_per_gene, "o-", color=BLUE,
              label=f"each gene adds new signal (slope {slope:.3f})")
    ax.loglog(KS, fixed_total, "s-", color=TEAL,
              label="same total signal, split k ways (slope 0)")
    ax.loglog(KS, fixed_per_gene[0] * np.sqrt(KS), ":", color=ORANGE, lw=2,
              label=r"$\sqrt{k}$ reference")
    ax.set_xlabel("number of marker genes k")
    ax.set_ylabel("total separation $d'$")
    ax.set_title("More genes help — but only if they carry more signal")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    stamp_seed(ax, SEED, "deterministic")
    paths.append(save_figure(fig, "week05_sqrt_k"))

    # --- Figure 2: Aim 1's deliverable -------------------------------------
    pred, meas, labels = [], [], []
    for depth in GRID_DEPTHS:
        for k in GRID_KS:
            p1, p2 = marker_profiles(k, FOLD, MASS * k)   # per-gene convention
            lam1, lam2 = depth * p1, depth * p2
            X, y = simulate_dataset([p1, p2], depth, n_cells, rng)
            llr = log_likelihood_ratio(X, lam1, lam2)
            correct = ((llr > 0) & (y == 0)).sum() + ((llr <= 0) & (y == 1)).sum()
            meas.append(correct / len(y))
            pred.append(float(accuracy_from_d_prime(d_prime_total(lam1, lam2))))
            labels.append((depth, k))
    pred = np.asarray(pred)
    meas = np.asarray(meas)
    resid = meas - pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = ax1.scatter(pred, meas, c=[np.log10(d) for d, _ in labels],
                     cmap="viridis", s=42, zorder=3)
    ax1.plot([0.45, 1.0], [0.45, 1.0], "--", color=ORANGE, lw=2, label="perfect agreement")
    ax1.set_xlabel(r"predicted accuracy  $\Phi(d'/2)$")
    ax1.set_ylabel("measured accuracy (optimal rule)")
    ax1.set_title("Aim 1, tested on 20 (depth, k) combinations")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    fig.colorbar(sc, ax=ax1, label=r"$\log_{10}$ depth")

    ax2.axhline(0.0, color=ORANGE, lw=2)
    ax2.axhspan(-0.01, 0.01, color=GRAY, alpha=0.15, label=r"$\pm$0.01")
    ax2.scatter(pred, resid, color=BLUE, s=36)
    ax2.set_xlabel("predicted accuracy")
    ax2.set_ylabel("measured $-$ predicted")
    ax2.set_title(f"Residuals: max |error| = {np.abs(resid).max():.4f}")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    stamp_seed(ax2, SEED, f"{n_cells} cells/type/point")
    fig.tight_layout()
    paths.append(save_figure(fig, "week05_prediction"))

    print(f"[week05] seed={SEED}, {n_cells} cells per type per point")
    print(f"[week05] sqrt(k) fitted slope = {slope:.5f}  (theory 0.5)")
    print("[week05]  depth    k   predicted  measured   diff")
    for (d, k), p, m in zip(labels, pred, meas):
        print(f"[week05] {d:7.0f} {k:4d}    {p:.4f}    {m:.4f}  {m - p:+.4f}")
    print(f"[week05] max |measured - predicted| = {np.abs(resid).max():.4f}")
    print(f"[week05] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
