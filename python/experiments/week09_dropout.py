"""Week 9 — dropout, and where the square-root law first bends.

Chapter: docs/s/week09.html (Lab 1, Lab 2). MILESTONE M3 figure.
Figures:
  figures/week09_dropout.png   — accuracy vs depth at three dropout levels,
      with each dropout curve compared to the plain curve at the reduced
      effective depth. If the collapse is exact, the curves lie on top of
      each other and dropout is "just less depth".
  figures/week09_breakdown.png — the low-depth regime, where the Gaussian
      approximation (not the sqrt law) fails: measured accuracy against
      Phi(d'/2) and against the exact Bayes accuracy.
Also writes:
  ../progress/week09_table.md — the collapse table, ready to paste into the
      week's progress log. Commit it.

Needs implemented: everything through Week 8, plus simulate.apply_dropout,
                   detection.bayes_error_exact.
Expected runtime: ~35 s.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, PURPLE, TEAL, accuracy_axes, save_figure, stamp_seed

from celldetect.detection import (
    accuracy_from_d_prime,
    bayes_error_exact,
    d_prime_total,
    log_likelihood_ratio,
)
from celldetect.simulate import marker_profiles, simulate_dataset

SEED = 109
DEPTHS = np.array([250, 500, 1000, 2000, 4000, 8000, 16000], float)
LOW_DEPTHS = np.array([10, 20, 40, 80, 160, 320, 640, 1280], float)
DROPOUTS = [0.0, 0.3, 0.6]
K = 8
FOLD = 1.5
MASS_PER_GENE = 0.004
PROGRESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, "progress")


def _measure(p1, p2, depth, dropout, n_cells, rng):
    """Optimal-rule accuracy on simulated data with the given dropout."""
    X, y = simulate_dataset([p1, p2], depth, n_cells, rng, dropout=dropout)
    eff = depth * (1.0 - dropout)
    llr = log_likelihood_ratio(X, eff * p1, eff * p2)
    return (((llr > 0) & (y == 0)).sum() + ((llr <= 0) & (y == 1)).sum()) / len(y)


def main(fast: bool = False):
    n_cells = 500 if fast else 6_000
    rng = np.random.default_rng(SEED)
    paths = []
    p1, p2 = marker_profiles(K, FOLD, MASS_PER_GENE * K)

    # --- Figure 1: does dropout collapse onto reduced depth? ---------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    table = []
    for q, c in zip(DROPOUTS, [BLUE, TEAL, PURPLE]):
        accs = [_measure(p1, p2, D, q, n_cells, rng) for D in DEPTHS]
        ax1.semilogx(DEPTHS, accs, "o-", color=c, label=f"dropout {q:.0%}")
        ax2.semilogx(DEPTHS * (1 - q), accs, "o", color=c, label=f"dropout {q:.0%}")
        for D, a in zip(DEPTHS, accs):
            eff = D * (1 - q)
            table.append((q, D, eff, a,
                          float(accuracy_from_d_prime(d_prime_total(eff * p1, eff * p2)))))
    fine = np.geomspace(DEPTHS[0] * 0.3, DEPTHS[-1] * 1.2, 200)
    theory = [float(accuracy_from_d_prime(d_prime_total(D * p1, D * p2))) for D in fine]
    ax1.semilogx(fine, theory, "-", color=ORANGE, lw=2, label="theory (no dropout)")
    ax2.semilogx(fine, theory, "-", color=ORANGE, lw=2, label=r"theory at $D_{\rm eff}$")
    accuracy_axes(ax1, "nominal depth D")
    accuracy_axes(ax2, r"effective depth $D(1-q)$")
    ax1.set_title("Dropout costs accuracy...")
    ax2.set_title("...but only through the effective depth")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.legend(fontsize=8, loc="lower right")
    stamp_seed(ax2, SEED, f"{n_cells} cells/type, k={K}")
    fig.tight_layout()
    paths.append(save_figure(fig, "week09_dropout"))

    # --- Figure 2: the low-depth breakdown ---------------------------------
    meas, gauss, exact = [], [], []
    for D in LOW_DEPTHS:
        meas.append(_measure(p1, p2, D, 0.0, n_cells, rng))
        gauss.append(float(accuracy_from_d_prime(d_prime_total(D * p1, D * p2))))
        # Exact accuracy for k independent identical genes is not a closed
        # form, but the single-gene exact result bounds how bad the Gaussian
        # step is; we report it for the k=1 slice as the diagnostic.
        exact.append(1.0 - bayes_error_exact(D * p1[0], D * p2[0]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax1.semilogx(LOW_DEPTHS, meas, "o-", color=BLUE, label=f"measured, k={K}")
    ax1.semilogx(LOW_DEPTHS, gauss, "--", color=ORANGE, lw=2,
                 label=r"$\Phi(d'/2)$, k=%d" % K)
    accuracy_axes(ax1, "depth D (low-depth regime)")
    ax1.set_title("Where the closed form stops being right")
    ax1.legend(fontsize=9, loc="lower right")

    ax2.semilogx(LOW_DEPTHS, np.array(meas) - np.array(gauss), "o-", color=PURPLE)
    ax2.axhline(0.0, color=ORANGE, lw=2)
    ax2.axhspan(-0.01, 0.01, color=GRAY, alpha=0.15, label=r"$\pm$0.01")
    ax2.set_xlabel("depth D")
    ax2.set_ylabel("measured $-$ Gaussian prediction")
    ax2.set_title("The residual, with a sign")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)
    stamp_seed(ax2, SEED)
    fig.tight_layout()
    paths.append(save_figure(fig, "week09_breakdown"))

    # --- The collapse table, as markdown -----------------------------------
    os.makedirs(PROGRESS, exist_ok=True)
    md = os.path.abspath(os.path.join(PROGRESS, "week09_table.md"))
    with open(md, "w") as f:
        f.write("# Week 9 — dropout collapse table\n\n")
        f.write(f"Generated by `experiments/week09_dropout.py`, SEED = {SEED}, "
                f"{n_cells} cells per type, k = {K} genes, fold change {FOLD}.\n\n")
        f.write("| dropout | nominal D | effective D | measured acc | "
                "theory at D_eff | diff |\n")
        f.write("|---|---|---|---|---|---|\n")
        for q, D, eff, a, t in table:
            f.write(f"| {q:.0%} | {D:.0f} | {eff:.0f} | {a:.4f} | {t:.4f} | "
                    f"{a - t:+.4f} |\n")
        f.write("\nClaim status: [observed] the collapse holds to within the "
                "table's last column. The chapter's proof makes it [proved] "
                "for the model; whether it holds for real dropout is Week 11.\n")

    print(f"[week09] seed={SEED}, {n_cells} cells per type")
    print("[week09] dropout  nomD   effD   measured  theory(eff)   diff")
    for q, D, eff, a, t in table:
        print(f"[week09]  {q:5.0%} {D:7.0f} {eff:7.0f}    {a:.4f}    {t:.4f}  {a - t:+.4f}")
    worst = max(abs(a - t) for _, _, _, a, t in table)
    print(f"[week09] worst collapse error = {worst:.4f}")
    print(f"[week09] wrote {md}")
    print(f"[week09] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
