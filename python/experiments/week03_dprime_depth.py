"""Week 3 — d' against depth: the square-root law, measured.

Chapter: docs/s/week03.html (Lab 1, Lab 2).
Figures:
  figures/week03_dprime_depth.png  — d' vs depth on log-log axes with the
      fitted slope printed on the plot. The claim is the SLOPE, not the plot.
  figures/week03_overlap.png       — the two count distributions at three
      depths, showing what "d' = 1" and "d' = 4" actually look like.
Needs implemented: detection.d_prime, counting.sample_counts.
Expected runtime: ~4 s.

Milestone M1 depends on this figure and on your being able to derive its
slope at the whiteboard without notes.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, TEAL, save_figure, stamp_seed

from celldetect.counting import sample_counts
from celldetect.detection import d_prime

SEED = 103
P1, P2 = 0.0040, 0.0055          # per-molecule rates of one marker gene
DEPTHS = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600], float)
SHOW_DEPTHS = [200, 1600, 12800]


def main(fast: bool = False):
    n = 3_000 if fast else 40_000
    rng = np.random.default_rng(SEED)
    paths = []

    # --- Figure 1: the law -------------------------------------------------
    # Predicted d' straight from the formula...
    predicted = np.array([float(d_prime(D * P1, D * P2)) for D in DEPTHS])
    # ...and measured d' from actual samples: (mean1 - mean2) / pooled sd.
    measured = []
    for D in DEPTHS:
        x1 = sample_counts(D * P1, n, rng)
        x2 = sample_counts(D * P2, n, rng)
        pooled_sd = np.sqrt(0.5 * (x1.var(ddof=1) + x2.var(ddof=1)))
        measured.append(abs(x1.mean() - x2.mean()) / pooled_sd)
    measured = np.asarray(measured)

    slope, intercept = np.polyfit(np.log(DEPTHS), np.log(measured), 1)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.loglog(DEPTHS, predicted, "-", color=ORANGE, lw=2,
              label=r"theory: $d' = \sqrt{D}\,|p_1-p_2|/\sqrt{\bar p}$")
    ax.loglog(DEPTHS, measured, "o", color=BLUE, ms=6, label="measured from samples")
    ax.loglog(DEPTHS, np.exp(intercept) * DEPTHS ** slope, ":", color=GRAY,
              label=f"fit: slope = {slope:.4f}")
    ax.set_xlabel("sequencing depth D (molecules per cell)")
    ax.set_ylabel("separation index $d'$")
    ax.set_title("Quadruple the depth, double the separation")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    stamp_seed(ax, SEED, f"{n} cells/point")
    paths.append(save_figure(fig, "week03_dprime_depth"))

    # --- Figure 2: what d' looks like --------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, D in zip(axes, SHOW_DEPTHS):
        l1, l2 = D * P1, D * P2
        x1 = sample_counts(l1, n, rng)
        x2 = sample_counts(l2, n, rng)
        hi = int(max(l1, l2) + 4 * np.sqrt(max(l1, l2)) + 5)
        bins = np.arange(-0.5, hi + 0.5)
        ax.hist(x1, bins=bins, density=True, color=BLUE, alpha=0.55, label="type 1")
        ax.hist(x2, bins=bins, density=True, color=TEAL, alpha=0.55, label="type 2")
        ax.set_title(f"D = {D}   $d'$ = {float(d_prime(l1, l2)):.2f}", fontsize=11)
        ax.set_xlabel("count of one marker gene")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Same two cell types, three sequencing depths")
    stamp_seed(axes[-1], SEED)
    fig.tight_layout()
    paths.append(save_figure(fig, "week03_overlap"))

    print(f"[week03] seed={SEED}, p1={P1}, p2={P2}, {n} cells per point")
    print(f"[week03] fitted slope = {slope:.5f}  (theory: 0.5)")
    for D, pr, me in zip(DEPTHS, predicted, measured):
        print(f"[week03]   D={D:7.0f}  predicted d'={pr:6.3f}  measured d'={me:6.3f}")
    print(f"[week03] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
