"""Week 7 — the simulator's first accuracy-vs-depth curve.

Chapter: docs/s/week07.html (Lab 1, Lab 2).
Figures:
  figures/week07_simulator.png — measured accuracy vs depth at three values
      of k, with the Aim 1 theory curve drawn through each. The first time
      simulation and theory appear on the same axes.
  figures/week07_cells.png     — what the simulated data actually looks like:
      a counts heatmap and the two-gene scatter for a handful of cells.
Needs implemented: simulate.marker_profiles, simulate_cells, simulate_dataset;
                   detection.d_prime_total, accuracy_from_d_prime,
                   log_likelihood_ratio.
Expected runtime: ~20 s.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, TEAL, accuracy_axes, save_figure, stamp_seed

from celldetect.detection import (
    accuracy_from_d_prime,
    d_prime_total,
    log_likelihood_ratio,
)
from celldetect.simulate import marker_profiles, simulate_cells, simulate_dataset

SEED = 107
DEPTHS = np.array([250, 500, 1000, 2000, 4000, 8000, 16000], float)
KS = [1, 4, 16]
FOLD = 1.5
MASS_PER_GENE = 0.004


def main(fast: bool = False):
    n_cells = 500 if fast else 5_000
    rng = np.random.default_rng(SEED)
    paths = []

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fine = np.geomspace(DEPTHS[0] * 0.8, DEPTHS[-1] * 1.2, 200)
    rows = []
    for k, c in zip(KS, [BLUE, TEAL, ORANGE]):
        p1, p2 = marker_profiles(k, FOLD, MASS_PER_GENE * k)
        theory = [float(accuracy_from_d_prime(d_prime_total(D * p1, D * p2)))
                  for D in fine]
        ax.semilogx(fine, theory, "-", color=c, lw=1.8, alpha=0.85)
        measured = []
        for D in DEPTHS:
            X, y = simulate_dataset([p1, p2], D, n_cells, rng)
            llr = log_likelihood_ratio(X, D * p1, D * p2)
            acc = (((llr > 0) & (y == 0)).sum() + ((llr <= 0) & (y == 1)).sum()) / len(y)
            measured.append(acc)
            rows.append((k, D, acc,
                         float(accuracy_from_d_prime(d_prime_total(D * p1, D * p2)))))
        ax.semilogx(DEPTHS, measured, "o", color=c, ms=6, label=f"k = {k} genes")
    accuracy_axes(ax, "sequencing depth D (molecules per cell)")
    ax.set_title("Simulation (points) against the Week 5 prediction (lines)")
    ax.legend(fontsize=9, loc="lower right")
    stamp_seed(ax, SEED, f"{n_cells} cells/type/point")
    paths.append(save_figure(fig, "week07_simulator"))

    # --- What the data looks like ------------------------------------------
    p1, p2 = marker_profiles(20, FOLD, MASS_PER_GENE * 20)
    X1 = simulate_cells(p1, 3000.0, 30, rng)
    X2 = simulate_cells(p2, 3000.0, 30, rng)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    im = ax1.imshow(np.vstack([X1, X2]), aspect="auto", cmap="magma",
                    interpolation="nearest")
    ax1.axhline(29.5, color="white", lw=1.5)
    ax1.set_xlabel("marker gene")
    ax1.set_ylabel("cell (type 1 above, type 2 below)")
    ax1.set_title("Simulated counts, depth 3000, 20 markers")
    fig.colorbar(im, ax=ax1, label="count")

    ax2.scatter(X1[:, 0], X1[:, 1], color=BLUE, s=28, label="type 1", alpha=0.8)
    ax2.scatter(X2[:, 0], X2[:, 1], color=TEAL, s=28, label="type 2", alpha=0.8)
    ax2.set_xlabel("count, gene 1")
    ax2.set_ylabel("count, gene 2")
    ax2.set_title("Two genes at a time: heavily overlapping")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    stamp_seed(ax2, SEED)
    fig.tight_layout()
    paths.append(save_figure(fig, "week07_cells"))

    print(f"[week07] seed={SEED}, {n_cells} cells per type per point")
    print("[week07]   k    depth   measured  predicted    diff")
    worst = 0.0
    for k, D, m, p in rows:
        worst = max(worst, abs(m - p))
        print(f"[week07] {k:3d} {D:8.0f}    {m:.4f}    {p:.4f}  {m - p:+.4f}")
    print(f"[week07] max |measured - predicted| = {worst:.4f}")
    print(f"[week07] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
