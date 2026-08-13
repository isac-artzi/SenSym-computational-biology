"""Week 2 — the thinning theorem: shallower sequencing is still Poisson.

Chapter: docs/s/week02.html (Lab 1).
Figures:
  figures/week02_thinning.png   — thinned Poisson(100) at q = 0.5 overlaid on
      a direct Poisson(50) draw, plus the pmf both should follow.
  figures/week02_thinning_fano.png — Fano factor of the thinned counts across
      a range of keep probabilities: flat at 1, which is the theorem.
Needs implemented: counting.sample_counts, counting.downsample_counts,
                   counting.poisson_pmf, counting.fano_factor.
Expected runtime: ~5 s.

Why this matters: every "less data" manipulation in the project — lower
depth, dropout, fewer beads — is a thinning. If thinning broke the Poisson
structure, the theory would have to be redone for each case. It does not.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, TEAL, save_figure, stamp_seed

from celldetect.counting import (
    downsample_counts,
    fano_factor,
    poisson_pmf,
    sample_counts,
)

SEED = 102
LAM = 100.0
KEEP = 0.5
KEEP_GRID = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]


def main(fast: bool = False):
    n = 5_000 if fast else 60_000
    rng = np.random.default_rng(SEED)
    paths = []

    # --- Figure 1: two routes to the same distribution ---------------------
    deep = sample_counts(LAM, n, rng)
    thinned = downsample_counts(deep, KEEP, rng)
    direct = sample_counts(LAM * KEEP, n, rng)

    k = np.arange(0, int(LAM * KEEP + 6 * np.sqrt(LAM * KEEP)) + 2)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bins = np.arange(k[0] - 0.5, k[-1] + 1.5)
    ax.hist(thinned, bins=bins, density=True, color=BLUE, alpha=0.55,
            label=f"Poisson({LAM:g}) thinned by q={KEEP}")
    ax.hist(direct, bins=bins, density=True, histtype="step", lw=2,
            color=TEAL, label=f"Poisson({LAM * KEEP:g}) drawn directly")
    ax.plot(k, poisson_pmf(k, LAM * KEEP), "o", ms=3, color=ORANGE,
            label=f"pmf of Poisson({LAM * KEEP:g})")
    ax.set_xlabel("count")
    ax.set_ylabel("density")
    ax.set_title("Thinning a Poisson gives a Poisson — exactly, not approximately")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    stamp_seed(ax, SEED, f"{n} draws")
    paths.append(save_figure(fig, "week02_thinning"))

    # --- Figure 2: Fano stays at 1 all the way down ------------------------
    fanos, means = [], []
    for q in KEEP_GRID:
        y = downsample_counts(sample_counts(LAM, n, rng), q, rng)
        fanos.append(fano_factor(y))
        means.append(y.mean())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(KEEP_GRID, means, "o-", color=BLUE, label="measured mean")
    ax1.plot(KEEP_GRID, [LAM * q for q in KEEP_GRID], "--", color=ORANGE,
             label=r"prediction $q\lambda$")
    ax1.set_xlabel("keep probability q")
    ax1.set_ylabel("mean count")
    ax1.set_title("The mean scales by q")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(KEEP_GRID, fanos, "o-", color=BLUE, label="measured Fano")
    ax2.axhline(1.0, color=ORANGE, ls="--", lw=2, label="Poisson truth")
    ax2.set_ylim(0.9, 1.1)
    ax2.set_xlabel("keep probability q")
    ax2.set_ylabel("Fano factor")
    ax2.set_title("...and the Fano factor does not move")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    stamp_seed(ax2, SEED)
    fig.tight_layout()
    paths.append(save_figure(fig, "week02_thinning_fano"))

    print(f"[week02] seed={SEED}, lambda={LAM}, n={n}")
    print(f"[week02]   thinned: mean={thinned.mean():.3f} fano={fano_factor(thinned):.4f}")
    print(f"[week02]   direct : mean={direct.mean():.3f} fano={fano_factor(direct):.4f}")
    print(f"[week02] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
