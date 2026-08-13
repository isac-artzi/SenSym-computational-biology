"""Week 1 — counting noise: variance equals mean.

Chapter: docs/s/week01.html (Lab 1, Lab 2).
Figures:
  figures/week01_counting_noise.png  — Poisson histograms at four rates,
      each annotated with its sample mean and variance.
  figures/week01_fano.png            — the Fano factor against sample size,
      showing both that it converges to 1 and how noisy it is at n = 50.
Needs implemented: counting.sample_counts, counting.fano_factor.
Expected runtime: ~5 s.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import BLUE, GRAY, ORANGE, save_figure, stamp_seed

from celldetect.counting import fano_factor, sample_counts

SEED = 101
RATES = [1.0, 5.0, 25.0, 200.0]
SAMPLE_SIZES = [20, 50, 100, 300, 1000, 3000, 10_000]
N_REPEATS = 200


def main(fast: bool = False):
    n_samples = 2_000 if fast else 20_000
    n_repeats = 40 if fast else N_REPEATS
    rng = np.random.default_rng(SEED)
    paths = []

    # --- Figure 1: the counts themselves -----------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    for ax, lam in zip(axes.ravel(), RATES):
        x = sample_counts(lam, n_samples, rng)
        hi = int(lam + 5 * np.sqrt(lam) + 5)
        ax.hist(x, bins=np.arange(-0.5, hi + 0.5), density=True,
                color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(x.mean(), color=ORANGE, lw=2)
        ax.set_title(f"$\\lambda$ = {lam:g}   mean = {x.mean():.2f}   "
                     f"var = {x.var(ddof=1):.2f}", fontsize=10)
        ax.set_xlabel("count")
        ax.grid(alpha=0.3)
    fig.suptitle("Poisson counts: the variance tracks the mean, always")
    stamp_seed(axes.ravel()[-1], SEED, f"{n_samples} draws/panel")
    fig.tight_layout()
    paths.append(save_figure(fig, "week01_counting_noise"))

    # --- Figure 2: how well can you MEASURE the Fano factor? ---------------
    # The point of this panel is not that Fano = 1 (it does), but that with
    # 50 cells your estimate scatters by ~20% — so "Fano is 1.3" from a small
    # sample is not evidence of overdispersion. Week 11 depends on knowing
    # this in advance.
    means, los, his = [], [], []
    for n in SAMPLE_SIZES:
        fs = [fano_factor(sample_counts(20.0, n, rng)) for _ in range(n_repeats)]
        fs = np.asarray(fs)
        means.append(fs.mean())
        los.append(np.percentile(fs, 5))
        his.append(np.percentile(fs, 95))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(SAMPLE_SIZES, los, his, color=BLUE, alpha=0.2,
                    label="5th-95th percentile")
    ax.plot(SAMPLE_SIZES, means, "o-", color=BLUE, label="mean estimate")
    ax.axhline(1.0, color=ORANGE, lw=2, ls="--", label="Poisson truth (1.0)")
    ax.set_xscale("log")
    ax.set_xlabel("number of cells in the sample")
    ax.set_ylabel("measured Fano factor (var / mean)")
    ax.set_title("A Fano factor from 50 cells is worth about $\\pm$20%")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    stamp_seed(ax, SEED, f"{n_repeats} repeats/point, $\\lambda$=20")
    paths.append(save_figure(fig, "week01_fano"))

    print(f"[week01] seed={SEED}, {n_samples} draws per rate")
    for lam in RATES:
        x = sample_counts(lam, n_samples, np.random.default_rng(SEED))
        print(f"[week01]   lambda={lam:6.1f}  mean={x.mean():8.3f}  "
              f"var={x.var(ddof=1):8.3f}  fano={fano_factor(x):.4f}")
    print(f"[week01] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
