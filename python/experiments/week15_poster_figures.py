"""Week 15 — the poster panel and the one-slide summary.

Chapter: docs/s/week15.html. MILESTONE M8 (`v1.0` tag).
Writes into figures/poster/:
    P1_question.png    the phenomenon and the question, in one panel
    P2_result.png      the headline: all lines of evidence, one axis
    P3_limits.png      where it fails — the panel that earns the trust

Poster figures are NOT the report figures scaled up. They carry larger type,
fewer curves, and one sentence of conclusion inside the axes, because a
poster is read from a metre away by someone who will give it thirty seconds.

Needs implemented: everything.
Expected runtime: ~20 s.
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from _common import (BLUE, FIGDIR, GRAY, MAGENTA, ORANGE, PURPLE, TEAL,
                     save_figure, stamp_seed)

from celldetect.beads import bead_d_prime, predicted_accuracy
from celldetect.counting import sample_counts
from celldetect.detection import (
    accuracy_from_d_prime,
    d_prime,
    d_prime_total,
    log_likelihood_ratio,
)
from celldetect.simulate import marker_profiles, simulate_dataset

SEED = 115
POSTER_RC = {
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "lines.linewidth": 2.6,
    "lines.markersize": 9,
}


def main(fast: bool = False):
    n_cells = 600 if fast else 6_000
    rng = np.random.default_rng(SEED)
    paths = []

    with mpl.rc_context(POSTER_RC):
        # --- P1: the question ----------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for lam, c in ((4.0, BLUE), (40.0, TEAL)):
            x = sample_counts(lam, 20_000, rng)
            ax1.hist(x, bins=np.arange(-0.5, lam + 5 * np.sqrt(lam) + 5),
                     density=True, color=c, alpha=0.65,
                     label=f"$\\lambda$={lam:g}: var/mean = {x.var(ddof=1) / x.mean():.2f}")
        ax1.set_xlabel("molecules counted")
        ax1.set_ylabel("density")
        ax1.set_title("A sequencer counts; counting is noisy")
        ax1.legend()
        ax1.grid(alpha=0.3)

        depths = np.geomspace(50, 20000, 60)
        for k, c in ((1, BLUE), (10, TEAL), (100, PURPLE)):
            p1, p2 = marker_profiles(k, 1.5, 0.004 * k)
            ax2.semilogx(depths,
                         [float(accuracy_from_d_prime(d_prime_total(D * p1, D * p2)))
                          for D in depths], color=c, label=f"{k} marker gene(s)")
        ax2.axhline(0.5, color=GRAY, ls=":", lw=1.5)
        ax2.set_ylim(0.45, 1.02)
        ax2.set_xlabel("sequencing depth per cell")
        ax2.set_ylabel("accuracy")
        ax2.set_title("How much data to tell two cell types apart?")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3, which="both")
        stamp_seed(ax2, SEED)
        fig.tight_layout()
        paths.append(save_figure(fig, "P1_question", subdir="poster"))

        # --- P2: the result --------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6.5))
        x = np.geomspace(1, 3000, 120)
        ax.loglog(x, np.sqrt(x) * 0.1, "-", color=ORANGE,
                  label=r"prediction: $d' \propto \sqrt{\rm data}$")
        # Simulation, in d' space.
        p1, p2 = marker_profiles(8, 1.5, 0.032)
        sim_depths = np.array([100, 300, 1000, 3000, 10000], float)
        sim_dp = []
        for D in sim_depths:
            X, y = simulate_dataset([p1, p2], D, n_cells, rng)
            llr = log_likelihood_ratio(X, D * p1, D * p2)
            acc = np.where(y == 0, llr > 0, llr <= 0).mean()
            from scipy.special import erfinv
            sim_dp.append(2 * np.sqrt(2) * erfinv(2 * np.clip(acc, 1e-9, 1 - 1e-9) - 1))
        # Rescale the x-axis of each line of evidence to a common "amount of
        # data" so the SLOPES can be compared; the offsets are meaningless
        # and the caption says so.
        ax.loglog(sim_depths / sim_depths[0], np.array(sim_dp) / sim_dp[0] * 0.1 * 1.0,
                  "o", color=BLUE, label="simulation (rescaled)")
        ns = np.array([10, 25, 50, 100, 200], float)
        bd = np.array([bead_d_prime(int(n), 0.30, 0.35) for n in ns])
        ax.loglog(ns / ns[0], bd / bd[0] * 0.1, "s", color=MAGENTA,
                  label="bead experiment (rescaled)")
        ax.set_xlabel("relative amount of data")
        ax.set_ylabel("separation $d'$  (rescaled to a common start)")
        ax.set_title("Four times the data, twice the separation")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left")
        ax.text(0.97, 0.06, "slope 1/2, three independent ways",
                transform=ax.transAxes, ha="right", fontsize=13, color=ORANGE,
                weight="bold")
        stamp_seed(ax, SEED)
        paths.append(save_figure(fig, "P2_result", subdir="poster"))

        # --- P3: the limits --------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6))
        depths = np.geomspace(50, 3e5, 80)
        p1s, p2s = 0.004, 0.006
        pois = [float(d_prime(D * p1s, D * p2s)) for D in depths]
        for phi, c in ((0.0, ORANGE), (0.1, TEAL), (0.5, PURPLE)):
            lbar = 0.5 * (depths * p1s + depths * p2s)
            dp = abs(p1s - p2s) * depths / np.sqrt(lbar + phi * lbar ** 2)
            ax.loglog(depths, dp, color=c, label=f"$\\phi$ = {phi}")
        ax.set_xlabel("sequencing depth per cell")
        ax.set_ylabel("separation $d'$")
        ax.set_title("Biological variability puts a ceiling on depth")
        ax.grid(alpha=0.3, which="both")
        ax.legend(title="dispersion", loc="upper left")
        ax.text(0.97, 0.06, "past the knee, deeper sequencing buys nothing",
                transform=ax.transAxes, ha="right", fontsize=13, color=PURPLE,
                weight="bold")
        stamp_seed(ax, SEED, "deterministic")
        paths.append(save_figure(fig, "P3_limits", subdir="poster"))

    print(f"[week15] seed={SEED}")
    print(f"[week15] poster figures in {os.path.join(FIGDIR, 'poster')}")
    print("[week15] REMINDER: P2's offsets are rescaled and meaningless; only "
          "the slopes are comparable. Say that on the poster, out loud, in the "
          "caption — a reviewer who spots it before you do has found your "
          "weakest moment.")
    print(f"[week15] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
