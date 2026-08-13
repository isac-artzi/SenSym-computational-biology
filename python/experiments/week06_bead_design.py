"""Week 6 — designing the bead experiment BEFORE running it.

Chapter: docs/s/week06.html. Protocol: docs/extras/bead-protocol.html.
Figures:
  figures/week06_bead_design.png — predicted accuracy vs n for several jar
      contrasts, with the achievable-precision band from the planned number
      of trials. This is what tells you whether the experiment can possibly
      detect what you want it to detect.
Also writes:
  data/beads/RUNTEMPLATE_plain.csv    — pre-randomized tally sheets, ready to
  data/beads/RUNTEMPLATE_dropout.csv    print. Fill in `guess` and `correct`.

Needs implemented: beads.bead_d_prime, beads.predicted_accuracy;
                   detection.normal_cdf, accuracy_from_d_prime.
Expected runtime: < 2 s.

Read the printed table before you buy any beads. If the design says 30/35
needs 350 trials per point to resolve, and you have time for 40, the honest
move is to change the jars (a bigger contrast), not to run 40 and hope.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from _common import DATADIR, GRAY, MAGENTA, ORANGE, TEAL, save_figure, stamp_seed

from celldetect.beads import (
    bead_d_prime,
    predicted_accuracy,
    trials_needed,
    write_tally_template,
)

SEED = 106
NS = [10, 25, 50, 100, 200]
CONTRASTS = [(0.30, 0.35), (0.30, 0.40), (0.30, 0.50)]
DROPOUT = 0.30
TRIALS_PER_N = 40


def main(fast: bool = False):
    rng = np.random.default_rng(SEED)
    paths = []
    grid = np.arange(1, 401)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for (pa, pb), c in zip(CONTRASTS, [MAGENTA, TEAL, ORANGE]):
        ax1.plot(grid, [predicted_accuracy(int(n), pa, pb) for n in grid],
                 color=c, lw=2, label=f"jars {pa:.0%} vs {pb:.0%}")
    ax1.axhline(0.5, color=GRAY, ls=":", lw=1)
    # The resolution the planned trial count buys, drawn as a band around the
    # 30/35 curve: if the curve's rise across the n-range is inside the band,
    # the experiment cannot see it.
    half = 1.96 * np.sqrt(0.25 / TRIALS_PER_N)
    base = np.array([predicted_accuracy(int(n), *CONTRASTS[0]) for n in grid])
    ax1.fill_between(grid, base - half, base + half, color=MAGENTA, alpha=0.15,
                     label=f"$\\pm$95% CI at {TRIALS_PER_N} trials/point")
    for n in NS:
        ax1.axvline(n, color=GRAY, lw=0.6, alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel("beads drawn per trial, n")
    ax1.set_ylabel("predicted accuracy")
    ax1.set_ylim(0.45, 1.0)
    ax1.set_title("Can this experiment see anything?")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="upper left")

    # d' is where the law lives: straight lines of slope 1/2 on log-log.
    for (pa, pb), c in zip(CONTRASTS, [MAGENTA, TEAL, ORANGE]):
        ax2.loglog(grid, [bead_d_prime(int(n), pa, pb) for n in grid],
                   color=c, lw=2, label=f"{pa:.0%} vs {pb:.0%}")
    ax2.loglog(grid, [bead_d_prime(int(n), *CONTRASTS[0], dropout=DROPOUT) for n in grid],
               "--", color=MAGENTA, lw=2,
               label=f"30/35 with {DROPOUT:.0%} dropout")
    ax2.set_xlabel("beads drawn per trial, n")
    ax2.set_ylabel("predicted $d'$")
    ax2.set_title("The same square-root law, on a kitchen table")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="upper left")
    stamp_seed(ax2, SEED, "deterministic")
    fig.tight_layout()
    paths.append(save_figure(fig, "week06_bead_design"))

    # --- Tally templates ---------------------------------------------------
    beaddir = os.path.join(DATADIR, "beads")
    os.makedirs(beaddir, exist_ok=True)
    t1 = write_tally_template(os.path.join(beaddir, "RUNTEMPLATE_plain.csv"),
                              "RUNTEMPLATE-plain", NS, TRIALS_PER_N, 0.0, rng)
    t2 = write_tally_template(os.path.join(beaddir, "RUNTEMPLATE_dropout.csv"),
                              "RUNTEMPLATE-dropout", NS, TRIALS_PER_N, DROPOUT, rng)

    print(f"[week06] seed={SEED}")
    print(f"[week06] budget check: {len(NS)} draw sizes x {TRIALS_PER_N} trials "
          f"x 2 arms = {len(NS) * TRIALS_PER_N * 2} trials")
    print(f"[week06] total beads drawn ~ {2 * TRIALS_PER_N * sum(NS)}")
    print("[week06]    n   acc(30/35)  acc(+dropout)   trials for +/-0.05")
    for n in NS:
        a = predicted_accuracy(n, *CONTRASTS[0])
        b = predicted_accuracy(n, *CONTRASTS[0], dropout=DROPOUT)
        print(f"[week06] {n:5d}    {a:.4f}      {b:.4f}         "
              f"{trials_needed(a, 0.05):5d}")
    print(f"[week06] wrote templates: {t1}, {t2}")
    print(f"[week06] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
