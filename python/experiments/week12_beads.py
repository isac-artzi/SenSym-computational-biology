"""Week 12 — the bead experiment, and all three lines of evidence together.

Chapter: docs/s/week12.html. MILESTONE M5 figure.
Figures:
  figures/week12_beads.png    — measured bead accuracy vs n with Wilson
      intervals, the theory curve, and the dropout arm.
  figures/week12_three_ways.png — THE figure of the project: d' against
      "amount of data" for theory, simulation, real data, and beads, all on
      one pair of log-log axes, each with its fitted exponent.

Data: every CSV in data/beads/ whose `run` column does not start with
RUNTEMPLATE. If no filled-in tallies exist yet, the script SIMULATES the
bead draws so the pipeline can be tested, and stamps the figure. A stamped
figure never goes in the report.

Needs implemented: beads.load_tally (provided), validate_tally,
                   accuracy_by_n, bead_d_prime, predicted_accuracy,
                   fit_power_law; stats.wilson_interval;
                   detection.normal_cdf, d_prime_from_auc.
Expected runtime: ~10 s.
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

from _common import (BLUE, DATADIR, GRAY, MAGENTA, ORANGE, TEAL,
                     accuracy_axes, save_figure, stamp_not_real, stamp_seed,
                     wilson_errorbar)

from celldetect.beads import (
    accuracy_by_n,
    bead_d_prime,
    fit_power_law,
    load_tally,
    predicted_accuracy,
    validate_tally,
)

SEED = 112
P_A, P_B = 0.30, 0.35
NS = [10, 25, 50, 100, 200]
DROPOUT = 0.30
TRIALS_PER_N = 40

# Set to True by main() when the script had to fall back to stand-in data.
# week14_report_figures.py reads this and refuses to put the figure in the
# report. Do not set it by hand.
USED_STAND_IN = False


def _accuracy_to_dprime(p):
    """Invert acc = Phi(d'/2): d' = 2 * Phi^{-1}(acc) = 2*sqrt(2)*erfinv(2acc-1).

    Accuracy at or below chance inverts to d' <= 0, which is not plottable on
    log axes; we return nan and the caller drops the point (and says so).
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    dp = 2.0 * np.sqrt(2.0) * erfinv(2.0 * p - 1.0)
    return np.where(dp > 0, dp, np.nan)


def load_all_tallies():
    """Read every non-template tally in data/beads/. Returns (records, files)."""
    records, files = [], []
    for path in sorted(glob.glob(os.path.join(DATADIR, "beads", "*.csv"))):
        try:
            rs = load_tally(path)
        except (KeyError, ValueError) as exc:
            print(f"[week12] !! could not read {os.path.basename(path)}: {exc}")
            continue
        rs = [r for r in rs if not r["run"].startswith("RUNTEMPLATE")]
        if rs:
            records.extend(rs)
            files.append(os.path.basename(path))
    return records, files


def simulate_tallies(rng):
    """Stand-in bead draws, for testing the pipeline before the real runs."""
    records, trial = [], 1
    for dropout in (0.0, DROPOUT):
        for n in NS:
            for _ in range(TRIALS_PER_N):
                jar = "A" if rng.random() < 0.5 else "B"
                p = P_A if jar == "A" else P_B
                kept = rng.binomial(n, 1 - dropout)
                red = rng.binomial(kept, p)
                # An idealized guesser using the optimal threshold.
                thresh = kept * 0.5 * (P_A + P_B)
                guess = "B" if red > thresh else "A"
                records.append({"run": "SIMULATED", "n": n, "dropout": dropout,
                                "trial": trial, "true_jar": jar, "guess": guess,
                                "correct": int(jar == guess)})
                trial += 1
    return records


def main(fast: bool = False):
    global USED_STAND_IN
    rng = np.random.default_rng(SEED)
    paths = []

    records, files = load_all_tallies()
    is_real = bool(records)
    USED_STAND_IN = not is_real
    if is_real:
        print(f"[week12] loaded {len(records)} trials from: {', '.join(files)}")
        problems = validate_tally(records)
        if problems:
            print(f"[week12] !! {len(problems)} PROBLEMS in the tally — fix before "
                  f"believing anything below:")
            for p in problems[:20]:
                print(f"[week12]    {p}")
        else:
            print("[week12] tally validates clean.")
    else:
        print("[week12] !! no filled-in tallies in data/beads/ — SIMULATING draws.")
        records = simulate_tallies(rng)

    # --- Figure 1: accuracy vs n, both arms --------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    grid = np.arange(2, 260)
    ax.plot(grid, [predicted_accuracy(int(n), P_A, P_B) for n in grid],
            "-", color=ORANGE, lw=2, label="theory, no dropout")
    ax.plot(grid, [predicted_accuracy(int(n), P_A, P_B, DROPOUT) for n in grid],
            "--", color=ORANGE, lw=2, label=f"theory, {DROPOUT:.0%} dropout")

    summary = {}
    for q, colour, marker in ((0.0, MAGENTA, "o"), (DROPOUT, TEAL, "s")):
        by_n = accuracy_by_n(records, dropout=q)
        if not by_n:
            continue
        ns = list(by_n.keys())
        ks = [by_n[n][0] for n in ns]
        tots = [by_n[n][1] for n in ns]
        wilson_errorbar(ax, ns, ks, tots, color=colour, marker=marker,
                        label=f"measured, dropout {q:.0%}")
        summary[q] = (ns, ks, tots)
    ax.set_xscale("log")
    accuracy_axes(ax, "beads drawn per trial, n")
    ax.set_title("The bead experiment" + ("" if is_real else "  (SIMULATED)"))
    ax.legend(fontsize=8, loc="upper left")
    stamp_seed(ax, SEED, f"{len(records)} trials")
    if not is_real:
        stamp_not_real(fig, "SIMULATED BEADS")
    paths.append(save_figure(fig, "week12_beads"))

    # --- Figure 2: three lines of evidence, one axis -----------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    # (1) Theory: exact, by construction a straight line of slope 1/2.
    ax.loglog(grid, [bead_d_prime(int(n), P_A, P_B) for n in grid], "-",
              color=ORANGE, lw=2, label="theory: $d' \\propto \\sqrt{n}$ (slope 0.5)")
    # (2) The bead measurements, converted to d'.
    fits = {}
    for q, colour, marker in ((0.0, MAGENTA, "o"), (DROPOUT, TEAL, "s")):
        if q not in summary:
            continue
        ns, ks, tots = summary[q]
        acc = np.array(ks, float) / np.array(tots, float)
        dps = _accuracy_to_dprime(acc)
        good = np.isfinite(dps)
        n_eff = np.array(ns, float)[good] * (1 - q)
        ax.loglog(n_eff, dps[good], marker, color=colour, ms=7,
                  label=f"beads, dropout {q:.0%}")
        if good.sum() >= 2:
            slope, intercept = fit_power_law(n_eff, dps[good])
            fits[q] = (slope, intercept, int((~good).sum()))
            ax.loglog(n_eff, np.exp(intercept) * n_eff ** slope, ":",
                      color=colour, lw=1.4)
    ax.set_xlabel("effective amount of data  (beads drawn $\\times$ (1-dropout))")
    ax.set_ylabel("separation index $d'$")
    ax.set_title("Everything, in $d'$ space, on log-log axes")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    stamp_seed(ax, SEED)
    if not is_real:
        stamp_not_real(fig, "SIMULATED BEADS")
    paths.append(save_figure(fig, "week12_three_ways"))

    print(f"[week12] seed={SEED}, real bead data: {is_real}")
    for q in sorted(summary):
        ns, ks, tots = summary[q]
        print(f"[week12] dropout {q:.0%}:")
        for n, k, t in zip(ns, ks, tots):
            print(f"[week12]    n={n:4d}  {k:3d}/{t:3d} = {k / t:.3f}   "
                  f"theory {predicted_accuracy(n, P_A, P_B, q):.3f}")
        if q in fits:
            slope, _, dropped = fits[q]
            note = f"  ({dropped} point(s) at or below chance dropped)" if dropped else ""
            print(f"[week12]    fitted exponent = {slope:.3f}  (theory 0.5){note}")
    print(f"[week12] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
