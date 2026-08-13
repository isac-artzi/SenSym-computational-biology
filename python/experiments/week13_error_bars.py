"""Week 13 — error bars, and an honest divergence table.

Chapter: docs/s/week13.html. MILESTONE M6 figure.
Figures:
  figures/week13_intervals.png — Wilson vs the naive normal interval across
      the whole range of proportions, at the trial counts this project
      actually has. Where the naive one leaves [0,1] is where the report
      would have been wrong.
  figures/week13_divergence.png — measured minus predicted, for every
      regime the project covers, with intervals. The figure that decides
      which claims survive into the report.
Also writes:
  ../progress/week13_claims.md — the claim ledger, every row labelled
      proved / cited / observed / conjectured. Edit it by hand afterwards;
      the script only fills in the numbers.

Needs implemented: stats.wilson_interval, bootstrap_ci, permutation_test;
                   everything from Weeks 5-9.
Expected runtime: ~30 s.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from _common import (BLUE, GRAY, ORANGE, PURPLE, RED, TEAL, save_figure,
                     stamp_seed)

from celldetect.detection import (
    accuracy_from_d_prime,
    d_prime_total,
    log_likelihood_ratio,
)
from celldetect.simulate import marker_profiles, simulate_dataset
from celldetect.stats import bootstrap_ci, permutation_test, wilson_interval

SEED = 113
TRIAL_COUNTS = [20, 40, 200]
REGIMES = [
    # (label, depth, k, dropout)
    ("very low depth", 50.0, 8, 0.0),
    ("low depth", 250.0, 8, 0.0),
    ("mid depth", 2000.0, 8, 0.0),
    ("high depth", 16000.0, 8, 0.0),
    ("many genes", 2000.0, 64, 0.0),
    ("heavy dropout", 2000.0, 8, 0.7),
]
PROGRESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, "progress")


def main(fast: bool = False):
    n_cells = 400 if fast else 4_000
    n_boot = 200 if fast else 2_000
    rng = np.random.default_rng(SEED)
    paths = []

    # --- Figure 1: the two intervals ---------------------------------------
    ps = np.linspace(0.0, 1.0, 41)
    fig, axes = plt.subplots(1, len(TRIAL_COUNTS), figsize=(13, 4),
                             sharey=True)
    for ax, n in zip(axes, TRIAL_COUNTS):
        ks = np.round(ps * n).astype(int)
        wil = np.array([wilson_interval(int(k), n) for k in ks])
        phat = ks / n
        naive_half = 1.96 * np.sqrt(phat * (1 - phat) / n)
        ax.fill_between(phat, wil[:, 0], wil[:, 1], color=BLUE, alpha=0.3,
                        label="Wilson")
        ax.plot(phat, phat - naive_half, "--", color=RED, lw=1.5, label="naive normal")
        ax.plot(phat, phat + naive_half, "--", color=RED, lw=1.5)
        ax.axhspan(-0.35, 0.0, color=GRAY, alpha=0.25)
        ax.axhspan(1.0, 1.35, color=GRAY, alpha=0.25)
        ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
        ax.set_xlabel("observed proportion")
        ax.set_title(f"n = {n} trials")
        ax.set_ylim(-0.35, 1.35)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("95% interval")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Grey = impossible probabilities. The naive interval goes there.")
    stamp_seed(axes[-1], SEED, "deterministic")
    fig.tight_layout()
    paths.append(save_figure(fig, "week13_intervals"))

    # --- Figure 2: divergence, with intervals ------------------------------
    labels, diffs, los, his, rows = [], [], [], [], []
    for label, depth, k, drop in REGIMES:
        p1, p2 = marker_profiles(k, 1.5, 0.004 * k)
        X, y = simulate_dataset([p1, p2], depth, n_cells, rng, dropout=drop)
        eff = depth * (1 - drop)
        llr = log_likelihood_ratio(X, eff * p1, eff * p2)
        correct = np.where(y == 0, llr > 0, llr <= 0).astype(float)
        measured = correct.mean()
        predicted = float(accuracy_from_d_prime(d_prime_total(eff * p1, eff * p2)))
        lo, hi = wilson_interval(int(correct.sum()), len(correct))
        labels.append(f"{label}\nD={depth:.0f}, k={k}" + (f", q={drop:.0%}" if drop else ""))
        diffs.append(measured - predicted)
        los.append(lo - predicted)
        his.append(hi - predicted)
        # A bootstrap over cells, to show it agrees with Wilson here — and to
        # have the number in hand when the report claims they agree.
        bl, bh = bootstrap_ci(correct, np.mean, n_boot, rng)
        rows.append((label, depth, k, drop, measured, predicted, lo, hi, bl, bh))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = np.arange(len(labels))
    ax.axhline(0.0, color=ORANGE, lw=2, label="theory")
    ax.axhspan(-0.01, 0.01, color=GRAY, alpha=0.18, label=r"$\pm$0.01 band")
    ax.errorbar(xs, diffs,
                yerr=[np.array(diffs) - np.array(los), np.array(his) - np.array(diffs)],
                fmt="o", color=PURPLE, capsize=4, ms=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("measured $-$ predicted accuracy")
    ax.set_title("Where the closed form holds, and where it does not")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    stamp_seed(ax, SEED, f"{n_cells} cells/type, Wilson 95%")
    fig.tight_layout()
    paths.append(save_figure(fig, "week13_divergence"))

    # --- A permutation test, as a worked example ---------------------------
    p1, p2 = marker_profiles(8, 1.5, 0.032)
    Xa, ya = simulate_dataset([p1, p2], 2000.0, n_cells, rng)
    llr_a = log_likelihood_ratio(Xa, 2000.0 * p1, 2000.0 * p2)
    ca = np.where(ya == 0, llr_a > 0, llr_a <= 0).astype(float)
    Xb, yb = simulate_dataset([p1, p2], 2400.0, n_cells, rng)
    llr_b = log_likelihood_ratio(Xb, 2400.0 * p1, 2400.0 * p2)
    cb = np.where(yb == 0, llr_b > 0, llr_b <= 0).astype(float)
    pval = permutation_test(ca, cb, 400 if fast else 2_000, rng)

    # --- The claim ledger --------------------------------------------------
    os.makedirs(PROGRESS, exist_ok=True)
    md = os.path.abspath(os.path.join(PROGRESS, "week13_claims.md"))
    with open(md, "w") as f:
        f.write("# Week 13 — claim ledger\n\n")
        f.write(f"Numbers generated by `experiments/week13_error_bars.py`, "
                f"SEED = {SEED}, {n_cells} cells per type.\n"
                f"**Edit the STATUS column by hand.** The script cannot know "
                f"whether you can defend a claim at the whiteboard.\n\n")
        f.write("| regime | D | k | dropout | measured | predicted | Wilson 95% | "
                "bootstrap 95% | agrees? | status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for label, depth, k, drop, m, p, lo, hi, bl, bh in rows:
            agrees = "yes" if lo <= p <= hi else "**NO**"
            f.write(f"| {label} | {depth:.0f} | {k} | {drop:.0%} | {m:.4f} | "
                    f"{p:.4f} | [{lo:.4f}, {hi:.4f}] | [{bl:.4f}, {bh:.4f}] | "
                    f"{agrees} | _fill in_ |\n")
        f.write(f"\nWorked permutation test (depth 2000 vs 2400, same k): "
                f"p = {pval:.4f}.\n")
        f.write("\nStatus vocabulary: **proved** (you can derive it on demand) · "
                "**cited** (someone else proved it; give the reference) · "
                "**observed** (measured here, no derivation) · "
                "**conjectured** (believed, stated as such).\n")

    print(f"[week13] seed={SEED}, {n_cells} cells/type, {n_boot} bootstrap resamples")
    print("[week13]  regime            measured  predicted   Wilson95            agrees")
    for label, depth, k, drop, m, p, lo, hi, bl, bh in rows:
        print(f"[week13]  {label:<17} {m:.4f}    {p:.4f}   "
              f"[{lo:.4f},{hi:.4f}]   {'yes' if lo <= p <= hi else 'NO'}")
    print(f"[week13] permutation test, depth 2000 vs 2400: p = {pval:.4f}")
    print(f"[week13] wrote {md}")
    print(f"[week13] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
