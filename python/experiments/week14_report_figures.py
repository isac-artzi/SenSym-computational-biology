"""Week 14 — freeze the report's figures.

Chapter: docs/s/week14.html. MILESTONE M7 (`report-draft` tag).
Writes into figures/report/:
    F1_counting_noise.png   the phenomenon: variance tracks mean
    F2_theory.png           Aim 1: the sqrt(depth * k) law and Phi(d'/2)
    F3_simulation.png       Aim 2a: simulation against theory
    F4_realdata.png         Aim 2b: PBMC downsampling, Poisson vs NB
    F5_beads.png            Aim 3: the physical experiment
    F6_divergence.png       where the theory fails, with error bars
    CAPTIONS.md             a caption for each, with its seed
    SEEDS.json              every seed used, machine-readable

The freeze test: delete figures/report/, run this once, and the output must
be byte-identical to what the report cites. If it is not, something is
unseeded and the report is not reproducible.

REFUSES to copy any figure whose source script reported surrogate or
simulated stand-in data. Fix the data, not this script.
"""

import json
import os
import shutil

from _common import FIGDIR

import week01_counting_noise
import week05_sqrt_k
import week07_simulator
import week11_overdispersion
import week12_beads
import week13_error_bars

REPORT = os.path.join(FIGDIR, "report")

# (destination name, module, index into the module's returned path list)
PLAN = [
    ("F1_counting_noise", week01_counting_noise, 0),
    ("F2_theory", week05_sqrt_k, 1),
    ("F3_simulation", week07_simulator, 0),
    ("F4_realdata", week11_overdispersion, 1),
    ("F5_beads", week12_beads, 0),
    ("F6_divergence", week13_error_bars, 1),
]

CAPTIONS = {
    "F1_counting_noise":
        "Poisson counts at four expression rates. The sample variance tracks "
        "the sample mean across two orders of magnitude — the fact the entire "
        "analysis rests on. Panel titles give both statistics.",
    "F2_theory":
        "Aim 1. Measured accuracy of the optimal (naive-Bayes) rule against "
        "the closed-form prediction Phi(d'/2), over 20 combinations of "
        "sequencing depth and marker-gene count. Points on the diagonal mean "
        "the prediction has no free parameters left to fit.",
    "F3_simulation":
        "Aim 2a. Simulated cells (points) against the Week 5 prediction "
        "(lines) at three marker-set sizes. Deviations at the low-depth end "
        "are the Gaussian approximation, not the square-root law, failing.",
    "F4_realdata":
        "Aim 2b. Accuracy on downsampled 10x PBMC 3k data against two "
        "predictions: Poisson (dashed) and negative binomial with the "
        "measured per-gene dispersion (solid). The gap between the dashed "
        "line and the points is the cost of assuming Poisson.",
    "F5_beads":
        "Aim 3. Bead-drawing accuracy against draw size, with Wilson 95% "
        "intervals, for the plain and dropout arms. Curves are the theory "
        "computed in Week 6, before any beads were drawn.",
    "F6_divergence":
        "Measured minus predicted accuracy across six regimes, with Wilson "
        "95% intervals. Points whose interval excludes zero are where the "
        "closed form is refuted, not merely imprecise.",
}


def main(fast: bool = False):
    os.makedirs(REPORT, exist_ok=True)
    seeds, written, refused = {}, [], []

    for dest, module, idx in PLAN:
        print(f"\n=== {dest}  <-  {module.__name__} ===")
        paths = module.main(fast=fast)
        src = paths[idx]
        seeds[dest] = {"script": module.__name__ + ".py",
                       "seed": getattr(module, "SEED", None),
                       "source_figure": os.path.basename(src)}
        # Refuse anything that stood in for data it does not have.
        if _looks_like_stand_in(module):
            refused.append(dest)
            print(f"!! REFUSING {dest}: {module.__name__} used stand-in data.")
            continue
        out = os.path.join(REPORT, dest + ".png")
        shutil.copyfile(src, out)
        written.append(dest)

    with open(os.path.join(REPORT, "SEEDS.json"), "w") as f:
        json.dump(seeds, f, indent=2, sort_keys=True)

    with open(os.path.join(REPORT, "CAPTIONS.md"), "w") as f:
        f.write("# Report figures — captions and provenance\n\n")
        f.write("Regenerate everything with `python experiments/"
                "week14_report_figures.py`. Seeds are in `SEEDS.json`.\n\n")
        for dest, module, _ in PLAN:
            status = "written" if dest in written else "REFUSED (stand-in data)"
            f.write(f"## {dest}  \n")
            f.write(f"*Source:* `experiments/{module.__name__}.py`, "
                    f"SEED = {getattr(module, 'SEED', 'n/a')} · *status:* {status}\n\n")
            f.write(CAPTIONS[dest] + "\n\n")
        if refused:
            f.write("---\n\n**Figures refused:** " + ", ".join(refused) +
                    ". These require real data (the PBMC cache and/or filled-in "
                    "bead tallies). The report cannot cite them until then.\n")

    print("\n" + "=" * 60)
    print(f"[week14] wrote {len(written)} figures into {REPORT}")
    if refused:
        print(f"[week14] REFUSED {len(refused)}: {', '.join(refused)}")
        print("[week14] -> run experiments/fetch_pbmc3k.py and/or fill in "
              "data/beads/*.csv, then re-run.")
    return [os.path.join(REPORT, d + ".png") for d in written]


def _looks_like_stand_in(module) -> bool:
    """True if the module fell back to surrogate/simulated data this run.

    The experiment scripts set a module-level flag when they do; older ones
    that cannot use stand-in data never set it, so absence means 'fine'.
    """
    return bool(getattr(module, "USED_STAND_IN", False))


if __name__ == "__main__":
    main()
