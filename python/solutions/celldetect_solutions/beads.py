"""The bead experiment: Aim 3's analysis code. [SOLUTIONS]

Reference implementation of `celldetect.beads`.
Used by: Weeks 6 and 12.

The experiment (protocol: docs/extras/bead-protocol.html):
  Two jars of beads. Jar A is 30% red, jar B is 35% red. A helper picks a
  jar at random and hands you n beads drawn from it with replacement; you
  guess which jar. Repeat.

The theory is the SAME theory as the sequencing problem, with counts of red
beads standing in for counts of transcripts. If the square-root law is real,
it will show up on a kitchen table.
"""

import csv
from typing import Dict, List, Tuple

import numpy as np

from .detection import accuracy_from_d_prime

TALLY_FIELDS = ["run", "n", "dropout", "trial", "true_jar", "guess", "correct"]


def load_tally(path: str) -> List[dict]:
    """[PROVIDED] Read a bead tally CSV into a list of dicts.

    Expected header (exactly these names, any order):
        run,n,dropout,trial,true_jar,guess,correct
    `n` = beads drawn, `dropout` = the discard probability in force (0 for
    the plain runs), `true_jar`/`guess` in {A, B}, `correct` in {0, 1}.

    The `correct` column is redundant with true_jar == guess, and that is on
    purpose: transcription errors show up as disagreements, and
    `validate_tally` finds them.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "run": r["run"].strip(),
            "n": int(r["n"]),
            "dropout": float(r["dropout"]),
            "trial": int(r["trial"]),
            "true_jar": r["true_jar"].strip().upper(),
            "guess": r["guess"].strip().upper(),
            "correct": int(r["correct"]),
        })
    return out


def validate_tally(records: List[dict]) -> List[str]:
    """Return a list of human-readable problems found in a tally.

    Checks, in order of how often they actually bite:
      1. `correct` disagrees with (true_jar == guess)  -> transcription error
      2. a jar label outside {A, B}
      3. duplicate (run, trial) pairs                  -> a copy-paste slip
      4. a run whose true_jar is constant              -> the helper was not
         randomizing, which silently destroys the experiment

    An empty list means the tally is clean. Run this BEFORE plotting
    anything; Week 12 makes it a verification item.
    """
    problems = []
    seen = set()
    by_run: Dict[str, List[str]] = {}
    for i, r in enumerate(records):
        line = i + 2   # +1 for 0-index, +1 for the header row
        if r["true_jar"] not in ("A", "B") or r["guess"] not in ("A", "B"):
            problems.append(f"row {line}: jar label not in {{A, B}}")
            continue
        expected = int(r["true_jar"] == r["guess"])
        if r["correct"] != expected:
            problems.append(
                f"row {line}: correct={r['correct']} but "
                f"true_jar={r['true_jar']} guess={r['guess']}")
        key = (r["run"], r["trial"])
        if key in seen:
            problems.append(f"row {line}: duplicate (run, trial) = {key}")
        seen.add(key)
        by_run.setdefault(r["run"], []).append(r["true_jar"])
    for run, jars in by_run.items():
        if len(jars) > 4 and len(set(jars)) == 1:
            problems.append(
                f"run {run}: every trial used jar {jars[0]} — "
                "the jar was not being randomized")
    return problems


def accuracy_by_n(records: List[dict], dropout: float = 0.0
                  ) -> Dict[int, Tuple[int, int]]:
    """Group trials by draw size n; return {n: (n_correct, n_trials)}.

    Only trials at the given `dropout` level are counted, so the plain and
    dropout arms of the experiment never get silently pooled.
    """
    out: Dict[int, Tuple[int, int]] = {}
    for r in records:
        if abs(r["dropout"] - dropout) > 1e-9:
            continue
        k, t = out.get(r["n"], (0, 0))
        out[r["n"]] = (k + int(r["correct"]), t + 1)
    return dict(sorted(out.items()))


def bead_d_prime(n: int, p_a: float, p_b: float, dropout: float = 0.0) -> float:
    """Predicted d' for distinguishing two jars from n drawn beads.

    Counting red beads in n draws gives Binomial(n, p), whose mean is n*p
    and variance n*p(1-p). The same construction as the Poisson case:

        d' = |n p_A - n p_B| / sqrt(n * pbar (1 - pbar))
           = sqrt(n) * |p_A - p_B| / sqrt(pbar (1 - pbar)).

    Dropout discards each drawn bead with probability q, so the number that
    survive is Binomial(n, 1-q) — in expectation n(1-q) beads, all still an
    unbiased sample of the jar. To first order this replaces n by n(1-q):
    dropout costs exactly as much as sequencing less deeply, and no more.
    (Exercise 12.3: why is the replacement only first order, and does the
    fluctuation in the number of survivors matter at n = 200?)
    """
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    n_eff = n * (1.0 - dropout)
    pbar = 0.5 * (p_a + p_b)
    var = pbar * (1.0 - pbar)
    if var <= 0 or n_eff <= 0:
        return 0.0
    return float(np.sqrt(n_eff) * abs(p_a - p_b) / np.sqrt(var))


def predicted_accuracy(n: int, p_a: float, p_b: float, dropout: float = 0.0) -> float:
    """Theory curve for the bead experiment: Phi(d'/2) with the bead d'."""
    return float(accuracy_from_d_prime(bead_d_prime(n, p_a, p_b, dropout)))


def fit_power_law(ns, values) -> Tuple[float, float]:
    """Least-squares fit of log(value) = slope * log(n) + intercept.

    Returns (slope, intercept). The whole project's hypothesis is a claim
    about ONE NUMBER: fit d' against n and the slope should be 0.5. Fit it
    against accuracy instead and you get nothing interpretable, because
    accuracy saturates at 1 — which is Week 12's most common mistake and
    why the chapter converts measured accuracy back to d' first.
    """
    ns = np.asarray(ns, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = (ns > 0) & (values > 0) & np.isfinite(values)
    if ok.sum() < 2:
        raise ValueError("need at least two positive (n, value) pairs to fit")
    slope, intercept = np.polyfit(np.log(ns[ok]), np.log(values[ok]), 1)
    return float(slope), float(intercept)


def trials_needed(p_true: float, half_width: float, z: float = 1.96) -> int:
    """[PROVIDED] How many trials to pin an accuracy of ~p_true to +/- half_width.

    n >= z^2 p(1-p) / half_width^2. Call this in Week 6 BEFORE running
    anything: at p = 0.65 and half_width = 0.05 it returns ~350, which is
    what sets the ~500-draw budget in the plan. Designing the sample size
    before collecting is what separates an experiment from a demonstration.
    """
    p = float(np.clip(p_true, 1e-6, 1 - 1e-6))
    return int(np.ceil(z ** 2 * p * (1 - p) / half_width ** 2))


def write_tally_template(path: str, run: str, ns, trials_per_n: int,
                         dropout: float, rng: np.random.Generator) -> str:
    """[PROVIDED] Write a pre-randomized tally sheet, ready to print and fill in.

    The `true_jar` column is filled in ADVANCE by this function and must be
    covered up (or held by the helper) during the run. Pre-randomizing beats
    "pick a jar at random each time" because humans are demonstrably bad at
    generating random sequences — they alternate too much — and that bias
    would show up as an accuracy above chance at n = 0.
    """
    rows = []
    trial = 1
    for n in ns:
        for _ in range(trials_per_n):
            rows.append({"run": run, "n": int(n), "dropout": dropout,
                         "trial": trial,
                         "true_jar": "A" if rng.random() < 0.5 else "B",
                         "guess": "", "correct": ""})
            trial += 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TALLY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path
