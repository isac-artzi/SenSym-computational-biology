"""The bead experiment: Aim 3's analysis code.

Used by: Weeks 6 and 12.

Build order:
    1. validate_tally      (W6 — write it BEFORE collecting data)
    2. accuracy_by_n       (W12)
    3. bead_d_prime        (W6)
    4. predicted_accuracy  (W6)
    5. fit_power_law       (W12)

The experiment (full protocol: docs/extras/bead-protocol.html):
  Two jars of beads. Jar A is 30% red, jar B is 35% red. A helper picks a jar
  at random and hands you n beads drawn from it with replacement; you guess
  which jar. Repeat.

The theory is the SAME theory as the sequencing problem, with counts of red
beads standing in for counts of transcripts. If the square-root law is real,
it will show up on a kitchen table — and if it does not, that is the most
interesting outcome available to this project.
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

    Write this in Week 6, BEFORE any data exists. A validator written after
    the fact tends to be a validator that happens to accept the data you got.

    Checks, in order of how often they actually bite:
      1. `correct` disagrees with (true_jar == guess)  -> transcription error
      2. a jar label outside {A, B}
      3. duplicate (run, trial) pairs                  -> a copy-paste slip
      4. a run whose true_jar never changes            -> the helper was not
         randomizing, which silently destroys the experiment

    Returns
    -------
    list of strings, one per problem, each naming the CSV LINE number
    (remember the header is line 1, so record i is line i+2). An empty list
    means the tally is clean.
    """
    # --- YOUR CODE HERE ---
    # 1. problems = []; seen = set(); by_run = {}.
    # 2. Loop with enumerate(records) so you have the index i; the CSV line
    #    number is i + 2.
    # 3. Check the jar labels first and `continue` if they are bad — the
    #    later checks assume they are valid.
    # 4. expected = int(r["true_jar"] == r["guess"]); if it differs from
    #    r["correct"], append a message quoting all three values.
    # 5. key = (run, trial): if already in `seen`, report a duplicate.
    # 6. Collect the true_jar values per run in by_run.
    # 7. After the loop: for each run with more than ~4 trials, if
    #    len(set(jars)) == 1, report that the jar was not randomized.
    # 8. Return problems.
    raise NotImplementedError("Implement me! See docs/s/week06.html — and try before peeking at python/solutions/.")


def accuracy_by_n(records: List[dict], dropout: float = 0.0
                  ) -> Dict[int, Tuple[int, int]]:
    """Group trials by draw size n; return {n: (n_correct, n_trials)}.

    Only trials at the given `dropout` level are counted, so the plain and
    dropout arms of the experiment never get silently pooled — which would
    flatten the very effect the dropout arm exists to measure.

    Returns a dict sorted by n (so plots come out in order without extra
    care at the call site).
    """
    # --- YOUR CODE HERE ---
    # 1. out = {}.
    # 2. For each record: skip it unless its dropout matches the requested
    #    one (compare floats with a tolerance: abs(a - b) > 1e-9).
    # 3. k, t = out.get(r["n"], (0, 0)); out[r["n"]] = (k + r["correct"], t + 1).
    # 4. Return dict(sorted(out.items())).
    raise NotImplementedError("Implement me! See docs/s/week12.html — and try before peeking at python/solutions/.")


def bead_d_prime(n: int, p_a: float, p_b: float, dropout: float = 0.0) -> float:
    """Predicted d' for distinguishing two jars from n drawn beads.

    Counting red beads in n draws gives Binomial(n, p), with mean n*p and
    variance n*p(1-p). Exactly the same construction as the Poisson case:

        d' = |n p_A - n p_B| / sqrt(n * pbar (1 - pbar))
           = sqrt(n) * |p_A - p_B| / sqrt(pbar (1 - pbar)).

    Dropout discards each drawn bead with probability q, so the number that
    survive is Binomial(n, 1-q) — in expectation n(1-q) beads, all still an
    unbiased sample of the jar. To first order this replaces n by n(1-q):
    dropout costs exactly as much as drawing fewer beads, and no more.
    (Exercise 12.3: why is the replacement only first order, and does the
    fluctuation in the number of survivors matter at n = 200?)

    Raises
    ------
    ValueError if dropout is not in [0, 1).
    """
    # --- YOUR CODE HERE ---
    # 1. Validate dropout.
    # 2. n_eff = n * (1 - dropout); pbar = (p_a + p_b) / 2.
    # 3. var = pbar * (1 - pbar). If var <= 0 or n_eff <= 0, return 0.0.
    # 4. Return sqrt(n_eff) * |p_a - p_b| / sqrt(var), as a float.
    #    Hand-check before trusting it: n = 100, p = .30 vs .35 must give
    #    10 * 0.05 / sqrt(0.325 * 0.675) = 0.688...
    raise NotImplementedError("Implement me! See docs/s/week06.html — and try before peeking at python/solutions/.")


def predicted_accuracy(n: int, p_a: float, p_b: float, dropout: float = 0.0) -> float:
    """Theory curve for the bead experiment: Phi(d'/2) with the bead d'.

    Run this in Week 6 for n in {10, 25, 50, 100, 200} before drawing a
    single bead. The numbers are sobering — even 200 beads gives well under
    80% — and knowing that in advance is what sets the trial budget instead
    of discovering it after a wasted month.
    """
    # --- YOUR CODE HERE ---
    # 1. One line: float(accuracy_from_d_prime(bead_d_prime(...))).
    raise NotImplementedError("Implement me! See docs/s/week06.html — and try before peeking at python/solutions/.")


def fit_power_law(ns, values) -> Tuple[float, float]:
    """Least-squares fit of log(value) = slope * log(n) + intercept.

    The whole project's hypothesis is a claim about ONE NUMBER: fit d'
    against n and the slope should be 0.5.

    Fit accuracy against n instead and you get nothing interpretable,
    because accuracy saturates at 1 — which is Week 12's most common mistake
    and why the chapter converts measured accuracy back into d' (via
    detection.d_prime_from_auc, or by inverting Phi(d'/2)) before fitting.

    Returns
    -------
    (slope, intercept).

    Raises
    ------
    ValueError unless at least two (n, value) pairs are strictly positive
    and finite — logs of zero or negative numbers are not a fit, they are a
    silent NaN.
    """
    # --- YOUR CODE HERE ---
    # 1. Convert both to float arrays.
    # 2. ok = (ns > 0) & (values > 0) & np.isfinite(values); raise ValueError
    #    if fewer than 2 entries survive.
    # 3. slope, intercept = np.polyfit(np.log(ns[ok]), np.log(values[ok]), 1).
    # 4. Return them as plain floats.
    raise NotImplementedError("Implement me! See docs/s/week12.html — and try before peeking at python/solutions/.")


def trials_needed(p_true: float, half_width: float, z: float = 1.96) -> int:
    """[PROVIDED] How many trials to pin an accuracy of ~p_true to +/- half_width.

    n >= z^2 p(1-p) / half_width^2. Call this in Week 6 BEFORE running
    anything: at p = 0.65 and half_width = 0.05 it returns 350, which is what
    sets the ~500-draw budget in the plan. Designing the sample size before
    collecting is what separates an experiment from a demonstration.
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
    would show up as an accuracy above chance even at n = 0.
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
