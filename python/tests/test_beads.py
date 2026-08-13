"""Tests for celldetect.beads — the tabletop experiment's analysis (W6, W12)."""

import math

import numpy as np
import pytest

from celldetect.beads import (
    accuracy_by_n,
    bead_d_prime,
    fit_power_law,
    load_tally,
    predicted_accuracy,
    trials_needed,
    validate_tally,
    write_tally_template,
)

P_A, P_B = 0.30, 0.35


def _rows(n_list, correct_pattern, dropout=0.0, run="r1"):
    """Build a synthetic tally in the in-memory format load_tally returns."""
    out, trial = [], 1
    for n in n_list:
        for c in correct_pattern:
            true_jar = "A" if trial % 2 else "B"
            guess = true_jar if c else ("B" if true_jar == "A" else "A")
            out.append({"run": run, "n": n, "dropout": dropout, "trial": trial,
                        "true_jar": true_jar, "guess": guess, "correct": int(c)})
            trial += 1
    return out


def test_bead_d_prime_scales_as_sqrt_n():
    """The bead experiment tests the SAME exponent as the sequencing one (W12)."""
    ns = np.array([10, 25, 50, 100, 200], dtype=float)
    dps = np.array([bead_d_prime(int(n), P_A, P_B) for n in ns])
    slope = np.polyfit(np.log(ns), np.log(dps), 1)[0]
    assert abs(slope - 0.5) < 1e-9
    assert math.isclose(bead_d_prime(200, P_A, P_B) / bead_d_prime(50, P_A, P_B),
                        2.0, rel_tol=1e-12)


def test_bead_d_prime_known_value():
    """Hand-checkable: n=100, p=.30 vs .35 gives sqrt(100)*0.05/sqrt(.325*.675)."""
    expected = 10.0 * 0.05 / math.sqrt(0.325 * 0.675)
    assert math.isclose(bead_d_prime(100, P_A, P_B), expected, rel_tol=1e-12)


def test_bead_d_prime_dropout_is_reduced_n():
    """Dropout q is exactly a factor (1-q) on the effective draw size (W12)."""
    assert math.isclose(bead_d_prime(100, P_A, P_B, dropout=0.5),
                        bead_d_prime(50, P_A, P_B), rel_tol=1e-12)
    with pytest.raises(ValueError):
        bead_d_prime(10, P_A, P_B, dropout=1.0)


def test_bead_d_prime_zero_when_jars_are_identical():
    assert bead_d_prime(500, 0.3, 0.3) == 0.0


def test_predicted_accuracy_is_chance_at_n_zero_and_rises():
    assert math.isclose(predicted_accuracy(0, P_A, P_B), 0.5, abs_tol=1e-12)
    accs = [predicted_accuracy(n, P_A, P_B) for n in (10, 25, 50, 100, 200)]
    assert all(a < b for a, b in zip(accs, accs[1:]))
    # A realistic sanity check for the actual experiment: even 200 beads is
    # only a modest edge. Knowing this BEFORE running saves a wasted month.
    assert 0.5 < accs[-1] < 0.8


def test_accuracy_by_n_groups_correctly():
    records = _rows([10, 50], [1, 1, 0, 1])
    got = accuracy_by_n(records)
    assert got == {10: (3, 4), 50: (3, 4)}


def test_accuracy_by_n_separates_dropout_arms():
    """Pooling the dropout and plain arms would silently ruin the analysis."""
    records = _rows([50], [1, 1, 1, 1]) + _rows([50], [0, 0, 0, 0], dropout=0.3, run="r2")
    assert accuracy_by_n(records, dropout=0.0) == {50: (4, 4)}
    assert accuracy_by_n(records, dropout=0.3) == {50: (0, 4)}


def test_validate_tally_accepts_a_clean_sheet():
    assert validate_tally(_rows([10, 25], [1, 0, 1, 0])) == []


def test_validate_tally_catches_transcription_errors():
    records = _rows([10], [1, 1, 1, 1])
    records[2]["correct"] = 0            # disagrees with true_jar == guess
    problems = validate_tally(records)
    assert len(problems) == 1 and "correct=" in problems[0]


def test_validate_tally_catches_duplicate_trials():
    records = _rows([10], [1, 1])
    records[1]["trial"] = records[0]["trial"]
    assert any("duplicate" in p for p in validate_tally(records))


def test_validate_tally_catches_an_unrandomized_jar():
    records = _rows([10], [1] * 6)
    for r in records:
        r["true_jar"] = "A"
        r["guess"] = "A"
        r["correct"] = 1
    assert any("not being randomized" in p for p in validate_tally(records))


def test_validate_tally_catches_bad_labels():
    records = _rows([10], [1])
    records[0]["true_jar"] = "C"
    assert any("not in {A, B}" in p for p in validate_tally(records))


def test_fit_power_law_recovers_a_planted_exponent():
    ns = np.array([10, 20, 40, 80, 160], dtype=float)
    vals = 3.0 * ns ** 0.5
    slope, intercept = fit_power_law(ns, vals)
    assert abs(slope - 0.5) < 1e-9
    assert abs(math.exp(intercept) - 3.0) < 1e-6


def test_fit_power_law_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_power_law([10], [1.0])
    with pytest.raises(ValueError):
        fit_power_law([10, 20], [0.0, -1.0])


def test_trials_needed_provided():
    """[PROVIDED] Sample-size planning, done before any beads are drawn."""
    assert trials_needed(0.65, 0.05) == 350
    assert trials_needed(0.5, 0.10) < trials_needed(0.5, 0.05)


def test_tally_roundtrip_through_a_csv(tmp_path):
    """write_tally_template -> fill in -> load_tally -> validate (W6)."""
    path = tmp_path / "run01.csv"
    write_tally_template(str(path), "run01", [10, 25], 3, 0.0,
                         np.random.default_rng(51))
    text = path.read_text().splitlines()
    assert text[0].split(",") == ["run", "n", "dropout", "trial",
                                  "true_jar", "guess", "correct"]
    assert len(text) == 1 + 6
    # Pre-randomization must actually randomize: not all one jar.
    jars = [line.split(",")[4] for line in text[1:]]
    assert len(set(jars)) == 2
    # Now "run the experiment": guess A every time, and score it.
    filled = ["run,n,dropout,trial,true_jar,guess,correct"]
    for line in text[1:]:
        f = line.split(",")
        f[5] = "A"
        f[6] = "1" if f[4] == "A" else "0"
        filled.append(",".join(f))
    path.write_text("\n".join(filled) + "\n")
    records = load_tally(str(path))
    assert len(records) == 6
    assert validate_tally(records) == []
    total = sum(t for _, t in accuracy_by_n(records).values())
    assert total == 6
