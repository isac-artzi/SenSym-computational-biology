# Bead experiment data — primary records

This is the only directory in the repository holding data that cannot be
regenerated. Treat it accordingly: never edit a committed tally to make a
figure look better; add a new run and say what changed.

## Files

| Pattern | What it is |
|---|---|
| `RUNTEMPLATE_*.csv` | Blank, pre-randomized sheets written by `experiments/week06_bead_design.py`. Print them, fill in `guess` and `correct` by hand. Ignored by the analysis. |
| `run01.csv`, `run02.csv`, … | Filled-in runs. These ARE the experiment. |

The analysis (`experiments/week12_beads.py`) reads every `*.csv` here and
skips any row whose `run` column starts with `RUNTEMPLATE`.

## Format

```
run,n,dropout,trial,true_jar,guess,correct
run01,10,0.0,1,A,B,0
run01,10,0.0,2,B,B,1
```

| Column | Meaning |
|---|---|
| `run` | a label for the sitting, e.g. `run03`. Keep it unique per file. |
| `n` | beads drawn for this trial |
| `dropout` | the discard probability in force (`0.0` for the plain arm) |
| `trial` | trial number within the run, starting at 1 |
| `true_jar` | `A` or `B` — **pre-randomized, and covered up during the run** |
| `guess` | `A` or `B` — what you said before looking |
| `correct` | `1` if `guess == true_jar`, else `0` |

`correct` is deliberately redundant with the two columns before it. That
redundancy is a checksum: `beads.validate_tally` flags every row where they
disagree, which is how transcription slips get caught instead of published.

## The rules that make this data worth having

1. **Randomize in advance.** The template fills `true_jar` before the run.
   Never let a human pick "randomly" in the moment — people alternate far
   more than chance does, and that bias alone can push accuracy above 50% at
   n = 0.
2. **Blind the guesser.** Whoever draws the beads must not see which jar,
   and whoever knows the jar must not see the guess being made.
3. **Transcribe the same day.** A tally sheet you cannot read in a month is
   a dataset you no longer have. Photograph the paper sheet into
   `notebook/beads/` and type the numbers here, then check the row count
   against the tick marks and record the check in the week's progress log.
4. **Never delete a trial.** If something went wrong — the wrong jar, a
   dropped bead, an interruption — write a note in the progress log and keep
   the row, or add a new run and explain. Silently dropping trials that went
   badly is how a null result turns into a positive one.
5. **Draw with replacement.** Put each bead back before the next draw, or
   the jar's composition shifts as you go and the binomial model is wrong.
   (`bead_d_prime` assumes replacement; exercise 12.5 asks how much it
   matters at n = 200 from a jar of 500.)
