# Reference solutions

`celldetect_solutions/` mirrors every module of the student package
`python/celldetect/` — same file names, same function names, same contracts —
but with complete, heavily commented implementations. The comments try to
teach *why* each line is there, not just what it does: where a result
licenses a numerical shortcut (flooring a rate before taking a log, guarding
a 0/0 in `d_prime`), the comment names the result and the week.

## Intended use

1. **Attempt first.** Open the scaffold in `python/celldetect/`, follow the
   step-by-step comments, and run the matching test file
   (`pytest tests/test_<module>.py`). The tests define "done".
2. **Get stuck for real** — meaning: you have re-read the chapter section,
   tried at least one concrete small example by hand, and can say *which*
   step you are stuck on. (See `docs/extras/unstuck.html`.)
3. **Then read the matching function here.** Read the comments, not just the
   code — the comments are where the teaching is.
4. **Close the file and rewrite your own version from understanding.**
   Copy-pasting runs; it does not teach.

## The policy, in one line

It is **not cheating to read this** — it is cheating to present it as yours
at a milestone. Full policy: `docs/extras/ai-policy.html`. Your mentor will
ask you to explain your code line by line at verification; code you
understand survives that, code you pasted does not.

## A note on differences

Where a function is marked `[PROVIDED]` in the student package
(`total_counts`, `empirical_moments`, `posterior_log_odds`, `expected_rates`,
`multi_type_profiles`, `cross_validated_accuracy`, `_sigmoid`, `load_pbmc`,
`surrogate_pbmc`, `load_tally`, `trials_needed`, `write_tally_template`,
`standard_error_of_proportion`, `bonferroni`, and all of `viz`), the two
copies are identical by design.

Some solutions are vectorized where the scaffold comments suggest a plain
loop (`knn_predict`'s distance matrix, `simulate_cells`'s broadcast). Both
are correct; the loop is the version to write first, and the scaffold says
so. Writing the loop, checking it against the fast version on a small
example, and only then keeping the fast one is a good habit and takes four
minutes.

## Verifying the solutions

The whole test suite runs green against this package:

```
cd python
python run_solution_check.py
```

That script temp-copies these files under the package name `celldetect` and
runs pytest against them — the same tests you run against your own code. If
it ever fails, the tests and the solutions have drifted apart and the mentor
needs to know before you waste an afternoon on it.
