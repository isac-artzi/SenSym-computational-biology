# The paper notebook, scanned

The paper notebook is the primary research record; this directory is its
backup and its shareable form.

## What belongs in the notebook

Everything. Derivations and failed derivations, computations, sketches of
distributions, conjectures, lab observations — and especially **failures**,
written in full sentences ("I tried to get $d'$ for the dropout case by
substituting $\lambda \to (1-p)\lambda$ directly; that is wrong because the
*variance* also changes, and here is why"). A written failure localizes a
difficulty; an unwritten one just recirculates. Date every entry.

## Scanning conventions

- One folder per course week: `notebook/week04/`, `notebook/week09/`, ...
- Pages as photos: `notebook/weekNN/weekNN_page1.jpg`, `weekNN_page2.jpg`, ...
  — or a single `notebook/weekNN/weekNN.pdf` per week if your scanner app
  produces one. Either is fine; be consistent.
- Scan weekly (five minutes) and commit. Milestone weeks call out specific
  scans as deliverables (e.g. the complete $\sqrt{k}$ derivation in
  `notebook/week05/`).

## The bead experiment is different

Bead tally sheets are **primary data**, not notes. They go in two places:

- the raw scan or photo of the paper sheet → `notebook/beads/runNN.jpg`
- the transcribed numbers → `python/data/beads/runNN.csv`
  (columns: `run,n,dropout,trial,true_jar,guess,correct` — the format
  `celldetect.beads.load_tally` expects)

Transcribe the same day you run. A tally sheet you cannot read in a month
is a dataset you no longer have. Cross-check: the CSV row count must equal
the number of tick marks on the sheet, and the check is recorded in the
week's progress log.

## The table-of-contents habit

Keep the first page (or first two) of the physical notebook as a running
table of contents: date, topic, page number, one line. Mirror it here as
`notebook/TOC.md` if you like. Future-you, writing the report in week 14,
will be looking for "where did I show the naive-Bayes rule is linear in the
counts?" — the TOC is how that search takes seconds instead of an afternoon.
