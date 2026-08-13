# How Much Data Does It Take to Tell Two Cell Types Apart?

A one-semester (15-week), mentor-guided research course for a serious
high-school student — an interactive textbook, a Python laboratory, a
tabletop physical experiment, and a genuine research project, all in one
repository.

**Student:** Vihaan Rao · **Mentor:** Dr. Isac Artzi · **Duration:** 15 weeks

**The research question, honestly stated.** Single-cell RNA sequencing reads
out only a few thousand molecules per cell — a tiny, noisy sample of that
cell's true transcriptome. So deciding what type a cell is, is not primarily
a machine-learning problem: it is a *statistical detection* problem, of
exactly the kind radar engineers solved in the 1950s. This project asks how
classification accuracy scales with (a) sequencing depth per cell and (b) the
number of informative marker genes, and answers it three independent ways —
by derivation, by simulation, and by a physical bead-drawing experiment on a
kitchen table.

The central prediction: discriminability grows as $\sqrt{\text{depth}\times k}$.
**Doubling the separation between two cell types costs four times the data.**

## Quickstart

1. Open the course site: **https://isac-artzi.github.io/SenSym-computational-biology/**
   (same content as `docs/index.html` — works from any browser, and offline).
2. Clone the repo to work through exercises, run Python, and commit progress.
3. Set up Python: follow `python/SETUP.md`, then see `python/README.md` for
   the experiment map and conventions.

## Repository map

| Path | What lives there |
|---|---|
| `docs/` | The course website: 15 weekly chapters (`s/`), extras (glossary, notation, bead protocol, AI policy, how to get unstuck), interactive labs |
| `python/` | The `celldetect` package (student scaffolds), `solutions/`, `tests/`, `experiments/`, `figures/`, `data/` |
| `plans/` | The governing research plan: milestones M1–M8, decision rules, fallback ladder, the 15-week schedule |
| `notebook/` | Scans of the paper notebook — derivations, failures, bead tally sheets (see `notebook/README.md`) |
| `progress/` | Weekly progress logs (see `progress/README.md` and `TEMPLATE.md`) |
| `.github/` | The biweekly check-in issue template |

## The three aims

| Aim | What it is | Weeks |
|---|---|---|
| **Aim 1 — Theory** | Poisson counting statistics → the separation index $d'$ → a closed-form accuracy prediction $\Phi(d'/2)$ with $d' \propto \sqrt{\text{depth}\cdot k}$ | 1–5 |
| **Aim 2 — Simulation & real data** | A synthetic-cell simulator, classifiers written from scratch, then the same measurement on downsampled 10x PBMC 3k data | 7–11 |
| **Aim 3 — Physical analog** | Two jars of coloured beads at 30/70 and 35/65; draw $n$, guess the jar, measure accuracy vs. $n$; add a dropout rule | 6–12 |

Aim 3 is an honest analog of the *sampling* process, not a substitute for
sequencing, and is presented as such throughout.

## The weekly workflow

1. **Read the chapter** for the week (`docs/s/weekNN.html`) and work its
   exercises in the paper notebook.
2. **Derive in the notebook** — everything goes in, including failed
   attempts; scan to `notebook/weekNN/`.
3. **Code**: implement the week's TODOs in `python/celldetect/`, get
   `pytest` green, run the week's experiment, commit code *and* figure
   (with the seed recorded).
4. **Bead runs** (weeks 6–12): tally sheets go in
   `python/data/beads/`, scanned originals in `notebook/beads/`.
5. **Log it**: copy `progress/TEMPLATE.md` to `progress/weekNN.md`, fill it
   in, commit.
6. **Every two weeks**: open a check-in issue from the template
   (`.github/ISSUE_TEMPLATE/checkin.md`) and meet with your mentor.

## Milestones

| Tag | Week | What it certifies |
|---|---|---|
| M1 | 3 | The $d' \propto \sqrt{\text{depth}}$ derivation, defended at the whiteboard |
| M2 · `theory-complete` | 5 | Aim 1 closed: the $\sqrt{k\cdot\text{depth}}$ law and the accuracy formula |
| M3 · `simulator-validated` | 9 | Simulation matches theory (or the disagreement is characterized) |
| M4 | 11 | Real PBMC 3k downsampling curve produced end to end |
| M5 · `evidence-complete` | 12 | All three lines of evidence on one axis |
| M6 | 13 | Every claim labelled *proved / cited / observed / conjectured*, with error bars |
| M7 · `report-draft` | 14 | The 6–8 page report, frozen |
| M8 · `v1.0` | 15 | Poster, talk, submission |

Convention: tag only when `pytest python/tests/` is fully green and the
relevant figures regenerate identically from a clean checkout.

## Success criteria

Success is **not** high classification accuracy — that is easy and
uninformative. Success is a quantitative comparison between a derived
prediction, a simulation, and a physical experiment, with an honest account
of where they diverge. A clean negative result (e.g. dropout breaks the
$\sqrt{n}$ law below some depth) is a strong outcome, and the report has a
section reserved for exactly that.

## For mentors

The check-in rhythm, milestone criteria, decision rules, and the fallback
ladder F1–F4 are in `plans/`. The AI-use policy the student follows — and
discloses under "Tools used" in every progress log — is
`docs/extras/ai-policy.html`. Reference implementations for every function
live in `python/solutions/`; reading them after a real attempt is expected
and is not cheating — presenting them as one's own at a milestone is.
