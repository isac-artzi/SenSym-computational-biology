# The `celldetect` Python project

The computational half of *How Much Data Does It Take to Tell Two Cell Types
Apart?* Every week's chapter (in `docs/`) points at code here; every figure
in the report regenerates from this directory with a recorded seed.

## Layout

```
python/
├── celldetect/               the STUDENT package — scaffolds with TODOs (you fill them in)
├── solutions/                complete reference implementations (read AFTER a real attempt)
├── tests/                    pytest suite; a red test defines "not done yet"
├── experiments/              the weekly driver scripts (table below)
├── figures/                  every generated figure lands here — and gets committed
├── data/                     bead tallies (primary data) + the PBMC cache (not committed)
├── SETUP.md                  installation walkthrough (incl. the Colab fallback)
└── run_solution_check.py     author/mentor tool: runs the tests against solutions/
```

Install once (see `SETUP.md`): `pip install -r requirements.txt`.
Work loop per week: implement the TODOs in `celldetect/`, then

```bash
cd python
pytest tests/test_<module>.py      # green = your code agrees with the theory
python experiments/<script>.py     # produces the week's figure(s)
```

## Conventions

- **Seeded.** Each experiment has a `SEED` constant at the top. That seed is
  part of the result — record it in the progress log and the figure caption.
  Irreproducible data is not data. Every figure carries its seed in the
  corner, drawn by `_common.stamp_seed`.
- **`main(fast=False)`.** Every experiment exposes `main(fast=False)` which
  runs the whole script and returns the list of figure paths it wrote. That
  is what lets `week14_report_figures.py` and `make_all_figures.py`
  regenerate everything in one command. `fast=True` is the tiny-parameter
  smoke mode — never for a figure you will show anyone.
- **`figures/`.** Scripts save into `../figures/` (dpi 150, tight box, never
  `plt.show()` — everything runs headless). Figures are deliverables: commit
  them. `figures/report/` is the Week 14 freeze set (`F1`–`F6`,
  `CAPTIONS.md`, `SEEDS.json`); `figures/poster/` is the Week 15 set.
- **Stand-in data is stamped.** Before the PBMC download exists, Weeks 10–11
  run on a synthetic surrogate; before the bead runs exist, Week 12
  simulates draws. Both stamp the figure with a large red "not a result",
  set a module flag `USED_STAND_IN`, and `week14_report_figures.py` refuses
  to copy such a figure into the report set. If a figure has the stamp, the
  claim it supports does not exist yet.
- **`[PROVIDED]` in a docstring** means the function is already written:
  read it, do not rewrite it.

## Experiment ↔ chapter ↔ figure map

| Experiment | Chapter | Figure(s) in `figures/` | Seed |
|---|---|---|---|
| `week01_counting_noise.py` | `docs/s/week01.html` | `week01_counting_noise.png`, `week01_fano.png` | 101 |
| `week02_thinning.py` | `docs/s/week02.html` | `week02_thinning.png`, `week02_thinning_fano.png` | 102 |
| `week03_dprime_depth.py` | `docs/s/week03.html` | `week03_dprime_depth.png`, `week03_overlap.png` | 103 |
| `week04_accuracy_roc.py` | `docs/s/week04.html` | `week04_accuracy_curve.png`, `week04_roc.png` | 104 |
| `week05_sqrt_k.py` | `docs/s/week05.html` | `week05_sqrt_k.png`, `week05_prediction.png` | 105 |
| `week06_bead_design.py` | `docs/s/week06.html` | `week06_bead_design.png` (+ tally templates into `data/beads/`) | 106 |
| `week07_simulator.py` | `docs/s/week07.html` | `week07_simulator.png`, `week07_cells.png` | 107 |
| `week08_classifiers.py` | `docs/s/week08.html` | `week08_classifiers.png`, `week08_curse.png` | 108 |
| `week09_dropout.py` | `docs/s/week09.html` | `week09_dropout.png`, `week09_breakdown.png` (+ `progress/week09_table.md`) | 109 |
| `week10_markers.py` | `docs/s/week10.html` | `week10_markers.png`, `week10_ranking.png` | 110 |
| `week11_overdispersion.py` | `docs/s/week11.html` | `week11_fano.png`, `week11_downsample.png` | 111 |
| `week12_beads.py` | `docs/s/week12.html` | `week12_beads.png`, `week12_three_ways.png` | 112 |
| `week13_error_bars.py` | `docs/s/week13.html` | `week13_intervals.png`, `week13_divergence.png` (+ `progress/week13_claims.md`) | 113 |
| `week14_report_figures.py` | `docs/s/week14.html` | `figures/report/F1`–`F6`, `CAPTIONS.md`, `SEEDS.json` | (each script's own) |
| `week15_poster_figures.py` | `docs/s/week15.html` | `figures/poster/P1`–`P3` | 115 |
| `fetch_pbmc3k.py` | `docs/s/week10.html` | — (writes `data/pbmc3k_cache.npz`) | — |

## The one command

```bash
python experiments/make_all_figures.py            # full parameters (minutes)
python experiments/make_all_figures.py --smoke    # tiny smoke run (< 1 min)
python experiments/make_all_figures.py --only week07_simulator week08_classifiers
```

Scripts whose functions are not implemented yet report as **skipped**, not
failed — early in the semester most of the list is expected to skip.

The freeze test (Week 14): delete `figures/report/`, run
`week14_report_figures.py` once, diff — identical figures, or it is not a
freeze.

## Module ↔ week map

| Module | Weeks | What it holds |
|---|---|---|
| `counting.py` | 1–2 | Poisson sampling, the Fano factor, the pmf, thinning |
| `detection.py` | 3–5 | $d'$, $\Phi(d'/2)$, the naive-Bayes log-ratio, ROC/AUC |
| `simulate.py` | 7, 9 | marker profiles, synthetic cells, dropout |
| `classify.py` | 8 | train/test split, logistic regression, kNN |
| `realdata.py` | 10–11 | marker selection, downsampling, overdispersion, NB $d'$ |
| `beads.py` | 6, 12 | tally files and validation, the bead $d'$, the power-law fit |
| `stats.py` | 13 | Wilson intervals, bootstrap, permutation test |
| `viz.py` | all | `[PROVIDED]` palette and figure furniture |
