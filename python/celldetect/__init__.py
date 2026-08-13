"""celldetect — the computational half of

    "How Much Data Does It Take to Tell Two Cell Types Apart?"

Every function here starts as a scaffold with a docstring, step-by-step
comments, and `raise NotImplementedError`. You implement them week by week;
the tests in ../tests/ define "done". Complete reference implementations
live in ../solutions/ — read them AFTER a real attempt.

Module map — which week needs which:

    counting.py    W1-W2   Poisson counts, the Fano factor, thinning
    detection.py   W3-W5   d', Phi(d'/2), the naive-Bayes log-ratio, ROC/AUC
    simulate.py    W7, W9  synthetic cells, marker profiles, dropout
    classify.py    W8      train/test split, logistic regression, kNN
    realdata.py    W10-W11 PBMC markers, downsampling, overdispersion
    beads.py       W6, W12 tally files, the bead d', the power-law fit
    stats.py       W13     Wilson intervals, bootstrap, permutation test
    viz.py         all     [PROVIDED] palette and figure furniture

Functions marked [PROVIDED] in a docstring are already written: read them,
do not rewrite them. Everything else is yours.
"""

__all__ = [
    "counting", "detection", "simulate", "classify",
    "realdata", "beads", "stats", "viz",
]
