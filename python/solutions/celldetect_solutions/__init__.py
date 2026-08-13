"""celldetect — reference solutions.

Complete implementations of every scaffold in `python/celldetect/`, with the
same module names, function names, and contracts. Read them AFTER a real
attempt (see solutions/README.md); the comments are where the teaching is.

Module map — which week needs which:

    counting.py    W1-W2   Poisson counts, the Fano factor, thinning
    detection.py   W3-W5   d', Phi(d'/2), the naive-Bayes log-ratio, ROC/AUC
    simulate.py    W7,W9   synthetic cells, marker profiles, dropout
    classify.py    W8      train/test split, logistic regression, kNN
    realdata.py    W10-W11 PBMC markers, downsampling, overdispersion
    beads.py       W6,W12  tally files, the bead d', the power-law fit
    stats.py       W13     Wilson intervals, bootstrap, permutation test
    viz.py         all     [PROVIDED] palette and figure furniture
"""

__all__ = [
    "counting", "detection", "simulate", "classify",
    "realdata", "beads", "stats", "viz",
]
