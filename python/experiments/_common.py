"""Shared plumbing for the celldetect experiment drivers.

Every experiment in this directory is a thin, seeded, headless script:
a SEED constant at the top, one main(fast=False) function that returns the
list of figure paths it wrote, and no command-line parsing beyond what a
chapter explicitly asks for.

This module only centralizes the boring parts: the figures/ directory, the
course colours, a headless-safe savefig, and the two "this is not real data"
stamps that keep an in-progress figure from being mistaken for a result.
"""

import os
import sys

# Make `import celldetect` work when a script is run as
#     python experiments/week01_counting_noise.py
# from the python/ directory. In that form Python puts experiments/ on the
# path but NOT its parent, so the package would not be found. Every
# experiment imports _common first, so this one insert covers all of them.
_PYTHON_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import matplotlib

matplotlib.use("Agg")  # headless-safe: never open a window
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

# Course palette (matches docs/assets/css/style.css and docs/assets/js/cd.js).
BLUE = "#2563eb"        # simulation
ORANGE = "#ea580c"      # theory
TEAL = "#0d9488"        # real data
MAGENTA = "#c026d3"     # beads
PURPLE = "#7c3aed"
GOLD = "#b45309"
GRAY = "#94a3b8"
RED = "#b91c1c"

# One colour per line of evidence, used in EVERY figure of the project so
# that the Week 14 report reads as one document.
AIM = {"theory": ORANGE, "simulation": BLUE, "real": TEAL, "beads": MAGENTA}

# figures/ and data/ sit next to experiments/.
_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_HERE, os.pardir, "figures")
DATADIR = os.path.join(_HERE, os.pardir, "data")


def save_figure(fig, name: str, subdir: str = "") -> str:
    """Save fig as <figures>/[subdir/]<name>.png (dpi=150, tight) and return the path."""
    outdir = os.path.join(FIGDIR, subdir) if subdir else FIGDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.abspath(os.path.join(outdir, name + ".png"))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def stamp_seed(ax, seed, extra: str = ""):
    """Write the seed in the corner of an axes. Every figure carries one."""
    text = f"seed {seed}" + (f" · {extra}" if extra else "")
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=GRAY)


def stamp_not_real(fig, what="SURROGATE DATA"):
    """Mark a figure that is NOT built from the thing it stands for. Loudly.

    Used for the surrogate PBMC data (no download yet) and for simulated
    bead draws (no physical runs yet). A figure with this stamp must never
    appear in the report; week14_report_figures.py refuses to copy one.
    """
    fig.text(0.5, 0.5, what + "\nnot a result", ha="center", va="center",
             fontsize=32, color=RED, alpha=0.16, rotation=24, weight="bold",
             zorder=100)


def accuracy_axes(ax, xlabel: str, chance: float = 0.5):
    """Standard accuracy panel: chance line drawn, y-limits fixed across figures."""
    ax.axhline(chance, color=GRAY, lw=1, ls=":", zorder=0)
    ax.set_ylim(chance - 0.03, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)


def wilson_errorbar(ax, xs, ks, ns, **kw):
    """Plot measured proportions k/n with Wilson intervals. Returns the proportions."""
    import numpy as np
    from celldetect.stats import wilson_interval
    p = np.array([k / n for k, n in zip(ks, ns)], dtype=float)
    lohi = np.array([wilson_interval(int(k), int(n)) for k, n in zip(ks, ns)])
    yerr = np.vstack([p - lohi[:, 0], lohi[:, 1] - p])
    ax.errorbar(np.asarray(xs, dtype=float), p, yerr=yerr, fmt="o", capsize=3, **kw)
    return p
