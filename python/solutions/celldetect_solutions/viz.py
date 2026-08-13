"""[PROVIDED] Plot helpers shared by the experiment drivers. [SOLUTIONS]

Identical to `celldetect.viz` by design — there is nothing to implement
here. Kept in both packages so `run_solution_check.py` can copy the
solutions wholesale and still have the imports resolve.
"""

import numpy as np

# Course palette — matches docs/assets/css/style.css and cd.js.
BLUE = "#2563eb"
ORANGE = "#ea580c"
TEAL = "#0d9488"
PURPLE = "#7c3aed"
GOLD = "#b45309"
GRAY = "#94a3b8"
MAGENTA = "#c026d3"
RED = "#b91c1c"

# The three aims always get the same colour, in every figure of the project.
AIM_COLORS = {"theory": ORANGE, "simulation": BLUE, "beads": MAGENTA,
              "real": TEAL}


def stamp_seed(ax, seed, extra: str = ""):
    """Write the seed in the corner of an axes. Every figure carries one."""
    text = f"seed {seed}" + (f" · {extra}" if extra else "")
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=GRAY)


def stamp_surrogate(fig):
    """Mark a figure built from surrogate rather than real data. Loudly."""
    fig.text(0.5, 0.5, "SURROGATE DATA\nnot a result", ha="center", va="center",
             fontsize=34, color=RED, alpha=0.16, rotation=24, weight="bold",
             zorder=100)


def accuracy_axes(ax, xlabel: str, chance: float = 0.5):
    """Standard accuracy panel: y in [chance - 0.03, 1.02], chance line drawn."""
    ax.axhline(chance, color=GRAY, lw=1, ls=":", zorder=0)
    ax.set_ylim(chance - 0.03, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)


def plot_with_wilson(ax, xs, ks, ns, **kw):
    """Scatter measured proportions k/n with Wilson error bars."""
    from .stats import wilson_interval
    xs = np.asarray(xs, dtype=float)
    p = np.array([k / n for k, n in zip(ks, ns)], dtype=float)
    lo_hi = np.array([wilson_interval(int(k), int(n)) for k, n in zip(ks, ns)])
    yerr = np.vstack([p - lo_hi[:, 0], lo_hi[:, 1] - p])
    ax.errorbar(xs, p, yerr=yerr, fmt="o", capsize=3, **kw)
    return p
