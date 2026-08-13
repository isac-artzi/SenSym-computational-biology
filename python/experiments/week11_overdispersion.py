"""Week 11 — real data II: the theory meets overdispersion.

Chapter: docs/s/week11.html (Lab 1, Lab 2). MILESTONE M4 figure.
Figures:
  figures/week11_fano.png          — the distribution of per-gene Fano
      factors against expression level. Poisson says 1. The data says
      otherwise, and by how much is the result.
  figures/week11_downsample.png    — the headline real-data curve: accuracy
      against downsampling fraction, with the Poisson prediction and the
      negative-binomial prediction both drawn. Which one tracks the data?

Runs on the cached PBMC 3k if present, otherwise on SURROGATE data with a
stamp. Note that the surrogate is deliberately overdispersed, so the SHAPE
of these figures is meaningful even before the download — but the numbers
are not, and the stamp says so.

Needs implemented: realdata.gene_fano, negative_binomial_dispersion,
                   d_prime_overdispersed, downsample_matrix, estimate_rates,
                   select_markers; the Week 8 classifiers.
Expected runtime: ~40 s.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import (BLUE, GRAY, ORANGE, PURPLE, TEAL, accuracy_axes,
                     save_figure, stamp_not_real, stamp_seed)

from celldetect.classify import LogisticRegression, accuracy, train_test_split
from celldetect.detection import accuracy_from_d_prime, combine_d_prime, d_prime
from celldetect.realdata import (
    d_prime_overdispersed,
    downsample_matrix,
    estimate_rates,
    gene_fano,
    negative_binomial_dispersion,
    select_markers,
)

from week10_markers import load_data

SEED = 111
KEEP_FRACTIONS = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
N_MARKERS = 32

# Set to True by main() when the script had to fall back to stand-in data.
# week14_report_figures.py reads this and refuses to put the figure in the
# report. Do not set it by hand.
USED_STAND_IN = False


def main(fast: bool = False):
    global USED_STAND_IN
    rng = np.random.default_rng(SEED)
    paths = []
    X, y, genes, types, is_real = load_data(rng, fast)
    USED_STAND_IN = not is_real
    print(f"[week11] dataset: {X.shape[0]} cells x {X.shape[1]} genes, real={is_real}")

    # --- Figure 1: is anything Poisson? ------------------------------------
    fano = gene_fano(X)
    mean = X.mean(axis=0)
    phi = negative_binomial_dispersion(X)
    ok = np.isfinite(fano) & (mean > 0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax1.loglog(mean[ok], fano[ok], ".", ms=4, color=BLUE, alpha=0.5)
    ax1.axhline(1.0, color=ORANGE, lw=2, label="Poisson prediction")
    ax1.set_xlabel("mean count per cell")
    ax1.set_ylabel("Fano factor (var / mean)")
    ax1.set_title(f"median Fano = {np.nanmedian(fano[ok]):.2f}")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=9)

    ax2.hist(np.log10(np.maximum(phi[ok], 1e-4)), bins=40, color=PURPLE, alpha=0.75)
    ax2.axvline(np.log10(max(np.nanmedian(phi[ok]), 1e-4)), color=ORANGE, lw=2,
                label=f"median $\\phi$ = {np.nanmedian(phi[ok]):.3f}")
    ax2.set_xlabel(r"$\log_{10}$ dispersion $\phi$   (Var $= m + \phi m^2$)")
    ax2.set_ylabel("genes")
    ax2.set_title("How far from Poisson, per gene")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    stamp_seed(ax2, SEED, "real data" if is_real else "SURROGATE")
    if not is_real:
        stamp_not_real(fig)
    fig.tight_layout()
    paths.append(save_figure(fig, "week11_fano"))

    # --- Figure 2: the downsampling curve ----------------------------------
    Xtr_full, Xte_full, ytr, yte = train_test_split(X, y, 0.3, rng)
    idx = select_markers(Xtr_full, ytr, N_MARKERS)     # honest: training only
    rates = estimate_rates(Xtr_full[:, idx], ytr)
    phi_m = negative_binomial_dispersion(Xtr_full[:, idx])
    phi_m = np.nan_to_num(phi_m, nan=0.0)

    measured, pois_pred, nb_pred = [], [], []
    for q in KEEP_FRACTIONS:
        Xtr = downsample_matrix(Xtr_full[:, idx], q, rng)
        Xte = downsample_matrix(Xte_full[:, idx], q, rng)
        clf = LogisticRegression(lr=0.5, n_iter=1200).fit(Xtr, ytr)
        measured.append(accuracy(yte, clf.predict(Xte)))
        # Thinning scales the rates by q; the dispersion phi is scale-free
        # under a gamma-Poisson mixture only in the limit — Week 11's
        # exercise 11.5 asks what actually happens to phi under thinning.
        r1, r2 = q * rates[0], q * rates[1]
        pois_pred.append(float(accuracy_from_d_prime(
            combine_d_prime(d_prime(r1, r2)))))
        nb_pred.append(float(accuracy_from_d_prime(
            combine_d_prime(d_prime_overdispersed(r1, r2, phi_m)))))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.semilogx(KEEP_FRACTIONS, pois_pred, "--", color=ORANGE, lw=2,
                label="Poisson theory")
    ax.semilogx(KEEP_FRACTIONS, nb_pred, "-", color=PURPLE, lw=2,
                label=r"negative-binomial theory (measured $\phi$)")
    ax.semilogx(KEEP_FRACTIONS, measured, "o", color=TEAL, ms=7,
                label="measured (logistic regression)")
    accuracy_axes(ax, "fraction of reads kept")
    ax.set_title(f"Downsampling {N_MARKERS} markers: which model tracks the data?")
    ax.legend(fontsize=9, loc="lower right")
    stamp_seed(ax, SEED, "real data" if is_real else "SURROGATE")
    if not is_real:
        stamp_not_real(fig)
    paths.append(save_figure(fig, "week11_downsample"))

    print(f"[week11] seed={SEED}, markers={N_MARKERS}, real={is_real}")
    print(f"[week11] median Fano = {np.nanmedian(fano[ok]):.3f}, "
          f"median phi = {np.nanmedian(phi[ok]):.4f}")
    print("[week11]  keep   measured   Poisson    NB     m-Pois    m-NB")
    for q, m, p, nb in zip(KEEP_FRACTIONS, measured, pois_pred, nb_pred):
        print(f"[week11] {q:5.2f}    {m:.4f}   {p:.4f}  {nb:.4f}  "
              f"{m - p:+.4f}  {m - nb:+.4f}")
    rmse_p = float(np.sqrt(np.mean((np.array(measured) - np.array(pois_pred)) ** 2)))
    rmse_nb = float(np.sqrt(np.mean((np.array(measured) - np.array(nb_pred)) ** 2)))
    print(f"[week11] RMSE vs Poisson = {rmse_p:.4f}, vs NB = {rmse_nb:.4f}")
    print(f"[week11] wrote {paths}")
    return paths


if __name__ == "__main__":
    main()
