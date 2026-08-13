"""Tests for celldetect.realdata — markers, downsampling, overdispersion (W10-W11).

These tests never touch the real PBMC download: they run on the surrogate
dataset and on hand-built matrices, so the suite is green on a laptop with no
network. What they check is that the CODE is right; whether the MODEL is
right is what the real data decides in Week 11.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from celldetect.realdata import (
    d_prime_overdispersed,
    downsample_matrix,
    estimate_rates,
    gene_fano,
    negative_binomial_dispersion,
    select_markers,
    surrogate_pbmc,
)


def test_select_markers_finds_the_planted_genes():
    """Genes 0-4 are planted with a large fold change; they must be picked (W10)."""
    rng = np.random.default_rng(41)
    n_genes = 60
    lam = np.full(n_genes, 5.0)
    X0 = rng.poisson(lam, size=(300, n_genes))
    lam2 = lam.copy()
    lam2[:5] = 40.0
    X1 = rng.poisson(lam2, size=(300, n_genes))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(300, int), np.ones(300, int)])
    idx = select_markers(X, y, 5)
    assert set(idx.tolist()) == {0, 1, 2, 3, 4}


def test_select_markers_respects_min_cells():
    """A gene detected in almost no cells is not a marker, however extreme (W10)."""
    rng = np.random.default_rng(42)
    X = rng.poisson(4.0, size=(200, 20))
    X[:, 7] = 0
    X[0, 7] = 500          # one cell, one huge count
    y = np.concatenate([np.zeros(100, int), np.ones(100, int)])
    idx = select_markers(X, y, 5, min_cells=10)
    assert 7 not in idx.tolist()


def test_select_markers_requires_two_classes():
    with pytest.raises(ValueError):
        select_markers(np.ones((10, 3)), np.zeros(10, int), 2)


def test_downsample_matrix_halves_the_library():
    rng = np.random.default_rng(43)
    X = rng.poisson(20.0, size=(400, 30))
    Y = downsample_matrix(X, 0.5, rng)
    assert Y.shape == X.shape
    assert (Y <= X).all()
    assert abs(Y.sum() / X.sum() - 0.5) < 0.01


def test_downsample_matrix_rejects_bad_probability():
    with pytest.raises(ValueError):
        downsample_matrix(np.ones((2, 2), int), -0.1, np.random.default_rng(0))


def test_estimate_rates_recovers_the_truth_with_enough_cells():
    """Estimated rates converge to the true ones — and are noisy before that (W11)."""
    rng = np.random.default_rng(44)
    lam0 = np.array([2.0, 10.0, 30.0])
    lam1 = np.array([4.0, 10.0, 15.0])
    X = np.vstack([rng.poisson(lam0, size=(4000, 3)),
                   rng.poisson(lam1, size=(4000, 3))])
    y = np.concatenate([np.zeros(4000, int), np.ones(4000, int)])
    R = estimate_rates(X, y)
    assert R.shape == (2, 3)
    npt.assert_allclose(R[0], lam0, rtol=0.06)
    npt.assert_allclose(R[1], lam1, rtol=0.06)


def test_gene_fano_is_one_for_poisson_and_large_for_nb():
    """The diagnostic that fails on real data (W11)."""
    rng = np.random.default_rng(45)
    X_pois = rng.poisson(15.0, size=(5000, 4))
    npt.assert_allclose(gene_fano(X_pois), 1.0, atol=0.12)
    g = rng.gamma(shape=2.0, scale=7.5, size=(5000, 4))
    X_nb = rng.poisson(g)
    assert (gene_fano(X_nb) > 3.0).all()


def test_negative_binomial_dispersion_recovers_phi():
    """Method of moments: Var = m + phi m^2 (W11)."""
    rng = np.random.default_rng(46)
    lam, phi = 20.0, 0.5
    g = rng.gamma(shape=1 / phi, scale=phi * lam, size=(40_000, 2))
    X = rng.poisson(g)
    est = negative_binomial_dispersion(X)
    npt.assert_allclose(est, phi, rtol=0.15)


def test_negative_binomial_dispersion_is_zero_for_poisson():
    rng = np.random.default_rng(47)
    X = rng.poisson(25.0, size=(20_000, 3))
    assert (negative_binomial_dispersion(X) < 0.02).all()


def test_d_prime_overdispersed_reduces_to_poisson_when_phi_is_zero():
    from celldetect.detection import d_prime
    lam1, lam2 = np.array([30.0, 5.0]), np.array([50.0, 8.0])
    npt.assert_allclose(d_prime_overdispersed(lam1, lam2, 0.0),
                        d_prime(lam1, lam2), rtol=1e-12)


def test_d_prime_overdispersed_saturates_with_depth():
    """The sharpest prediction of Week 11: with overdispersion, deeper
    sequencing eventually buys nothing at all."""
    p1, p2, phi = 0.004, 0.006, 0.4
    depths = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    dps = np.array([float(d_prime_overdispersed(D * p1, D * p2, phi)) for D in depths])
    assert (np.diff(dps) > 0).all()                 # still increasing...
    ceiling = abs(p1 - p2) / (0.5 * (p1 + p2) * math.sqrt(phi))
    assert dps[-1] < ceiling                         # ...but bounded
    assert dps[-1] / dps[-2] < 1.05                  # and essentially flat
    # By contrast the Poisson d' would have grown by sqrt(10) ~ 3.16.


def test_surrogate_pbmc_provided():
    """[PROVIDED] The surrogate is overdispersed on purpose — it is a
    stand-in for messy data, and must not look like the clean simulator."""
    X, y, genes, types = surrogate_pbmc(n_cells=400, n_genes=80,
                                        rng=np.random.default_rng(48))
    assert X.shape == (400, 80)
    assert set(np.unique(y)) == {0, 1}
    assert len(genes) == 80 and len(types) == 2
    expressed = X.mean(axis=0) > 1.0
    assert np.nanmedian(gene_fano(X)[expressed]) > 1.5
