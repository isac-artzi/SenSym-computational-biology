#!/usr/bin/env python3
"""Download the 10x PBMC 3k dataset once and write a compact cache.

Usage:
    pip install -r requirements-optional.txt      # scanpy + anndata
    python experiments/fetch_pbmc3k.py

Writes python/data/pbmc3k_cache.npz containing:
    X      (n_cells, n_genes) int32 raw counts, genes filtered to those
           detected in at least MIN_CELLS cells
    y      integer cell-type labels
    genes  gene names
    types  the label names, in order

Why a cache and not the .h5ad: the raw download is ~35 MB and scanpy is a
heavy dependency. Everything after this script runs on numpy alone, so the
project's reproducibility does not depend on a scanpy version.

Why these two types: CD14+ monocytes and CD4 T cells are the two largest,
cleanest populations in PBMC 3k and are separated by well-known markers
(LYZ, S100A8 vs IL7R, CD3D). Choosing an EASY pair is deliberate — the
project is measuring how accuracy scales with depth, not how hard a
particular pair is, and an easy pair keeps the accuracy curve away from the
saturating ends where everything looks the same.

The cell-type labels come from the standard Leiden clustering + marker
annotation in the scanpy PBMC 3k tutorial. They are ANNOTATIONS, not ground
truth: the report must say so. Anything you conclude is conditional on that
labelling being right.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "data", "pbmc3k_cache.npz"))
MIN_CELLS = 10

# Marker sets used to name the clusters (scanpy tutorial conventions).
MARKERS = {
    "CD14_monocytes": ["LYZ", "S100A8", "S100A9", "CST3"],
    "CD4_T_cells": ["IL7R", "CD3D", "CD3E", "LTB"],
}


def main() -> int:
    try:
        import scanpy as sc
    except ImportError:
        print("scanpy is not installed. Run:\n"
              "    pip install -r requirements-optional.txt", file=sys.stderr)
        return 1

    print("downloading / loading PBMC 3k ...")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    raw_counts = adata.X.copy()          # keep the RAW counts; we need integers

    # Standard preprocessing ONLY to obtain cluster labels. The counts we
    # cache are the raw ones — normalizing them would destroy the Poisson
    # structure the entire project is about. This separation is the single
    # most important line in this script.
    proc = adata.copy()
    sc.pp.filter_cells(proc, min_genes=200)
    sc.pp.filter_genes(proc, min_cells=3)
    sc.pp.normalize_total(proc, target_sum=1e4)
    sc.pp.log1p(proc)
    sc.pp.highly_variable_genes(proc, n_top_genes=2000)
    proc = proc[:, proc.var.highly_variable]
    sc.pp.scale(proc, max_value=10)
    sc.tl.pca(proc, svd_solver="arpack")
    sc.pp.neighbors(proc, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(proc, resolution=0.5, key_added="leiden", flavor="igraph",
                 n_iterations=2, directed=False)

    # Score each cluster on the two marker sets and take the argmax.
    import pandas as pd
    scores = {}
    for name, genes in MARKERS.items():
        present = [g for g in genes if g in proc.raw.var_names] if proc.raw is not None else []
        if not present:
            present = [g for g in genes if g in adata.var_names]
        expr = np.asarray(adata[:, present].X.todense()
                          if hasattr(adata[:, present].X, "todense")
                          else adata[:, present].X)
        expr = np.log1p(expr / np.maximum(expr.sum(axis=1, keepdims=True), 1) * 1e4)
        s = pd.Series(expr.mean(axis=1), index=adata.obs_names)
        scores[name] = s.reindex(proc.obs_names).values

    df = pd.DataFrame(scores, index=proc.obs_names)
    df["leiden"] = proc.obs["leiden"].values
    per_cluster = df.groupby("leiden").mean()
    assignment = per_cluster.idxmax(axis=1)
    # Only keep clusters whose winning score clearly beats the other one;
    # ambiguous clusters are dropped rather than guessed at.
    margin = per_cluster.max(axis=1) - per_cluster.min(axis=1)
    keep_clusters = per_cluster.index[margin > 0.25]
    print("cluster assignment:\n", per_cluster.round(3))
    print("keeping clusters:", list(keep_clusters))

    types = list(MARKERS.keys())
    leiden = np.asarray(proc.obs["leiden"].values).astype(str)
    keep = np.isin(leiden, np.asarray(keep_clusters).astype(str))
    lab = np.array([types.index(assignment[c]) if c in assignment.index else -1
                    for c in leiden])
    keep &= lab >= 0

    # Map the kept processed cells back to raw counts.
    idx = {n: i for i, n in enumerate(adata.obs_names)}
    rows = np.array([idx[n] for n in np.asarray(proc.obs_names)[keep]])
    X = raw_counts[rows]
    X = np.asarray(X.todense() if hasattr(X, "todense") else X).astype(np.int32)
    y = lab[keep].astype(np.int32)

    detected = (X > 0).sum(axis=0) >= MIN_CELLS
    X = X[:, detected]
    genes = np.asarray(adata.var_names)[detected]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, genes=genes, types=np.array(types))
    print(f"wrote {OUT}")
    print(f"  cells: {X.shape[0]}  ({np.bincount(y)} per type: {types})")
    print(f"  genes: {X.shape[1]} (detected in >= {MIN_CELLS} cells)")
    print(f"  median library size: {np.median(X.sum(axis=1)):.0f} counts/cell")
    print("\nNOTE: the labels are cluster annotations, not ground truth. "
          "Say so in the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
