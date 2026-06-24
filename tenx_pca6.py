#!/usr/bin/env python3
"""
TENx 1M-cell mouse brain: in-memory sparse PCA
scipy.sparse + fbpca randomized SVD

Memory-conservative version: avoids the scipy CSC->CSR format conversion
that occurs internally during row indexing (X[rows, :] on a CSC matrix).
On a 60 GB machine that conversion would peak at 2x matrix size (~63 GB)
and trigger OOM. Instead, rows are selected directly from CSC format using
boolean masking and np.add.reduceat.

Usage:
    python tenx_pca6.py [--kgenes 3] [--kpcs 50]

    --kgenes  HVGs to select in thousands (default: 3 -> 3000 genes)
    --kpcs    Number of principal components (default: 50)

Environment variables:
    TENX_MBRAIN_H5_PATH  (required) path to the 10x HDF5 file
    TENX_OUT_DIR         (optional) output directory, default: cwd
    TENX_HDF5_GROUP      (optional) HDF5 group name, default: mm10

Requirements:
    pip install h5py numpy scipy fbpca
"""

import argparse
import os
import sys
import h5py
import numpy as np
import scipy.sparse as sp
import fbpca
import time

# -- Arguments ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--kgenes", type=int, default=3,
                    help="HVGs to select in thousands (default: 3 -> 3000 genes)")
parser.add_argument("--kpcs", type=int, default=50,
                    help="Number of principal components to compute (default: 50)")
args = parser.parse_args()

N_HVG  = args.kgenes * 1000
N_PCS  = args.kpcs
SUFFIX = f"_{args.kgenes}K"

# -- Paths from environment variables -----------------------------------------
H5_PATH = os.environ.get("TENX_MBRAIN_H5_PATH")
if not H5_PATH:
    sys.exit("Error: environment variable TENX_MBRAIN_H5_PATH is not set.\n"
             "  export TENX_MBRAIN_H5_PATH=/path/to/136be1528461d_1039")

OUT_DIR = os.environ.get("TENX_OUT_DIR", os.getcwd())
GROUP   = os.environ.get("TENX_HDF5_GROUP", "mm10")

print(f"HVGs: {N_HVG:,}  |  PCs: {N_PCS}  |  output suffix: {SUFFIX}", flush=True)
print(f"H5:   {H5_PATH}", flush=True)
print(f"Out:  {OUT_DIR}", flush=True)

# -- 0. Inspect HDF5 structure ------------------------------------------------
print("\nHDF5 structure:", flush=True)
with h5py.File(H5_PATH, "r") as f:
    f.visit(lambda name: print(f"  {name}  "
                               f"{f[name].dtype if hasattr(f[name], 'dtype') else ''}"))

# -- 1. Read sparse arrays from HDF5 -----------------------------------------
print("\nReading HDF5 ...", flush=True)
t0 = time.perf_counter()

with h5py.File(H5_PATH, "r") as f:
    grp      = f[GROUP]
    data     = grp["data"][:].astype(np.float32)
    indices  = grp["indices"][:]        # already int64
    indptr   = grp["indptr"][:]         # already int64
    barcodes = grp["barcodes"][:].astype(str)
    genes    = grp["gene_names"][:].astype(str)

n_genes = len(genes)
n_cells = len(barcodes)
print(f"  {n_genes:,} genes x {n_cells:,} cells, {len(data):,} nonzeros "
      f"({time.perf_counter()-t0:.1f}s)", flush=True)

# -- 2. Build CSC matrix ------------------------------------------------------
print("Building CSC matrix ...", flush=True)

X = sp.csc_matrix((data, indices, indptr), shape=(n_genes, n_cells))
del data, indices, indptr
print(f"  shape={X.shape}, nnz={X.nnz:,}, "
      f"index dtype={X.indices.dtype}, indptr dtype={X.indptr.dtype}", flush=True)

# -- 3. Library-size normalization + log1p ------------------------------------
# In-place column scaling via indptr -- avoids X.multiply() which copies the
# full data array (~10 GB). Chunked at 100K columns (~400 MB temporaries).
print("Normalizing ...", flush=True)
t0 = time.perf_counter()

col_sums  = np.asarray(X.sum(axis=0)).ravel().clip(1.0)
col_scale = (1e4 / col_sums).astype(np.float32)
del col_sums

for j in range(0, n_cells, 100_000):
    jj           = min(j + 100_000, n_cells)
    p0, p1       = int(X.indptr[j]), int(X.indptr[jj])
    if p1 > p0:
        nnz_per_col   = np.diff(X.indptr[j:jj + 1]).astype(np.int32)
        X.data[p0:p1] *= np.repeat(col_scale[j:jj], nnz_per_col)

del col_scale
X.data = np.log1p(X.data)
print(f"  Done ({time.perf_counter()-t0:.1f}s)", flush=True)

# -- 4. Highly variable gene selection ----------------------------------------
# Variance computed via chunked np.bincount on X.indices -- avoids X.power(2)
# which would copy the full indices array (~21 GB).
print(f"Selecting top {N_HVG:,} HVGs ...", flush=True)
t0 = time.perf_counter()

row_sum    = np.zeros(n_genes, dtype=np.float64)
row_sum_sq = np.zeros(n_genes, dtype=np.float64)

for start in range(0, X.nnz, 50_000_000):
    end  = min(start + 50_000_000, X.nnz)
    d    = X.data[start:end].astype(np.float64)
    idx  = X.indices[start:end].astype(np.intp)
    row_sum    += np.bincount(idx, weights=d,    minlength=n_genes)
    row_sum_sq += np.bincount(idx, weights=d**2, minlength=n_genes)

variance  = row_sum_sq / n_cells - (row_sum / n_cells)**2
del row_sum, row_sum_sq
top_genes = np.argsort(variance)[::-1][:N_HVG]
genes_hvg = genes[top_genes]

# -- 4b. Row selection directly from CSC -------------------------------------
# X[top_genes, :] on a CSC matrix internally calls self.tocsr()[rows], creating
# a full 31.5 GB CSR copy before selecting. On a 60 GB machine this peaks at
# ~63 GB and triggers OOM.
#
# Instead, select rows from CSC directly:
#   1. Map each nonzero's row index to its new index (-1 if not selected)
#   2. Build a boolean keep mask
#   3. Rebuild indptr using np.add.reduceat in column chunks (no full cumsum)
#   4. Filter data and indices
#
# Peak RAM: X (31.5 GB) + new_row_for_nz (10.5 GB) + keep (2.6 GB) +
#           new_data (3.4 GB) + new_indices (2.8 GB) ≈ 48 GB.

print(f"  Selecting rows from CSC without format conversion ...", flush=True)

row_map = np.full(n_genes, -1, dtype=np.int32)
row_map[top_genes] = np.arange(N_HVG, dtype=np.int32)

# new row index for every nonzero (-1 = not in HVG set)
new_row_for_nz = row_map[X.indices]          # int32, ~10.5 GB
del row_map
keep = new_row_for_nz >= 0                   # bool,  ~2.6 GB

# Rebuild indptr: for each column, count how many of its nonzeros are kept.
# np.add.reduceat handles zero-length columns (identity = 0 for add).
new_nnz_per_col = np.empty(n_cells, dtype=np.int32)
for j in range(0, n_cells, 100_000):
    jj  = min(j + 100_000, n_cells)
    p0  = int(X.indptr[j])
    p1  = int(X.indptr[jj])
    starts = (X.indptr[j:jj] - p0).astype(np.intp)
    new_nnz_per_col[j:jj] = np.add.reduceat(
        keep[p0:p1].view(np.uint8), starts)

new_indptr = np.empty(n_cells + 1, dtype=np.int64)
new_indptr[0] = 0
np.cumsum(new_nnz_per_col, out=new_indptr[1:])
del new_nnz_per_col

new_data    = X.data[keep]                              # float32, ~3.4 GB
new_indices = new_row_for_nz[keep].astype(np.int64)    # int64,   ~2.8 GB
del new_row_for_nz, keep
del X   # free 31.5 GB

X_hvg = sp.csc_matrix(
    (new_data, new_indices, new_indptr),
    shape=(N_HVG, n_cells)
)
del new_data, new_indices, new_indptr

print(f"  HVG subset: {X_hvg.shape}, nnz={X_hvg.nnz:,}  "
      f"({time.perf_counter()-t0:.1f}s)", flush=True)

# -- 5. PCA via fbpca ---------------------------------------------------------
# fbpca.pca() with raw=True calls A.dot(x) and A.T.dot(x) via scipy sparse
# CSR. raw=True skips centering (which would densify via A.conj().T).
# Convention: fbpca expects (n_samples, n_features) -> transpose to (cells, genes).

Xt = X_hvg.T.tocsr()    # (n_cells, N_HVG) -- .T on CSC is free, tocsr is small
del X_hvg
n_c, n_g = Xt.shape

print(f"Running fbpca (k={N_PCS}, n_iter=4) on {n_c:,} x {n_g:,} CSR matrix ...",
      flush=True)
t0 = time.perf_counter()

U, s, Va = fbpca.pca(Xt, k=N_PCS, raw=True, n_iter=4)

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# -- 6. Results ---------------------------------------------------------------
cell_pcs  = (U * s).astype(np.float32)
gene_load = Va.astype(np.float32)

var_explained     = s**2 / (n_c - 1)
var_explained_pct = 100 * var_explained / var_explained.sum()

print(f"\nResults")
print(f"  cell_pcs  : {cell_pcs.shape}  (cells x PCs)")
print(f"  gene_load : {gene_load.shape} (PCs x genes)")
print(f"  Top 10 singular values : {np.round(s[:10], 2)}")
print(f"  Variance explained (%) : {np.round(var_explained_pct[:10], 2)}")

# -- 7. Save ------------------------------------------------------------------
np.save(f"{OUT_DIR}/tenx_cell_pcs{SUFFIX}.npy",      cell_pcs)
np.save(f"{OUT_DIR}/tenx_gene_load{SUFFIX}.npy",     gene_load)
np.save(f"{OUT_DIR}/tenx_singular_vals{SUFFIX}.npy", s)
np.savetxt(f"{OUT_DIR}/tenx_hvg_names{SUFFIX}.txt",  genes_hvg, fmt="%s")

print(f"\nSaved to {OUT_DIR}:")
print(f"  tenx_cell_pcs{SUFFIX}.npy       {cell_pcs.nbytes/1e6:.0f} MB")
print(f"  tenx_gene_load{SUFFIX}.npy      {gene_load.nbytes/1e6:.0f} MB")
print(f"  tenx_singular_vals{SUFFIX}.npy")
print(f"  tenx_hvg_names{SUFFIX}.txt")
