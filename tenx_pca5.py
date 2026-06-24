#!/usr/bin/env python3
"""
TENx 1M-cell mouse brain: in-memory sparse PCA
scipy.sparse + fbpca randomized SVD

Usage:
    python tenx_pca5.py [--kgenes 3]

    --kgenes  Number of highly variable genes to select, in thousands (default: 3).
              e.g. --kgenes 5 selects 5000 HVGs and writes outputs with _5K suffix.
    --kpcs    Number of principal components to compute (default: 50).

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
# Required:
#   TENX_MBRAIN_H5_PATH  path to the 10x 1M-cell mouse brain HDF5 file
# Optional:
#   TENX_OUT_DIR         directory for output .npy/.txt files (default: cwd)
#   TENX_HDF5_GROUP      HDF5 group name for the matrix (default: mm10)

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
    f.visit(lambda name: print(f"  {name}  {f[name].dtype if hasattr(f[name], 'dtype') else ''}"))

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
# full 2.62B-element data array into a new sparse matrix (~10 GB).
# Chunked at 100K columns to keep np.repeat temporaries ~400 MB each.
print("Normalizing ...", flush=True)
t0 = time.perf_counter()

col_sums  = np.asarray(X.sum(axis=0)).ravel().clip(1.0)
col_scale = (1e4 / col_sums).astype(np.float32)

for j in range(0, n_cells, 100_000):
    jj           = min(j + 100_000, n_cells)
    p0, p1       = int(X.indptr[j]), int(X.indptr[jj])
    if p1 > p0:
        nnz_per_col   = np.diff(X.indptr[j:jj + 1]).astype(np.int32)
        X.data[p0:p1] *= np.repeat(col_scale[j:jj], nnz_per_col)

X.data = np.log1p(X.data)
print(f"  Done ({time.perf_counter()-t0:.1f}s)", flush=True)

# -- 4. Highly variable gene selection ----------------------------------------
# Accumulate row sums and sums-of-squares via np.bincount on X.indices.
# Avoids X.power(2) which copies the full 2.62B indices array (~21 GB).
# Chunked at 50M nonzeros (~800 MB temporaries per iteration).
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
top_genes = np.argsort(variance)[::-1][:N_HVG]
X_hvg     = X[top_genes, :]
genes_hvg = genes[top_genes]
print(f"  HVG subset: {X_hvg.shape}, nnz={X_hvg.nnz:,}  ({time.perf_counter()-t0:.1f}s)", flush=True)
del X

# -- 5. PCA via fbpca ---------------------------------------------------------
# fbpca.pca() with raw=True calls A.dot(x) and A.T.dot(x) internally, both of
# which scipy sparse CSR supports efficiently. raw=True skips centering (which
# would require A.conj().T and densify the result).
# Convention: fbpca expects (n_samples, n_features) -> transpose to (cells, genes).

Xt = X_hvg.T.tocsr()       # (n_cells, N_HVG) -- CSR for efficient row matvecs
del X_hvg
n_c, n_g = Xt.shape

print(f"Running fbpca (k={N_PCS}, n_iter=4) on {n_c:,} x {n_g:,} CSR matrix ...", flush=True)
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
