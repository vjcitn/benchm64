#!/usr/bin/env python3
"""
TENx 1M-cell mouse brain: in-memory sparse PCA
scipy.sparse + fbpca randomized SVD

Memory-conservative version for machines with ~60 GB RAM and no swap.

Key changes vs tenx_pca7:
  - X.indices stored as int32 (row indices 0..27997 fit): 10.5 GB not 21 GB.
  - Row selection uses a two-pass chunked approach that never materialises the
    full new_row_for_nz (10.5 GB) or keep (2.6 GB) arrays globally:
      Pass 1 (count): compute new_indptr in 50K-column chunks — tiny RAM.
      Pass 2 (fill):  pre-allocate new_data + new_indices, fill chunk by chunk.
    Peak: X (21 GB) + new_data (6.9 GB) + new_indices (6.9 GB) + ~0.5 GB
    chunk temporaries + ~17 GB other processes ≈ 52 GB.

Usage:
    python tenx_pca8.py [--kgenes 3] [--kpcs 50]

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
    indices  = grp["indices"][:]
    indptr   = grp["indptr"][:]
    barcodes = grp["barcodes"][:].astype(str)
    genes    = grp["gene_names"][:].astype(str)

n_genes = len(genes)
n_cells = len(barcodes)
print(f"  {n_genes:,} genes x {n_cells:,} cells, {len(data):,} nonzeros "
      f"({time.perf_counter()-t0:.1f}s)", flush=True)

# -- 2. Build CSC matrix ------------------------------------------------------
# Row indices (0..27997) fit in int32: saves 10.5 GB vs int64 (21 GB -> 10.5 GB).
# Column pointers reach 2.62B so indptr must stay int64.
print("Building CSC matrix ...", flush=True)

X = sp.csc_matrix(
    (data,
     indices.astype(np.int32),   # int32: 10.5 GB  (int64 would be 21 GB)
     indptr.astype(np.int64)),
    shape=(n_genes, n_cells)
)
del data, indices, indptr
print(f"  shape={X.shape}, nnz={X.nnz:,}, "
      f"index dtype={X.indices.dtype}, indptr dtype={X.indptr.dtype}", flush=True)
print(f"  footprint: data {X.data.nbytes/1e9:.1f} GB  "
      f"indices {X.indices.nbytes/1e9:.1f} GB", flush=True)

# -- 3. Library-size normalization + log1p ------------------------------------
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

# -- 4b. Two-pass chunked row selection from CSC ------------------------------
# Avoids materialising new_row_for_nz (10.5 GB) and keep (2.6 GB) globally.
#
# Pass 1 (count): for each 50K-column chunk, map row indices via row_map,
#   count kept nonzeros per column, accumulate into new_indptr.
#   Peak extra RAM: ~0.5 GB chunk temporaries only.
#
# Pass 2 (fill): pre-allocate new_data + new_indices from total_kept,
#   fill them chunk by chunk recomputing the mapping on the fly.
#   Peak: X (21 GB) + new_data (6.9 GB) + new_indices (6.9 GB) + ~0.5 GB = 35 GB.

CHUNK_COLS = 50_000

row_map = np.full(n_genes, -1, dtype=np.int32)
row_map[top_genes] = np.arange(N_HVG, dtype=np.int32)

print(f"  Pass 1: counting kept nonzeros per column ...", flush=True)
new_nnz_per_col = np.zeros(n_cells, dtype=np.int32)

for j in range(0, n_cells, CHUNK_COLS):
    jj = min(j + CHUNK_COLS, n_cells)
    p0 = int(X.indptr[j])
    p1 = int(X.indptr[jj])
    if p1 <= p0:
        continue
    chunk_keep = row_map[X.indices[p0:p1]] >= 0        # bool, ≤100M
    # Per-column keep counts via cumsum diff (handles empty columns correctly)
    cs = np.zeros(p1 - p0 + 1, dtype=np.int32)
    np.cumsum(chunk_keep.astype(np.int32), out=cs[1:])  # ≤100M × 4 = 400 MB
    col_s = (X.indptr[j:jj]     - p0).astype(np.intp)
    col_e = (X.indptr[j+1:jj+1] - p0).astype(np.intp)
    new_nnz_per_col[j:jj] = cs[col_e] - cs[col_s]

new_indptr = np.zeros(n_cells + 1, dtype=np.int64)
np.cumsum(new_nnz_per_col, out=new_indptr[1:])
del new_nnz_per_col
total_kept = int(new_indptr[-1])
print(f"  total kept nonzeros: {total_kept:,}", flush=True)

print(f"  Pass 2: filling pre-allocated arrays ...", flush=True)
new_data    = np.empty(total_kept, dtype=np.float32)  # 6.9 GB
new_indices = np.empty(total_kept, dtype=np.int32)    # 6.9 GB
# Peak: X (21) + new_data (6.9) + new_indices (6.9) + chunk temps (~0.5) = 35 GB

out_pos = 0
for j in range(0, n_cells, CHUNK_COLS):
    jj = min(j + CHUNK_COLS, n_cells)
    p0 = int(X.indptr[j])
    p1 = int(X.indptr[jj])
    if p1 <= p0:
        continue
    chunk_new_row = row_map[X.indices[p0:p1]]   # int32, ≤100M × 4 = 400 MB
    chunk_keep    = chunk_new_row >= 0            # bool,  ≤100M × 1 = 100 MB
    n_kept = int(chunk_keep.sum())
    if n_kept > 0:
        new_data[out_pos:out_pos + n_kept]    = X.data[p0:p1][chunk_keep]
        new_indices[out_pos:out_pos + n_kept] = chunk_new_row[chunk_keep]
    out_pos += n_kept

del X, row_map   # free 21 GB

X_hvg = sp.csc_matrix(
    (new_data, new_indices, new_indptr),
    shape=(N_HVG, n_cells)
)
del new_data, new_indices, new_indptr

print(f"  HVG subset: {X_hvg.shape}, nnz={X_hvg.nnz:,}  "
      f"({time.perf_counter()-t0:.1f}s)", flush=True)

# -- 5. PCA via fbpca ---------------------------------------------------------
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
