#!/usr/bin/env python3
"""
TENx 1M-cell mouse brain: in-memory sparse PCA
scipy.sparse + choice of sklearn or fbpca randomized SVD

Memory-conservative version for machines with ~60 GB RAM and no swap.

Key design decisions (vs earlier versions):
  - X.indices kept as int64 (as stored in HDF5). Casting to int32 before
    construction is counterproductive: scipy sees nnz=2.62B > INT32_MAX and
    silently upcasts back to int64, creating a triple allocation peak of
    data(10.5) + indices_int64(21) + indices_int32(10.5) + scipy_copy(21)
    = 63 GB. Passing int64 directly gives a 31.5 GB peak.
  - Two-pass chunked row selection avoids materialising full keep/row_map
    arrays globally (peak ~46 GB numpy during fill).
  - SVD backend selectable via --svd:
      sklearn (default): uses A.T (no-copy view for real sparse matrices).
      fbpca:             uses A.conj().T which copies the full sparse matrix
                         (~13.7 GB) per power iteration. Safe on machines with
                         ample RAM and swap; dangerous on no-swap servers.

Usage:
    python tenx_pca9.py [--kgenes 3] [--kpcs 50] [--svd {sklearn,fbpca}]

    --kgenes  HVGs to select in thousands (default: 3 -> 3000 genes)
    --kpcs    Number of principal components (default: 50)
    --svd     SVD backend: sklearn (default) or fbpca
              WARNING: fbpca copies the full sparse matrix (~13.7 GB) in
              each power iteration via A.conj().T. On machines with no swap
              this will OOM; use sklearn there.

Environment variables:
    TENX_MBRAIN_H5_PATH  (required) path to the 10x HDF5 file
    TENX_OUT_DIR         (optional) output directory, default: cwd
    TENX_HDF5_GROUP      (optional) HDF5 group name, default: mm10

Requirements:
    pip install h5py numpy scipy scikit-learn
    pip install fbpca   # only needed for --svd fbpca
"""

import argparse
import os
import sys
import h5py
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import time

# -- Arguments ----------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--kgenes", type=int, default=3,
                    help="HVGs to select in thousands (default: 3 -> 3000 genes)")
parser.add_argument("--kpcs", type=int, default=50,
                    help="Number of principal components to compute (default: 50)")
parser.add_argument("--svd", choices=["sklearn", "fbpca"], default="sklearn",
                    help=("SVD backend (default: sklearn). "
                          "WARNING: fbpca copies the full sparse matrix per "
                          "power iteration via A.conj().T (~13.7 GB each); "
                          "only use on machines with ample RAM and swap."))
args = parser.parse_args()

N_HVG  = args.kgenes * 1000
N_PCS  = args.kpcs
SUFFIX = f"_{args.kgenes}K"

if args.svd == "fbpca":
    try:
        import fbpca
    except ImportError:
        sys.exit("Error: fbpca is not installed. Run: pip install fbpca")
    print("WARNING: fbpca copies the full sparse matrix (~13.7 GB) per power "
          "iteration.\n         Use --svd sklearn on machines without swap.",
          flush=True)

# -- Paths from environment variables -----------------------------------------
H5_PATH = os.environ.get("TENX_MBRAIN_H5_PATH")
if not H5_PATH:
    sys.exit("Error: environment variable TENX_MBRAIN_H5_PATH is not set.\n"
             "  export TENX_MBRAIN_H5_PATH=/path/to/136be1528461d_1039")

OUT_DIR = os.environ.get("TENX_OUT_DIR", os.getcwd())
GROUP   = os.environ.get("TENX_HDF5_GROUP", "mm10")

print(f"HVGs: {N_HVG:,}  |  PCs: {N_PCS}  |  SVD: {args.svd}  "
      f"|  suffix: {SUFFIX}", flush=True)
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
# Pass indices and indptr as-is (both int64 from HDF5).
# Do NOT cast indices to int32: scipy sees nnz=2.62B > INT32_MAX and silently
# upcasts back to int64, creating a 63 GB peak (original int64 + int32 temp +
# scipy's int64 copy). Passing int64 directly peaks at 31.5 GB.
print("Building CSC matrix ...", flush=True)

X = sp.csc_matrix(
    (data, indices, indptr),
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
# X.indices is int64 (21 GB). new_indices is int32 (6.9 GB) since output row
# indices are 0..N_HVG-1, which fits in int32.
# Peak: X (31.5 GB) + new_data (6.9 GB) + new_indices (6.9 GB) + ~0.5 GB = 46 GB.
CHUNK_COLS = 50_000

row_map = np.full(n_genes, -1, dtype=np.int32)
row_map[top_genes] = np.arange(N_HVG, dtype=np.int32)

print(f"  Pass 1: counting kept nonzeros per column ...", flush=True)
new_nnz_per_col = np.zeros(n_cells, dtype=np.int32)

for j in range(0, n_cells, CHUNK_COLS):
    jj = min(j + CHUNK_COLS, n_cells)
    p0, p1 = int(X.indptr[j]), int(X.indptr[jj])
    if p1 <= p0:
        continue
    chunk_keep = row_map[X.indices[p0:p1]] >= 0
    cs = np.zeros(p1 - p0 + 1, dtype=np.int32)
    np.cumsum(chunk_keep.astype(np.int32), out=cs[1:])
    col_s = (X.indptr[j:jj]     - p0).astype(np.intp)
    col_e = (X.indptr[j+1:jj+1] - p0).astype(np.intp)
    new_nnz_per_col[j:jj] = cs[col_e] - cs[col_s]

new_indptr = np.zeros(n_cells + 1, dtype=np.int64)
np.cumsum(new_nnz_per_col, out=new_indptr[1:])
del new_nnz_per_col
total_kept = int(new_indptr[-1])
print(f"  total kept nonzeros: {total_kept:,}", flush=True)

print(f"  Pass 2: filling pre-allocated arrays ...", flush=True)
new_data    = np.empty(total_kept, dtype=np.float32)
new_indices = np.empty(total_kept, dtype=np.int32)

out_pos = 0
for j in range(0, n_cells, CHUNK_COLS):
    jj = min(j + CHUNK_COLS, n_cells)
    p0, p1 = int(X.indptr[j]), int(X.indptr[jj])
    if p1 <= p0:
        continue
    chunk_new_row = row_map[X.indices[p0:p1]]
    chunk_keep    = chunk_new_row >= 0
    n_kept = int(chunk_keep.sum())
    if n_kept > 0:
        new_data[out_pos:out_pos + n_kept]    = X.data[p0:p1][chunk_keep]
        new_indices[out_pos:out_pos + n_kept] = chunk_new_row[chunk_keep]
    out_pos += n_kept

del X, row_map

X_hvg = sp.csc_matrix(
    (new_data, new_indices, new_indptr),
    shape=(N_HVG, n_cells)
)
del new_data, new_indices, new_indptr

print(f"  HVG subset: {X_hvg.shape}, nnz={X_hvg.nnz:,}  "
      f"({time.perf_counter()-t0:.1f}s)", flush=True)

# -- 5. PCA -------------------------------------------------------------------
# sklearn: uses A.T (no-copy view for real sparse matrices). Safe on all machines.
# fbpca:   uses A.conj().T which copies the full sparse matrix (~13.7 GB) per
#          power iteration. Only use on machines with sufficient RAM and swap.

print(f"Running SVD (k={N_PCS}, n_iter=4, backend={args.svd}) ...", flush=True)
t0 = time.perf_counter()

if args.svd == "sklearn":
    # X_hvg.T is a CSR-compatible view of X_hvg (no copy, no tocsr needed).
    # sklearn accepts scipy sparse matrices and uses A.T not A.conj().T.
    Xt = X_hvg.T
    del X_hvg
    n_c, n_g = Xt.shape
    U, s, Vt = randomized_svd(Xt, n_components=N_PCS, n_iter=4, random_state=0)

else:  # fbpca
    # fbpca expects (n_samples, n_features); needs CSR for efficient row matvecs.
    Xt = X_hvg.T.tocsr()
    del X_hvg
    n_c, n_g = Xt.shape
    U, s, Vt = fbpca.pca(Xt, k=N_PCS, raw=True, n_iter=4)

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# -- 6. Results ---------------------------------------------------------------
cell_pcs  = (U * s).astype(np.float32)
gene_load = Vt.astype(np.float32)

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
