# benchm64

Benchmarking and exploration of large-matrix representations and algorithms
in R and Python, with a focus on matrices exceeding R's native 2^31-1 element
indexing limit.

## tenx_pca5.py

Sparse PCA on the 10x Genomics 1M-cell mouse brain dataset
([TENxBrainData / EH1039](https://bioconductor.org/packages/TENxBrainData)),
a 27,998 × 1,306,127 matrix with ~2.62 billion nonzero integer counts.

### What it does

1. Reads the 10x HDF5 sparse format directly via `h5py`
2. Builds a `scipy.sparse` CSC matrix with `int64` indices (the `indptr` array
   exceeds the int32 range at 2.62B nonzeros)
3. Normalizes to 10,000 counts per cell and applies log1p — in-place on
   `X.data` to avoid copying the full nonzero array
4. Selects the top `--kgenes` × 1000 highly variable genes by per-gene
   variance, computed via chunked `np.bincount` on the nonzero indices
5. Runs randomized SVD via `fbpca.pca()` with `raw=True` (operates directly
   on the sparse CSR matrix without densification)
6. Saves cell PC coordinates, gene loadings, singular values, and HVG names

### Dependencies

```
pip install h5py numpy scipy fbpca
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `TENX_MBRAIN_H5_PATH` | **yes** | Path to the 10x 1M-cell mouse brain HDF5 file |
| `TENX_OUT_DIR` | no | Output directory for `.npy` and `.txt` files (default: current working directory) |
| `TENX_HDF5_GROUP` | no | HDF5 group name for the matrix (default: `mm10`) |

The HDF5 file can be obtained via Bioconductor's ExperimentHub in R:

```r
library(ExperimentHub)
hub  <- ExperimentHub()
path <- path(hub[["EH1039"]])   # prints the local cache path
```

### Usage

```bash
export TENX_MBRAIN_H5_PATH=/path/to/136be1528461d_1039
export TENX_OUT_DIR=/path/to/output    # optional
export TENX_HDF5_GROUP=mm10            # optional

# defaults: 3000 HVGs, 50 PCs -- outputs named *_3K.*
python3 tenx_pca5.py

# 5000 HVGs, 100 PCs -- outputs named *_5K.*
python3 tenx_pca5.py --kgenes 5 --kpcs 100
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--kgenes` | `3` | Number of highly variable genes to select, in thousands |
| `--kpcs` | `50` | Number of principal components to compute |

### Outputs

Output files are written to `TENX_OUT_DIR` with a suffix reflecting `--kgenes`:

| File | Description |
|---|---|
| `tenx_cell_pcs_<k>K.npy` | float32 array (n\_cells × k\_pcs): cell coordinates in PC space |
| `tenx_gene_load_<k>K.npy` | float32 array (k\_pcs × n\_hvg): gene loadings |
| `tenx_singular_vals_<k>K.npy` | float64 array (k\_pcs,): singular values |
| `tenx_hvg_names_<k>K.txt` | selected HVG gene names, one per line |

### Design notes

- `indptr` in the HDF5 file reaches ~2.62 × 10⁹, exceeding the int32 maximum
  (2.147 × 10⁹). Both `indices` and `indptr` are read/stored as `int64`.
- Column scaling and HVG variance computation are chunked to avoid large
  intermediate allocations; peak extra RAM is ~800 MB per chunk.
- `fbpca` is called with `raw=True` because `raw=False` internally calls
  `A.conj().T` in a way that densifies sparse input. Centering has modest
  effect on top PCs for log-normalized HVG data.
