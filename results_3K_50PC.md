# PCA results: TENx 1M-cell mouse brain (3K HVGs, 50 PCs)

Run with `python3 tenx_pca5.py --kgenes 3 --kpcs 50` on a macOS M4 machine.

## Dataset

| Property | Value |
|---|---|
| Source | TENxBrainData (Bioconductor ExperimentHub EH1039) |
| Dimensions | 27,998 genes × 1,306,127 cells |
| Nonzeros | 2,624,828,308 |
| Sparsity | 92.8% zeros |
| Storage type | int32 counts in 10x HDF5 sparse format |

## Runtime

| Step | Time |
|---|---|
| Read HDF5 | 51.6 s |
| Build CSC matrix | — |
| Normalize (log1p, lib-size) | 11.0 s |
| HVG selection (top 3,000) | 57.3 s |
| fbpca SVD (k=50, n\_iter=4) | 247.4 s |
| **Total** | **~367 s (~6 min)** |

## Variance explained

PC1 captures 92.7% of variance — typical for raw count data where library size
dominates even after per-cell normalization to 10,000 counts. PCs 2–50 each
explain less than 2%, with cumulative variance reaching 97.7% by PC10 and
99.9% by PC50.

| PC | Singular value | % Variance | Cumulative % |
|---|---|---|---|
| 1 | 59258.12 | 92.70 | 92.70 |
| 2 | 8347.96 | 1.84 | 94.54 |
| 3 | 5406.20 | 0.77 | 95.31 |
| 4 | 4701.77 | 0.58 | 95.90 |
| 5 | 4221.86 | 0.47 | 96.37 |
| 6 | 3649.07 | 0.35 | 96.72 |
| 7 | 3506.43 | 0.32 | 97.04 |
| 8 | 3262.27 | 0.28 | 97.32 |
| 9 | 2925.90 | 0.23 | 97.55 |
| 10 | 2683.83 | 0.19 | 97.74 |
| 11 | 2608.46 | 0.18 | 97.92 |
| 12 | 2458.93 | 0.16 | 98.08 |
| 13 | 2221.73 | 0.13 | 98.21 |
| 14 | 2089.03 | 0.12 | 98.32 |
| 15 | 1959.46 | 0.10 | 98.43 |
| 16 | 1930.46 | 0.10 | 98.52 |
| 17 | 1897.06 | 0.10 | 98.62 |
| 18 | 1832.79 | 0.09 | 98.71 |
| 19 | 1789.18 | 0.08 | 98.79 |
| 20 | 1679.66 | 0.07 | 98.87 |
| 21 | 1645.90 | 0.07 | 98.94 |
| 22 | 1627.89 | 0.07 | 99.01 |
| 23 | 1525.09 | 0.06 | 99.07 |
| 24 | 1507.68 | 0.06 | 99.13 |
| 25 | 1458.93 | 0.06 | 99.19 |
| 26 | 1450.69 | 0.06 | 99.24 |
| 27 | 1387.50 | 0.05 | 99.29 |
| 28 | 1366.85 | 0.05 | 99.34 |
| 29 | 1327.90 | 0.05 | 99.39 |
| 30 | 1282.24 | 0.04 | 99.43 |
| 31 | 1265.13 | 0.04 | 99.47 |
| 32 | 1246.63 | 0.04 | 99.51 |
| 33 | 1220.47 | 0.04 | 99.55 |
| 34 | 1171.64 | 0.04 | 99.59 |
| 35 | 1152.73 | 0.04 | 99.62 |
| 36 | 1120.64 | 0.03 | 99.66 |
| 37 | 1081.33 | 0.03 | 99.69 |
| 38 | 1052.76 | 0.03 | 99.72 |
| 39 | 1037.88 | 0.03 | 99.75 |
| 40 | 1017.01 | 0.03 | 99.77 |
| 41 | 998.43 | 0.03 | 99.80 |
| 42 | 963.68 | 0.02 | 99.82 |
| 43 | 947.46 | 0.02 | 99.85 |
| 44 | 936.68 | 0.02 | 99.87 |
| 45 | 932.04 | 0.02 | 99.89 |
| 46 | 918.85 | 0.02 | 99.92 |
| 47 | 905.06 | 0.02 | 99.94 |
| 48 | 890.18 | 0.02 | 99.96 |
| 49 | 883.10 | 0.02 | 99.98 |
| 50 | 875.46 | 0.02 | 100.00 |

## Top gene loadings (PCs 1–5)

Top 5 genes by absolute loading value per PC.

| PC | Gene | Loading | Gene | Loading | Gene | Loading | Gene | Loading | Gene | Loading |
|---|---|---|---|---|---|---|---|---|---|---|
| PC1 | Malat1 | +0.098 | Tmsb4x | +0.087 | Tuba1a | +0.085 | mt-Atp6 | +0.083 | Ptma | +0.079 |
| PC2 | Dbi | −0.153 | Fabp7 | −0.141 | Tubb3 | +0.140 | Stmn2 | +0.119 | Vim | −0.110 |
| PC3 | Meg3 | −0.162 | Tmsb10 | −0.106 | Nfix | +0.102 | mt-Atp6 | −0.094 | Igfbpl1 | +0.090 |
| PC4 | Igfbpl1 | +0.125 | Tubb2a | −0.107 | Ly6h | −0.106 | Meis2 | +0.103 | Stmn2 | −0.103 |
| PC5 | Neurod6 | +0.107 | Nfib | +0.107 | Gria2 | +0.102 | Fabp7 | +0.097 | Neurod2 | +0.097 |

**Notes:**
- PC1 is dominated by broadly expressed housekeeping and cytoskeletal genes
  (*Malat1*, *Tmsb4x*, *Tuba1a*) and mitochondrial transcripts (*mt-Atp6*),
  consistent with library size residual variation.
- PC2 separates glia (*Dbi*, *Fabp7*, *Vim* — astrocyte/radial glia markers)
  from neurons (*Tubb3*, *Stmn2*).
- PC5 loads heavily on neuronal differentiation factors (*Neurod6*, *Neurod2*,
  *Nfib*, *Gria2*), suggesting it captures neuronal subtype variation.

## Output files

| File | Size | Description |
|---|---|---|
| `tenx_cell_pcs_3K.npy` | 261 MB | float32 (1,306,127 × 50) cell coordinates |
| `tenx_gene_load_3K.npy` | 1 MB | float32 (50 × 3,000) gene loadings |
| `tenx_singular_vals_3K.npy` | — | float64 (50,) singular values |
| `tenx_hvg_names_3K.txt` | — | 3,000 HVG gene names |
