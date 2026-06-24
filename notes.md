  On the M4 Mac, macOS used its NVMe-backed virtual memory to handle the overcommit:
  - The theoretical peak of 63 GB exceeds the 32 GB of physical RAM
  - macOS silently swapped cold pages (parts of X.indices, old X.data) to the NVMe SSD
  - The M4's memory compression (similar to zswap) further reduces pressure by compressing cold pages in RAM before they
  hit disk
  - Apple Silicon NVMe is fast enough (~5 GB/s) that this is slow but survivable

  So the Mac didn't succeed because it had enough RAM — it succeeded because macOS gave it a 63 GB virtual address space
  backed by 32 GB RAM + fast SSD swap. The X[top_genes, :] step likely caused a burst of swap activity that made it
  slower than expected, but it completed.

  The Linux server has 60 GB physical RAM — nearly twice the Mac — but is actually more fragile for this workload
  because it has zero swap. Any single allocation that pushes past available physical RAM is an immediate kill.

  The int32 fix is still the right thing to do: it removes the swap dependency on Mac (making it faster) and fits within
  physical RAM on Linux.
