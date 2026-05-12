# Running UMAP on huge datasets (memmap + alternatives)

Context: discovered while migrating EpiClass `utils/embedding/compute_umap.py`
to use `LazyHdf5Loader.as_mmap()`. Question: does mmap-backing the input X
actually let UMAP scale beyond RAM?

## Short answer

- **IncrementalPCA + memmap**: yes, true streaming. Officially supported.
- **UMAP + memmap**: technically works, but it's "graceful degradation,"
  not real streaming. Watch out on the first kNN build for huge datasets.

## What `as_mmap()` actually does

Returns `np.load(path, mmap_mode="r")` — a 2-D ndarray whose data lives on
disk. The OS pages 4 KB blocks in on demand and evicts under memory pressure.
The array *looks* like a normal ndarray to consumers.

- Sequential access: fast (OS prefetches the next pages).
- Random access: slow (every miss is a disk read). On Linux with very
  large arrays, this can be "incredibly slow" without
  `madvise(MADV_RANDOM)` — see numpy issue #13172.

## IncrementalPCA: clean win

From the sklearn docs (verbatim):

> This algorithm has constant memory complexity, on the order of
> `batch_size * n_features`, enabling use of `np.memmap` files without
> loading the entire file into memory.

`IncrementalPCA.fit_transform(memmap_array)`:
- Reads `batch_size` consecutive rows → `partial_fit` → next batch.
- Peak RAM ≈ `batch_size × n_features × dtype_bytes`. Truly bounded.
- Access pattern is sequential → mmap is happy.

Default `batch_size = 5 * n_features` if not provided. EpiClass script
passes 30 000 explicitly.

## UMAP: works but with caveats

Three phases:

1. **kNN graph construction (NNDescent / pynndescent)** — the expensive
   part. Approximate algorithm that compares many random pairs of rows.
   Access pattern is **random**, not sequential. With mmap on a dataset
   bigger than RAM, this means thrashing (pages evicted then re-read).
   Still completes; can be 10–100× slower than in-RAM. Officially
   undocumented — the UMAP FAQ has no mention of memmap.
2. **Spectral init** — small, fine.
3. **SGD on embeddings + kNN graph** — operates on the low-dim
   embedding and graph, doesn't re-scan X. Memory-friendly.

Key insight from EpiClass `compute_umap.py`: the script already supports
`--load_knn` to persist + reuse the kNN graph. So phase 1 only hurts on
the **first run**. Subsequent runs are fine even with massive X.

## The right pattern for genuinely huge UMAP

This is industry-standard practice (scRNA-seq, ML, etc.):

```
HDF5 signals (huge, mmap-backed)
    ↓ IncrementalPCA  (streaming, batch_size × n_features RAM)
PCA components (e.g. n=50, fits easily in RAM)
    ↓ UMAP
embedding
```

PCA acts as a lossy compressor that shrinks `n_features` from 30 000 →
50 while preserving most variance. UMAP then runs on the small dense
matrix without memory pressure.

EpiClass has both `compute_pca.py` and `compute_umap.py`; chaining them
(saving the PCA output to disk, loading it as `X` for UMAP) is the
established workflow.

## Escape hatches if X still doesn't fit

- `UMAP(low_memory=True)` — switches NNDescent to a slower, less RAM-
  hungry approach. Already set in EpiClass `compute_umap.py` params.
- Subsample → fit UMAP on subset → `umap_model.transform()` the rest in
  batches. Caveat: there's an open issue (lmcinnes/umap#535) about
  `transform()` leaking memory over many calls — batch carefully.
- GPU UMAP via RAPIDS cuML — much faster, GPU RAM bound rather than
  system RAM bound.
- `madvise(MADV_RANDOM)` hint on the memmap if NNDescent thrashing is
  the actual bottleneck (not currently exposed by `as_mmap()`).

## Practical decision tree

```
n_samples × n_features × 4 bytes < free RAM?
├─ yes → just load in memory, mmap doesn't matter
└─ no → first reduce dims:
        IncrementalPCA with mmap → save components → UMAP on components
        (if components still too big: subsample + transform, or cuML)
```

For EpiClass-scale data:
- 100 kb resolution: ~30 000 features × 4 bytes = ~120 KB/sample
- 100 000 samples × 120 KB = ~12 GB → fits in most workstations
- 1 M samples × 120 KB = ~120 GB → PCA-first becomes mandatory

## Sources

- [sklearn IncrementalPCA docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
- [UMAP FAQ — low_memory](https://umap-learn.readthedocs.io/en/latest/faq.html)
- [pynndescent docs](https://pynndescent.readthedocs.io/en/latest/how_pynndescent_works.html)
- [numpy memmap docs](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [numpy issue #13172 — slow random access on memmap](https://github.com/numpy/numpy/issues/13172)
- [UMAP issue #535 — transform memory leak](https://github.com/lmcinnes/umap/issues/535)
- [NVIDIA RAPIDS cuML UMAP on GPU](https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/)
