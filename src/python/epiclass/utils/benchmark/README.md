# Interpreting `benchmark_results.csv`

Reference for the output of `epiclass.utils.benchmark.benchmark_prefetch`. Read this before drawing conclusions from a sweep — several columns are easy to misread in isolation.

The benchmark answers one question: **what is the cheapest DataLoader configuration that keeps the GPU from waiting on data?** It sweeps `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`, `batch_size`, the number of CPU cores actually usable (via affinity capping), the page-cache regime, and which model consumes the batches. Each configuration runs in a fresh subprocess so affinity, workers, CUDA and page-cache state never bleed between runs.

## Column reference

### Identity

| Column | Meaning |
| --- | --- |
| `row` | CSV index, no meaning beyond ordering. |
| `config_id` | `cfg<NNN>_rep<R>` — matches the per-run subdirectory in the log dir, which holds `run_spec.json` (exact inputs) and `result.json` (raw output). Go there to debug a surprising row. |

### Swept knobs (what was requested)

| Column | Meaning |
| --- | --- |
| `num_workers` | DataLoader worker processes. `0` means loading happens **in the main process, synchronously** — no overlap with GPU compute. This is the single most consequential knob. |
| `num_physical_cpus` | Cores the run was allowed to use. Enforced by `os.sched_setaffinity`, which worker processes inherit, so a run configured for 1 core genuinely contends for 1 core even when the job allocated 4. |
| `prefetch_factor` | Batches each worker keeps queued ahead. **Empty/`NaN` when `num_workers == 0`** — torch rejects it there, so it is structurally inapplicable, not "tuned off". This is the most commonly misread column. |
| `pin_memory` | Page-locked host buffers for faster host→device transfer. |
| `persistent_workers` | Keep workers alive between epochs instead of re-forking. Forced `False` when `num_workers == 0`. |
| `batch_size` | Samples per batch. |
| `drop_cache_between_epochs` | `True` = page cache evicted before **every** epoch (cold, simulates data ≫ RAM). `False` = evicted only before epoch 0, so epochs 1+ run warm (matches production on a RAM-fitting dataset). Always recorded even when not swept. |
| `model` | Which model consumed the batches: `classifier` (`LightningDenseClassifier`) or `ave` (`LightningAVE`). The data path is identical across models; only GPU compute changes. |

Note that `num_workers` is the **clamped** value: configurations requesting more workers than `num_physical_cpus` are reduced to the core count, since exceeding it only oversubscribes workers onto the same cores. The CSV therefore records what ran, not what was requested.

### Run shape

| Column | Meaning |
| --- | --- |
| `n_train_samples` | Training samples in the fold, after oversampling. |
| `n_batches` | Batches per epoch (`limit_train_batches` applies here if set). |
| `epochs` | Epochs run. `epochs - 1` is how many feed the steady-state statistics. |
| `n_params` | Model parameter count — the sanity check on how much GPU work a row represents. |
| `mmap_bytes` | Size of the preloaded signals mmap. Compare against the job's `--mem` to know whether the page cache could hold the dataset. |
| `cores_pinned` | Cores actually pinned. Should equal `num_physical_cpus`; a mismatch means the cap could not be applied and the row is not trustworthy. |
| `device` | GPU name, or `cpu`. **CPU rows are not representative** — prefetching only matters against a real GPU. |
| `error` | Present only when the subprocess failed; timing columns are then empty. |

### Timing — the actual measurements

| Column | Meaning |
| --- | --- |
| `first_epoch_s` | Epoch 0. Includes worker spin-up and the coldest reads, so it is **always** slower and is reported separately rather than averaged in. |
| `steady_epoch_s` | Mean of epochs 1+. **The primary metric.** |
| `total_fit_s` | Whole `trainer.fit` wall time, including setup. |
| `samples_per_s` | `n_batches × batch_size / steady_epoch_s`. Convenience only — monotone with `steady_epoch_s`, so it ranks configs identically. |
| `dataloader_only_s` | One full cold pass over the DataLoader **with no model**. Pure data-delivery cost. Its cache is always dropped first, regardless of `drop_cache_between_epochs`. |

Compare `dataloader_only_s` against `steady_epoch_s` to see the headroom: if delivery is several times faster than a full epoch, the GPU is the bottleneck and no amount of prefetching will help.

### Dispersion — is a difference real?

All computed over the **steady** epochs only (epoch 0 excluded, since its spin-up cost would inflate every spread).

| Column | Meaning |
| --- | --- |
| `epoch_std_s` | Sample standard deviation (ddof=1). |
| `epoch_min_s`, `epoch_max_s` | Range. |
| `epoch_median_s` | Median — compare to `steady_epoch_s`; a large gap means one outlier epoch is dragging the mean. |
| `epoch_q1_s`, `epoch_q3_s`, `epoch_iqr_s` | Quartiles and IQR (numpy's linear-interpolation convention). |
| `epoch_cv_pct` | Std as a percent of the mean — the one dispersion column comparable across configs with different absolute times. |
| `n_steady_epochs` | How many epochs the statistics rest on. **Check this first.** With `epochs: 4` it is 3, which makes the std directional at best and the IQR nearly meaningless. Use `epochs: 8` or more when a small difference matters. |
| `steady_epoch_times_s` | The raw per-epoch times, `;`-joined, so anything above can be recomputed. |

The rule: **a gap between two configs smaller than their combined `epoch_std_s` is not a real difference.** The printed summary applies this automatically and labels such rows `GPU-bound (noise)`.

## How to read a sweep

1. Filter to one `(model, drop_cache_between_epochs)` group. Epoch times are only comparable within a group — a heavier model or a colder cache shifts every row.
2. Find the minimum `steady_epoch_s`: the GPU-bound floor.
3. Everything within the combined spread of that floor is tied. Among the tied configs, **pick the one requesting the fewest cores and workers** — that is the answer, since it frees allocation for other jobs at no throughput cost.
4. Cross-check with `dataloader_only_s`. Large headroom versus `steady_epoch_s` confirms the workload is GPU-bound and the ranking is not about I/O at all.


## Running a sweep

See `input-format/benchmark_prefetch.json` for the config schema and `src/bash_utils/benchmark_prefetch_template.sh` for the SLURM launcher. Request the maximum core count once — the benchmark sweeps subsets internally by capping affinity, so there is no need for one job per `--cpus-per-task` value.

Findings are intentionally not kept here: they are specific to a dataset, resolution, node and model, and go stale as soon as any of those change. Record them next to the run's `benchmark_results.csv`, or in a local `RESULTS.md` (gitignored).
