# Test suite performance: measurements

Running record of what has actually been *measured* about `pytest` wall time in this repo, and what was proposed, tested, and rejected.

The reason this file exists: several plausible-sounding optimizations here made the suite **slower**, including the textbook-correct one. Two of them were landed before being measured. Treat "this test is slow, so running it first will help" as a hypothesis, not a fact — time the suite with and without the change, three rounds minimum, before committing it.

## Scope: when do these numbers stop being true

All measurements below were taken on **2026-08-27 and 2026-08-28** on one workstation. They are specific to this hardware and this dependency set. Re-measure rather than trusting this file if any of the following changed.

Hardware: Intel i7-6700, **4 physical cores, 8 logical**, spinning-rust-free local disk. `pytest -n auto` resolves to **4**, the *physical* count, because `psutil` is installed; without `psutil`, xdist would pick 8 and oversubscribe.

Background load: BOINC normally runs at nice priority on this machine. It yields within a few seconds once tests start, but it still adds noise. Every number here was taken with BOINC suspended, and the differences below that are called "noise" are differences that vanish when it is running.

Dependency versions the conclusions actually depend on:

| Package | Version | Why it matters |
| --- | --- | --- |
| `umap-learn` | 0.5.12 | 117 `@numba.njit` decorators, **none** with `cache=True` — this is the single biggest driver of suite time |
| `pynndescent` | 0.6.0 | 227 `@numba.njit`, only 20 with `cache=True` |
| `numba` | 0.63.1 | JIT backend; `NUMBA_CACHE_DIR` only helps functions declared `cache=True` |
| `pytest-xdist` | 3.8.0 | The opening-chunk distribution described below is an implementation detail of this version |
| `pytest` | 8.3.5 | |
| `psutil` | 7.2.1 | Present ⇒ `-n auto` means 4, not 8 |

**What would invalidate the central finding:** if `umap-learn` ships `cache=True` on its jitted functions in a future release, UMAP compilation becomes cacheable across processes, and the "co-locate the two UMAP tests" conclusion loses most of its value. Re-run the check with:

```bash
python - <<'EOF'
import re, pathlib, importlib.util
for pkg in ("umap", "pynndescent"):
    d = pathlib.Path(importlib.util.find_spec(pkg).origin).parent
    cached = uncached = 0
    for f in d.rglob("*.py"):
        for m in re.finditer(r"@numba\.njit\((.*?)\)|@njit\((.*?)\)", f.read_text(errors="ignore"), re.S):
            args = m.group(1) or m.group(2) or ""
            if "cache=True" in args: cached += 1
            else: uncached += 1
    print(f"{pkg}: njit cached={cached} uncached={uncached}")
EOF
```

## Vocabulary

**Makespan** — the wall time of the slowest worker, which is what the run's duration actually is. Minimising total CPU work and minimising makespan are different goals, and this suite is a case where they conflict.

**LPT (Longest Processing Time first)** — the classic list-scheduling heuristic: sort jobs longest-first, hand each to the next free worker. Provably within 4/3 of optimal makespan *when workers pull work one job at a time*. That precondition does not hold under xdist's default distribution, which is why LPT loses badly here (see below).

**Opening chunk** — xdist's `--dist load` does not hand out tests one at a time at the start. It gives each worker a block of consecutive tests up front. Size and consequences are detailed below.

## Suite shape

As of 2026-08-27: **572 tests collected**, 562 passed + 10 skipped on a clean tree.

Markers, declared in `src/python/pyproject.toml`:

- `slow` — 22 tests. **Not** auto-skipped; a plain `pytest tests` runs them. Opt out with `make test-fast` / `-m "not slow"` (550 tests).
- `embedding` — the PCA/UMAP smoke tests, described in `pyproject.toml` as "JIT-heavy, dominate suite time". That description turned out to be exactly right, and the reason is quantified below.

## Total work and the theoretical floor

From one instrumented run (`-n auto --durations=0`, plus a `pytest_fixture_setup` timing plugin — see reproduction section):

| Phase | Total |
| --- | --- |
| `call` (190 tests over 5ms) | 258.2s |
| `setup` (118 fixtures over 5ms) | 64.9s |
| **Total work** | **323.1s** |

Ideal 4-worker makespan is 323.1 / 4 = **80.8s**. Observed wall is 121-131s, so roughly 25s is scheduling imbalance and ~25s is process startup plus collection.

Work is extremely skewed — the top 4 tests are 48% of all work, the top 16 are 76%:

| Test | setup + call |
| --- | --- |
| `utils/compute_umap_test.py::test_compute_umap_single_sample` | 71.9s |
| `utils/compute_umap_test.py::test_compute_umap_chunked` | 48.3s |
| `core/shap_test.py::Test_NN_SHAP_Handler::test_compute_shaps` | 20.7s |
| `utils/benchmark_shap_test.py::test_benchmark_shap_pipeline` | 14.9s |

## Finding 1: the dominant cost is numba JIT, not computation

`umap-learn` 0.5.12 has **117 `@numba.njit` decorators and zero with `cache=True`**; `pynndescent` 0.6.0 has 207 of 227 uncached. Compilation therefore happens in **every process** that touches UMAP and **cannot be persisted to disk** — numba only writes a cache for functions declared `cache=True`, so setting `NUMBA_CACHE_DIR` does nothing for this.

Decomposing `epiclass.utils.embedding.compute_umap.main()` inside a single process, sweeping sample count:

| Samples | KNN build | UMAP fit | Total `main()` |
| --- | --- | --- | --- |
| 25 (first call in process) | 33.4s | 4.8s | 33.8s |
| 50 | 0.1s | 0.5s | 2.2s |
| 100 | 0.2s | 0.1s | 2.0s |
| 100 (repeat) | 0.2s | 0.1s | 2.1s |

Actual numeric work is about **2 seconds and flat in sample count**. That is unsurprising once you look at the data: saccer3 is a ~12 Mb genome at 100 kb resolution, so `SACCER3_SUBSET_N = 100` samples gives a matrix of roughly 100 x 120. Everything above ~2s is compilation.

Three "obvious" fixes are therefore worthless, and should not be re-proposed without new evidence:

- **Reducing `SACCER3_SUBSET_N`** (the 100-sample subset fixture in `tests/conftest.py`) — cost is flat in sample count.
- **Setting `NUMBA_CACHE_DIR`** — nothing to cache, per above.
- **Trimming the embedding sweep** — the test already passes `--max_embeddings 1`, and `compute_umap.py` hardcodes `nn_knn = 100` for the KNN build regardless of that cap, so the cap does not reach the expensive part.

## Finding 2: JIT is paid per process, so co-location beats balance

| | alone in its own process | second, same process |
| --- | --- | --- |
| `test_compute_umap_single_sample` | 35.2s | 35.5s |
| `test_compute_umap_chunked` | **37.3s** | **4.1s** |

Both UMAP tests in one process: 61.8s wall. In separate processes: 52.3 + 57.3 = 109.6s. **Co-locating them saves about 33s of total work** in isolation, and ~43s measured inside the full suite (47.2s vs 4.4s for the chunked test), because the second test reuses the first's compiled kernels.

This inverts the usual scheduling intuition. Normally you spread the two biggest jobs across workers. Here you want them in the *same* worker.

## Finding 3: the ordering hook, and how it got two things wrong

`tests/conftest.py` has a `pytest_collection_modifyitems` hook that hoists selected tests to the front of collection order. Its history is the main cautionary tale in this file.

**Version 1 — hoist by marker.** `LONG_FIRST_MARKERS = ("embedding", "slow")`, sorting all 22 `slow`-marked tests to the front, on the reasoning that xdist hands out tests in collection order so the longest should go first. Landed unmeasured. When finally measured on the full 572-test suite:

| | Round 1 | Round 2 |
| --- | --- | --- |
| Hoist all `slow` tests first | 164.3s | 166.4s |
| No hoist | 141.9s | 146.9s |

**15% slower**, reproduced twice. Reverted. At the time the mechanism was unknown and thread oversubscription was suspected; that suspicion was wrong (see the thread-pinning null result below). The real mechanism is in Finding 4.

**Version 2 — hoist one test by nodeid.** Narrowed to `LONG_FIRST_NODEIDS = ("compute_umap_test.py::test_compute_umap_single_sample",)` and committed as `4585800`, again **without measuring it**, on the reasoning that this one test is longer than the whole suite's ideal makespan so it must start first.

Finding 2 says this is backwards, and measurement confirms it. The hoist moves `single_sample` to index 0 while `chunked` stays where it was, so the two tests land in **different workers** and the compile is paid twice. Doing nothing leaves them adjacent in collection order, hence in the same worker.

**Version 3 — hoist both, or hoist neither.** Three orderings, 3 rounds each, round-robin. `chunked`'s own call time is a direct probe of co-location: ~4s means it reused `single_sample`'s compiled kernels, ~47s means it compiled from scratch.

| Ordering | `chunked` call time | Wall |
| --- | --- | --- |
| Hoist `single_sample` only (as committed in `4585800`) | 47.2 / 47.5 / 46.8s | 137.6 / 132.7 / 134.9s |
| Hoist **both** UMAP tests to indices 0 and 1 | 4.20 / 4.40 / 4.40s | 114.8 / 117.7 / 112.6s |
| No hoist at all (natural collection order) | 4.37 / 4.37 / 4.39s | 118.5 / 118.4 / 112.9s |

The committed hoist separates them **deterministically, in every run** — not occasionally, and not as a scheduling coin flip. It costs ~43s of redundant compilation and ~17-20s of wall time, roughly 13-15%.

Hoisting both and hoisting neither are statistically indistinguishable (overlapping ranges, medians 114.8s vs 118.4s). Both work for the same reason: the two tests stay **adjacent in collection order**, so they land in the same opening chunk and therefore the same process.

The actionable rule is therefore not "hoist the long tests" but **"keep the two UMAP tests adjacent"**. Hoisting both was adopted because it is the smaller diff against the existing hook and is nominally fastest; reverting the hook entirely would have been equally defensible.

A caution about comparing across campaigns: this same committed configuration measured 121-123s in one campaign and 133-141s in another, on different days. Within a round-robin campaign the comparisons are clean, but **absolute wall times are not comparable between campaigns** on this machine. The `chunked` call-time probe is immune to that drift, which is why it is the number to trust here.

## Finding 4: why longest-first ordering is catastrophic here

From `xdist/scheduler/load.py` (pytest-xdist 3.8.0):

```python
items_per_node = len(self.collection) // len(self.node2pending)
node_chunksize = min(items_per_node // 4, self.maxschedchunk)
node_chunksize = max(node_chunksize, 2)
for node in self.nodes:
    self._send_tests(node, node_chunksize)
```

The opening distribution hands each worker a block of **consecutive** tests: `572 // 4 // 4` = **35** go to gw0 before any other worker receives anything. Only once a worker drains its block does it pull more.

So sorting the collection longest-first drops the 35 most expensive tests — 285s of the 323s total — into a single worker, which runs them serially while the other three finish everything else and idle. LPT's guarantee assumes one-job-at-a-time pull; the opening chunk breaks that assumption, and the heuristic inverts from near-optimal to worst-measured.

This is the mechanism behind the Version 1 regression in Finding 3: hoisting the 22 `slow` tests put all of them inside gw0's opening 35.

`--maxschedchunk 1` caps `node_chunksize` (floored at 2) and restores one-at-a-time pull. It is *still* slower (+20%, table below), because breaking up consecutive runs separates same-file tests and duplicates their JIT and fixture setup. The coarse default chunking is accidentally doing the right thing for this suite.

## Finding 5: scheduling knobs, all neutral or worse

Median of 3 rounds, full suite, `-n auto` (= 4), BOINC suspended:

| Configuration | Median wall | vs default |
| --- | --- | --- |
| `--dist load` (default) | 125.5s | — |
| `--dist loadfile` | 122.5s | within noise |
| `--dist worksteal` | 124.2s | within noise |
| `--dist loadscope` | 124.4s | within noise |
| `-n 6` | 148.2s | +18% |
| `-n 8` | 155.0s | +24% |
| `--maxschedchunk 1` | 146.0s | +20% |
| LPT order + `--maxschedchunk 1` | 167.7s | +37% |
| LPT order (longest-first) | 199.3s | +64% |

Round spreads were tight for the losing configurations (`-n 6`: 147.9 / 148.4 / 148.2; `-n 8`: 155.0 / 154.4 / 155.1), so those regressions are real. Oversubscribing beyond the 4 physical cores is reliably bad on this machine.

The four `--dist` modes all overlap and **cannot be separated at this sample size**. Do not read a winner into a 3s gap.

## Finding 6: collection and imports

Collection is ~16s per worker, and every worker collects the whole suite — importing every test module even though it will run about a quarter of them.

Cold import costs, measured in isolated interpreters: `umap` 8.55s, `lightning` 5.19s, `comet_ml` 2.28s, `shap` 1.81s, `torch` 1.41s, `sklearn` 0.77s.

`utils/compute_umap_test.py` imports `epiclass.utils.embedding.compute_umap` at module scope, which pulls `umap` into **all four workers during collection**, though only one will ever run a UMAP test. Moving that import inside the test body would remove it from three workers' critical path — worth up to ~8.5s. `tests/import_test.py` already demonstrates the pattern: every one of its imports sits inside `test_imports()` rather than at module level, so collecting it costs 0.10s against `compute_umap_test.py`'s 8.58s.

**Untested.** Listed as the most promising remaining candidate, not as a recommendation.

## Finding 7: duplicated fixture setup

Per-fixture instrumentation, one 4-worker run:

| Fixture | Scope | Built in | Cost each |
| --- | --- | --- | --- |
| `test_epiatlas_data_handler` | session | gw0, gw2, gw3 | 4.1 / 7.1 / 4.5s |
| `test_data` (`TestEpiAtlasFoldFactory`) | class | gw0 | 7.9s |
| `test_data` (`Test_Hdf5Loader`) | class | gw3 | 8.2s |
| `test_data` (`TestLazyEpiAtlasMetadata`) | **function** | gw0 | 4.0 + 3.7s |
| `extracted_hdf5_dir` | session | gw1, gw2 | 4.9 / 1.6s |
| `saccer3_chunked_dir` | session | gw2 | 5.3s |
| `big_test_data` | function | gw0 | 6.6s |

Session scope in pytest means once per *worker process*, not once per run, so a 4-worker run pays session fixtures up to four times. That is inherent to xdist and not a bug.

Two things are not inherent:

`TestLazyEpiAtlasMetadata`'s `test_data` fixture (`core/epiatlas_treatment_test.py:438`) is **function-scoped**, while the two sibling classes use `scope="class"` for the identical `EpiAtlasTreatmentTestData.test_data()` call. It rebuilds at ~4s per test. This looks like an oversight — but check it against the mutation caveat first: `core/metadata_test.py`'s `test_meta` fixture hands out `test_epiatlas_data_handler.epiatlas_dataset.metadata` and the `env_filtering` tests mutate it in place. That is presumably why several of these fixtures build private copies instead of sharing one, so "just make them all session-scoped" is **not** safe.

`extracted_hdf5_dir` untars `fixtures/saccer3/hdf5/saccer3_2016-07.tar.xz` into pytest's per-run `basetemp`, once per worker, every run. An already-extracted copy sits in `fixtures/saccer3/hdf5/saccer3_2016-07/` and `make clean-saccer3` still targets that path, so the tree was clearly extracted there at some point in the past. A stable, lock-guarded cache directory would make this free after the first run, for every worker and every future run.

## Earlier investigation (2026-08-27)

### The mmap cache race — real bug, fixed

`core/epiatlas_treatment_test.py` failed intermittently under `-n auto` and never under `-n 1`. Root cause: `LazyHdf5Loader`'s default `mmap_dir` was the cwd-relative `./mmap_cache`, which is process-independent, so all four xdist workers shared one cache directory. Workers register different sample sets, so they took turns unlinking and rebuilding the same `signals_*.npy`; on top of that, each worker's `pytest_sessionstart` wipes the cache, so one worker could delete the `.tmp` file another was mid-write on. Symptom: `FileNotFoundError` inside `os.replace()` in `preload_all()`.

Fix (commit `54a95f8`): the loader honours `$EPICLASS_MMAP_DIR`, and `tests/conftest.py` points each worker at `./mmap_cache/<worker-id>`. This is the one change in the whole investigation that was unambiguously worth making. It also gives HPC runs somewhere to put the cache — point it at `$SLURM_TMPDIR`.

### `--dist loadgroup` — ~15% slower, rejected

Adding `--dist loadgroup` plus an `xdist_group` marker was the first attempt at forcing `core/epiatlas_treatment_test.py` to run serially, before the race above was understood. Measured on the non-slow subset: `-n 3 --dist load` gave 57.7 / 56.6 / 63.3s against `-n 3 --dist loadgroup` at 64.3 / 68.8 / 70.3s — non-overlapping ranges.

A discriminating experiment — leaving the marker in place but running under `--dist load`, where the marker is inert — showed the cost comes from the **scheduler mode itself**, not from serializing that one group.

It was harmful for a second reason too: putting `--dist loadgroup` in `addopts` broke `make test-last-failed`, which runs `-p no:xdist` and then fails on the now-unrecognized `--dist`. Both the marker and the addopts were reverted; fixing the race removed the need for either.

### Pinning thread counts — no measurable effect

Motivation was to avoid spawning too many threads when many torch-heavy tests run at once. Torch honours `OMP_NUM_THREADS`/`MKL_NUM_THREADS` at import; confirmed `torch.get_num_threads()` returns 4 / 1 / 2 under default / `OMP=1` / `OMP=2`.

Nine torch-heavy tests, three rounds per arm, reported test time:

| Arm | R1 | R2 | R3 |
| --- | --- | --- | --- |
| A default (4 threads) | 36.2s | 29.3s | 29.4s |
| B `OMP_NUM_THREADS=1` | 29.3s | 35.0s | 28.0s |
| C `OMP_NUM_THREADS=2` | 33.3s | 30.6s | 27.9s |

Within-arm spread (up to 7s) exceeds every between-arm difference. **Null result.** The useful consequence: since pinning costs nothing measurable, it is safe to pin low if you want to bound thread counts for other reasons — it just will not make the suite faster.

This is also the result that ruled out thread oversubscription as the explanation for the Finding 3 regression, which is what left that mechanism unexplained until Finding 4.

Scope note: `shap.DeepExplainer` is torch underneath, not an independently threaded pool, so the torch setting covers it. UMAP passes `random_state`, which forces single-threaded layout optimization anyway.

### `-n 3` vs `-n auto` (4) — within noise

An earlier 13% gap was attributed to leaving a core free for the OS. Re-measured with BOINC suspended: ~2.6s apart with overlapping ranges. The original gap was BOINC contention. **Retracted.**

## Reproducing these measurements

All of it runs from `src/python/` with the project venv active. Nothing below modifies the repo — the instrumentation lives in throwaway plugins on `PYTHONPATH`, which is the recommended way to test ordering and scheduling changes.

**Total work, and per-test setup/call split:**

```bash
pytest tests -n auto --durations=0 -q > run.txt
grep -E "^[0-9]+\.[0-9]+s (setup|call|teardown)" run.txt | awk \
  '{t=$1+0; tot[$2]+=t; all+=t; n[$2]++} END {for (k in tot) printf "%-9s n=%-4d %7.1fs\n", k, n[k], tot[k]; printf "ALL %7.1fs\n", all}'
```

**Per-fixture setup cost, per worker** — save as `bench_fixtures.py` somewhere outside the repo, put that directory on `PYTHONPATH`, and run with `-p bench_fixtures`:

```python
import os, pathlib, time
import pytest

LOGDIR = pathlib.Path(os.environ["BENCH_LOG_DIR"]); LOGDIR.mkdir(parents=True, exist_ok=True)
LOG = LOGDIR / f"fixtures_{os.environ.get('PYTEST_XDIST_WORKER', 'main')}.tsv"

@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    start = time.perf_counter()
    yield
    dur = time.perf_counter() - start
    if dur >= 0.05:
        with open(LOG, "a", encoding="utf8") as fh:
            fh.write(f"{dur:.3f}\t{fixturedef.scope}\t{fixturedef.argname}\t{request.node.nodeid}\n")
```

**Testing a different test order** — same trick. A plugin's `pytest_collection_modifyitems` runs *after* the repo `conftest.py`'s, so it has the last word, and you never touch the working tree:

```python
def pytest_collection_modifyitems(config, items):
    items.sort(key=lambda it: ...)   # must be deterministic across workers
```

The determinism requirement is not optional: every xdist worker re-runs the hook independently and xdist aborts with a collection mismatch if they disagree.

**Isolating whether a cost is JIT or compute** — run the tests together in one process, then each alone, and compare:

```bash
pytest tests/utils/compute_umap_test.py -p no:xdist -q --durations=0            # together
pytest tests/utils/compute_umap_test.py::test_compute_umap_chunked -p no:xdist -q --durations=0   # alone
```

A test that is far cheaper in the first form is paying somebody else's compilation, not doing its own work.

## Method notes

Things that burned time and are worth not repeating.

Always benchmark with BOINC suspended and check `uptime` first; background load produced a spurious 13% effect that survived three rounds and was reported as real before being retracted.

Give every run a unique output filename. Two benchmark jobs once wrote to the same file and silently corrupted a round.

Verify a process actually died before starting the next run — a `pkill` that matched nothing let a full-suite run overlap with the next benchmark.

Three rounds minimum. Several arms have within-arm spread of 5-7s, so a single round distinguishes nothing, and differences under ~5s are not resolvable on this machine at all.

Benchmark the configuration people actually run. An early conclusion that the ordering hook "contributes exactly zero" was measured against `-m "not slow"`, which excludes the very tests the hook reorders.

Watch for warming drift. In the 2026-08-27 campaign the first configuration's first round ran 131.1s and its third ran 122.8s, consistent with page-cache warming. Round-robin across configurations so drift is shared rather than charged to whichever arm happened to run first.

Run `pre-commit` from the repo root. Passing `src/python/...` paths from inside `src/python` makes every hook report "no files to check" and exit 0, which looks exactly like success.
