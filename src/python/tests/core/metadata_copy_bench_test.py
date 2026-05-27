"""Benchmark deepcopy vs marshal-based copy on a worst-case Metadata.

Motivation: scripts like analyze_shaps_kfold.py need many independent filtered
views of the same metadata. copy.deepcopy on ~thousands of nested dicts is the
hot path. Before we expose `Metadata.copy()` / override `__deepcopy__` to use
marshal, this test confirms marshal is materially faster than deepcopy on
realistic worst-case input.

Run with output visible:
    pytest tests/core/metadata_copy_bench_test.py -m slow -s
"""
from __future__ import annotations

import copy
import marshal
import statistics
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict

import pytest

from epiclass.core.metadata import Metadata

# Worst-case scale: dfreeze v2 is a few thousand entries; pick 5000 to give
# deepcopy room to lose. ~30 keys/dataset is realistic for IHEC-style metadata.
N_ENTRIES = 5000
N_KEYS_PER_DATASET = 30
N_RUNS = 5


def _build_metadata(n_entries: int, n_keys: int) -> Metadata:
    """Synthesize a Metadata with `n_entries` × `n_keys` string fields each."""
    base_keys = [
        "assay",
        "assay_epiclass",
        "cell_type",
        "harmonized_sample_ontology_intermediate",
        "track_type",
        "uuid",
        "epirr_id",
        "disease",
        "donor_health_status",
        "data_generating_centre",
        "paired_end_mode",
        "analyzed_as_stranded",
        "antibody",
        "experiment_type",
        "tissue_type",
    ]
    # Pad up to n_keys with synthetic columns to match worst-case widths.
    keys = list(base_keys)
    while len(keys) < n_keys:
        keys.append(f"extra_col_{len(keys)}")
    keys = keys[:n_keys]

    meta_dict: Dict[str, dict] = {}
    for i in range(n_entries):
        signal_id = uuid.uuid4().hex  # 32 chars, satisfies Metadata invariant
        dset = {k: f"value_{i}_{k}" for k in keys}
        dset["md5sum"] = signal_id
        meta_dict[signal_id] = dset
    return Metadata.from_dict(meta_dict)


def _time(fn: Callable[[], object], n_runs: int = N_RUNS) -> Dict[str, float]:
    """Run fn() n_runs times, return min/median/mean/std/iqr wall-clock in seconds."""
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    if len(samples) >= 4:
        quartiles = statistics.quantiles(samples, n=4)
        iqr = quartiles[2] - quartiles[0]
    else:
        iqr = float("nan")
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) >= 2 else float("nan"),
        "iqr": iqr,
    }


@pytest.fixture(scope="module", name="big_metadata")
def fixture_big_metadata() -> Metadata:
    """Worst-case synthetic metadata. Built once per module."""
    return _build_metadata(N_ENTRIES, N_KEYS_PER_DATASET)


@pytest.mark.slow
def test_copy_benchmark(big_metadata: Metadata, tmp_path: Path):
    """Compare ways of producing an independent copy of Metadata.

    Two phases:
      1. Raw dict ops on `meta._metadata` (apples-to-apples, no Metadata
         construction overhead).
      2. Full Metadata-producing clones (what the call site actually pays).

    Output is printed (run with `pytest -s` to see). The test passes
    regardless of ordering — its job is to surface the numbers, not to
    enforce a winner. Conclusions get baked into the API afterwards.
    """
    # pylint: disable=protected-access
    inner = big_metadata._metadata

    # Pre-serialize once for the "amortized" variants (mirrors the
    # analyze_shaps_kfold pattern: dump once, read back per iteration).
    cached_marshal_bytes = marshal.dumps(inner)
    cached_marshal_file = (
        tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
            mode="wb", delete=False, dir=tmp_path
        )
    )
    cached_marshal_file.write(cached_marshal_bytes)
    cached_marshal_file.close()

    # ---- Phase 1: raw dict ops producing a {signal_id: dataset} dict. ----
    def raw_deepcopy() -> Dict[str, dict]:
        return copy.deepcopy(inner)

    def raw_marshal_inmem() -> Dict[str, dict]:
        return marshal.loads(marshal.dumps(inner))

    def raw_marshal_disk_full() -> Dict[str, dict]:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=tmp_path) as fh:
            marshal.dump(inner, fh)
            fname = fh.name
        with open(fname, "rb") as fh:
            return marshal.load(fh)

    def raw_marshal_disk_read_only() -> Dict[str, dict]:
        # Amortized: serialization is paid once outside the loop. This is the
        # pattern in analyze_shaps_kfold (save once, from_marshal N times).
        with open(cached_marshal_file.name, "rb") as fh:
            return marshal.load(fh)

    def raw_marshal_inmem_buf_cached() -> Dict[str, dict]:
        # Same amortized pattern but bytes stay in memory.
        return marshal.loads(cached_marshal_bytes)

    def raw_outer_only() -> Dict[str, dict]:
        # UNSAFE: shares inner dataset dicts. Reference baseline.
        return dict(inner)

    def raw_one_level_deep() -> Dict[str, dict]:
        # Independent outer dict + independent inner dicts (one level deep is
        # sufficient since Metadata datasets are flat key→str maps).
        return {sid: dict(dset) for sid, dset in inner.items()}

    phase1 = {
        "deepcopy": raw_deepcopy,
        "marshal-inmem": raw_marshal_inmem,
        "marshal-disk (full)": raw_marshal_disk_full,
        "marshal-disk (read-only)": raw_marshal_disk_read_only,
        "marshal-inmem (buf cached)": raw_marshal_inmem_buf_cached,
        "outer-only (unsafe)": raw_outer_only,
        "one-level-deep": raw_one_level_deep,
    }

    # ---- Phase 2: full Metadata clones via each strategy. ----
    # NB: Metadata.from_dict itself does copy.deepcopy internally, which adds
    # a constant cost; this phase shows the real-world cost a caller pays
    # if we keep that constructor as-is. Cheaper alternatives bypass it.

    def full_deepcopy() -> Metadata:
        return copy.deepcopy(big_metadata)

    def full_marshal_inmem() -> Metadata:
        obj = Metadata.__new__(Metadata)
        obj._metadata = marshal.loads(marshal.dumps(inner))
        obj._rest = dict(big_metadata._rest)
        return obj

    def full_one_level_deep() -> Metadata:
        obj = Metadata.__new__(Metadata)
        obj._metadata = {sid: dict(dset) for sid, dset in inner.items()}
        obj._rest = dict(big_metadata._rest)
        return obj

    phase2 = {
        "deepcopy (Metadata)": full_deepcopy,
        "marshal-inmem (Metadata)": full_marshal_inmem,
        "one-level-deep (Metadata)": full_one_level_deep,
    }

    def report(title: str, results: Dict[str, Dict[str, float]]) -> None:
        print(f"\n{title}")
        header = (
            f"{'approach':<28}{'min (ms)':>12}{'median (ms)':>14}"
            f"{'mean (ms)':>12}{'std (ms)':>12}{'iqr (ms)':>12}"
        )
        print(header)
        print("-" * len(header))
        for name, t in results.items():
            print(
                f"{name:<28}{t['min']*1e3:>12.2f}{t['median']*1e3:>14.2f}"
                f"{t['mean']*1e3:>12.2f}{t['std']*1e3:>12.2f}{t['iqr']*1e3:>12.2f}"
            )

    print(
        f"\n\nMetadata copy benchmark "
        f"(N_entries={N_ENTRIES}, keys/dataset={N_KEYS_PER_DATASET}, "
        f"runs={N_RUNS})"
    )
    phase1_results = {name: _time(fn) for name, fn in phase1.items()}
    report("Phase 1 — raw dict clone", phase1_results)
    phase2_results = {name: _time(fn) for name, fn in phase2.items()}
    report("Phase 2 — full Metadata clone", phase2_results)
    print()

    # Sanity check: independence holds for the safe candidates.
    a = raw_one_level_deep()
    sid = next(iter(a))
    a[sid]["assay"] = "MUTATED"
    assert inner[sid]["assay"] != "MUTATED", "one-level-deep is not actually independent"
