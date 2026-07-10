"""Unit tests for the prefetch-benchmark config expansion and page-cache helper.

The full single-run path needs a GPU + real data and is exercised via the SLURM
template, not here. These fast tests cover the orchestrator-side logic that must
be correct for the sweep to be meaningful.
"""
from pathlib import Path

import pytest

from epiclass.utils.benchmark.benchmark_prefetch import (
    _normalize_config,
    drop_page_cache,
    expand_configs,
    resolve_mmap_path,
)


def test_cartesian_product_count():
    """Every combination of independent knobs is produced."""
    sweep = {
        "num_workers": [2, 4],
        "num_physical_cpus": [1, 2],
        "prefetch_factor": [2, 4],
        "pin_memory": [True],
        "persistent_workers": [True],
        "batch_size": [64],
    }
    configs = expand_configs(sweep)
    assert len(configs) == 2 * 2 * 2  # workers x cpus x prefetch


def test_zero_workers_normalized_and_deduped():
    """num_workers==0 collapses prefetch/persistent, removing spurious dupes."""
    cfg = _normalize_config(
        {
            "num_workers": 0,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "batch_size": 64,
        }
    )
    assert cfg["prefetch_factor"] is None
    assert cfg["persistent_workers"] is False

    sweep = {
        "num_workers": [0],
        "num_physical_cpus": [1],
        "prefetch_factor": [2, 4],  # both must collapse to a single config
        "pin_memory": [True],
        "persistent_workers": [True, False],
        "batch_size": [64],
    }
    configs = expand_configs(sweep)
    assert len(configs) == 1
    assert configs[0]["prefetch_factor"] is None
    assert configs[0]["persistent_workers"] is False


def test_mixed_zero_and_nonzero_workers():
    """Zero-worker configs dedupe while worker configs keep their prefetch axis."""
    sweep = {
        "num_workers": [0, 4],
        "num_physical_cpus": [1],
        "prefetch_factor": [2, 4],
        "pin_memory": [True],
        "persistent_workers": [True],
        "batch_size": [64],
    }
    configs = expand_configs(sweep)
    # 1 collapsed zero-worker config + 2 four-worker configs (prefetch 2 and 4).
    assert len(configs) == 3
    n_workers = sorted(c["num_workers"] for c in configs)
    assert n_workers == [0, 4, 4]


def test_resolve_mmap_path(tmp_path: Path):
    """The signals*.npy mmap is discovered; absence returns None."""
    assert resolve_mmap_path(tmp_path) is None
    mmap_file = tmp_path / "signals_raw.npy"
    mmap_file.write_bytes(b"\x00")
    assert resolve_mmap_path(tmp_path) == mmap_file


def test_drop_page_cache_is_safe(tmp_path: Path):
    """drop_page_cache tolerates None and real files without raising."""
    drop_page_cache(None)  # no-op, must not raise
    real = tmp_path / "signals_raw.npy"
    real.write_bytes(b"\x00" * 4096)
    drop_page_cache(real)  # should silently succeed (or warn) regardless of OS


@pytest.mark.parametrize("missing", [True, False])
def test_drop_page_cache_missing_file(tmp_path: Path, missing: bool):
    """A missing path is handled gracefully (warning, no exception)."""
    path = tmp_path / "nope.npy"
    if not missing:
        path.write_bytes(b"\x00")
    drop_page_cache(path)
