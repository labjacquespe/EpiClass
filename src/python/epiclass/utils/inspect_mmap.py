"""Inspect a preloaded HDF5 mmap cache to diagnose reuse hangs / corruption.

Run this on the *actual* ``signals_*.npy`` left behind by a crashed/segfaulted
run to get ground truth instead of guessing:

    python -m epiclass.utils.inspect_mmap /path/to/mmap_dir/signals_norm.npy
    python -m epiclass.utils.inspect_mmap /path/to/mmap_dir          # finds the .npy

It reports:
  - true on-disk state: logical size vs allocated blocks (``st_blocks`` -> sparse
    detection, which logical size alone hides), mtime, sibling ``.tmp``/``core`` files;
  - the ``.npy`` header (shape, dtype, header length) and expected vs actual bytes;
  - whether *reading* a few rows completes within a timeout. The read runs in a
    child process: if the child times out, reads themselves block -> the problem is
    at the mmap/filesystem layer (e.g. the file, the disk, or a stale lock), not in
    downstream code. If reads complete, the hang reported elsewhere is NOT this file.

This deliberately does no repair; it only observes.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np

from epiclass.core.lazy.lazy_hdf5_loader import read_npy_header


def _read_probe(path: str, row_idxs: list[int], queue: Queue) -> None:
    """Child-process worker: force-read the given rows and report stats."""
    try:
        arr = np.load(path, mmap_mode="r")
        stats = {}
        for i in row_idxs:
            row = np.asarray(arr[i])  # materialize -> forces the page fault
            stats[i] = {
                "nonzero_frac": float(np.count_nonzero(row)) / row.size,
                "has_nan": bool(np.isnan(row).any()),
                "has_inf": bool(np.isinf(row).any()),
            }
        queue.put(("ok", stats))
    except Exception as err:  # pylint: disable=broad-except
        queue.put(("err", repr(err)))


def _scan_zeros(path: str, queue: Queue) -> None:
    """Child-process worker: count exact-zero values over the whole array.

    A normalized real signal has ~no exact zeros, so zeros mark sparse holes /
    lost data. Counts individual values (not all-zero rows) because page-sized
    holes land mid-row, leaving the row partly real — a row-level check misses them.
    """
    try:
        arr = np.load(path, mmap_mode="r")
        n_zero = 0
        rows_hit = 0
        for start in range(0, arr.shape[0], 512):  # chunk to bound memory
            block = np.asarray(arr[start : start + 512]) == 0
            n_zero += int(block.sum())
            rows_hit += int(np.count_nonzero(block.any(axis=1)))
        queue.put(("ok", (int(arr.shape[0]), int(arr.size), n_zero, rows_hit)))
    except Exception as err:  # pylint: disable=broad-except
        queue.put(("err", repr(err)))


def _resolve_npy(target: Path) -> Path:
    """Accept a .npy file or a directory containing exactly one signals_*.npy."""
    if target.is_dir():
        candidates = sorted(target.glob("signals_*.npy"))
        if not candidates:
            raise FileNotFoundError(f"No signals_*.npy under {target}")
        return candidates[0]
    return target


def _run_child(target, args, timeout: float) -> tuple[str, object]:
    """Run a probe worker in a child process so a blocking read can't hang us."""
    queue: Queue = Queue()
    proc = Process(target=target, args=(*args, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ("hang", None)
    if queue.empty():
        return ("err", "no result")
    return queue.get()


def _print_file_state(npy_path: Path) -> tuple[int, ...] | None:
    """Print on-disk state + header; return the array shape (None if unreadable)."""
    stat = npy_path.stat()
    allocated = stat.st_blocks * 512  # st_blocks is always 512-byte units
    print(f"  logical size : {stat.st_size:,} bytes")
    if stat.st_size:
        print(f"  allocated    : {allocated:,} bytes ({allocated / stat.st_size:.1%})")
        if allocated < stat.st_size * 0.99:
            print("  -> SPARSE: unallocated holes (zero content or never written).")
    print(f"  mtime        : {time.ctime(stat.st_mtime)}")

    siblings = list(npy_path.parent.glob("*.tmp")) + list(npy_path.parent.glob("core*"))
    if siblings:
        print(f"  siblings     : {[s.name for s in siblings]}")

    try:
        shape, dtype, header_end = read_npy_header(npy_path)
    except Exception as err:  # pylint: disable=broad-except
        print(f"  header       : UNREADABLE ({err!r})")
        return None
    expected = header_end + int(np.prod(shape)) * dtype.itemsize
    print(f"  header shape : {shape}  dtype={dtype}  header_end={header_end}")
    print(f"  expected size: {expected:,} bytes (header + data)")
    if stat.st_size < expected:
        print("  -> TRUNCATED: file shorter than the header declares.")
    return shape


def _print_read_probe(npy_path: Path, n_rows: int, n: int, timeout: float) -> None:
    """Probe a few rows in a child process and print the result."""
    row_idxs = sorted({0, n // 2, n - 1, *range(min(n_rows, n))})
    print(f"  probing reads of rows {row_idxs} (timeout {timeout}s)...")
    status, payload = _run_child(_read_probe, (str(npy_path), row_idxs), timeout)
    if status == "hang":
        print("  -> READS HANG: the read never returned within the timeout.")
        print("     The problem is at the mmap/filesystem layer, not downstream.")
    elif status == "err":
        print(f"  -> READ ERROR: {payload}")
    else:
        assert isinstance(payload, dict)
        for i, stats in payload.items():  # pylint: disable=no-member
            print(
                f"     row {i}: nonzero={stats['nonzero_frac']:.3f} "
                f"nan={stats['has_nan']} inf={stats['has_inf']}"
            )
        print("  reads completed: this file is readable (hang is elsewhere).")


def _print_full_scan(npy_path: Path, timeout: float) -> None:
    """Scan the whole array for exact-zero values (filesystem-independent truth)."""
    scan_timeout = max(timeout, 300.0)
    print(f"  full scan for zero values (timeout {scan_timeout}s)...")
    status, payload = _run_child(_scan_zeros, (str(npy_path),), scan_timeout)
    if status == "hang":
        print("  -> SCAN HANG: full read never returned.")
    elif status == "err":
        print(f"  -> SCAN ERROR: {payload}")
    else:
        assert isinstance(payload, tuple)
        n_rows, total, n_zero, rows_hit = payload
        if n_zero:
            print(
                f"  -> {n_zero}/{total} values are EXACTLY ZERO ({n_zero / total:.1%}) "
                f"across {rows_hit}/{n_rows} rows (sparse holes / lost data)."
            )
        else:
            print(f"  no zero values: all {total} values in {n_rows} rows are real.")


def inspect(npy_path: Path, n_rows: int, timeout: float, full_scan: bool) -> None:
    """Print on-disk state, header, sibling files, and a timed read probe."""
    print(f"== Inspecting {npy_path} ==")
    if not npy_path.exists():
        print("  MISSING: file does not exist.")
        return

    shape = _print_file_state(npy_path)
    if not shape:
        return

    _print_read_probe(npy_path, n_rows, shape[0], timeout)
    if full_scan:
        _print_full_scan(npy_path, timeout)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="signals_*.npy file or its directory")
    parser.add_argument(
        "--rows", type=int, default=4, help="How many leading rows to probe."
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-read-probe timeout (s)."
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Read the whole array and count all-zero (unwritten) rows.",
    )
    args = parser.parse_args()
    inspect(_resolve_npy(args.path), args.rows, args.timeout, args.full_scan)


if __name__ == "__main__":
    main()
