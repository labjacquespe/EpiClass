#!/usr/bin/env python3
"""
Two-phase HDF5 file integrity checker:

  Phase 1 — h5check:   Verify file format compliance against the HDF5 spec.
                        This catches low-level corruption (bad superblocks,
                        broken B-trees, invalid object headers, etc.).

  Phase 2 — h5dump -H: Verify that all expected chromosome datasets
                        (chr1-22, chrX, chrY) are present in the metadata.

Usage:
    python check_hdf5_integrity.py file_list.txt -t 8
"""
import argparse
import logging
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXPECTED_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# Matches lines like:  DATASET "chr1" {
DATASET_RE = re.compile(r'DATASET\s+"([^"]+)"')

# Captures dataset name and its DATATYPE from the header block.
# Example:  DATASET "chr1" {
#              DATATYPE  H5T_IEEE_F32LE
DATASET_DTYPE_RE = re.compile(
    r'DATASET\s+"([^"]+)"\s*\{[^}]*?DATATYPE\s+(H5T_\S+)',
    re.DOTALL,
)


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Two-phase HDF5 integrity checker: h5check (format) + h5dump -H (datasets)."
    )
    parser.add_argument(
        "file_list",
        help="Path to a text file with one HDF5 file path per line.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Number of parallel threads (default: 4).",
    )
    parser.add_argument(
        "--h5check-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for h5check per file (default: 300).",
    )
    parser.add_argument(
        "--h5dump-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for h5dump per file (default: 120).",
    )
    parser.add_argument(
        "--check-dtype",
        action="store_true",
        default=False,
        help="Verify that each chromosome dataset has DATATYPE H5T_IEEE_F32LE (float32).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path (includes per-file DEBUG detail).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N files (default: 50).",
    )
    return parser.parse_args()


# ── Logging setup ───────────────────────────────────────────────────────────


def setup_logging(logfile=None):
    """Configure logging to stderr and optionally to a file."""
    logger = logging.getLogger("h5integrity")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Always log to stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    # Optionally log to file (includes DEBUG for per-file detail)
    if logfile:
        file_handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


# ── Progress tracker (thread-safe) ──────────────────────────────────────────


class ProgressTracker:
    """Thread-safe counter that logs periodic progress updates."""

    def __init__(self, total, logger, log_every=50):
        self.total = total
        self.logger = logger
        self.log_every = log_every
        self._lock = threading.Lock()
        self._done = 0
        self._passed = 0
        self._failed = 0
        self._start = time.monotonic()

    def record(self, ok):
        """Record a completed file and log progress at intervals."""
        with self._lock:
            self._done += 1
            if ok:
                self._passed += 1
            else:
                self._failed += 1
            done = self._done
            passed = self._passed
            failed = self._failed

        if done % self.log_every == 0 or done == self.total:
            elapsed = time.monotonic() - self._start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (self.total - done) / rate if rate > 0 else 0
            self.logger.info(
                "Progress: %d/%d (%.1f%%) | passed=%d failed=%d | "
                "%.1f files/s | elapsed %s | ETA %s",
                done,
                self.total,
                done * 100 / self.total,
                passed,
                failed,
                rate,
                self._fmt_time(elapsed),
                self._fmt_time(eta),
            )

    @staticmethod
    def _fmt_time(seconds):
        """Format seconds into a human-readable string."""
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}h{mins:02d}m{secs:02d}s"
        if mins:
            return f"{mins}m{secs:02d}s"
        return f"{secs}s"

    def summary(self):
        """Log final summary with totals and throughput."""
        elapsed = time.monotonic() - self._start
        rate = self._done / elapsed if elapsed > 0 else 0
        self.logger.info(
            "Finished: %d/%d passed, %d failed | total time %s | avg %.1f files/s",
            self._passed,
            self.total,
            self._failed,
            self._fmt_time(elapsed),
            rate,
        )


## ── HDF5 checks ────────────────────────────────────────────────────────────


def run_h5check(filepath, timeout=300):
    """
    Run h5check on the file to verify format compliance.

    Returns dict with 'passed' (bool) and 'error' (str or None).
    """
    try:
        proc = subprocess.run(
            ["h5check", filepath],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {
            "passed": False,
            "error": "h5check not found — install it separately "
            "from https://github.com/HDFGroup/h5check",
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"h5check timed out (>{timeout}s)"}

    if proc.returncode != 0:
        output = (proc.stderr or proc.stdout).strip()
        first_lines = "\n".join(output.splitlines()[:5])
        return {"passed": False, "error": f"h5check non-compliant: {first_lines[:300]}"}

    return {"passed": True, "error": None}


def run_h5dump_check(filepath, timeout=120, check_dtype=False):
    """
    Run h5dump -H and verify expected chromosome datasets exist.

    Optionally verify that each chromosome dataset has DATATYPE H5T_IEEE_F32LE.

    Returns dict with 'passed', 'missing', 'found', 'wrong_dtype', 'error'.
    """
    result_base = {
        "passed": False,
        "missing": [],
        "found": [],
        "wrong_dtype": [],
        "error": None,
    }

    try:
        proc = subprocess.run(
            ["h5dump", "-H", filepath],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {**result_base, "error": "h5dump not found — install HDF5 tools"}
    except subprocess.TimeoutExpired:
        return {**result_base, "error": f"h5dump timed out (>{timeout}s)"}

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        return {
            **result_base,
            "error": f"h5dump failed (rc={proc.returncode}): {stderr[:300]}",
        }

    dataset_names = set(DATASET_RE.findall(proc.stdout))
    expected_set = set(EXPECTED_CHROMOSOMES)
    found = sorted(dataset_names & expected_set)
    missing = [c for c in EXPECTED_CHROMOSOMES if c not in dataset_names]

    # Optional dtype check: each chromosome dataset must be F32LE
    wrong_dtype = []
    if check_dtype and not missing:
        dataset_dtypes = dict(DATASET_DTYPE_RE.findall(proc.stdout))
        for chrom in EXPECTED_CHROMOSOMES:
            dtype = dataset_dtypes.get(chrom)
            if dtype and dtype != "H5T_IEEE_F32LE":
                wrong_dtype.append(f"{chrom}={dtype}")

    return {
        "passed": not missing and not wrong_dtype,
        "missing": missing,
        "found": found,
        "wrong_dtype": wrong_dtype,
        "error": None,
    }


def check_file(filepath, h5check_timeout=300, h5dump_timeout=120, check_dtype=False):
    """Run h5check then h5dump -H on a single file, returning a result dict."""
    result = {
        "file": filepath,
        "h5check_ok": False,
        "datasets_ok": False,
        "ok": False,
        "missing": [],
        "wrong_dtype": [],
        "error": None,
    }

    if not Path(filepath).is_file():
        result["error"] = "File not found"
        return result

    # Phase 1: format compliance
    h5c = run_h5check(filepath, timeout=h5check_timeout)
    result["h5check_ok"] = h5c["passed"]
    if not h5c["passed"]:
        result["error"] = h5c["error"]
        return result

    # Phase 2: dataset verification (+ optional dtype check)
    h5d = run_h5dump_check(filepath, timeout=h5dump_timeout, check_dtype=check_dtype)
    result["datasets_ok"] = h5d["passed"]
    if not h5d["passed"]:
        if h5d["error"]:
            result["error"] = h5d["error"]
        else:
            result["missing"] = h5d["missing"]
            result["wrong_dtype"] = h5d["wrong_dtype"]
        return result

    result["ok"] = True
    return result


def _format_failure(res):
    """Build a human-readable failure reason string from a result dict."""
    if res["error"]:
        return res["error"]

    reasons = []
    if res["missing"]:
        missing = ", ".join(res["missing"])
        reasons.append(f"missing {missing}")

    if res["wrong_dtype"]:
        wrong_dtype = ", ".join(res["wrong_dtype"])
        reasons.append(f"wrong dtype {wrong_dtype}")

    return "; ".join(reasons)


def _log_result(log, res, failed_files):
    """Log a single file result and append to failed_files if needed."""
    fname = res["file"]
    if res["ok"]:
        log.debug("OK    %s", fname)
        return
    if res["error"]:
        phase = "h5check" if not res["h5check_ok"] else "h5dump"
        log.warning("FAIL  %s  [%s] %s", fname, phase, res["error"])
    else:
        log.warning("FAIL  %s  [datasets] %s", fname, _format_failure(res))
    failed_files.append(res)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    """Main."""
    args = cli()

    log = setup_logging(args.log_file)

    list_path = Path(args.file_list)
    if not list_path.is_file():
        sys.exit(f"Error: file list {args.file_list} not found.")

    files = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not files:
        sys.exit("Error: file list is empty.")

    log.info(
        "Starting integrity check: %d files, %d threads",
        len(files),
        args.threads,
    )
    log.info(
        "  h5check timeout=%ds, h5dump timeout=%ds",
        args.h5check_timeout,
        args.h5dump_timeout,
    )
    if args.check_dtype:
        log.info(
            "  dtype check enabled: expecting H5T_IEEE_F32LE for all chromosome datasets"
        )

    progress = ProgressTracker(len(files), log, log_every=args.log_every)
    failed_files = []

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(
                check_file,
                f,
                args.h5check_timeout,
                args.h5dump_timeout,
                args.check_dtype,
            ): f
            for f in files
        }
        for future in as_completed(futures):
            res = future.result()
            _log_result(log, res, failed_files)
            progress.record(res["ok"])

    progress.summary()

    if failed_files:
        log.info("Failed files (%d):", len(failed_files))
        for res in failed_files:
            log.info("  %s: %s", res["file"], _format_failure(res))


if __name__ == "__main__":
    main()
