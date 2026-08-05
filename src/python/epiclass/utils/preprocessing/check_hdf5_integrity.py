#!/usr/bin/env python3
"""
Three-phase HDF5 file integrity checker:

  Phase 1 — h5check:   Verify file format compliance against the HDF5 spec.
                        This catches low-level corruption (bad superblocks,
                        broken B-trees, invalid object headers, etc.).

  Phase 2 — h5dump -H: Verify that all expected chromosome datasets
                        (chr1-22, chrX, chrY) are present in the metadata.

  Phase 3 — value scan (opt-in, via --outlier-threshold): read the signal with
                        h5py and flag values whose absolute amplitude exceeds a
                        threshold, plus any non-finite (NaN / +-Inf) values.
                        Phases 1 and 2 never read the data, so this is the only
                        phase that catches corrupted signal in a valid file.

Optional dtype verification (via --dtype-mode):
  float : accept any float dtype; flag 64-bit (F64) as oversized.
  int   : accept any (signed/unsigned) int dtype; flag 64-bit as oversized.
  auto  : accept either float *or* int per file (auto-detected from the
          file's own datasets), flag 64-bit, and flag files that mix
          float and int across chromosomes.

Usage:
    python check_hdf5_integrity.py file_list.txt -t 8 --dtype-mode float

    # format + datasets + amplitude scan, offending values written to CSV
    python check_hdf5_integrity.py file_list.txt -t 12 \\
        --outlier-threshold 1e10 --outlier-csv scan_results.csv
"""
# pylint: disable=too-many-branches,too-many-positional-arguments
import argparse
import csv
import logging
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Set

import h5py
import numpy as np

EXPECTED_CHROMOSOMES_HUMAN = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
EXPECTED_CHROMOSOMES_MOUSE = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]

# Matches lines like:  DATASET "chr1" {
DATASET_RE = re.compile(r'DATASET\s+"([^"]+)"')

# Captures dataset name and its DATATYPE from the header block.
# Example:  DATASET "chr1" {
#              DATATYPE  H5T_IEEE_F32LE
DATASET_DTYPE_RE = re.compile(
    r'DATASET\s+"([^"]+)"\s*\{[^}]*?DATATYPE\s+(H5T_\S+)',
    re.DOTALL,
)

# Accepted dtype families.
# - 32-bit floats:  H5T_IEEE_F32LE / F32BE
# - 64-bit floats:  H5T_IEEE_F64LE / F64BE  (flagged as oversized)
# - 32-bit ints:    H5T_STD_{I,U}32{LE,BE}
# - 64-bit ints:    H5T_STD_{I,U}64{LE,BE}  (flagged as oversized)
FLOAT32_RE = re.compile(r"^H5T_IEEE_F32[LB]E$")
FLOAT64_RE = re.compile(r"^H5T_IEEE_F64[LB]E$")
INT32_RE = re.compile(r"^H5T_STD_[IU]32[LB]E$")
INT64_RE = re.compile(r"^H5T_STD_[IU]64[LB]E$")


def classify_dtype(dtype):
    """
    Classify an h5dump DATATYPE string.

    Returns one of: 'float32', 'float64', 'int32', 'int64', 'other'.
    """
    if FLOAT32_RE.match(dtype):
        return "float32"
    if FLOAT64_RE.match(dtype):
        return "float64"
    if INT32_RE.match(dtype):
        return "int32"
    if INT64_RE.match(dtype):
        return "int64"
    return "other"


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "HDF5 integrity checker: h5check (format) + h5dump -H (datasets) "
            "+ optional value scan (--outlier-threshold)."
        )
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
        "--dtype-mode",
        choices=["off", "float", "int", "auto"],
        default="auto",
        help=(
            "Dtype verification mode. "
            "'off': skip dtype checks. "
            "'float': require all chromosome datasets to be float (32-bit ok, "
            "64-bit flagged as oversized). "
            "'int': require all chromosome datasets to be int (32-bit ok, "
            "64-bit flagged as oversized — unlikely to be needed for rank data). "
            "'auto'  (default): accept either float or int per file (auto-detected), flag "
            "64-bit, and flag files that mix float and int across chromosomes."
        ),
    )
    parser.add_argument(
        "--chromosomes",
        choices=["human", "mouse"],
        default="human",
        help=(
            "Set expected chromosome names. "
            "'human' (default): chr1-22, chrX, chrY. "
            "'mouse': chr1-19, chrX, chrY."
        ),
    )
    parser.add_argument(
        "--allow-missing-chry",
        action="store_true",
        default=False,
        help=(
            "Treat chrY as optional: a file missing only chrY still passes. "
            "Dtype checks (if enabled) run on the remaining chromosomes. "
            "Files where chrY is absent are logged as info, not as failures."
        ),
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=None,
        help=(
            "Enable the value scan (phase 3): flag values whose absolute "
            "amplitude exceeds this threshold, plus any non-finite (NaN/Inf) "
            "value. Reads the signal with h5py, so it is much slower than the "
            "metadata-only phases. Disabled by default."
        ),
    )
    parser.add_argument(
        "--outlier-csv",
        type=Path,
        default=None,
        help=(
            "Write offending values found by the value scan to this CSV "
            "(columns: file, dataset, index, value, kind). Requires "
            "--outlier-threshold. An existing file is overwritten."
        ),
    )
    parser.add_argument(
        "--max-outliers-per-file",
        type=int,
        default=None,
        help=(
            "Cap on individual offending values recorded per file. Unlimited "
            "by default. Records are held in memory until the file's results "
            "are written, so a fully corrupted fine-resolution file (millions "
            "of bins) can cost ~1 GB per in-flight file: cap it on memory-"
            "constrained jobs. Reported counts stay exact regardless."
        ),
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
    args = parser.parse_args()

    if args.outlier_csv is not None and args.outlier_threshold is None:
        parser.error("--outlier-csv requires --outlier-threshold")
    if args.max_outliers_per_file is not None and args.max_outliers_per_file < 1:
        parser.error("--max-outliers-per-file must be >= 1")
    if args.outlier_threshold is not None:
        args.outlier_threshold = abs(args.outlier_threshold)

    return args


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


def verify_dtypes(
    expected_chromosomes: set[str], dataset_dtypes: dict[str, str], mode: str
):
    """
    Verify chromosome dataset dtypes against the requested mode.

    Parameters
    ----------
    expected_chromosomes : set[str]
        Set of expected chromosome names.
    dataset_dtypes : dict[str, str]
        Map of dataset name -> DATATYPE string (from h5dump -H).
    mode : str
        One of 'float', 'int', 'auto'.

    Returns
    -------
    dict with keys:
        wrong_dtype : list[str]   # "chrom=DTYPE" entries (unrecognized/disallowed family)
        oversized   : list[str]   # "chrom=DTYPE" entries (64-bit variants)
        mixed       : list[str]   # "chrom=DTYPE" entries that broke file-level consistency (auto only)
        family      : str or None # resolved family for this file ('float' or 'int'), auto mode only
    """
    wrong_dtype = []
    oversized = []
    mixed = []
    family = None  # for auto mode

    # In auto mode, the first classified chromosome sets the expected family
    # for the rest of the file. Subsequent chromosomes of the *other* family
    # are reported as 'mixed'. Sort to give "first" a stable meaning — callers
    # often pass a set, whose iteration order is hash-dependent.
    for chrom in sorted(expected_chromosomes):
        dtype = dataset_dtypes.get(chrom)
        if dtype is None:
            # Missing datasets are handled by the caller; skip here.
            continue

        kind = classify_dtype(
            dtype
        )  # 'float32' | 'float64' | 'int32' | 'int64' | 'other'

        if kind == "other":
            wrong_dtype.append(f"{chrom}={dtype}")
            continue

        this_family = "float" if kind.startswith("float") else "int"
        is_64bit = kind.endswith("64")

        if mode == "float":
            if this_family != "float":
                wrong_dtype.append(f"{chrom}={dtype}")
            elif is_64bit:
                oversized.append(f"{chrom}={dtype}")
        elif mode == "int":
            if this_family != "int":
                wrong_dtype.append(f"{chrom}={dtype}")
            elif is_64bit:
                oversized.append(f"{chrom}={dtype}")
        elif mode == "auto":
            if family is None:
                family = this_family
            if this_family != family:
                mixed.append(f"{chrom}={dtype}")
            if is_64bit:
                oversized.append(f"{chrom}={dtype}")

    return {
        "wrong_dtype": wrong_dtype,
        "oversized": oversized,
        "mixed": mixed,
        "family": family,
    }


def run_h5dump_check(
    filepath,
    expected_chromosomes: Set[str],
    timeout=120,
    dtype_mode="off",
    allow_missing_chry=False,
):
    """
    Run h5dump -H and verify expected chromosome datasets exist.

    Optionally verify dtypes according to dtype_mode ('off', 'float', 'int', 'auto').

    When allow_missing_chry is True, a file missing only chrY is still considered
    to have passed the dataset check; chrY is reported in 'missing_allowed'
    (informational) rather than 'missing' (failure).

    Returns dict with 'passed', 'missing', 'missing_allowed', 'found',
    'wrong_dtype', 'oversized', 'mixed', 'family', 'error'.
    """
    result_base = {
        "passed": False,
        "missing": [],
        "missing_allowed": [],
        "found": [],
        "wrong_dtype": [],
        "oversized": [],
        "mixed": [],
        "family": None,
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
    expected_set = set(expected_chromosomes)
    found = sorted(dataset_names & expected_set)

    # Split missing chromosomes into failing vs. informational (allowed).
    # chrY is optional when allow_missing_chry is True.
    all_missing = [c for c in expected_chromosomes if c not in dataset_names]
    missing_allowed = []
    missing = []
    for c in all_missing:
        if c == "chrY" and allow_missing_chry:
            missing_allowed.append(c)
        else:
            missing.append(c)

    wrong_dtype = []
    oversized = []
    mixed = []
    family = None

    # Only run dtype verification if enabled and no *required* chromosomes are
    # missing. chrY in missing_allowed does not block the dtype pass; the
    # per-chrom loop in verify_dtypes naturally skips datasets that aren't
    # present in the dtype map.
    if dtype_mode != "off" and not missing:
        dataset_dtypes = dict(DATASET_DTYPE_RE.findall(proc.stdout))
        dv = verify_dtypes(expected_chromosomes, dataset_dtypes, dtype_mode)
        wrong_dtype = dv["wrong_dtype"]
        oversized = dv["oversized"]
        mixed = dv["mixed"]
        family = dv["family"]

    # A file passes only if nothing went wrong. 'oversized' is treated as a
    # failure condition (64-bit is flagged). 'mixed' is a failure in auto mode.
    # 'missing_allowed' is informational and does not affect 'passed'.
    passed = not missing and not wrong_dtype and not oversized and not mixed

    return {
        "passed": passed,
        "missing": missing,
        "missing_allowed": missing_allowed,
        "found": found,
        "wrong_dtype": wrong_dtype,
        "oversized": oversized,
        "mixed": mixed,
        "family": family,
        "error": None,
    }


def run_value_scan(
    filepath,
    expected_chromosomes: Set[str],
    threshold: float,
    max_records: int | None = None,
):
    """
    Read chromosome datasets and flag out-of-range and non-finite values.

    Two conditions are reported, both of which indicate corrupted signal:
      - outlier   : abs(value) > threshold (strictly greater).
      - nonfinite : NaN or +-Inf. Checked separately because abs(nan) > t is
                    False, so a pure threshold test silently misses NaN.

    Only datasets named in `expected_chromosomes` are scanned; the file is
    expected to hold exactly one top-level group (the single-sample layout
    produced by epigeec), and a different group count is reported as an error.

    Every offending value is recorded by default. `max_records` caps how many
    are materialised per file, for memory-constrained runs over fine-resolution
    data where a fully corrupted file holds millions of them; the reported
    counts stay exact either way.

    Returns dict with 'passed', 'n_outliers', 'n_nonfinite', 'records', 'error'.
    Each record is {file, dataset, index, value, kind}.
    """
    result = {
        "passed": False,
        "n_outliers": 0,
        "n_nonfinite": 0,
        "records": [],
        "error": None,
    }

    records = []
    n_outliers = 0
    n_nonfinite = 0

    def _collect(dataset_name, data, mask, kind):
        """Count masked entries and record them, honouring the per-file cap."""
        count = int(np.count_nonzero(mask))
        if not count:
            return 0
        room = count if max_records is None else max_records - len(records)
        if room > 0:
            for idx in np.flatnonzero(mask)[:room]:
                records.append(
                    {
                        "file": str(filepath),
                        "dataset": dataset_name,
                        "index": int(idx),
                        "value": float(data[idx]),
                        "kind": kind,
                    }
                )
        return count

    try:
        with h5py.File(filepath, "r") as h5file:
            groups = [obj for _, obj in h5file.items() if isinstance(obj, h5py.Group)]
            if len(groups) != 1:
                return {
                    **result,
                    "error": f"expected 1 group, found {len(groups)}",
                }

            group = groups[0]
            for name in expected_chromosomes:
                dset = group.get(name)  # type: ignore[union-attr]
                if not isinstance(dset, h5py.Dataset):
                    # Dataset presence is phase 2's job; nothing to scan here.
                    continue

                data = np.asarray(dset[...]).ravel()
                finite = np.isfinite(data)
                n_nonfinite += _collect(name, data, ~finite, "nonfinite")
                over = finite & (np.abs(data) > threshold)
                n_outliers += _collect(name, data, over, "outlier")
    except (OSError, KeyError, ValueError) as err:
        return {**result, "error": str(err)}

    return {
        "passed": not n_outliers and not n_nonfinite,
        "n_outliers": n_outliers,
        "n_nonfinite": n_nonfinite,
        "records": records,
        "error": None,
    }


def check_file(
    filepath,
    expected_chromosomes: Set[str],
    h5check_timeout=300,
    h5dump_timeout=120,
    dtype_mode="off",
    allow_missing_chry=False,
    outlier_threshold=None,
    max_outliers_per_file=None,
):
    """Run h5check, h5dump -H, then (optionally) the value scan on a single file.

    The value scan runs only when `outlier_threshold` is not None, and only if
    the earlier phases passed — there is no point reading signal out of a file
    that is already known to be malformed. When it is disabled, 'values_ok'
    stays True so the overall verdict is unchanged.

    Returns a result dict.
    """
    result = {
        "file": filepath,
        "h5check_ok": False,
        "datasets_ok": False,
        "values_ok": True,
        "ok": False,
        "missing": [],
        "missing_allowed": [],
        "wrong_dtype": [],
        "oversized": [],
        "mixed": [],
        "family": None,
        "n_outliers": 0,
        "n_nonfinite": 0,
        "records": [],
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
    h5d = run_h5dump_check(
        filepath,
        expected_chromosomes=expected_chromosomes,
        timeout=h5dump_timeout,
        dtype_mode=dtype_mode,
        allow_missing_chry=allow_missing_chry,
    )
    result["datasets_ok"] = h5d["passed"]
    result["family"] = h5d["family"]
    # missing_allowed is informational; carry it through regardless of pass/fail.
    result["missing_allowed"] = h5d["missing_allowed"]
    if not h5d["passed"]:
        if h5d["error"]:
            result["error"] = h5d["error"]
        else:
            result["missing"] = h5d["missing"]
            result["wrong_dtype"] = h5d["wrong_dtype"]
            result["oversized"] = h5d["oversized"]
            result["mixed"] = h5d["mixed"]
        return result

    # Phase 3: value scan (opt-in)
    if outlier_threshold is not None:
        scan = run_value_scan(
            filepath,
            expected_chromosomes=expected_chromosomes,
            threshold=outlier_threshold,
            max_records=max_outliers_per_file,
        )
        result["values_ok"] = scan["passed"]
        result["n_outliers"] = scan["n_outliers"]
        result["n_nonfinite"] = scan["n_nonfinite"]
        result["records"] = scan["records"]
        if not scan["passed"]:
            if scan["error"]:
                result["error"] = scan["error"]
            return result

    result["ok"] = True
    return result


def format_failure(res):
    """Build a human-readable failure reason string from a result dict."""
    if res["error"]:
        return res["error"]

    reasons = []
    if res["missing"]:
        reasons.append(f"missing {', '.join(res['missing'])}")
    if res["wrong_dtype"]:
        reasons.append(f"wrong dtype {', '.join(res['wrong_dtype'])}")
    if res["oversized"]:
        reasons.append(f"oversized dtype {', '.join(res['oversized'])}")
    if res["mixed"]:
        reasons.append(f"mixed dtype families {', '.join(res['mixed'])}")
    if res.get("n_outliers"):
        reasons.append(f"{res['n_outliers']} values above threshold")
    if res.get("n_nonfinite"):
        reasons.append(f"{res['n_nonfinite']} non-finite values")

    return "; ".join(reasons)


def _log_result(log, res, failed_files):
    """Log a single file result and append to failed_files if needed."""
    fname = res["file"]
    if res["ok"]:
        if res.get("missing_allowed"):
            log.debug(
                "OK    %s  (allowed missing: %s)",
                fname,
                ", ".join(res["missing_allowed"]),
            )
        else:
            log.debug("OK    %s", fname)
        return
    if res["error"]:
        if not res["h5check_ok"]:
            phase = "h5check"
        elif not res["datasets_ok"]:
            phase = "h5dump"
        else:
            phase = "values"
        log.warning("FAIL  %s  [%s] %s", fname, phase, res["error"])
    else:
        phase = "datasets" if not res["datasets_ok"] else "values"
        log.warning("FAIL  %s  [%s] %s", fname, phase, format_failure(res))
    failed_files.append(res)


OUTLIER_CSV_FIELDS = ["file", "dataset", "index", "value", "kind"]


@contextmanager
def _open_outlier_writer(csv_path: Path, log):
    """Yield a csv.DictWriter for offending values, header already written."""
    if csv_path.exists():
        log.warning("Output file %s already exists and will be overwritten.", csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTLIER_CSV_FIELDS)
        writer.writeheader()
        yield writer


def _run_checks(files, expected_chromosomes, args, log, progress, writer=None):
    """Check every file in parallel, returning (failed_files, allowed_missing_count).

    Results are consumed on the calling thread, so `writer` needs no locking.
    """
    failed_files = []
    allowed_missing_count = 0

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(
                check_file,
                f,
                expected_chromosomes,
                args.h5check_timeout,
                args.h5dump_timeout,
                args.dtype_mode,
                args.allow_missing_chry,
                args.outlier_threshold,
                args.max_outliers_per_file,
            ): f
            for f in files
        }
        for future in as_completed(futures):
            res = future.result()
            _log_result(log, res, failed_files)
            progress.record(res["ok"])
            if res["ok"] and res.get("missing_allowed"):
                allowed_missing_count += 1
            if writer is not None and res.get("records"):
                writer.writerows(res["records"])

    return failed_files, allowed_missing_count


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

    # Determine expected chromosomes based on the --chromosomes argument
    if args.chromosomes == "human":
        expected_chromosomes = set(EXPECTED_CHROMOSOMES_HUMAN)
    elif args.chromosomes == "mouse":
        expected_chromosomes = set(EXPECTED_CHROMOSOMES_MOUSE)
    else:
        sys.exit(f"Error: unknown chromosome set '{args.chromosomes}'")

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
    if args.dtype_mode != "off":
        log.info(
            "  dtype check enabled (mode=%s): 32-bit accepted, 64-bit flagged as oversized%s",
            args.dtype_mode,
            "; mixed float/int families flagged" if args.dtype_mode == "auto" else "",
        )
    if args.allow_missing_chry:
        log.info("  chrY is optional: files missing only chrY will pass")
    if args.outlier_threshold is not None:
        cap = (
            "all recorded"
            if args.max_outliers_per_file is None
            else f"max {args.max_outliers_per_file} recorded per file"
        )
        log.info(
            "  value scan enabled: flagging abs(value) > %e and non-finite values (%s)",
            args.outlier_threshold,
            cap,
        )

    progress = ProgressTracker(len(files), log, log_every=args.log_every)

    with ExitStack() as stack:
        writer = None
        if args.outlier_csv is not None:
            writer = stack.enter_context(_open_outlier_writer(args.outlier_csv, log))
        failed_files, allowed_missing_count = _run_checks(
            files, expected_chromosomes, args, log, progress, writer
        )

    progress.summary()

    if args.outlier_csv is not None:
        log.info("Offending values written to %s", args.outlier_csv)

    if allowed_missing_count:
        log.info(
            "Passed files with allowed-missing chromosomes: %d (chrY treated as optional)",
            allowed_missing_count,
        )

    if failed_files:
        log.info("Failed files (%d):", len(failed_files))
        for res in failed_files:
            log.info("  %s: %s", res["file"], format_failure(res))


if __name__ == "__main__":
    main()
