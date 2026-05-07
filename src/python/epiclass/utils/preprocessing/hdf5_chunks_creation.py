#!/usr/bin/env python3
"""Convert single-sample HDF5 files to chunked HDF5 format.

Reads per-sample HDF5 files (one file per sample, with per-chromosome
datasets), concatenates chromosomes, optionally normalizes, and writes
the results into chunked HDF5 files containing many samples each.

Output HDF5 structure (per chunk):
    chunk_NNNN.h5
    ├── signals:    shape (n_samples, signal_length), dtype float32
    ├── sample_ids: shape (n_samples,), dtype variable-length str
    └── attrs
    │   ├── signal_length: int
    │   ├── normalized: bool
    │   └── source_chrom_file: str

Usage examples:

    # Convert all files, 10000 samples per chunk:
    python convert_to_chunked.py \\
        --hdf5-list /path/to/file_list.txt \\
        --chrom-file /path/to/chroms.txt \\
        --output-dir /path/to/chunks/ \\
        --samples-per-chunk 10000

    # Convert with normalization, custom chunk size:
    python convert_to_chunked.py \\
        --hdf5-list /path/to/file_list.txt \\
        --chrom-file /path/to/chroms.txt \\
        --output-dir /path/to/chunks/ \\
        --samples-per-chunk 5000 \\
        --normalize

    # Convert only specific samples:
    python convert_to_chunked.py \\
        --hdf5-list /path/to/file_list.txt \\
        --chrom-file /path/to/chroms.txt \\
        --output-dir /path/to/chunks/ \\
        --samples-per-chunk 10000 \\
        --sample-ids-file /path/to/wanted_ids.txt

    # Dry run to check sizes:
    python convert_to_chunked.py \\
        --hdf5-list /path/to/file_list.txt \\
        --chrom-file /path/to/chroms.txt \\
        --output-dir /path/to/chunks/ \\
        --dry-run

    # Override HDF5 directory (e.g. files copied to $SLURM_TMPDIR):
    python convert_to_chunked.py \\
        --hdf5-list /path/to/file_list.txt \\
        --chrom-file /path/to/chroms.txt \\
        --output-dir /path/to/chunks/ \\
        --hdf5-dir "$SLURM_TMPDIR/hdf5s"
"""
# pylint: disable=too-many-branches, too-many-positional-arguments, duplicate-code
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np


# Some repeated code from the chunked loader is duplicated here to avoid importing epiclass
def load_chroms(chrom_file: Path) -> List[str]:
    """Return sorted chromosome names from a chromosome file."""
    with open(chrom_file, "r", encoding="utf-8") as f:
        chroms = []
        for line in f:
            line = line.rstrip()
            if line:
                chroms.append(line.split()[0])
    chroms.sort()
    return chroms


def read_hdf5_list(data_file: Path) -> Dict[str, Path]:
    """Return {sample_id: file_path} dict from a file-of-paths.

    Expects one HDF5 path per line. The sample ID is extracted from
    the filename: if the filename starts with a 32-character hex string
    followed by '_', that prefix is used as the ID; otherwise the
    file stem is used.
    """
    files: Dict[str, Path] = {}
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            path = Path(line.rstrip())
            sample_id = _extract_sample_id(path)
            files[sample_id] = path
    return files


def _extract_sample_id(file_path: Path) -> str:
    """Extract sample ID from filename."""
    candidate = file_path.name.split("_")[0]
    if len(candidate) == 32:
        try:
            int(candidate, 16)
            return candidate
        except ValueError:
            pass
    return file_path.stem


def read_signal(
    hdf5_path: Path,
    chroms: List[str],
    normalize: bool,
) -> np.ndarray:
    """Read a single-sample HDF5, concatenate chromosomes, return signal.

    Args:
        hdf5_path: Path to single-sample HDF5 file.
        chroms: Sorted list of chromosome names to read.
        normalize: Whether to z-score normalize the signal.

    Returns:
        1-D float32 array of concatenated chromosome signals.
    """
    with h5py.File(hdf5_path, "r") as f:
        try:
            header = list(f.keys())[0]
        except IndexError as e:
            raise OSError(f"No groups found in {hdf5_path}") from e

        group = f[header]
        chrom_signals: List[Sequence[float]] = [
            group[chrom][...] for chrom in chroms  # type: ignore[index]
        ]
        # pylint: disable-next=unexpected-keyword-arg
        signal = np.concatenate(chrom_signals, dtype=np.float32)

    if normalize:
        with np.errstate(all="raise"):
            signal = (signal - signal.mean()) / signal.std()

    return signal


def write_chunk(
    chunk_path: Path,
    sample_ids: List[str],
    signals: np.ndarray,
    signal_length: int,
    normalized: bool,
    chrom_file: str,
) -> None:
    """Write a single chunk HDF5 file.

    Args:
        chunk_path: Output file path.
        sample_ids: List of sample IDs in this chunk.
        signals: Array of shape (n_samples, signal_length).
        signal_length: Length of each signal vector.
        normalized: Whether signals were normalized.
        chrom_file: Source chromosome file path (stored as metadata).
    """
    with h5py.File(chunk_path, "w") as f:
        f.create_dataset("signals", data=signals, dtype=np.float32)
        f.create_dataset(
            "sample_ids",
            data=np.array(sample_ids, dtype=h5py.string_dtype()),
        )
        f.attrs["signal_length"] = signal_length
        f.attrs["normalized"] = normalized
        f.attrs["source_chrom_file"] = chrom_file


def load_sample_id_filter(path: Path) -> List[str]:
    """Load a list of sample IDs from a text file (one per line)."""
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def check_disk_space(
    output_dir: Path,
    total_bytes: int,
    headroom: float = 0.9,
) -> None:
    """Raise OSError if output directory lacks sufficient disk space."""
    usage = shutil.disk_usage(output_dir)
    if total_bytes > usage.free * headroom:
        raise OSError(
            f"Insufficient disk space in {output_dir}: "
            f"need {total_bytes / 1e9:.2f} GB, "
            f"have {usage.free / 1e9:.2f} GB available "
            f"({headroom:.0%} headroom policy)"
        )


def convert(
    hdf5_list: Path,
    chrom_file: Path,
    output_dir: Path,
    samples_per_chunk: int,
    normalize: bool = False,
    hdf5_dir: Optional[Path] = None,
    sample_id_filter: Optional[List[str]] = None,
    dry_run: bool = False,
    strict: bool = True,
) -> List[Path]:
    """Convert single-sample HDF5 files to chunked format.

    Args:
        hdf5_list: Path to text file listing HDF5 file paths.
        chrom_file: Path to chromosome names file.
        output_dir: Directory for output chunk files.
        samples_per_chunk: Maximum number of samples per chunk file.
        normalize: Whether to z-score normalize signals.
        hdf5_dir: Override directory for HDF5 files.
        sample_id_filter: If provided, only convert these sample IDs.
        dry_run: If True, compute sizes and print plan without writing.
        strict: If True, raise on any read error; otherwise skip.

    Returns:
        List of created chunk file paths.
    """
    chroms = load_chroms(chrom_file)
    files = read_hdf5_list(hdf5_list)

    if hdf5_dir is not None:
        files = {sid: hdf5_dir / path.name for sid, path in files.items()}

    if sample_id_filter is not None:
        filter_set = set(sample_id_filter)
        filtered = {sid: path for sid, path in files.items() if sid in filter_set}
        missing = filter_set - set(filtered.keys())
        if missing:
            print(
                f"Warning: {len(missing)} requested sample ID(s) "
                "not found in HDF5 list",
                file=sys.stderr,
            )
            if len(missing) <= 10:
                for sid in sorted(missing):
                    print(f"  {sid}", file=sys.stderr)
        files = filtered

    n_samples = len(files)
    if n_samples == 0:
        print("No samples to convert.", file=sys.stderr)
        return []

    # Determine signal length by trying to find a readable file.
    signal_length = None
    for probe_id, probe_path in files.items():
        try:
            first_signal = read_signal(probe_path, chroms, normalize)
            signal_length = len(first_signal)
            break
        except (OSError, KeyError) as err:
            if strict:
                raise RuntimeError(
                    f"Error reading {probe_id} ({probe_path}): {err}"
                ) from err
            print(
                f"Warning: skipping {probe_id} during probe: {err}",
                file=sys.stderr,
            )
            continue

    if signal_length is None:
        print("Error: no readable files found", file=sys.stderr)
        return []

    n_chunks = (n_samples + samples_per_chunk - 1) // samples_per_chunk
    bytes_per_sample = signal_length * np.dtype(np.float32).itemsize
    total_bytes = n_samples * bytes_per_sample

    print("Conversion plan:")
    print(f"  Samples:          {n_samples}")
    print(f"  Signal length:    {signal_length}")
    print(f"  Bytes per sample: {bytes_per_sample:,}")
    print(f"  Samples/chunk:    {samples_per_chunk}")
    print(f"  Chunk files:      {n_chunks}")
    print(f"  Total size:       {total_bytes / 1e9:.2f} GB")
    print(f"  Normalize:        {normalize}")
    print(f"  Output dir:       {output_dir}")

    if dry_run:
        print("\nDry run — no files written.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(output_dir, total_bytes)

    ordered_ids = list(files.keys())
    chunk_paths: List[Path] = []
    total_written = 0
    total_skipped = 0
    t_start = time.monotonic()

    for chunk_idx in range(n_chunks):
        start = chunk_idx * samples_per_chunk
        end = min(start + samples_per_chunk, n_samples)
        chunk_ids = ordered_ids[start:end]
        chunk_size = len(chunk_ids)

        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.h5"

        if chunk_path.exists():
            print(f"  Skipping {chunk_path} (already exists)")
            chunk_paths.append(chunk_path)
            total_written += chunk_size
            continue

        print(
            f"  Writing chunk {chunk_idx + 1}/{n_chunks}: "
            f"{chunk_size} samples → {chunk_path.name}"
        )

        signals = np.empty((chunk_size, signal_length), dtype=np.float32)
        valid_ids: List[str] = []
        valid_count = 0

        for i, sample_id in enumerate(chunk_ids):
            hdf5_path = files[sample_id]

            try:
                signal = read_signal(hdf5_path, chroms, normalize)
            except (OSError, FloatingPointError, KeyError) as err:
                msg = f"Error reading {sample_id} ({hdf5_path}): {err}"
                if strict:
                    # Clean up partial chunk file
                    if chunk_path.exists():
                        chunk_path.unlink()
                    raise RuntimeError(msg) from err
                print(f"    {msg}", file=sys.stderr)
                total_skipped += 1
                continue

            if len(signal) != signal_length:
                msg = (
                    f"Signal length mismatch for {sample_id}: "
                    f"expected {signal_length}, got {len(signal)}. "
                    f"File: {hdf5_path}"
                )
                if strict:
                    if chunk_path.exists():
                        chunk_path.unlink()
                    raise ValueError(msg)
                print(f"    {msg}", file=sys.stderr)
                total_skipped += 1
                continue

            signals[valid_count] = signal
            valid_ids.append(sample_id)
            valid_count += 1

            if (i + 1) % 1000 == 0:
                elapsed = time.monotonic() - t_start
                rate = (total_written + valid_count) / elapsed
                print(f"    [{i + 1}/{chunk_size}] " f"({rate:.0f} samples/s)")

        # Trim if some samples were skipped
        if valid_count < chunk_size:
            signals = signals[:valid_count]

        if valid_count == 0:
            print(f"    No valid samples for chunk {chunk_idx}, skipping")
            continue

        write_chunk(
            chunk_path,
            valid_ids,
            signals,
            signal_length,
            normalize,
            str(chrom_file),
        )

        chunk_paths.append(chunk_path)
        total_written += valid_count

    elapsed = time.monotonic() - t_start
    print("\nConversion complete:")
    print(f"  Written:  {total_written} samples")
    print(f"  Skipped:  {total_skipped} samples")
    print(f"  Chunks:   {len(chunk_paths)} files")
    print(f"  Time:     {elapsed:.1f}s")
    if elapsed > 0:
        print(f"  Rate:     {total_written / elapsed:.0f} samples/s")

    return chunk_paths


def verify_chunks(
    chunk_dir: Path,
    expected_samples: Optional[int] = None,
) -> bool:
    """Verify integrity of chunk files in a directory.

    Checks that all chunk files can be opened, have the required
    datasets, and have consistent signal lengths. Optionally checks
    that the total sample count matches an expected value.

    Args:
        chunk_dir: Directory containing chunk HDF5 files.
        expected_samples: If provided, verify total sample count.

    Returns:
        True if all checks pass, False otherwise.
    """
    chunk_files = sorted(list(chunk_dir.glob("*.h5")) + list(chunk_dir.glob("*.hdf5")))

    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}", file=sys.stderr)
        return False

    print(f"Verifying {len(chunk_files)} chunk file(s)...")

    total_samples = 0
    signal_length: Optional[int] = None
    all_ids: List[str] = []
    ok = True

    for chunk_file in chunk_files:
        try:
            with h5py.File(chunk_file, "r") as f:
                if "signals" not in f or "sample_ids" not in f:
                    print(
                        f"  FAIL {chunk_file.name}: " f"missing required datasets",
                        file=sys.stderr,
                    )
                    ok = False
                    continue

                sigs: h5py.Dataset = f["signals"]  # type: ignore[assignment]
                ids: h5py.Dataset = f["sample_ids"]  # type: ignore[assignment]

                if len(sigs) != len(ids):
                    print(
                        f"  FAIL {chunk_file.name}: "
                        f"signals ({len(sigs)}) != "
                        f"sample_ids ({len(ids)})",
                        file=sys.stderr,
                    )
                    ok = False
                    continue

                file_signal_length = sigs.shape[1]  # pylint: disable=no-member
                if signal_length is None:
                    signal_length = file_signal_length
                elif file_signal_length != signal_length:
                    print(
                        f"  FAIL {chunk_file.name}: "
                        f"signal length {file_signal_length} != "
                        f"expected {signal_length}",
                        file=sys.stderr,
                    )
                    ok = False
                    continue

                n = len(sigs)
                total_samples += n

                chunk_ids = [
                    sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)
                    for sid in ids[:]
                ]
                all_ids.extend(chunk_ids)

                print(f"  OK   {chunk_file.name}: {n} samples")

        except OSError as err:
            print(
                f"  FAIL {chunk_file.name}: {err}",
                file=sys.stderr,
            )
            ok = False

    # Check for duplicate IDs
    id_set = set(all_ids)
    if len(id_set) != len(all_ids):
        n_dupes = len(all_ids) - len(id_set)
        print(
            f"\n  WARNING: {n_dupes} duplicate sample ID(s) found",
            file=sys.stderr,
        )
        ok = False

    if expected_samples is not None and total_samples != expected_samples:
        print(
            f"\n  FAIL: expected {expected_samples} samples, " f"found {total_samples}",
            file=sys.stderr,
        )
        ok = False

    print(f"\nTotal: {total_samples} samples, signal length {signal_length}")
    if ok:
        print("All checks passed.")
    else:
        print("Some checks failed.", file=sys.stderr)

    return ok


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert single-sample HDF5 files to chunked format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --hdf5-list files.txt --chrom-file chroms.txt "
            "--output-dir chunks/\n"
            "  %(prog)s --hdf5-list files.txt --chrom-file chroms.txt "
            "--output-dir chunks/ --normalize --samples-per-chunk 5000\n"
            "  %(prog)s --verify chunks/\n"
            "  %(prog)s --verify chunks/ --expected-samples 50000\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- convert subcommand ---
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert single-sample HDF5 files to chunked format.",
    )
    convert_parser.add_argument(
        "--hdf5-list",
        type=Path,
        required=True,
        help="Text file listing HDF5 file paths (one per line).",
    )
    convert_parser.add_argument(
        "--chrom-file",
        type=Path,
        required=True,
        help="Chromosome names file (sorted, one per line).",
    )
    convert_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for chunk files.",
    )
    convert_parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=10000,
        help="Maximum samples per chunk file (default: 10000).",
    )
    convert_parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Z-score normalize each signal.",
    )
    convert_parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=None,
        help=(
            "Override directory for HDF5 files. Useful when files "
            "have been copied to a local disk (e.g. $SLURM_TMPDIR)."
        ),
    )
    convert_parser.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help="Text file listing sample IDs to include (one per line).",
    )
    convert_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print conversion plan without writing files.",
    )
    convert_parser.add_argument(
        "--no-strict",
        action="store_true",
        default=False,
        help="Skip samples that fail to read instead of aborting.",
    )

    # --- verify subcommand ---
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify integrity of chunk files.",
    )
    verify_parser.add_argument(
        "chunk_dir",
        type=Path,
        help="Directory containing chunk HDF5 files.",
    )
    verify_parser.add_argument(
        "--expected-samples",
        type=int,
        default=None,
        help="Expected total number of samples across all chunks.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the conversion script."""
    args = parse_args(argv)

    if args.command == "convert":
        sample_id_filter = None
        if args.sample_ids_file is not None:
            sample_id_filter = load_sample_id_filter(args.sample_ids_file)
            print(f"Filtering to {len(sample_id_filter)} sample IDs")

        try:
            convert(
                hdf5_list=args.hdf5_list,
                chrom_file=args.chrom_file,
                output_dir=args.output_dir,
                samples_per_chunk=args.samples_per_chunk,
                normalize=args.normalize,
                hdf5_dir=args.hdf5_dir,
                sample_id_filter=sample_id_filter,
                dry_run=args.dry_run,
                strict=not args.no_strict,
            )
        except (OSError, ValueError, RuntimeError) as err:
            print(f"\nError: {err}", file=sys.stderr)
            return 1

    elif args.command == "verify":
        ok = verify_chunks(
            chunk_dir=args.chunk_dir,
            expected_samples=args.expected_samples,
        )
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
