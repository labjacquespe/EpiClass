#!/usr/bin/env python3
"""Sum stranded RNA-Seq HDF5 pairs into unstranded single-sample / chunked HDF5.

Each biological RNA-Seq sample (one EpiRR) is stored as two stranded
single-sample HDF5 files -- a ``plusRaw`` track and a ``minusRaw`` track --
each with per-chromosome datasets under a single top-level group. This script
reads a list of such pairs and produces the *unstranded* representation by
adding the two strands element-wise (per chromosome), so the summed signal
keeps the exact per-chromosome bin length and is directly predictable by the
same models the stranded tracks were trained on.

Summation is commutative, so the order of the two files in a pair does not
matter and no plus/minus detection is needed. Each summed sample is given a
deterministic, order-independent ID -- the md5sum of its two (sorted) source
filenames -- and the script writes a **mapping TSV** recording, per pair, the
two source files, their source IDs, the new ID, and (in per-pair mode) the
output path. That mapping is the bridge back to biology: downstream, a summed
prediction's ID resolves to its source IDs, which resolve to an EpiRR via the
metadata JSON -- so nothing here depends on any filename convention.

Two independent, complementary output modes (at least one is required):

    --per-pair   one summed single-sample HDF5 per pair (per-chromosome group
                 layout preserved), named ``<new_id>.hdf5``. Feed to predict.py
                 single-sample mode (with its own --chromsize). Always written
                 RAW -- normalization happens at predict time, matching how the
                 stranded inputs live on disk.

    --chunked    one chunked multi-sample HDF5 (chunk_NNNN.h5 with `signals` +
                 `sample_ids`), same format as hdf5_chunks_creation.py, sample
                 IDs = the new IDs. Feed to predict.py --chunked. This is a
                 "precomputed" matrix baked once at write time, so --normalize
                 applies HERE only.

Input pair list (--pair-list): one pair per line, whitespace/comma separated,
with an optional leading explicit-ID column (overrides the md5 ID):

    <path_A>      <path_B>                # 2 cols: ID = md5(sorted filenames)
    <new_id>      <path_A>   <path_B>     # 3 cols: explicit ID

Usage examples:

    # Both outputs + mapping TSV, normalized chunked matrix:
    python sum_stranded_rna_hdf5.py \\
        --pair-list /path/to/pairs.txt \\
        --output-dir /path/to/unstranded/ \\
        --per-pair \\
        --chunked --chrom-file /path/to/chroms.txt --normalize

    # Per-pair single-sample files only (raw):
    python sum_stranded_rna_hdf5.py \\
        --pair-list /path/to/pairs.txt \\
        --output-dir /path/to/unstranded/ \\
        --per-pair
"""
# pylint: disable=too-many-branches, too-many-positional-arguments, too-many-arguments
# pylint: disable=too-many-locals, duplicate-code
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# Chromosome (name, size) reading goes through the codebase's canonical reader so the
# [assembly].[name].chrom.sizes format and ordering match the rest of the pipeline.
from epiclass.core.data_source import EpiDataSource

# Reuse the sibling converter's helpers so the chunked output format stays identical.
from epiclass.utils.preprocessing.hdf5_chunks_creation import (
    _extract_sample_id,
    check_disk_space,
    write_chunk,
)

MAPPING_COLUMNS = ("new_id", "file_a", "file_b", "id_a", "id_b", "per_pair_path")


def load_chrom_names(chrom_file: Path) -> List[str]:
    """Chromosome names in the codebase's canonical concat order.

    Delegates to ``EpiDataSource.load_external_chrom_file`` (reads a
    ``[assembly].[name].chrom.sizes`` file of tab-separated ``name<TAB>size``
    pairs and returns them *sorted* -- the same order the lazy loader
    concatenates chromosomes in, so the chunked matrix matches the model's
    feature order). We take the first column (chromosome name).
    """
    return [name for name, _size in EpiDataSource.load_external_chrom_file(chrom_file)]


def parse_pair_list(pair_list: Path) -> List[Tuple[Optional[str], Path, Path]]:
    """Parse the pair list into ``(explicit_id, path_a, path_b)`` tuples.

    Each non-empty, non-comment line holds two HDF5 paths (whitespace/comma
    separated), optionally preceded by an explicit ID column. ``explicit_id`` is
    ``None`` when only two paths are given (the ID is then derived as the md5 of
    the two sorted filenames). Order of the two paths is irrelevant -- summation
    is commutative and the derived ID sorts the names first.
    """
    pairs: List[Tuple[Optional[str], Path, Path]] = []
    with open(pair_list, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = [tok for tok in re.split(r"[,\s]+", line) if tok]
            if len(fields) == 2:
                pairs.append((None, Path(fields[0]), Path(fields[1])))
            elif len(fields) == 3:
                pairs.append((fields[0], Path(fields[1]), Path(fields[2])))
            else:
                raise ValueError(
                    f"pair list line {lineno}: expected 2 or 3 fields, "
                    f"got {len(fields)}: {line!r}"
                )
    return pairs


def derive_pair_id(path_a: Path, path_b: Path) -> str:
    """Deterministic, order-independent ID: md5 of the two sorted basenames."""
    key = "\n".join(sorted((path_a.name, path_b.name)))
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def read_group(hdf5_path: Path) -> Tuple[str, Dict[str, np.ndarray]]:
    """Return ``(group_name, {chrom: 1-D float32 array})`` for a single-sample file."""
    with h5py.File(hdf5_path, "r") as f:
        try:
            header = list(f.keys())[0]
        except IndexError as e:
            raise OSError(f"No groups found in {hdf5_path}") from e
        group = f[header]
        datasets = {
            name: np.asarray(group[name][...], dtype=np.float32)  # type: ignore[index]
            for name in group.keys()  # type: ignore[union-attr]
        }
    return header, datasets


def sum_pair(path_a: Path, path_b: Path) -> Tuple[str, Dict[str, np.ndarray]]:
    """Element-wise sum the two files, returning ``(group_name, {chrom: sum})``.

    Validates that both files expose the same chromosome dataset names and
    per-chromosome shapes before adding. The group name of the first file is
    reused for the output. Summation is commutative, so file order is irrelevant.
    """
    header_a, data_a = read_group(path_a)
    _, data_b = read_group(path_b)

    if data_a.keys() != data_b.keys():
        raise ValueError(
            f"chromosome datasets differ between {path_a.name} and "
            f"{path_b.name}: {sorted(data_a.keys())} vs {sorted(data_b.keys())}"
        )

    summed: Dict[str, np.ndarray] = {}
    for chrom, arr_a in data_a.items():
        arr_b = data_b[chrom]
        if arr_a.shape != arr_b.shape:
            raise ValueError(
                f"shape mismatch for {chrom} between {path_a.name} "
                f"({arr_a.shape}) and {path_b.name} ({arr_b.shape})"
            )
        summed[chrom] = arr_a + arr_b
    return header_a, summed


def write_single_sample(
    out_path: Path, group_name: str, datasets: Dict[str, np.ndarray]
) -> None:
    """Write a summed single-sample HDF5, preserving the per-chromosome layout."""
    with h5py.File(out_path, "w") as f:
        group = f.create_group(group_name)
        for chrom, arr in datasets.items():
            group.create_dataset(chrom, data=arr, dtype=np.float32)


def concat_signal(
    datasets: Dict[str, np.ndarray], chroms: List[str], normalize: bool
) -> np.ndarray:
    """Concatenate per-chromosome arrays in ``chroms`` order into one 1-D vector."""
    missing = [c for c in chroms if c not in datasets]
    if missing:
        raise ValueError(f"summed sample missing chromosomes {missing}")
    # pylint: disable-next=unexpected-keyword-arg
    signal = np.concatenate([datasets[c] for c in chroms], dtype=np.float32)
    if normalize:
        with np.errstate(all="raise"):
            signal = (signal - signal.mean()) / signal.std()
    return signal


def write_mapping(mapping_path: Path, rows: List[Dict[str, str]]) -> None:
    """Write the pair -> new-ID mapping TSV (the bridge back to source samples)."""
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MAPPING_COLUMNS), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def process(
    pair_list: Path,
    output_dir: Path,
    per_pair: bool,
    chunked: bool,
    chrom_file: Optional[Path] = None,
    samples_per_chunk: int = 10000,
    normalize: bool = False,
    hdf5_dir: Optional[Path] = None,
    mapping_file: Optional[Path] = None,
    dry_run: bool = False,
    strict: bool = True,
) -> Optional[Path]:
    """Sum stranded RNA pairs into unstranded HDF5; write outputs + mapping TSV.

    Returns the path to the mapping TSV (or ``None`` on a dry run / no pairs).
    """
    pairs = parse_pair_list(pair_list)

    if hdf5_dir is not None:
        pairs = [(eid, hdf5_dir / a.name, hdf5_dir / b.name) for eid, a, b in pairs]

    n_pairs = len(pairs)
    if n_pairs == 0:
        print("No pairs to process.", file=sys.stderr)
        return None

    chroms = (
        load_chrom_names(chrom_file) if (chunked and chrom_file is not None) else None
    )

    print("Summing plan:")
    print(f"  Pairs:            {n_pairs}")
    print(f"  Per-pair files:   {per_pair}")
    print(f"  Chunked output:   {chunked}")
    if chunked:
        print(f"  Samples/chunk:    {samples_per_chunk}")
        print(f"  Normalize chunks: {normalize}")
    print(f"  Output dir:       {output_dir}")

    if dry_run:
        print("\nDry run -- no files written.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    per_pair_dir = output_dir / "per_pair" if (per_pair and chunked) else output_dir
    if per_pair:
        per_pair_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators for the chunked path.
    chunk_ids: List[str] = []
    chunk_signals: List[np.ndarray] = []
    signal_length: Optional[int] = None

    mapping_rows: List[Dict[str, str]] = []
    total_written = 0
    total_skipped = 0
    t_start = time.monotonic()

    for idx, (explicit_id, path_a, path_b) in enumerate(pairs):
        new_id = (
            explicit_id if explicit_id is not None else derive_pair_id(path_a, path_b)
        )
        try:
            group_name, summed = sum_pair(path_a, path_b)
        except (OSError, ValueError) as err:
            msg = f"pair {idx} ({new_id}): {err}"
            if strict:
                raise RuntimeError(msg) from err
            print(f"  Skipping {msg}", file=sys.stderr)
            total_skipped += 1
            continue

        per_pair_path = ""
        if per_pair:
            out_path = per_pair_dir / f"{new_id}.hdf5"
            per_pair_path = str(out_path)
            if out_path.exists():
                print(f"  Skipping {out_path.name} (already exists)")
            else:
                write_single_sample(out_path, group_name, summed)

        if chunked:
            assert chroms is not None
            signal = concat_signal(summed, chroms, normalize)
            if signal_length is None:
                signal_length = len(signal)
            elif len(signal) != signal_length:
                msg = (
                    f"pair {idx} ({new_id}): signal length {len(signal)} != "
                    f"expected {signal_length}"
                )
                if strict:
                    raise ValueError(msg)
                print(f"  {msg}", file=sys.stderr)
                total_skipped += 1
                continue
            chunk_ids.append(new_id)
            chunk_signals.append(signal)

        mapping_rows.append(
            {
                "new_id": new_id,
                "file_a": str(path_a),
                "file_b": str(path_b),
                "id_a": _extract_sample_id(path_a),
                "id_b": _extract_sample_id(path_b),
                "per_pair_path": per_pair_path,
            }
        )
        total_written += 1
        if (idx + 1) % 500 == 0:
            elapsed = time.monotonic() - t_start
            rate = (idx + 1) / elapsed if elapsed else 0.0
            print(f"  [{idx + 1}/{n_pairs}] ({rate:.0f} pairs/s)")

    if chunked and chunk_signals:
        assert signal_length is not None
        bytes_total = len(chunk_signals) * signal_length * np.dtype(np.float32).itemsize
        check_disk_space(output_dir, bytes_total)
        _write_chunks(
            output_dir,
            chunk_ids,
            chunk_signals,
            signal_length,
            samples_per_chunk,
            normalize,
            chrom_file,
        )

    mapping_path = mapping_file or (output_dir / "pair_mapping.tsv")
    write_mapping(mapping_path, mapping_rows)

    elapsed = time.monotonic() - t_start
    print("\nSumming complete:")
    print(f"  Written:  {total_written} pairs")
    print(f"  Skipped:  {total_skipped} pairs")
    print(f"  Mapping:  {mapping_path}")
    print(f"  Time:     {elapsed:.1f}s")
    return mapping_path


def _write_chunks(
    output_dir: Path,
    ids: List[str],
    signals: List[np.ndarray],
    signal_length: int,
    samples_per_chunk: int,
    normalize: bool,
    chrom_file: Optional[Path],
) -> None:
    """Split accumulated signals into chunk_NNNN.h5 files via write_chunk."""
    n = len(ids)
    n_chunks = (n + samples_per_chunk - 1) // samples_per_chunk
    for chunk_idx in range(n_chunks):
        start = chunk_idx * samples_per_chunk
        end = min(start + samples_per_chunk, n)
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.h5"
        if chunk_path.exists():
            print(f"  Skipping {chunk_path.name} (already exists)")
            continue
        block = np.asarray(signals[start:end], dtype=np.float32)
        print(
            f"  Writing chunk {chunk_idx + 1}/{n_chunks}: "
            f"{end - start} samples -> {chunk_path.name}"
        )
        write_chunk(
            chunk_path,
            ids[start:end],
            block,
            signal_length,
            normalize,
            str(chrom_file) if chrom_file is not None else "",
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sum stranded RNA-Seq HDF5 pairs into unstranded HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pair-list",
        type=Path,
        required=True,
        help="Text file listing HDF5 pairs to sum (one pair per line).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for summed files and the mapping TSV.",
    )
    parser.add_argument(
        "--per-pair",
        action="store_true",
        default=False,
        help="Write one summed single-sample HDF5 per pair (always raw).",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        default=False,
        help="Write a chunked multi-sample HDF5 (chunk_NNNN.h5).",
    )
    parser.add_argument(
        "--chrom-file",
        type=Path,
        default=None,
        help=(
            "Chromosome sizes file in [assembly].[name].chrom.sizes format "
            "(tab-separated name<TAB>size; required for --chunked)."
        ),
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=10000,
        help="Maximum samples per chunk file (default: 10000).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Z-score normalize the concatenated signal (chunked output only).",
    )
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=None,
        help=(
            "Override directory for input HDF5 files. Useful when files have "
            "been copied to a local disk (e.g. $SLURM_TMPDIR). Basenames (and "
            "thus derived IDs) are unaffected."
        ),
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Path for the pair->new-ID mapping TSV (default: <output-dir>/pair_mapping.tsv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the plan without writing files.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        default=False,
        help="Skip pairs that fail to read/sum instead of aborting.",
    )

    args = parser.parse_args(argv)

    if not args.per_pair and not args.chunked:
        parser.error("at least one of --per-pair / --chunked is required")
    if args.chunked and args.chrom_file is None:
        parser.error("--chrom-file is required when --chunked is set")
    if args.normalize and not args.chunked:
        parser.error("--normalize only applies to --chunked output")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    try:
        process(
            pair_list=args.pair_list,
            output_dir=args.output_dir,
            per_pair=args.per_pair,
            chunked=args.chunked,
            chrom_file=args.chrom_file,
            samples_per_chunk=args.samples_per_chunk,
            normalize=args.normalize,
            hdf5_dir=args.hdf5_dir,
            mapping_file=args.mapping_file,
            dry_run=args.dry_run,
            strict=not args.no_strict,
        )
    except (OSError, ValueError, RuntimeError) as err:
        print(f"\nError: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
