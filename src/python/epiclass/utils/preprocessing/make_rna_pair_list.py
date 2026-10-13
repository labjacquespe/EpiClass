#!/usr/bin/env python3
"""Build the stranded RNA-Seq pair list for sum_stranded_rna_hdf5.py from a metadata JSON.

One RNA-Seq experiment is stored as two stranded tracks, ``Unique_plusRaw`` and
``Unique_minusRaw``, which share the experiment's ``uuid``. This script groups
the metadata's RNA tracks by uuid and writes one line per experiment in the
3-column pair list format read by sum_stranded_rna_hdf5.py:

    <summed_id>,<plus_hdf5_path>,<minus_hdf5_path>

with ``summed_id = <EpiRR without version>_<uuid>_rna_unique_sum_unstranded``.

Why the uuid is in the ID: an EpiRR (one biological sample) can hold more than
one RNA experiment, e.g. both an ``rna_seq`` and an ``mrna_seq`` one, as happens
in EpiATLAS. An EpiRR-only ID makes those experiments collide on the same output
name. The uuid is shared by the two strands of one experiment
only, so ``EpiRR + uuid`` is unique per pair while keeping the EpiRR readable.

Tracks are selected by ``assay_epiclass`` (``rna_seq`` / ``mrna_seq``, any
case) and ``track_type`` (the two stranded tracks); other tracks, such as
``Unique_raw``, are ignored, as are tracks without an ``epirr_id``. Every
problem found (a uuid missing a strand or holding one twice, a uuid spanning
several EpiRRs or assays) is reported together and nothing is written. Once
those checks pass, the summed IDs are unique by construction (one pair per uuid,
and the uuid is part of the ID).

HDF5 paths are built as ``<hdf5_dir>/<md5sum>_<suffix>``. Only their basenames
matter when sum_stranded_rna_hdf5.py is given --hdf5-dir.

Other outputs, from the same validated pairs:

    --no-ids     2-column pair list (plus path, minus path); the summing
                 script then names each pair by an md5 of its filenames.
    --table      TSV with a header, one row per pair: summed ID, EpiRR,
                 assay, uuid, plus md5sum, minus md5sum.
    --md5-list   sorted md5sums of every stranded track, one per line.

Usage example:

    python make_rna_pair_list.py \\
        /path/to/metadata.json \\
        /path/to/rna_pairs.csv \\
        --hdf5-dir /path/to/stranded_hdf5 \\
        --suffix 1kb_all_none_value.hdf5 \\
        --table /path/to/rna_pairs.tsv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from epiclass.core.metadata import Metadata

RNA_ASSAYS = ("rna_seq", "mrna_seq")
PLUS_TRACK = "Unique_plusRaw"
MINUS_TRACK = "Unique_minusRaw"
ID_SUFFIX = "rna_unique_sum_unstranded"
PAIR_TABLE_COLUMNS = ("summed_id", "epirr_id", "assay", "uuid", "plus_md5", "minus_md5")


@dataclass(frozen=True)
class RnaPair:
    """The two stranded tracks (md5sums) of one RNA-Seq experiment."""

    epirr_id: str
    assay: str
    uuid: str
    plus_md5: str
    minus_md5: str

    @property
    def summed_id(self) -> str:
        """ID of the summed (unstranded) sample: EpiRR without version + uuid."""
        epirr = self.epirr_id.split(".")[0]
        return f"{epirr}_{self.uuid}_{ID_SUFFIX}"


def collect_rna_pairs(datasets: Iterable[dict]) -> List[RnaPair]:
    """Group the stranded RNA tracks of ``datasets`` into one pair per uuid.

    Raises:
        ValueError: listing every inconsistent uuid.
    """
    tracks_per_uuid: Dict[str, List[dict]] = defaultdict(list)
    for dset in datasets:
        if str(dset.get("assay_epiclass", "")).lower() not in RNA_ASSAYS:
            continue
        if dset.get("track_type") not in (PLUS_TRACK, MINUS_TRACK):
            continue
        if not dset.get("epirr_id"):
            continue
        if not dset.get("uuid"):
            raise ValueError(f"RNA track {dset['md5sum']} has no uuid.")
        tracks_per_uuid[dset["uuid"]].append(dset)

    problems: List[str] = []
    pairs: List[RnaPair] = []
    for uuid, tracks in sorted(tracks_per_uuid.items()):
        track_types = sorted(dset["track_type"] for dset in tracks)
        if track_types != sorted((PLUS_TRACK, MINUS_TRACK)):
            problems.append(
                f"uuid {uuid}: expected one plus and one minus track, got {track_types}"
            )
            continue
        epirrs = {dset["epirr_id"] for dset in tracks}
        assays = {dset["assay_epiclass"].lower() for dset in tracks}
        if len(epirrs) > 1 or len(assays) > 1:
            problems.append(
                f"uuid {uuid}: strands disagree on EpiRR/assay: {sorted(epirrs)}, {sorted(assays)}"
            )
            continue
        md5_by_track = {dset["track_type"]: dset["md5sum"] for dset in tracks}
        pairs.append(
            RnaPair(
                epirr_id=epirrs.pop(),
                assay=assays.pop(),
                uuid=uuid,
                plus_md5=md5_by_track[PLUS_TRACK],
                minus_md5=md5_by_track[MINUS_TRACK],
            )
        )

    if problems:
        raise ValueError(
            f"{len(problems)} problem(s) in the RNA metadata:\n" + "\n".join(problems)
        )
    return sorted(pairs, key=lambda pair: pair.summed_id)


def write_pair_list(
    pairs: Iterable[RnaPair],
    out_path: Path,
    hdf5_dir: Path,
    suffix: str,
    with_ids: bool = True,
) -> None:
    """Write ``pairs`` in sum_stranded_rna_hdf5.py's pair list format.

    3 columns (summed ID, plus path, minus path) by default. ``with_ids=False``
    drops the ID column; sum_stranded_rna_hdf5.py then derives an md5 ID per pair.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            fields = [
                str(hdf5_dir / f"{pair.plus_md5}_{suffix}"),
                str(hdf5_dir / f"{pair.minus_md5}_{suffix}"),
            ]
            if with_ids:
                fields.insert(0, pair.summed_id)
            f.write(",".join(fields) + "\n")


def write_pair_table(pairs: Iterable[RnaPair], out_path: Path) -> None:
    """Write one TSV row per pair with its metadata, for joining results back to EpiRRs."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(PAIR_TABLE_COLUMNS) + "\n")
        for pair in pairs:
            row = [getattr(pair, column) for column in PAIR_TABLE_COLUMNS]
            f.write("\t".join(row) + "\n")


def write_md5_list(pairs: Iterable[RnaPair], out_path: Path) -> None:
    """Write the sorted md5sums of every stranded track, one per line."""
    md5s = sorted(md5 for pair in pairs for md5 in (pair.plus_md5, pair.minus_md5))
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(md5 + "\n" for md5 in md5s)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the stranded RNA-Seq pair list for sum_stranded_rna_hdf5.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("metadata", type=Path, help="Metadata JSON (epigeec format).")
    parser.add_argument("output", type=Path, help="Pair list to write (CSV).")
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        required=True,
        help="Directory of the stranded HDF5 files, used as the path prefix.",
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help="Filename suffix after '<md5sum>_', ex: 1kb_all_none_value.hdf5",
    )
    parser.add_argument(
        "--no-ids",
        action="store_true",
        help="Write the pair list without the summed ID column (2 columns).",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help=f"Also write a TSV of the pairs' metadata ({', '.join(PAIR_TABLE_COLUMNS)}).",
    )
    parser.add_argument(
        "--md5-list",
        type=Path,
        default=None,
        help="Also write the md5sums of every stranded track, one per line.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    try:
        pairs = collect_rna_pairs(Metadata(args.metadata).datasets)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    write_pair_list(
        pairs, args.output, args.hdf5_dir, args.suffix, with_ids=not args.no_ids
    )
    if args.table is not None:
        write_pair_table(pairs, args.table)
    if args.md5_list is not None:
        write_md5_list(pairs, args.md5_list)

    n_epirr = len({pair.epirr_id.split(".")[0] for pair in pairs})
    print(f"Wrote {len(pairs)} pairs ({n_epirr} distinct EpiRRs) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
