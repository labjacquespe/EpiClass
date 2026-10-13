"""Tests for make_rna_pair_list.py (stranded RNA pair list from a metadata JSON)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest

from epiclass.utils.preprocessing.make_rna_pair_list import (
    MINUS_TRACK,
    PLUS_TRACK,
    collect_rna_pairs,
    main,
)
from epiclass.utils.preprocessing.sum_stranded_rna_hdf5 import (
    derive_pair_id,
    main as sum_main,
    parse_pair_list,
)


def _track(
    md5: str, uuid: str, track_type: str, epirr="IHECRE0000001.2", assay="rna_seq"
):
    return {
        "md5sum": md5,
        "uuid": uuid,
        "track_type": track_type,
        "epirr_id": epirr,
        "assay_epiclass": assay,
    }


def _experiment(uuid: str, epirr="IHECRE0000001.2", assay="rna_seq") -> List[dict]:
    """Plus and minus tracks of one experiment; md5s derived from the uuid."""
    return [
        _track(f"{uuid}_plus", uuid, PLUS_TRACK, epirr, assay),
        _track(f"{uuid}_minus", uuid, MINUS_TRACK, epirr, assay),
    ]


def test_strands_of_one_experiment_form_one_pair():
    """Plus/minus md5s land in the right fields, whatever the track order."""
    datasets = _experiment("u1")[::-1]

    (pair,) = collect_rna_pairs(datasets)

    assert (pair.plus_md5, pair.minus_md5) == ("u1_plus", "u1_minus")
    assert pair.summed_id == "IHECRE0000001_u1_rna_unique_sum_unstranded"


def test_rna_and_mrna_in_one_epirr_get_distinct_ids():
    """The case that made EpiRR-only IDs collide."""
    datasets = _experiment("u1", assay="rna_seq") + _experiment("u2", assay="mrna_seq")

    pairs = collect_rna_pairs(datasets)

    assert {pair.assay for pair in pairs} == {"rna_seq", "mrna_seq"}
    assert len({pair.summed_id for pair in pairs}) == 2


def test_only_stranded_rna_tracks_with_epirr_are_used():
    """Other assays, other RNA tracks and tracks without EpiRR are dropped; assay case is ignored."""
    datasets = (
        _experiment("u1")
        + [_track("raw", "u1", "Unique_raw")]
        + _experiment("chip", assay="h3k4me3")
        + _experiment("no_epirr", epirr=None)
        + _experiment("upper", assay="mRNA_seq")
    )

    pairs = collect_rna_pairs(datasets)

    assert [pair.uuid for pair in pairs] == ["u1", "upper"]


@pytest.mark.parametrize(
    "datasets, expected",
    [
        pytest.param(_experiment("u1")[:1], "u1: expected one plus", id="missing_strand"),
        pytest.param(
            _experiment("u1") + [_track("extra", "u1", PLUS_TRACK)],
            "u1: expected one plus",
            id="duplicated_strand",
        ),
        pytest.param(
            [
                _track("p", "u1", PLUS_TRACK, epirr="IHECRE0000001.1"),
                _track("m", "u1", MINUS_TRACK, epirr="IHECRE0000002.1"),
            ],
            "u1: strands disagree",
            id="strands_in_two_epirrs",
        ),
        pytest.param(
            [
                _track("p", "u1", PLUS_TRACK, assay="rna_seq"),
                _track("m", "u1", MINUS_TRACK, assay="mrna_seq"),
            ],
            "u1: strands disagree",
            id="strands_in_two_assays",
        ),
    ],
)
def test_inconsistent_metadata_raises(datasets, expected):
    """A uuid that is not exactly one plus/minus pair of one EpiRR and assay is an error."""
    with pytest.raises(ValueError, match=expected):
        collect_rna_pairs(datasets)


def test_all_problems_reported_together():
    """Every bad uuid is listed in one error, not just the first."""
    datasets = _experiment("u1")[:1] + _experiment("u2")[1:] + _experiment("u3")

    with pytest.raises(ValueError, match="2 problem") as excinfo:
        collect_rna_pairs(datasets)

    assert "u1" in str(excinfo.value) and "u2" in str(excinfo.value)


def test_main_writes_list_read_by_summing_script(tmp_path: Path):
    """End to end: metadata JSON -> pair list parsed by sum_stranded_rna_hdf5.py."""
    datasets = _experiment("u1", assay="rna_seq") + _experiment("u2", assay="mrna_seq")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
    out = tmp_path / "pairs.csv"
    hdf5_dir = tmp_path / "hdf5"

    assert (
        main(
            [str(metadata), str(out), "--hdf5-dir", str(hdf5_dir), "--suffix", "1kb.hdf5"]
        )
        == 0
    )

    assert parse_pair_list(out) == [
        (
            "IHECRE0000001_u1_rna_unique_sum_unstranded",
            hdf5_dir / "u1_plus_1kb.hdf5",
            hdf5_dir / "u1_minus_1kb.hdf5",
        ),
        (
            "IHECRE0000001_u2_rna_unique_sum_unstranded",
            hdf5_dir / "u2_plus_1kb.hdf5",
            hdf5_dir / "u2_minus_1kb.hdf5",
        ),
    ]


def test_main_alt_formats(tmp_path: Path):
    """--no-ids list is read as ID-less pairs; --table and --md5-list match the pairs."""
    datasets = _experiment("u1", assay="rna_seq") + _experiment("u2", assay="mrna_seq")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
    out, table, md5_list = (tmp_path / name for name in ("p.csv", "p.tsv", "md5.list"))
    argv = [str(metadata), str(out), "--hdf5-dir", "d", "--suffix", "s.hdf5", "--no-ids"]
    argv += ["--table", str(table), "--md5-list", str(md5_list)]

    assert main(argv) == 0

    assert parse_pair_list(out) == [
        (None, Path("d/u1_plus_s.hdf5"), Path("d/u1_minus_s.hdf5")),
        (None, Path("d/u2_plus_s.hdf5"), Path("d/u2_minus_s.hdf5")),
    ]
    assert table.read_text(encoding="utf-8").splitlines() == [
        "summed_id\tepirr_id\tassay\tuuid\tplus_md5\tminus_md5",
        "IHECRE0000001_u1_rna_unique_sum_unstranded\tIHECRE0000001.2\trna_seq\tu1\tu1_plus\tu1_minus",
        "IHECRE0000001_u2_rna_unique_sum_unstranded\tIHECRE0000001.2\tmrna_seq\tu2\tu2_plus\tu2_minus",
    ]
    assert md5_list.read_text(encoding="utf-8").split() == [
        "u1_minus",
        "u1_plus",
        "u2_minus",
        "u2_plus",
    ]


@pytest.mark.parametrize("with_ids", [True, False], ids=["3col", "2col"])
def test_pair_list_feeds_summing_script(
    extracted_hdf5_dir: Path, tmp_path: Path, with_ids: bool
):
    """Both pair list formats run through sum_stranded_rna_hdf5.py on real HDF5 files.

    The saccer3 fixtures are not real strand pairs; they stand in for the plus/minus
    tracks of two experiments, named <md5sum>_10kb_all_none.hdf5 like epigeec outputs.
    """
    suffix = "10kb_all_none.hdf5"
    md5s = [path.name.split("_")[0] for path in sorted(extracted_hdf5_dir.glob("*.hdf5"))]
    datasets = [
        _track(md5s[0], "u1", PLUS_TRACK),
        _track(md5s[1], "u1", MINUS_TRACK),
        _track(md5s[2], "u2", PLUS_TRACK, assay="mrna_seq"),
        _track(md5s[3], "u2", MINUS_TRACK, assay="mrna_seq"),
    ]
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
    pair_list = tmp_path / "pairs.csv"
    argv = [str(metadata), str(pair_list), "--hdf5-dir", str(extracted_hdf5_dir)]
    argv += ["--suffix", suffix] + ([] if with_ids else ["--no-ids"])
    assert main(argv) == 0

    out = tmp_path / "out"
    assert (
        sum_main(["--pair-list", str(pair_list), "--output-dir", str(out), "--per-pair"])
        == 0
    )

    with open(out / "pair_mapping.tsv", "r", encoding="utf-8") as f:
        mapping = list(csv.DictReader(f, delimiter="\t"))
    assert len(mapping) == 2
    for row, (plus, minus) in zip(mapping, [(md5s[0], md5s[1]), (md5s[2], md5s[3])]):
        plus_path = extracted_hdf5_dir / f"{plus}_{suffix}"
        minus_path = extracted_hdf5_dir / f"{minus}_{suffix}"
        uuid = "u1" if plus == md5s[0] else "u2"
        expected_id = (
            f"IHECRE0000001_{uuid}_rna_unique_sum_unstranded"
            if with_ids
            else derive_pair_id(plus_path, minus_path)
        )
        assert row["new_id"] == expected_id

        with h5py.File(plus_path, "r") as fp, h5py.File(minus_path, "r") as fm, h5py.File(
            row["per_pair_path"], "r"
        ) as fo:
            assert list(fo.keys()) == [expected_id]
            gp, gm = fp[list(fp.keys())[0]], fm[list(fm.keys())[0]]
            for chrom, dset in fo[expected_id].items():  # type: ignore[union-attr]
                np.testing.assert_array_equal(
                    dset[...], gp[chrom][...] + gm[chrom][...]  # type: ignore[index]
                )


def test_main_writes_nothing_on_bad_metadata(tmp_path: Path):
    """No partial pair list is left behind when the metadata is inconsistent."""
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"datasets": _experiment("u1")[:1]}), encoding="utf-8")
    out = tmp_path / "pairs.csv"

    assert main([str(metadata), str(out), "--hdf5-dir", "x", "--suffix", "y"]) == 1
    assert not out.exists()
