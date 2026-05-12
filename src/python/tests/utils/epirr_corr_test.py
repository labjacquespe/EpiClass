"""Sanity-check integration test for epirr_corr.py after lazy migration.

The script filters metadata by assay_epiclass + track_type and groups samples
by epirr_id. Saccer3 metadata has none of those fields, so the fixture below
synthesizes them from a subset of saccer3 entries.
"""
import json
import sys
from pathlib import Path

import pytest

from epiclass.utils.metadata_utils import EPIATLAS_ASSAYS
from epiclass.utils.metrics.epirr_corr import main as main_module
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("epirr_corr")


@pytest.fixture(name="adapted_metadata")
def fixture_adapted_metadata(test_dir: Path) -> Path:
    """Tag saccer3 metadata with assay_epiclass / track_type / epirr_id.

    We pick a handful of entries and assign 2 EpiRR ids, 2 assays from the
    accepted list, and a track_type that survives the script's filter.
    """
    src = FIXTURES_DIR / "saccer3" / "saccer3_2016-07_metadata.json"
    meta = json.loads(src.read_text())

    # Two EpiRRs × two assays each = enough for a small correlation
    assays = [EPIATLAS_ASSAYS[0], EPIATLAS_ASSAYS[1]]  # e.g. h3k27ac, h3k27me3
    entries = meta["datasets"][:20]
    for i, entry in enumerate(entries):
        entry["assay_epiclass"] = assays[i % 2]
        entry["track_type"] = "pval"  # in accepted_tracks
        entry["epirr_id"] = f"EPIRR{(i // 2) % 5:02d}"

    meta["datasets"] = entries
    out = test_dir / "saccer3_meta_epirr.json"
    out.write_text(json.dumps(meta))
    return out


@pytest.fixture(name="adapted_hdf5_list")
def fixture_adapted_hdf5_list(
    test_dir: Path, saccer3_hdf5_file_list: Path, adapted_metadata: Path
) -> Path:
    """Restrict the HDF5 list to the md5s actually kept in adapted_metadata."""
    meta = json.loads(adapted_metadata.read_text())
    wanted_md5s = {entry["md5sum"] for entry in meta["datasets"]}
    lines = [
        line
        for line in saccer3_hdf5_file_list.read_text().splitlines()
        if any(md5 in line for md5 in wanted_md5s)
    ]
    out = test_dir / "epirr_hdf5_list.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


@pytest.mark.slow
def test_epirr_corr_runs(test_dir: Path, adapted_hdf5_list: Path, adapted_metadata: Path):
    """End-to-end: filter metadata, load signals lazily, write correlation CSV."""
    chroms = FIXTURES_DIR / "saccer3" / "saccer3.can.chrom.sizes"
    sys.argv = [
        "epirr_corr.py",
        str(adapted_hdf5_list),
        str(chroms),
        str(adapted_metadata),
        str(test_dir),
    ]
    main_module()

    assert (test_dir / "epirr_correlated_signals.csv").is_file()
    assert (test_dir / "epirr_correlated_signals.md5").is_file()
