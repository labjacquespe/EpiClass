"""Sanity-check integration test for analyze_hdf5_vals.py after lazy migration.

The script hardcodes ASSAY="assay_epiclass" and TRACK_TYPE="track_type"; the
saccer3 fixture only has "assay" and no track_type. We adapt the metadata at
test time so the script reaches the loader path.
"""
import json
import sys
from pathlib import Path

import pytest

from epiclass.utils.metrics.analyze_hdf5_vals import (
    main as main_module,
    make_plots as _original_make_plots,
)
from tests.epilap_test_data import FIXTURES_DIR

# kaleido calls the deprecated setDaemon() when spawning its export process.
# Third-party, reached only through plotly's static image export.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:setDaemon\(\) is deprecated:DeprecationWarning"
)


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("analyze_hdf5_vals")


@pytest.fixture(name="adapted_metadata")
def fixture_adapted_metadata(test_dir: Path) -> Path:
    """Tag saccer3 metadata with assay_epiclass + a track_type ACCEPTED_TRACKS sees."""
    src = FIXTURES_DIR / "saccer3" / "saccer3_2016-07_metadata.json"
    meta = json.loads(src.read_text())
    for entry in meta["datasets"]:
        entry["assay_epiclass"] = entry.get("assay", "unknown")
        entry["track_type"] = "pval"  # in ACCEPTED_TRACKS
    out = test_dir / "saccer3_meta_adapted.json"
    out.write_text(json.dumps(meta))
    return out


@pytest.mark.slow
def test_analyze_hdf5_vals_runs(
    monkeypatch,
    test_dir: Path,
    saccer3_small_hdf5_file_list: Path,
    adapted_metadata: Path,
):
    """End-to-end: filter metadata, register HDF5s, preload mmap, write plots.

    Uses the 100-sample subset and wraps ``make_plots`` to cap the metrics
    set at 2 — kaleido PNG export is by far the dominant cost (~120 files
    on a full run) and the test only asserts that PNGs and HTMLs get
    written, so the full sweep adds runtime without coverage.
    """

    def make_plots_capped(stats_df, metrics, logdir, name=""):
        capped = set(sorted(metrics)[:2])
        return _original_make_plots(stats_df, capped, logdir, name)

    monkeypatch.setattr(
        "epiclass.utils.metrics.analyze_hdf5_vals.make_plots", make_plots_capped
    )

    chroms = FIXTURES_DIR / "saccer3" / "saccer3.can.chrom.sizes"
    sys.argv = [
        "analyze_hdf5_vals.py",
        str(saccer3_small_hdf5_file_list),
        str(chroms),
        str(adapted_metadata),
        str(test_dir),
    ]
    main_module()

    # make_plots writes per-metric PNG + HTML files
    assert list(test_dir.glob("*.png")), "Expected at least one violin PNG."
    assert list(test_dir.glob("*.html")), "Expected at least one violin HTML."
