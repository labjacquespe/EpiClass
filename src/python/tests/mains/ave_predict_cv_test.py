"""Test module for ave_predict_CV.py (AVE ensemble scoring over all fold models)."""
# Deliberately parallel to its classifier counterpart.
# pylint: disable=duplicate-code
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epiclass.core.model_checkpoint import last_checkpoint_path
from epiclass.mains.ave_predict_CV import main as main_module
from epiclass.mains.predict_CV import write_concatenated
from tests.epilap_test_data import SACCER3_AVE_DIR

REAL_CKPT = last_checkpoint_path(SACCER3_AVE_DIR)


def _make_cv_root(tmp_path: Path, n_folds: int = 2) -> Path:
    """Build a fake CV run dir: split0..splitN each pointing at the AVE fixture ckpt."""
    cv_root = tmp_path / "ave_cv_run"
    for i in range(n_folds):
        fold_dir = cv_root / f"split{i}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "best_checkpoint.list").write_text(
            f"{REAL_CKPT} 2026-08-25 14:00:00.000000\n", encoding="utf-8"
        )
    return cv_root


def test_write_concatenated_default_name_unchanged(tmp_path: Path):
    """The name_prefix parameter must not change predict_CV.py's own filename."""
    csv = tmp_path / "split0_tag_test_prediction_data.csv"
    pd.DataFrame({"ID": ["a"], "Predicted class": ["x"]}).to_csv(csv, index=False)

    concat_path = write_concatenated([csv], tmp_path, "data")
    assert concat_path.name == "concatenated_test_prediction_data.csv"


@pytest.mark.slow
def test_ave_predict_cv_chunked(tmp_path: Path, saccer3_chunked_dir: Path):
    """End-to-end ensemble scoring: per-fold CSVs, concatenated CSV, and summary."""
    cv_root = _make_cv_root(tmp_path, n_folds=2)
    sys.argv = [
        "ave_predict_CV.py",
        str(saccer3_chunked_dir),  # directory of chunk_*.h5
        str(cv_root),
        "--chunked",
    ]
    main_module()

    out_dir = cv_root / "ave_predictionsCV"
    assert out_dir.is_dir()

    per_fold = sorted(out_dir.glob("split*_ave_scores_*.csv"))
    assert len(per_fold) == 2, f"Expected one CSV per fold, got {per_fold}"
    # Filenames carry each fold model's training provenance: run id + checkpoint stem.
    assert REAL_CKPT is not None
    assert all(REAL_CKPT.parent.parent.name in p.name for p in per_fold)
    assert all(REAL_CKPT.stem in p.name for p in per_fold)

    fold_df = pd.read_csv(per_fold[0], index_col="ID")
    assert list(fold_df.columns) == ["reconstruction_error"]
    n_samples = len(fold_df)
    assert n_samples > 0

    # hdf5 stem depends on the fixture dir name; resolve by glob to stay robust.
    concat_matches = list(out_dir.glob("concatenated_ave_scores_*.csv"))
    assert len(concat_matches) == 1
    concat = pd.read_csv(concat_matches[0])
    assert "origin" in concat.columns
    assert concat["origin"].nunique() == 2  # one source file per fold
    assert len(concat) == 2 * n_samples

    summary_matches = list(out_dir.glob("ave_scores_summary_*.csv"))
    assert len(summary_matches) == 1
    summary = pd.read_csv(summary_matches[0], index_col="ID")
    assert list(summary.columns) == ["split0", "split1", "mean", "std"]
    assert len(summary) == n_samples
    # Both folds hold the same fixture model here, so per-fold errors agree exactly.
    np.testing.assert_allclose(
        summary["mean"].to_numpy(), summary["split0"].to_numpy(), rtol=1e-6
    )
    assert np.all(np.isfinite(summary["std"].to_numpy()))
