"""Test module for predict_CV.py (ensemble prediction over all CV fold models)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

from epiclass.mains.predict_CV import discover_fold_dirs, main as main_module
from tests.epilap_test_data import FIXTURES_DIR

SACCER3_FIXTURES_DIR = FIXTURES_DIR / "saccer3"
REAL_CKPT = (
    SACCER3_FIXTURES_DIR
    / "EpiLaP"
    / "35d1e5aed6bc4b589ccb23325d75201f"
    / "checkpoints"
    / "epoch=1-step=57.ckpt"
)


def _make_cv_root(tmp_path: Path, n_folds: int = 2) -> Path:
    """Build a fake CV run dir: split0..splitN each pointing at the fixture checkpoint."""
    cv_root = tmp_path / "cv_run"
    for i in range(n_folds):
        fold_dir = cv_root / f"split{i}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "best_checkpoint.list").write_text(
            f"{REAL_CKPT} 2023-05-08 16:39:54.954455\n", encoding="utf-8"
        )
    return cv_root


def test_discover_fold_dirs_skips_non_fold_dirs(tmp_path: Path):
    """Only split*/fold_* dirs holding best_checkpoint.list are returned, in order."""
    cv_root = _make_cv_root(tmp_path, n_folds=3)
    (cv_root / "predictionsCV").mkdir()  # output dir must be ignored
    (cv_root / "mmap_cache").mkdir()  # cache dir must be ignored
    fold_dirs = discover_fold_dirs(cv_root)
    assert [d.name for d in fold_dirs] == ["split0", "split1", "split2"]


@pytest.mark.slow
def test_predict_cv_chunked(tmp_path: Path, saccer3_chunked_dir: Path):
    """End-to-end ensemble prediction: per-fold CSVs + concatenated CSV with origin."""
    cv_root = _make_cv_root(tmp_path, n_folds=2)
    sys.argv = [
        "predict_CV.py",
        str(saccer3_chunked_dir),  # directory of chunk_*.h5
        str(cv_root),
        "--chunked",
    ]
    main_module()

    out_dir = cv_root / "predictionsCV"
    assert out_dir.is_dir()

    per_fold = sorted(out_dir.glob("split*_test_prediction_*.csv"))
    assert len(per_fold) == 2, f"Expected one CSV per fold, got {per_fold}"
    # Filenames carry each fold model's training provenance: the original training comet
    # experiment id (from the checkpoint path) and the checkpoint stem.
    assert all("35d1e5aed6bc4b589ccb23325d75201f" in p.name for p in per_fold)
    assert all("epoch=1-step=57" in p.name for p in per_fold)

    fold_df = pd.read_csv(per_fold[0], index_col="ID")
    assert "Predicted class" in fold_df.columns
    assert "True class" not in fold_df.columns
    n_samples = len(fold_df)
    assert n_samples > 0

    # hdf5 stem depends on the fixture dir name; resolve by glob to stay robust.
    concat_matches = list(out_dir.glob("concatenated_test_prediction_*.csv"))
    assert len(concat_matches) == 1
    concat = pd.read_csv(concat_matches[0])
    assert "origin" in concat.columns
    assert concat["origin"].nunique() == 2  # one source file per fold
    assert len(concat) == 2 * n_samples
