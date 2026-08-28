"""Integration test for ave_general_training.py using saccer3 fixtures."""
# pylint: disable=duplicate-code, protected-access
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from epiclass.core.lazy.general_fold_factory import GeneralFoldFactory
from epiclass.core.metadata import Metadata
from epiclass.mains.ave_general_training import main as main_module
from tests.epilap_test_data import SACCER3_DIR

# torch deprecates `isinstance(x, LeafSpec)`; lightning's _pytree helper still
# uses it, so every Lightning training loop warns. Upstream on both sides.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:.*isinstance\(treespec, LeafSpec\).*:FutureWarning"
)


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("ave_general_training")


def _write_ave_hparams(path: Path) -> Path:
    """Write a tiny AVE hyperparameters file for a fast CPU run."""
    hparams = {
        "batch_size": 4,
        "training_epochs": 2,
        "early_stop_limit": 1,
        "measure_frequency": 1,
        "learning_rate": 1e-3,
        "l2_scale": 0.0,
        "dropout": 0.0,
        "kl_weight": 0.1,
        "fusion_weight": 0.5,
        "latent_dim": 8,
        "contamination_rate": 0.1,
        "oversample": True,
    }
    path.write_text(json.dumps(hparams), encoding="utf-8")
    return path


@pytest.mark.filterwarnings("ignore:Resolution not found in HDF5:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:The 'val_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.filterwarnings("ignore:The number of training batches")
@pytest.mark.slow
def test_ave_general_training(
    test_dir: Path,
    saccer3_small_training_data: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    """AVE training + scoring succeeds on saccer3 (no-UUID) data with assay folds."""
    # monkeypatch, not os.environ: a leaked MAX_SPLIT would silently cut a later
    # test's fold loop short.
    monkeypatch.setenv("MAX_SPLIT", "0")  # only run the first fold to keep the test fast

    hdf5_list, metadata = saccer3_small_training_data
    hparams_file = _write_ave_hparams(test_dir / "ave_hparams.json")

    # fmt: off
    sys.argv = [
        "ave_general_training.py",
        "assay",
        str(hparams_file),
        str(hdf5_list),
        str(SACCER3_DIR / "saccer3.can.chrom.sizes"),
        str(metadata),
        str(test_dir),
        "--n_fold", "2",
        "--min_class_size", "10",
        "--offline",
    ]
    # fmt: on
    main_module()

    split_dir = test_dir / "split0"
    assert split_dir.is_dir()
    assert (split_dir / "best_checkpoint.list").is_file()

    scores_csv = split_dir / "ave_validation_scores.csv"
    assert scores_csv.is_file(), "ave_validation_scores.csv was not created"

    scores = pd.read_csv(scores_csv)
    assert list(scores.columns) == [
        "sample_id",
        "reconstruction_error",
        "outlier_flag",
    ]
    assert len(scores) > 0
    assert scores["outlier_flag"].isin([0, 1]).all()


def _split_md5s_into_folds(metadata_path: Path, n_folds: int = 2) -> dict:
    """Build a fold-definitions dict keyed by md5sum from a metadata file."""
    md5s = sorted(d["md5sum"] for d in json.loads(metadata_path.read_text())["datasets"])
    folds = {f"fold{i}": {"md5sum": []} for i in range(n_folds)}
    for j, md5 in enumerate(md5s):
        folds[f"fold{j % n_folds}"]["md5sum"].append(md5)
    return folds


@pytest.mark.filterwarnings("ignore:Resolution not found in HDF5:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:The 'val_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.filterwarnings("ignore:The number of training batches")
@pytest.mark.slow
def test_ave_general_training_with_folds(
    test_dir: Path,
    saccer3_small_training_data: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    """--folds runs leave-one-fold-out: split0 scores exactly fold0's samples."""
    monkeypatch.setenv("MAX_SPLIT", "0")  # only run the first fold to keep the test fast

    hdf5_list, metadata = saccer3_small_training_data
    hparams_file = _write_ave_hparams(test_dir / "ave_hparams.json")

    folds = _split_md5s_into_folds(metadata, n_folds=2)
    folds_file = test_dir / "folds.json"
    folds_file.write_text(json.dumps(folds), encoding="utf-8")

    # fmt: off
    sys.argv = [
        "ave_general_training.py",
        "assay",
        str(hparams_file),
        str(hdf5_list),
        str(SACCER3_DIR / "saccer3.can.chrom.sizes"),
        str(metadata),
        str(test_dir),
        "--folds", str(folds_file),
        "--offline",
    ]
    # fmt: on
    main_module()

    split_dir = test_dir / "split0"
    assert split_dir.is_dir()
    assert (split_dir / "best_checkpoint.list").is_file()

    scores_csv = split_dir / "ave_validation_scores.csv"
    assert scores_csv.is_file(), "ave_validation_scores.csv was not created"

    scores = pd.read_csv(scores_csv)
    # Leave-one-fold-out: split0 validates on fold0 exactly.
    assert set(scores["sample_id"]) == set(folds["fold0"]["md5sum"])


def test_resolve_folds_md5sum_identity(tmp_path: Path):
    """md5sum id-key resolves each value to itself."""
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"md5sum": "a", "uuid": "u1", "assay": "x"},
                    {"md5sum": "b", "uuid": "u1", "assay": "y"},
                    {"md5sum": "c", "uuid": "u2", "assay": "z"},
                ]
            }
        )
    )
    meta = Metadata(meta_path)
    fold_defs = {"fold0": {"md5sum": ["a", "c"]}, "fold1": {"md5sum": ["b"]}}

    id_key, resolved = GeneralFoldFactory._resolve_folds(meta, fold_defs)

    assert id_key == "md5sum"
    assert resolved == {"fold0": ["a", "c"], "fold1": ["b"]}


def test_resolve_folds_errors_on_missing_and_ambiguous(tmp_path: Path):
    """Absent values and non-unique id-keys both raise (no silent drop)."""
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"md5sum": "a", "uuid": "u1", "assay": "x"},
                    {"md5sum": "b", "uuid": "u1", "assay": "y"},
                ]
            }
        )
    )
    meta = Metadata(meta_path)

    # Absent md5sum value.
    with pytest.raises(ValueError, match="matched no loaded signal"):
        GeneralFoldFactory._resolve_folds(
            meta, {"fold0": {"md5sum": ["a"]}, "fold1": {"md5sum": ["missing"]}}
        )

    # uuid "u1" maps to two signals -> ambiguous.
    with pytest.raises(ValueError, match="matched multiple signals"):
        GeneralFoldFactory._resolve_folds(
            meta, {"fold0": {"uuid": ["u1"]}, "fold1": {"uuid": ["u1"]}}
        )
