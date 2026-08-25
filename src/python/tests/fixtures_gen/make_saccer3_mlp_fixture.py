"""Regenerate the saccer3 MLP classifier fixture (fixtures/models/saccer3_mlp/).

    cd src/python && python -m tests.fixtures_gen.make_saccer3_mlp_fixture

Runs a short CPU cross-validation with ``general_training`` on the saccer3
fixture data, then installs fold 0's checkpoint plus the training mapping and
hyperparameters into the fixture directory.

The checkpoint is filed under ``EpiLaP/<32-hex id>/checkpoints/`` even though
this run uses a CSV logger, not comet-ml: that is the path shape
``experiment_id_from_checkpoint`` parses, and the predict tests assert the id
shows up in the prediction CSV filenames. The id below is fixed so re-running
this script does not churn those paths.

Consumed by: tests/mains/predict_test.py, tests/mains/predict_cv_test.py,
tests/core/restore_model_test.py.
"""
# The two generators are deliberately parallel: same steps, different model.
# pylint: disable=duplicate-code
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from epiclass.core.model_checkpoint import last_checkpoint_path
from tests.fixtures_gen.fixture_gen_common import (
    SACCER3_CHROMS,
    SACCER3_METADATA,
    install_checkpoint,
    model_fixture_dir,
    repack_reminder,
    write_hdf5_list,
)

FIXTURE_NAME = "saccer3_mlp"
# Stand-in for a comet-ml experiment key (32 lowercase hex chars); see module docstring.
RUN_ID = "35d1e5aed6bc4b589ccb23325d75201f"
CATEGORY = "assay"

HPARAMS = {
    "batch_size": 32,
    "early_stop_limit": 1,
    "is_training": True,
    "keep_prob": 0.5,
    "l2_scale": 0.01,
    "learning_rate": 1e-04,
    "measure_frequency": 1,
    "training_epochs": 2,
}
HL_UNITS = 1000
NB_LAYER = 1
N_FOLD = 2
MIN_CLASS_SIZE = 10


def main() -> None:
    """Train a short CV run and install fold 0 as the saccer3_mlp fixture."""
    model_dir = model_fixture_dir(FIXTURE_NAME)
    hparams_file = model_dir / "saccer3_hparams.json"
    hparams_file.write_text(json.dumps(HPARAMS, indent=2) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="saccer3_mlp_fixture_") as tmp:
        tmp_dir = Path(tmp)
        hdf5_list = write_hdf5_list(tmp_dir / "saccer3.list")
        run_dir = tmp_dir / "run"
        run_dir.mkdir()

        # One trained model is all a fixture needs; the fold factory just refuses
        # fewer than 2 folds, so bound the loop instead. Overridable from the shell.
        os.environ.setdefault("MAX_SPLIT", "0")

        # Imported here so the heavyweight training stack is only loaded when running.
        from epiclass.mains.general_training import (  # pylint: disable=import-outside-toplevel
            main as train_main,
        )

        # fmt: off
        sys.argv = [
            "general_training.py",
            CATEGORY,
            str(hparams_file),
            str(hdf5_list),
            str(SACCER3_CHROMS),
            str(SACCER3_METADATA),
            str(run_dir),
            "--n_fold", str(N_FOLD),
            "--hl_units", str(HL_UNITS),
            "--nb_layer", str(NB_LAYER),
            "--min_class_size", str(MIN_CLASS_SIZE),
            "--mmap_dir", str(tmp_dir / "mmap_cache"),
        ]
        # fmt: on
        print("Running:", " ".join(sys.argv))
        train_main()

        fold_dir = run_dir / "fold_0"
        checkpoint = last_checkpoint_path(fold_dir)
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"Training produced no checkpoint in {fold_dir}")

        install_checkpoint(checkpoint, model_dir, logger_name="EpiLaP", run_id=RUN_ID)
        shutil.copy2(
            fold_dir / "training_mapping.tsv", model_dir / "training_mapping.tsv"
        )

    repack_reminder()


if __name__ == "__main__":
    main()
