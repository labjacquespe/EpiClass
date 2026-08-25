"""Generate the saccer3 AVE anomaly-detection fixture (fixtures/models/saccer3_ave/).

    cd src/python && python -m tests.fixtures_gen.make_saccer3_ave_fixture

Counterpart to make_saccer3_mlp_fixture.py: runs a short CPU cross-validation
with ``ave_general_training`` (offline comet logger) on the saccer3 fixture data
and installs fold 0's checkpoint.

The latent dimension is deliberately small — the checkpoint is committed inside
fixtures.tar.zstd, and the AVE's first projection dominates its size.

Consumed by: tests/mains/ave_predict_test.py, tests/mains/ave_predict_cv_test.py.
"""
# The two generators are deliberately parallel: same steps, different model.
# pylint: disable=duplicate-code
from __future__ import annotations

import json
import os
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

FIXTURE_NAME = "saccer3_ave"
# Stand-in for a comet-ml experiment key (32 lowercase hex chars): the offline logger
# generates a fresh one per run, so pin it here to keep the fixture path stable.
RUN_ID = "a7e4c0b219f34d6ea1c85b3d70f2e6c9"
CATEGORY = "assay"

HPARAMS = {
    "batch_size": 32,
    "training_epochs": 2,
    "early_stop_limit": 1,
    "measure_frequency": 1,
    "learning_rate": 1e-3,
    "l2_scale": 0.0,
    "dropout": 0.0,
    "kl_weight": 0.1,
    "fusion_weight": 0.5,
    # Tiny network on purpose: the checkpoint lives in the committed fixtures
    # archive, and the AVE's first projection dominates its size. The default
    # bounded sub-linear sizing would give (612, 306) here -- ~3M parameters.
    "ae_hidden": [64, 32],
    "vae_hidden": [64, 32],
    "latent_dim": 8,
    "contamination_rate": 0.05,
    "oversample": True,
}
N_FOLD = 2
MIN_CLASS_SIZE = 10


def main() -> None:
    """Train a short AVE CV run and install fold 0 as the saccer3_ave fixture."""
    model_dir = model_fixture_dir(FIXTURE_NAME)
    hparams_file = model_dir / "saccer3_ave_hparams.json"
    hparams_file.write_text(json.dumps(HPARAMS, indent=2) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="saccer3_ave_fixture_") as tmp:
        tmp_dir = Path(tmp)
        hdf5_list = write_hdf5_list(tmp_dir / "saccer3.list")
        run_dir = tmp_dir / "run"
        run_dir.mkdir()

        # One trained model is all a fixture needs; the fold factory just refuses
        # fewer than 2 folds, so bound the loop instead. Overridable from the shell.
        os.environ.setdefault("MAX_SPLIT", "0")

        # Imported here so the heavyweight training stack is only loaded when running.
        from epiclass.mains.ave_general_training import (  # pylint: disable=import-outside-toplevel
            main as train_main,
        )

        # fmt: off
        sys.argv = [
            "ave_general_training.py",
            CATEGORY,
            str(hparams_file),
            str(hdf5_list),
            str(SACCER3_CHROMS),
            str(SACCER3_METADATA),
            str(run_dir),
            "--offline",
            "--n_fold", str(N_FOLD),
            "--min_class_size", str(MIN_CLASS_SIZE),
            "--mmap_dir", str(tmp_dir / "mmap_cache"),
        ]
        # fmt: on
        print("Running:", " ".join(sys.argv))
        train_main()

        split_dir = run_dir / "split0"
        checkpoint = last_checkpoint_path(split_dir)
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"Training produced no checkpoint in {split_dir}")

        install_checkpoint(checkpoint, model_dir, logger_name="AVE", run_id=RUN_ID)

    repack_reminder()


if __name__ == "__main__":
    main()
