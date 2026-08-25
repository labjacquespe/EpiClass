"""Shared helpers for the trained-model fixture generators.

Both generators do the same three things around an otherwise ordinary short
training run:

1. materialize the saccer3 HDF5s (they ship as a tar.xz) and write a list file;
2. copy the resulting checkpoint into ``fixtures/models/<name>/`` under a
   ``<logger>/<run-id>/checkpoints/`` path;
3. write ``best_checkpoint_template.list`` with ``THIS_FOLDER`` placeholders,
   which ``pytest_sessionstart`` expands into an absolute ``best_checkpoint.list``.

Step 2's directory shape is not cosmetic: ``experiment_id_from_checkpoint``
(core/prediction_files.py) recovers a run id from the checkpoint's grandparent
directory when it looks like a 32-hex comet-ml key, and the prediction CSV
filenames carry it. Keeping that shape is what lets the predict tests assert on
provenance without a live comet-ml run.
"""
from __future__ import annotations

import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List

import torch

from tests.epilap_test_data import FIXTURES_DIR, MODELS_DIR, SACCER3_DIR

SACCER3_CHROMS = SACCER3_DIR / "saccer3.can.chrom.sizes"
SACCER3_METADATA = SACCER3_DIR / "saccer3_2016-07_metadata.json"
SACCER3_HDF5_ARCHIVE = SACCER3_DIR / "hdf5" / "saccer3_2016-07.tar.xz"


def extract_saccer3_hdf5s() -> Path:
    """Extract the saccer3 HDF5 dump (idempotent) and return its directory."""
    extracted_dir = SACCER3_HDF5_ARCHIVE.parent / "saccer3_2016-07"
    if not extracted_dir.is_dir() or not any(extracted_dir.glob("*.hdf5")):
        print(f"Extracting {SACCER3_HDF5_ARCHIVE} ...")
        with tarfile.open(SACCER3_HDF5_ARCHIVE, "r:xz") as tar:
            tar.extractall(path=SACCER3_HDF5_ARCHIVE.parent)
    return extracted_dir


def write_hdf5_list(out_path: Path, limit: int | None = None) -> Path:
    """Write a list file of saccer3 HDF5 absolute paths; return the list path."""
    hdf5_files: List[Path] = sorted(extract_saccer3_hdf5s().glob("*.hdf5"))
    if limit is not None:
        hdf5_files = hdf5_files[:limit]
    if not hdf5_files:
        raise FileNotFoundError("No saccer3 HDF5 files found after extraction.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in hdf5_files) + "\n", encoding="utf-8")
    print(f"Wrote {len(hdf5_files)} HDF5 paths to {out_path}")
    return out_path


# Training-only state, worth roughly twice the weights themselves for an AdamW model.
# The fixtures are only ever restored for inference, and the committed archive is the
# one place where every extra megabyte is paid for permanently.
_TRAINING_ONLY_KEYS = ("optimizer_states", "lr_schedulers")


def _copy_inference_checkpoint(src: Path, dest: Path) -> None:
    """Copy a Lightning checkpoint, dropping the training-only state."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    dropped = [key for key in _TRAINING_ONLY_KEYS if key in checkpoint]
    for key in dropped:
        del checkpoint[key]
    torch.save(checkpoint, dest)
    if dropped:
        print(f"Dropped training-only checkpoint keys: {', '.join(dropped)}")


def model_fixture_dir(name: str) -> Path:
    """Return (and create) ``fixtures/models/<name>``."""
    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def install_checkpoint(
    checkpoint: Path, model_dir: Path, logger_name: str, run_id: str
) -> Path:
    """Copy ``checkpoint`` into the fixture and write its template list.

    The checkpoint lands at ``<model_dir>/<logger_name>/<run_id>/checkpoints/<name>``
    and ``best_checkpoint_template.list`` gets a single ``THIS_FOLDER``-relative entry
    pointing at it. Any previous content of ``<model_dir>/<logger_name>`` is replaced,
    and a stale generated ``best_checkpoint.list`` is removed so the next pytest run
    regenerates it from the template.
    """
    run_dir = model_dir / logger_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    dest_dir = run_dir / run_id / "checkpoints"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / checkpoint.name
    _copy_inference_checkpoint(checkpoint, dest)

    # Same "<ckpt_path> <iso_timestamp>" line format MyTrainer.save_model_path writes.
    relative = dest.relative_to(model_dir)
    template = model_dir / "best_checkpoint_template.list"
    template.write_text(f"THIS_FOLDER/{relative} {datetime.now()}\n", encoding="utf-8")

    generated = model_dir / "best_checkpoint.list"
    if generated.exists():
        generated.unlink()

    print(f"Installed checkpoint: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote template: {template}")
    return dest


def repack_reminder() -> None:
    """Print the repack step — fixtures/ is gitignored, the tarball is committed."""
    print(
        "\nDone. The fixtures/ tree is gitignored, so repack the committed archive:\n"
        f"  cd {FIXTURES_DIR.parent} && bash pack_fixtures.sh"
    )
