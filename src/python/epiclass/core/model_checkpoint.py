"""Shared checkpoint-restoration logic for Lightning models.

``MyTrainer.save_model_path`` (core/trainer.py) appends the best checkpoint path
to ``<model_dir>/best_checkpoint.list`` with the format
``"<ckpt_path> <iso_timestamp>"``. This module reads that file back and loads
the model, so every LightningModule that wants restore-from-last-run behaviour
shares one implementation instead of duplicating it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Type, TypeVar

import lightning as pl

T = TypeVar("T", bound=pl.LightningModule)


def restore_from_checkpoint_list(
    model_cls: Type[T], model_dir, verbose: bool = True
) -> T:
    """Load the checkpoint of the best model from the last run.

    Reads ``<model_dir>/best_checkpoint.list``, takes the last line, splits off
    the leading checkpoint path (``"<ckpt_path> <iso_timestamp>"``), validates
    it, and loads it via ``model_cls.load_from_checkpoint``.

    Raises:
        FileNotFoundError: if the list is missing, empty, has a blank last
            entry, or points at a checkpoint that is not on disk.
    """
    path = Path(model_dir) / "best_checkpoint.list"

    if verbose:
        print("Reading checkpoint list and taking last line.")
    with open(path, "r", encoding="utf-8") as ckpt_file:
        lines = ckpt_file.read().splitlines()
        if not lines:
            raise FileNotFoundError(f"Empty checkpoint list: {path}")
        ckpt_path = lines[-1].split(" ", maxsplit=1)[0]

    if not ckpt_path:
        raise FileNotFoundError(
            f"Last entry of {path} has no checkpoint path. "
            "Training likely did not save a checkpoint."
        )
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"Checkpoint path from {path} does not exist: {ckpt_path}"
        )

    if verbose:
        print(f"Loading model from {ckpt_path}")
    return model_cls.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=ckpt_path
    )
