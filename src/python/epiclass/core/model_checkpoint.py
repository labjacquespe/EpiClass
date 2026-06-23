"""Shared checkpoint-restoration logic for Lightning models.

``MyTrainer.save_model_path`` (core/trainer.py) appends the best checkpoint path
to ``<model_dir>/best_checkpoint.list`` with the format
``"<ckpt_path> <iso_timestamp>"``. This module reads that file back and loads
the model, so every LightningModule that wants restore-from-last-run behaviour
shares one implementation instead of duplicating it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Type, TypeVar

import lightning as pl

T = TypeVar("T", bound=pl.LightningModule)


def last_checkpoint_path(model_dir) -> Optional[Path]:
    """Return the checkpoint path from the last line of ``best_checkpoint.list``.

    Reads ``<model_dir>/best_checkpoint.list`` (written by
    ``MyTrainer.save_model_path``, one ``"<ckpt_path> <iso_timestamp>"`` line per
    saved model) and returns the leading checkpoint path of its last entry.

    Returns ``None`` when the list is missing, empty, or has a blank last entry.
    Existence of the checkpoint file itself is *not* checked here -- callers that
    need to load it should validate (see ``restore_from_checkpoint_list``).
    """
    path = Path(model_dir) / "best_checkpoint.list"
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    ckpt_path = lines[-1].split(" ", maxsplit=1)[0]
    return Path(ckpt_path) if ckpt_path else None


def resolve_checkpoint_spec(model, verbose: bool = True) -> Path:
    """Resolve a model spec to a checkpoint file path.

    ``model`` may be either:
      - a ``.ckpt`` file, returned as-is (loaded directly, bypassing any list); or
      - a directory containing ``best_checkpoint.list``, in which case the leading
        path of the list's last entry is resolved (see :func:`last_checkpoint_path`).

    Raises:
        FileNotFoundError: if ``model`` is neither a file nor a directory, if a
            directory's list is missing/empty/blank, or if a directory's list
            points at a checkpoint that is not on disk.
    """
    model_path = Path(model)
    if model_path.is_file():
        return model_path
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model path is neither a checkpoint file nor a directory: {model_path}"
        )

    list_path = model_path / "best_checkpoint.list"
    if verbose:
        print("Reading checkpoint list and taking last line.")
    if not list_path.is_file():
        raise FileNotFoundError(f"Checkpoint list not found: {list_path}")

    ckpt_path = last_checkpoint_path(model_path)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Empty checkpoint list or blank last entry: {list_path}. "
            "Training likely did not save a checkpoint."
        )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint path from {list_path} does not exist: {ckpt_path}"
        )
    return ckpt_path


def restore_from_checkpoint_file(
    model_cls: Type[T], ckpt_path, verbose: bool = True
) -> T:
    """Load ``model_cls`` directly from a checkpoint file.

    Raises:
        FileNotFoundError: if ``ckpt_path`` is not on disk.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
    if verbose:
        print(f"Loading model from {ckpt_path}")
    return model_cls.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=str(ckpt_path)
    )


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
    ckpt_path = resolve_checkpoint_spec(model_dir, verbose=verbose)
    return restore_from_checkpoint_file(model_cls, ckpt_path, verbose=verbose)
