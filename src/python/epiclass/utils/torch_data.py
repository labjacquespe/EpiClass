"""Utility functions to create PyTorch datasets and dataloaders."""
# pylint: disable=import-outside-toplevel
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from epiclass.core.data.dataset import DataSet
from epiclass.core.lazy.lazy_data_classes import LazyData


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var, accepting 1/0, true/false, yes/no (case-insensitive).

    Returns ``default`` when the variable is unset or empty.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def create_torch_datasets(
    data: DataSet, batch_size: int
) -> Dict[str, Tuple[Dataset, DataLoader]]:
    """Return (dataset, DataLoader) pairs for non-empty sets.

    Automatically detects lazy vs eager data. For lazy data (LazyKnownData /
    LazyUnknownData), delegates to create_lazy_dataloaders so that signals are
    loaded on-demand rather than materialised into RAM.

    Warning: last batch dropped for training set to ensure consistent batch sizes.
    """
    if isinstance(data.train, LazyData):
        return _create_lazy(data, batch_size)
    return _create_eager(data, batch_size)


# ---------------------------------------------------------------------------
# Lazy path
# ---------------------------------------------------------------------------


def _create_lazy(data: DataSet, batch_size: int) -> Dict[str, Tuple[Dataset, DataLoader]]:
    # Local import: lazy_torch_dataset pulls in torch, avoid loading at module level
    from epiclass.core.lazy.lazy_torch_dataset import create_lazy_dataloaders

    # DataLoader knobs are env-overridable so a benchmark (or an HPC job) can
    # sweep them without touching call sites. When unset, defaults reproduce the
    # historical behaviour exactly. The num_workers==0 guards inside
    # create_lazy_dataloaders force prefetch/persistent off, so passing them
    # here is always safe.
    num_workers = int(
        os.getenv("EPICLASS_NUM_WORKERS", os.getenv("SLURM_CPUS_PER_TASK", "4"))
    )
    prefetch_factor = int(os.getenv("EPICLASS_PREFETCH_FACTOR", "2"))
    pin_memory = _env_bool("EPICLASS_PIN_MEMORY", torch.cuda.is_available())
    persistent_workers = _env_bool("EPICLASS_PERSISTENT_WORKERS", num_workers > 0)

    train = data.train if data.train.num_examples else None
    val = data.validation if data.validation.num_examples else None
    test = data.test if data.test.num_examples else None

    return create_lazy_dataloaders(
        train_data=train,
        val_data=val,
        test_data=test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


# ---------------------------------------------------------------------------
# Eager path (unchanged)
# ---------------------------------------------------------------------------


def _create_eager(
    data: DataSet, batch_size: int
) -> Dict[str, Tuple[TensorDataset, DataLoader]]:
    torch_dsets = []
    for data_split in [data.train, data.validation, data.test]:
        try:
            dset = TensorDataset(
                torch.from_numpy(data_split.signals).float(),
                torch.from_numpy(data_split.encoded_labels),
            )
            torch_dsets.append(dset)
        except AttributeError:
            torch_dsets.append(None)

    datasets_pairs = {}
    train_dset = torch_dsets[0]
    if (train_dset is not None) and (len(train_dset) > 0):
        train_dataloader = DataLoader(
            train_dset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        datasets_pairs["training"] = (train_dset, train_dataloader)

    for name, torch_dset in zip(["validation", "test"], torch_dsets[1:]):
        if (torch_dset is not None) and (len(torch_dset) > 0):
            dataloader = DataLoader(
                torch_dset, batch_size=len(torch_dset), pin_memory=True
            )
            datasets_pairs[name] = (torch_dset, dataloader)

    return datasets_pairs
