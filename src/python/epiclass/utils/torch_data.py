"""Utility functions to create PyTorch datasets and dataloaders."""
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from epiclass.core.data.data import DataSet


def create_torch_datasets(
    data: DataSet, batch_size: int
) -> Dict[str, Tuple[TensorDataset, DataLoader]]:
    """Return (dataset, DataLoader) pairs for non empty sets.

    Warning: last batch dropped for training set to ensure consistent batch sizes.
    """
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
