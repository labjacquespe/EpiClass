"""PyTorch Dataset implementations for lazy-loaded HDF5 data."""
# pylint: disable=too-many-positional-arguments
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .lazy_data_classes import LazyKnownData, LazyUnknownData


class LazyHdf5Dataset(Dataset):
    """PyTorch Dataset that loads HDF5 files on-demand.

    This dataset works with LazyKnownData or LazyUnknownData objects
    and loads signals only when requested by the DataLoader.
    """

    def __init__(
        self,
        data: LazyKnownData | LazyUnknownData,
        transform: Optional[callable] = None,
        return_id: bool = False,
    ):
        """Initialize lazy dataset.

        Args:
            data: LazyKnownData or LazyUnknownData object
            transform: Optional transform to apply to signals
            return_id: If True, return (signal, label, sample_id) instead of (signal, label)
        """
        self.data = data
        self.transform = transform
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, str]:
        """Load and return a single sample.

        This is where the actual loading happens - signals are loaded
        from disk only when this method is called.
        """
        signal, label, _ = self.data[idx]

        if self.transform is not None:
            signal = self.transform(signal)

        signal_tensor = torch.from_numpy(signal).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_id:
            sample_id = self.data.get_id(idx)
            return signal_tensor, label_tensor, sample_id

        return signal_tensor, label_tensor


def create_lazy_dataloaders(
    train_data: Optional[LazyKnownData | LazyUnknownData] = None,
    val_data: Optional[LazyKnownData | LazyUnknownData] = None,
    test_data: Optional[LazyKnownData | LazyUnknownData] = None,
    batch_size: int = 32,
    # 0 = load in the main process. Benchmarked as the best setting for the mmap
    # data path; see utils/torch_data._create_lazy for the reasoning.
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> Dict[str, Tuple[Dataset, DataLoader]]:
    """Create DataLoader objects for lazy-loaded datasets.

    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs.
            Set True for chunked HDF5 format — each worker keeps its file
            handles open across batches, avoiding repeated open/close overhead.

    Returns:
        Dictionary mapping split names to (dataset, dataloader) tuples
    """
    dataloaders = {}

    if train_data is not None and len(train_data) > 0:
        train_dataset = LazyHdf5Dataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=True,
        )
        dataloaders["training"] = (train_dataset, train_loader)

    if val_data is not None and len(val_data) > 0:
        val_dataset = LazyHdf5Dataset(val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        dataloaders["validation"] = (val_dataset, val_loader)

    if test_data is not None and len(test_data) > 0:
        test_dataset = LazyHdf5Dataset(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        dataloaders["test"] = (test_dataset, test_loader)

    return dataloaders
