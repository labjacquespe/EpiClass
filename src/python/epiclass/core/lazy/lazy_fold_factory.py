"""Lazy-loading cross-validation fold factory for EpiAtlas datasets.

This modernized version works with lazy-loaded data, creating splits
without loading signals into memory.
"""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Generator, List, Tuple

import numpy as np
import numpy.typing as npt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedGroupKFold

from epiclass.core.data import dataset
from epiclass.core.data_source import EpiDataSource
from epiclass.core.epiatlas_constants import EPIRR_LABEL
from epiclass.core.metadata import UUIDMetadata

from .lazy_data_classes import LazyKnownData
from .lazy_hdf5_loader import LazyHdf5Loader

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[int]
NDArrayBool = npt.NDArray[bool]


class LazyEpiAtlasDataset:
    """Lazy-loading EpiAtlas dataset handler.

    This version registers HDF5 files without loading them, enabling
    work with datasets larger than available RAM.
    """

    def __init__(
        self,
        datasource: EpiDataSource,
        label_category: str,
        label_list: List[str] | None = None,
        min_class_size: int = 10,
        md5_list: List[str] | None = None,
        force_filter: bool = True,
        metadata: UUIDMetadata | None = None,
        mmap_dir: Path | str | None = None,
    ):
        """Initialize lazy EpiAtlas dataset.

        Args:
            datasource: EpiDataSource object
            label_category: Target label category
            label_list: Optional list of labels to include
            min_class_size: Minimum samples per class
            md5_list: Optional list of MD5s to include
            force_filter: Filter metadata even if md5_list provided
            metadata: Optional pre-loaded metadata
            mmap_dir: Directory for the memory-mapped .npy file. Defaults to
                ./mmap_cache. On HPC, set to $SLURM_TMPDIR for fast local storage.
        """
        self._datasource = datasource
        self._label_category = label_category
        self._label_list = label_list

        # Load metadata
        meta = metadata
        if meta is None:
            meta = UUIDMetadata(self._datasource.metadata_file)
        if md5_list:
            try:
                meta = UUIDMetadata.from_dict({md5: meta[md5] for md5 in md5_list})
            except KeyError as e:
                raise KeyError(f"md5 {e} from md5 list not found in metadata") from e

        if force_filter or not md5_list:
            meta = self._filter_metadata(min_class_size, meta, verbose=True)

        self._metadata = meta

        # Classes info
        self._classes = self._metadata.unique_classes(self._label_category)
        self._classes_mapping = {label: i for i, label in enumerate(self._classes)}

        # UUID info
        self._metadata.display_uuid_per_class(self._label_category)
        self._uuid_mapping = self._metadata.uuid_to_md5()

        # Create lazy loader (doesn't load signals!)
        self._loader = self._create_loader(mmap_dir)

        # Create dataset object (no signal loading)
        md5s = list(self._loader.file_paths.keys())
        labels = [self._metadata[md5][self._label_category] for md5 in md5s]

        self._dataset: LazyKnownData = LazyKnownData(
            ids=md5s,
            loader=self._loader,
            y_str=labels,
            y=[self._classes_mapping[label] for label in labels],
            metadata=self._metadata,
        )

    @property
    def datasource(self) -> EpiDataSource:
        """Return given datasource."""
        return self._datasource

    @property
    def target_category(self) -> str:
        """Return given label category."""
        return self._label_category

    @property
    def label_list(self) -> List[str] | None:
        """Return given target labels inclusion list."""
        return self._label_list

    @property
    def classes(self) -> List[str]:
        """Return target classes."""
        return self._classes

    @property
    def metadata(self) -> UUIDMetadata:
        """Return a copy of current metadata."""
        return copy.deepcopy(self._metadata)

    @property
    def loader(self) -> LazyHdf5Loader:
        """Return the lazy loader."""
        return self._loader

    @property
    def dataset(self) -> LazyKnownData:
        """Return lazy dataset."""
        return self._dataset

    def _create_loader(self, mmap_dir: Path | str | None) -> LazyHdf5Loader:
        """Create, register, and preload the lazy loader."""
        loader = LazyHdf5Loader(
            chrom_file=self.datasource.chromsize_file,
            normalization=True,
            mmap_dir=mmap_dir,
        )
        loader.register_hdf5s(
            data_file=self.datasource.hdf5_file,
            md5s=self.metadata.md5s,
            strict=True,
            verbose=True,
        )
        loader.preload_all()
        return loader

    def _filter_metadata(
        self, min_class_size: int, metadata: UUIDMetadata, verbose: bool
    ) -> UUIDMetadata:
        """Filter entry metadata."""
        files = LazyHdf5Loader.read_list(self.datasource.hdf5_file)

        # Remove metadata not associated with files
        metadata.apply_filter(lambda item: item[0] in files)

        metadata.remove_missing_labels(self.target_category)
        if self.label_list is not None:
            metadata.select_category_subsets(self.target_category, self.label_list)
        metadata.remove_small_classes(
            min_class_size, self.target_category, verbose, using_uuid=True
        )
        return metadata


class LazyEpiAtlasMetadata(LazyEpiAtlasDataset):
    """Metadata-only variant that doesn't create a loader or call preload_all."""

    def _create_loader(self, mmap_dir: Path | str | None) -> LazyHdf5Loader:
        """Create a minimal loader without registering files."""
        loader = LazyHdf5Loader(
            chrom_file=self.datasource.chromsize_file,
            normalization=True,
            mmap_dir=mmap_dir,
        )
        # Don't register files for metadata-only usage
        return loader


class LazyEpiAtlasFoldFactory:
    """Lazy-loading cross-validation fold factory.

    Creates data splits without loading signals into memory.
    Signals are loaded on-demand during training.
    """

    def __init__(
        self,
        epiatlas_dataset: LazyEpiAtlasDataset,
        n_fold: int = 10,
        test_ratio: float = 0,
    ):
        """Initialize fold factory.

        Args:
            epiatlas_dataset: LazyEpiAtlasDataset instance
            n_fold: Number of cross-validation folds
            test_ratio: Ratio of data reserved for final test
        """
        self.k = n_fold
        if n_fold < 2:
            raise ValueError(
                f"Need at least two folds for cross-validation. Got {n_fold}."
            )
        self.test_ratio = test_ratio
        if test_ratio < 0 or test_ratio > 1:
            raise ValueError(f"test_ratio must be between 0 and 1. Got {test_ratio}.")

        self._epiatlas_dataset = epiatlas_dataset
        self._classes = self._epiatlas_dataset.classes

        self._train_val, self._test = self._reserve_test()
        if len(self._train_val) == 0:
            raise ValueError("No data in training and validation.")

    @classmethod
    def from_datasource(
        cls,
        datasource: EpiDataSource,
        label_category: str,
        label_list: List[str] | None = None,
        min_class_size: int = 10,
        test_ratio: float = 0,
        n_fold: int = 10,
        md5_list: List[str] | None = None,
        force_filter: bool = True,
        metadata: UUIDMetadata | None = None,
        mmap_dir: Path | str | None = None,
    ):
        """Create fold factory from datasource with lazy loading."""
        epiatlas_dataset = LazyEpiAtlasDataset(
            datasource,
            label_category,
            label_list,
            min_class_size,
            md5_list,
            force_filter,
            metadata,
            mmap_dir,
        )
        return cls(epiatlas_dataset, n_fold, test_ratio)

    @property
    def n_fold(self) -> int:
        """Returns expected number of folds."""
        return self.k

    @property
    def epiatlas_dataset(self) -> LazyEpiAtlasDataset:
        """Returns source LazyEpiAtlasDataset."""
        return self._epiatlas_dataset

    @property
    def classes(self) -> List[str]:
        """Returns classes."""
        return self._classes

    @property
    def train_val_dset(self) -> LazyKnownData:
        """Returns training dataset for cross-validation."""
        return self._train_val

    @property
    def test_dset(self) -> LazyKnownData:
        """Returns test dataset."""
        return self._test

    @staticmethod
    def _label_uuid(dset: LazyKnownData) -> Tuple[NDArray, NDArray, NDArrayInt]:
        """Return uuids, unique uuids and uuid to int mapping."""
        uuids = [dset.metadata[md5]["uuid"] for md5 in dset.ids]
        unique_uuids, uuid_to_int = np.unique(uuids, return_inverse=True)
        return np.array(uuids), unique_uuids, uuid_to_int

    @staticmethod
    def _label_epirr(dset: LazyKnownData) -> Tuple[NDArray, NDArray, NDArrayInt]:
        """Return epirrs, unique epirrs and epirr-to-int mapping for grouping."""
        epirrs = [dset.metadata[md5][EPIRR_LABEL] for md5 in dset.ids]
        unique_epirrs, epirr_to_int = np.unique(epirrs, return_inverse=True)
        return np.array(epirrs), unique_epirrs, epirr_to_int

    @staticmethod
    def _uuid_to_epirr_groups(
        dset: LazyKnownData,
        uuids_unique: NDArray,
    ) -> NDArrayInt:
        """For each unique UUID, return the integer index of its parent EpiRR."""
        uuid_epirr = {}
        for md5 in dset.ids:
            meta = dset.metadata[md5]
            uuid_epirr[meta["uuid"]] = meta[EPIRR_LABEL]

        epirr_per_uuid = [uuid_epirr[uuid] for uuid in uuids_unique]
        _, epirr_inverse = np.unique(epirr_per_uuid, return_inverse=True)
        return epirr_inverse

    def _reserve_test(self) -> Tuple[LazyKnownData, LazyKnownData]:
        """Reserve test data for final evaluation."""
        dset = self._epiatlas_dataset.dataset
        if self.test_ratio == 0:
            return dset, LazyKnownData.empty_collection()

        n_splits = int(1 / self.test_ratio)
        if self.epiatlas_dataset.target_category == "track_type":
            train_val, test = next(self._split_by_track_type(dset, n_splits))
        else:
            train_val, test = next(self._split_dataset(dset, n_splits, oversample=False))
        return train_val, test

    def _split_by_track_type(
        self, dset: LazyKnownData, n_splits: int
    ) -> Generator[Tuple[LazyKnownData, LazyKnownData], None, None]:
        """Split dataset by track_type."""
        _, _, uuids_inverse = self._label_uuid(dset)

        # Force track type as the class label
        labels = [dset.metadata[md5]["track_type"] for md5 in dset.ids]

        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Create dummy X since we're not loading signals
        dummy_X = np.zeros((len(dset), 1))

        for train_idxs, valid_idxs in skf.split(
            X=dummy_X, y=labels, groups=uuids_inverse
        ):
            train_set = dset.subsample(list(train_idxs))
            valid_set = dset.subsample(list(valid_idxs))

            yield train_set, valid_set

    def _split_dataset(
        self, dset: LazyKnownData, n_splits: int, oversample: bool = False
    ) -> Generator[Tuple[LazyKnownData, LazyKnownData], None, None]:
        """Split dataset with stratification, keeping all UUIDs from the same
        EpiRR in the same fold (mirrors EpiAtlasFoldFactory._split_dataset)."""
        uuids, uuids_unique, uuids_inverse = self._label_uuid(dset)
        labels_unique = [dset.encoded_labels[uuids == uuid][0] for uuid in uuids_unique]

        # Group unique UUIDs by their parent EpiRR so all tracks from the same
        # experiment land in the same fold.
        epirr_groups = self._uuid_to_epirr_groups(dset, uuids_unique)

        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idxs_unique, valid_idxs_unique in skf.split(
            X=np.empty(shape=(len(uuids_unique), 1)),
            y=labels_unique,
            groups=epirr_groups,
        ):
            # Expand UUID-level indices back to sample-level
            train_idxs: NDArrayInt = np.concatenate(
                [np.where(uuids_inverse == idx)[0] for idx in train_idxs_unique]
            )
            valid_idxs: NDArrayInt = np.concatenate(
                [np.where(uuids_inverse == idx)[0] for idx in valid_idxs_unique]
            )

            if oversample:
                # Oversample at UUID level (not sample level, not EpiRR level)
                ros = RandomOverSampler(random_state=42)
                train_uuids_resampled, _ = ros.fit_resample(
                    np.array(uuids_unique[train_idxs_unique]).reshape(-1, 1),
                    np.array(labels_unique)[train_idxs_unique],
                )
                train_idxs: NDArrayInt = np.concatenate(
                    [
                        np.where(uuids == uuid)[0]
                        for uuid in train_uuids_resampled.flatten()
                    ]
                )

            train_set = dset.subsample(list(train_idxs))
            valid_set = dset.subsample(list(valid_idxs))

            yield train_set, valid_set

    def yield_split(
        self, oversample: bool = True
    ) -> Generator[dataset.DataSet, None, None]:
        """Yield train and valid datasets for cross-validation.

        No signals are loaded - they'll be loaded on-demand during training.
        """
        dset = self._train_val

        if self.epiatlas_dataset.target_category == "track_type":
            generator = self._split_by_track_type(dset, self.k)
        else:
            generator = self._split_dataset(dset, self.k, oversample=oversample)

        for train_set, valid_set in generator:
            yield dataset.DataSet(
                training=train_set,
                validation=valid_set,
                test=LazyKnownData.empty_collection(),
                sorted_classes=self.classes,
            )

    def create_total_data(self, oversample: bool = True) -> LazyKnownData:
        """Create combined train+val dataset for final training.

        Used for training on all available data after cross-validation.
        """
        train_set = self._train_val

        uuids, uuids_unique, uuids_inverse = self._label_uuid(train_set)
        labels_unique = [
            train_set.encoded_labels[uuids == uuid][0] for uuid in uuids_unique
        ]

        if oversample:
            # Oversample in UUID space
            ros = RandomOverSampler(random_state=42)
            resampled_uuid_idxs, _ = ros.fit_resample(
                np.array(range(len(uuids_unique))).reshape(-1, 1),
                np.array(labels_unique),
            )
            resampled_uuid_idxs = resampled_uuid_idxs.flatten()

            # Map to sample space
            train_idxs = np.concatenate(
                [np.where(uuids_inverse == idx)[0] for idx in resampled_uuid_idxs]
            )

            train_set = train_set.subsample(list(train_idxs))

        return train_set
