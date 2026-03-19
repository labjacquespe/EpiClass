"""Functions to split epiatlas datasets properly, keeping track types together in the different sets."""
# pylint: disable=too-many-positional-arguments
# TODO: Proper Data vs TestData typing
from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import numpy.typing as npt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedGroupKFold

from epiclass.core.data import dataset
from epiclass.core.data.eager import KnownData
from epiclass.core.data_source import EpiDataSource
from epiclass.core.loaders.hdf5_loader import Hdf5Loader
from epiclass.core.metadata import UUIDMetadata

TRACKS_MAPPING = {
    "raw": ["pval", "fc"],
    "ctl_raw": [],
    "Unique_plusRaw": ["Unique_minusRaw"],
    "gembs_pos": ["gembs_neg"],
}

ACCEPTED_TRACKS = list(TRACKS_MAPPING.keys()) + list(
    itertools.chain.from_iterable(TRACKS_MAPPING.values())
)

LEADER_TRACKS = frozenset(["raw", "Unique_plusRaw", "gembs_pos"])
OTHER_TRACKS = frozenset(ACCEPTED_TRACKS) - LEADER_TRACKS

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[int]  # type: ignore # np.int_ is deprecated NumPy >= 1.20.0
NDArrayBool = npt.NDArray[bool]  # type: ignore # np.bool_ is deprecated NumPy >= 1.20.0

EPIRR_LABEL = "epirr_id"


class EpiAtlasDataset:
    """Class that handles how epiatlas data signals are linked together.

    Parameters
    ----------
    datasource : EpiDataSource
        Where everything is read from.
    label_category : str
        The target category of labels to use.
    label_list : List[str], optional
        List of labels/classes to include from given category
    min_class_size : int, optional
        Minimum number of samples per class.
    md5_list : List[str], optional
        List of datasource md5s to include in the dataset. If None, everything is used and usual filter methods are used.
        (using min_class_size and label_list)
    force_filter : bool, optional
        If True, will filter the metadata even if md5_list is given. If False, will not filter the metadata if md5_list.
    metadata : UUIDMetadata, optional
        If given, will use this metadata instead of loading it from the datasource.
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
    ):
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

        # Load signals and create proper dataset
        self._signals = self._load_signals()

        md5s = list(self._signals.keys())
        labels = [self._metadata[md5][self._label_category] for md5 in md5s]

        self._dataset: KnownData = KnownData(
            ids=md5s,
            x=list(self._signals.values()),
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
        """Return given label category (e.g. assay)"""
        return self._label_category

    @property
    def label_list(self) -> List[str] | None:
        """Return given target labels inclusion list."""
        return self._label_list

    @property
    def classes(self) -> List[str]:
        """Return target classes"""
        return self._classes

    @property
    def metadata(self) -> UUIDMetadata:
        """Return a copy of current metadata held"""
        return copy.deepcopy(self._metadata)

    @property
    def signals(self) -> Dict[str, np.ndarray]:
        """Return loaded signals."""
        return self._signals

    @property
    def dataset(self) -> KnownData:
        """Return dataset."""
        return self._dataset

    def _load_signals(self) -> Dict[str, np.ndarray]:
        """Load signals from given datasource."""
        loader = Hdf5Loader(chrom_file=self.datasource.chromsize_file, normalization=True)
        loader = loader.load_hdf5s(
            data_file=self.datasource.hdf5_file,
            md5s=self.metadata.md5s,  # type: ignore
            strict=True,
            verbose=True,
        )
        return loader.signals

    def _filter_metadata(
        self, min_class_size: int, metadata: UUIDMetadata, verbose: bool
    ) -> UUIDMetadata:
        """Filter entry metadata for given files, assay list and label_category."""
        files = Hdf5Loader.read_list(self.datasource.hdf5_file)

        # Remove metadata not associated with files
        metadata.apply_filter(lambda item: item[0] in files)

        metadata.remove_missing_labels(self.target_category)
        if self.label_list is not None:
            metadata.select_category_subsets(self.target_category, self.label_list)
        metadata.remove_small_classes(
            min_class_size, self.target_category, verbose, using_uuid=True
        )
        return metadata


class EpiAtlasMetadata(EpiAtlasDataset):
    """Class that handles how epiatlas data ids are linked together.

    Parameters
    ----------
    datasource : EpiDataSource
        Where everything is read from.
    label_category : str
        The target category of labels to use.
    label_list : List[str], optional
        List of labels/classes to include from given category
    min_class_size : int, optional
        Minimum number of samples per class.
    md5_list : List[str], optional
        List of datasource md5s to include in the dataset. If None, everything is used and usual filter methods are used.
        (using min_class_size and label_list)
    force_filter : bool, optional
        If True, will filter the metadata even if md5_list is given. If False, will not filter the metadata if md5_list.
    metadata : UUIDMetadata, optional
        If given, will use this metadata instead of loading it from the datasource.
    """

    def _load_signals(self) -> Dict[str, np.ndarray]:
        """Load empty signals as no signals are needed for metadata."""
        return {md5: np.ndarray(0) for md5 in self._metadata.md5s}


class EpiAtlasFoldFactory:
    """Class that handles how epiatlas data is split into training, validation, and testing sets.

    Parameters
    ----------
    epiatlas_dataset : EpiAtlasDataset
        Source container for epiatlas data.
    n_fold : int, optional
        Number of folds for cross-validation.
    test_ratio : float, optional
        Ratio of data kept for test (not used for training or validation)
    """

    def __init__(
        self,
        epiatlas_dataset: EpiAtlasDataset,
        n_fold: int = 10,
        test_ratio: float = 0,
    ):
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
    ):
        """Create EpiAtlasFoldFactory from a given EpiDataSource,
        directly create the intermediary EpiAtlasDataset. See
        EpiAtlasDataset init parameters for more details.
        """
        epiatlas_dataset = EpiAtlasDataset(
            datasource,
            label_category,
            label_list,
            min_class_size,
            md5_list,
            force_filter,
            metadata,
        )
        return cls(epiatlas_dataset, n_fold, test_ratio)

    @property
    def n_fold(self) -> int:
        """Returns expected number of folds."""
        return self.k

    @property
    def epiatlas_dataset(self) -> EpiAtlasDataset:
        """Returns source EpiAtlasDataset."""
        return self._epiatlas_dataset

    @property
    def classes(self) -> List[str]:
        """Returns classes."""
        return self._classes

    @property
    def train_val_dset(self) -> KnownData:
        """Returns training dataset for cross-validation."""
        return self._train_val

    @property
    def test_dset(self) -> KnownData:
        """Returns test dataset, not used in cross-validation."""
        return self._test

    @staticmethod
    def _label_uuid(
        dset: KnownData,
    ) -> Tuple[NDArray, NDArray, NDArrayInt]:
        """Return uuids, unique uuids and uuid to int mapping (for stratified group k-fold)

        Args:
            dset (data.KnownData): The dataset from which the UUIDs are to be extracted.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - uuids (np.ndarray): All the UUIDs for the dataset's samples. Length n.
                - unique_uuids (np.ndarray): Unique UUIDs present in the dataset.
                - uuid_to_int (np.ndarray): The indices to reconstruct the original array from the unique array. Length n.
        """
        uuids = [dset.metadata[md5]["uuid"] for md5 in dset.ids]
        unique_uuids, uuid_to_int = np.unique(uuids, return_inverse=True)  # type: ignore
        return np.array(uuids), unique_uuids, uuid_to_int

    @staticmethod
    def _label_epirr(
        dset: KnownData,
    ) -> Tuple[NDArray, NDArray, NDArrayInt]:
        """Return epirrs, unique epirrs and epirr-to-int mapping for grouping. Mirrors uuid mapping function."""
        epirrs = [dset.metadata[md5][EPIRR_LABEL] for md5 in dset.ids]
        unique_epirrs, epirr_to_int = np.unique(epirrs, return_inverse=True)
        return np.array(epirrs), unique_epirrs, epirr_to_int

    @staticmethod
    def _uuid_to_epirr_groups(
        dset: KnownData,
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

    def _reserve_test(
        self,
    ) -> Tuple[KnownData, KnownData]:
        """Return training data from cross-validation and test data for final evaluation."""
        dset = self._epiatlas_dataset.dataset
        if self.test_ratio == 0:
            return dset, KnownData.empty_collection()

        n_splits = int(1 / self.test_ratio)
        if self.epiatlas_dataset.target_category == "track_type":
            train_val, test = next(self._split_by_track_type(dset, n_splits))
        else:
            train_val, test = next(self._split_dataset(dset, n_splits, oversample=False))
        return train_val, test

    def _split_by_track_type(
        self, dset: KnownData, n_splits: int
    ) -> Generator[Tuple[KnownData, KnownData], None, None,]:
        """Split dataset by track_type. Oversampling not implemented."""
        _, _, uuids_inverse = self._label_uuid(dset)

        # forcing track type as the class label
        labels = [dset.metadata[md5]["track_type"] for md5 in dset.ids]

        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idxs, valid_idxs in skf.split(
            X=dset.signals, y=labels, groups=uuids_inverse
        ):
            train_set = dset.subsample(list(train_idxs))
            valid_set = dset.subsample(list(valid_idxs))

            yield train_set, valid_set

    def _split_dataset(
        self,
        dset: KnownData,
        n_splits: int,
        oversample: bool = False,
    ) -> Generator[Tuple[KnownData, KnownData], None, None]:
        uuids, uuids_unique, uuids_inverse = self._label_uuid(dset)
        labels_unique = [dset.encoded_labels[uuids == uuid][0] for uuid in uuids_unique]

        # Group unique UUIDs by their parent EpiRR
        epirr_groups = self._uuid_to_epirr_groups(dset, uuids_unique)

        # Split in UUID-space, but constrain so all UUIDs from the same EpiRR
        # land in the same fold
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idxs_unique, valid_idxs_unique in skf.split(
            X=np.empty(
                shape=(len(uuids_unique), 1)
            ),  # dummy X since we only care about y and groups
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
                train_uuids_resampled, _ = ros.fit_resample(  # type: ignore
                    np.array(uuids_unique[train_idxs_unique]).reshape(-1, 1),
                    np.array(labels_unique)[train_idxs_unique],
                )
                train_idxs: NDArrayInt = np.concatenate(
                    [
                        np.where(uuids == uuid)[0]
                        for uuid in train_uuids_resampled.flatten()  # type: ignore
                    ]
                )

            train_set = dset.subsample(list(train_idxs))
            valid_set = dset.subsample(list(valid_idxs))

            yield train_set, valid_set

    def yield_split(
        self, oversample: bool = True
    ) -> Generator[dataset.DataSet, None, None]:
        """Yield train and valid tensor datasets for one split.

        Depends on given init parameters.
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
                test=KnownData.empty_collection(),
                sorted_classes=self.classes,
            )

    def create_total_data(self, oversample: bool = True) -> KnownData:
        """Create a single dataset from the training and validation data.

        Will not oversample properly if all samples from the same UUID do not share target label (e.g. track type)

        Used for final training, with no validation.
        """
        train_set = self._train_val

        # Convert the labels and groups (uuids) into numpy arrays
        uuids, uuids_unique, uuids_inverse = self._label_uuid(train_set)
        labels_unique = [
            train_set.encoded_labels[uuids == uuid][0] for uuid in uuids_unique
        ]

        if oversample:
            # Oversample in the UUID space, not the sample space
            ros = RandomOverSampler(random_state=42)
            resampled_uuid_idxs, _ = ros.fit_resample(  # type: ignore
                np.array(range(len(uuids_unique))).reshape(-1, 1),
                np.array(labels_unique),
            )
            resampled_uuid_idxs = resampled_uuid_idxs.flatten()  # type: ignore

            # Map back to the sample space
            train_idxs = np.concatenate(
                [np.where(uuids_inverse == idx)[0] for idx in resampled_uuid_idxs]
            )

            train_set = train_set.subsample(list(train_idxs))

        return train_set

    # TODO: needed for tune_estimator
    # def split(
    #     self,
    #     total_data: data.KnownData,
    #     X=None,
    #     y=None,
    #     groups=None,
    # ) -> Generator[tuple[List, List], None, None]:
    #     """Generate indices to split total data into training and validation set.

    #     Indexes match positions in output of create_total_data()
    #     X, y and groups :
    #         Always ignored, exist for compatibility.
    #     """
    #     md5_mapping = {md5: i for i, md5 in enumerate(total_data.ids)}

    #     raw_dset = self.epiatlas_dataset.raw_dataset
    #     skf = StratifiedKFold(n_splits=self.k, shuffle=False)
    #     for train_idxs, valid_idxs in skf.split(
    #         np.zeros((raw_dset.train.num_examples, len(self.classes))),
    #         list(raw_dset.train.encoded_labels),
    #     ):
    #         # The "complete" refers to the fact that the indexes are sampling over total data.
    #         complete_train_idxs = self._find_other_tracks(
    #             train_idxs, self._raw_dset.train, resample=True, md5_mapping=md5_mapping  # type: ignore
    #         )

    #         complete_valid_idxs = self._find_other_tracks(
    #             valid_idxs, self._raw_dset.train, resample=False, md5_mapping=md5_mapping  # type: ignore
    #         )

    #         yield complete_train_idxs, complete_valid_idxs
