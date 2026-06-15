"""Stratified k-fold cross-validation without UUID grouping.

General-data counterpart to :class:`LazyEpiAtlasFoldFactory`
(``core/lazy/lazy_fold_factory.py``). Designed for datasets like saccer3 where
samples are independent (no UUID/track_type/EPIRR requirement).

Also supports *pre-specified folds*: an explicit fold-membership mapping
(``fold_definitions``) overrides the stratified split, ``n_fold``,
``min_class_size`` and oversampling. See :meth:`GeneralFoldFactory._resolve_folds`.
"""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold

from epiclass.core.data.dataset import DataSet
from epiclass.core.data_source import EpiDataSource
from epiclass.core.lazy.lazy_data_classes import LazyKnownData
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.core.metadata import Metadata

# A fold-membership mapping: {fold_name: {id_key: [id_value, ...]}}.
FoldDefinitions = Dict[str, Dict[str, List[str]]]


class GeneralFoldFactory:
    """Stratified k-fold cross-validation without UUID grouping.

    Loads signals once and yields DataSet objects per fold.

    When ``fold_definitions`` is given, folds come from that explicit mapping
    instead of a stratified split, and ``n_fold`` / ``min_class_size`` /
    oversampling are ignored.
    """

    def __init__(
        self,
        datasource: EpiDataSource,
        label_category: str,
        min_class_size: int = 3,
        n_fold: int = 4,
        mmap_dir: Path | None = None,
        fold_definitions: FoldDefinitions | None = None,
    ):
        self._label_category = label_category
        self._fold_signal_ids: "OrderedDict[str, List[str]] | None" = None

        # Load metadata, keeping only signals present in the HDF5 list.
        meta = Metadata(datasource.metadata_file)
        files = LazyHdf5Loader.read_list(datasource.hdf5_file)
        meta.apply_filter(lambda item: item[0] in files)

        if fold_definitions is not None:
            self._setup_predefined_folds(meta, label_category, fold_definitions)
        else:
            if n_fold < 2:
                raise ValueError(f"Need at least 2 folds. Got {n_fold}.")
            self.k = n_fold
            meta.remove_missing_labels(label_category)
            meta.remove_small_classes(min_class_size, label_category)

        self._metadata = meta
        self._classes = meta.unique_classes(label_category)
        self._classes_mapping = {label: i for i, label in enumerate(self._classes)}

        # Register HDF5 paths lazily (no signals loaded yet)
        loader = LazyHdf5Loader(
            chrom_file=datasource.chromsize_file,
            normalization=True,
            mmap_dir=mmap_dir,
        )
        loader.register_hdf5s(
            data_file=datasource.hdf5_file,
            signal_ids=list(meta.signal_ids),
            strict=True,
            verbose=True,
        )
        loader.preload_all()

        signal_ids = list(loader.file_paths.keys())
        labels = [meta[sid][label_category] for sid in signal_ids]

        self._dataset = LazyKnownData(
            ids=signal_ids,
            loader=loader,
            y_str=labels,
            y=np.array(
                [self._classes_mapping[label] for label in labels], dtype=np.int64
            ),
            metadata=meta,
        )

        print(f"\nLoaded {len(signal_ids)} samples across {len(self._classes)} classes.")
        meta.display_labels(label_category)

    def _setup_predefined_folds(
        self,
        meta: Metadata,
        label_category: str,
        fold_definitions: FoldDefinitions,
    ) -> None:
        """Resolve explicit folds and restrict ``meta`` to their union (in place).

        Pre-specified folds override n_fold / min_class_size / oversampling.
        """
        _id_key, fold_signal_ids = self._resolve_folds(meta, fold_definitions)

        if len(fold_signal_ids) < 2:
            raise ValueError(f"Need at least 2 folds. Got {len(fold_signal_ids)}.")

        # Ignore samples not listed in any fold.
        keep = {sid for ids in fold_signal_ids.values() for sid in ids}
        meta.apply_filter(lambda item: item[0] in keep)

        # Labels drive only the (AVE-vestigial) class mapping; never drop a
        # listed sample silently — a missing label is an error.
        missing = sorted(sid for sid in keep if meta[sid].get(label_category) is None)
        if missing:
            raise ValueError(
                f"{len(missing)} fold-listed sample(s) lack the '{label_category}' "
                f"label: {missing}"
            )

        self.k = len(fold_signal_ids)
        self._fold_signal_ids = fold_signal_ids

    @staticmethod
    def _detect_id_key(fold_definitions: FoldDefinitions) -> str:
        """Return the single id key shared by every fold's ``{id_key: [ids]}``."""
        id_keys = set()
        for fold_name, content in fold_definitions.items():
            if not isinstance(content, dict) or len(content) != 1:
                raise ValueError(
                    f"Fold '{fold_name}' must map to a single-key dict "
                    f"{{id_key: [ids]}}; got {content!r}."
                )
            id_keys.add(next(iter(content)))
        if len(id_keys) != 1:
            raise ValueError(
                f"All folds must use the same id key; found {sorted(id_keys)}."
            )
        return id_keys.pop()

    @staticmethod
    def _resolve_folds(
        metadata: Metadata, fold_definitions: FoldDefinitions
    ) -> Tuple[str, "OrderedDict[str, List[str]]"]:
        """Map ``fold_definitions`` id-values to signal IDs (md5sums).

        The JSON shape is ``{fold_name: {id_key: [value, ...]}}``. The single
        ``id_key`` (same across folds) names a metadata category; each value
        must match exactly one signal in ``metadata``. Returns
        ``(id_key, {fold_name: [signal_id, ...]})``.
        """
        if not fold_definitions:
            raise ValueError("fold_definitions is empty.")

        id_key = GeneralFoldFactory._detect_id_key(fold_definitions)

        # Validate the id key is a real metadata category (public API).
        categories = metadata.get_categories()
        if id_key not in categories:
            raise ValueError(
                f"Fold id key '{id_key}' is not a metadata category. "
                f"Available categories: {categories}"
            )

        # Build value -> [signal_id] lookup over the (HDF5-filtered) metadata.
        lookup: Dict[str, List[str]] = defaultdict(list)
        for signal_id, dset in metadata.items:
            value = dset.get(id_key)
            if value is not None:
                lookup[value].append(signal_id)

        resolved: "OrderedDict[str, List[str]]" = OrderedDict()
        missing: List[str] = []
        ambiguous: List[str] = []
        for fold_name, content in fold_definitions.items():
            fold_ids: List[str] = []
            for value in content[id_key]:
                matches = lookup.get(value, [])
                if len(matches) == 0:
                    missing.append(f"{value} (fold {fold_name})")
                elif len(matches) > 1:
                    ambiguous.append(f"{value} -> {matches} (fold {fold_name})")
                else:
                    fold_ids.append(matches[0])
            resolved[fold_name] = fold_ids

        errors: List[str] = []
        if missing:
            errors.append(
                f"{len(missing)} fold value(s) matched no loaded signal "
                f"(absent or not in the HDF5 list): {missing}"
            )
        if ambiguous:
            errors.append(
                f"{len(ambiguous)} fold value(s) matched multiple signals "
                f"(id key '{id_key}' is not unique): {ambiguous}"
            )
        if errors:
            raise ValueError(" ; ".join(errors))

        return id_key, resolved

    @property
    def classes(self) -> List[str]:
        """Get list of class labels."""
        return self._classes

    def yield_split(self, oversample: bool = True) -> Generator[DataSet, None, None]:
        """Yield a DataSet per fold.

        With pre-specified folds, yields leave-one-fold-out splits (fold i as
        validation, the rest as training) and ignores ``oversample``.
        Otherwise, yields stratified k-fold splits.
        """
        if self._fold_signal_ids is not None:
            yield from self._yield_predefined_splits()
            return

        dset = self._dataset
        y = dset.encoded_labels

        # StratifiedKFold only inspects sample count; passing a placeholder
        # avoids materializing all signals just to split on indices.
        x_placeholder = np.zeros((len(y), 1), dtype=np.float32)

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        for train_idxs, valid_idxs in skf.split(x_placeholder, y):
            train_idxs = list(train_idxs)
            valid_idxs = list(valid_idxs)

            if oversample:
                ros = RandomOverSampler(random_state=42)
                resampled, _ = ros.fit_resample(
                    np.arange(len(train_idxs)).reshape(-1, 1),
                    y[train_idxs],
                )
                train_idxs = [train_idxs[i] for i in resampled.flatten()]

            train_set = dset.subsample(train_idxs)
            valid_set = dset.subsample(valid_idxs)

            yield DataSet(
                training=train_set,
                validation=valid_set,
                test=LazyKnownData.empty_collection(),
                sorted_classes=self._classes,
            )

    def _yield_predefined_splits(self) -> Generator[DataSet, None, None]:
        """Leave-one-fold-out splits from pre-specified fold membership."""
        assert self._fold_signal_ids is not None
        dset = self._dataset
        id_to_index = {sid: i for i, sid in enumerate(dset.ids)}
        fold_indices = {
            name: [id_to_index[sid] for sid in ids]
            for name, ids in self._fold_signal_ids.items()
        }
        fold_names = list(fold_indices.keys())

        for valid_name in fold_names:
            valid_idxs = fold_indices[valid_name]
            train_idxs = [
                idx
                for name in fold_names
                if name != valid_name
                for idx in fold_indices[name]
            ]
            yield DataSet(
                training=dset.subsample(train_idxs),
                validation=dset.subsample(valid_idxs),
                test=LazyKnownData.empty_collection(),
                sorted_classes=self._classes,
            )
