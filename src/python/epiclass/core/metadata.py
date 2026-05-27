"""Module from Metadata class and HealthyCategory."""

# pylint: disable=unnecessary-lambda-assignment,too-many-public-methods
from __future__ import annotations

import copy
import json
import marshal
import os
import sys
from collections import Counter, defaultdict
from collections.abc import ItemsView, KeysView, ValuesView
from difflib import SequenceMatcher as SM
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


class Metadata:
    """
    Wrapper around metadata signal_id:dataset dict.

    path (Path): Path to json file containing metadata for some datasets.
    """

    def __init__(self, path: Path):
        self._metadata = self._load_metadata(path)
        self._rest = {}

    @classmethod
    def from_dict(
        cls, metadata: Dict[str, dict], allow_variable_length_id: bool = False
    ) -> Metadata:
        """Creates an object from a dict conforming to {signal_id:dset} format."""
        first_key = list(metadata.keys())[0]
        if len(first_key) != 32:
            message = f"Unexpected signal ID format (expected 32-char string). First key is: {first_key}"
            if not allow_variable_length_id:
                raise ValueError(
                    f"Unexpected signal ID format (expected 32-char string). First key is: {first_key}"
                )
            print(message, file=sys.stderr)

        obj = cls.__new__(cls)
        obj._metadata = copy.deepcopy(metadata)
        obj._rest = {}
        return obj

    @classmethod
    def from_marshal(cls, path: Path | str) -> Metadata:
        """Load a metadata dict from a marshal file format."""
        with open(path, "rb") as file:
            metadata_dict = marshal.load(file)

        first_key = list(metadata_dict.keys())[0]
        if len(first_key) != 32:
            raise ValueError(
                f"Unexpected signal ID format (expected 32-char string). Is: {first_key}"
            )

        obj = cls.__new__(cls)
        obj._metadata = metadata_dict
        obj._rest = {}
        return obj

    def empty(self):
        """Remove all entries."""
        self._metadata = {}

    def __setitem__(self, signal_id, value):
        self._metadata[signal_id] = value

    def __getitem__(self, signal_id):
        return self._metadata[signal_id]

    def __delitem__(self, signal_id):
        del self._metadata[signal_id]

    def __contains__(self, signal_id):
        return signal_id in self._metadata

    def __len__(self):
        return len(self._metadata)

    def __eq__(self, other):
        if isinstance(other, Metadata):
            return self._metadata == other._metadata and self._rest == other._rest
        return False

    def get(self, signal_id, default=None) -> Dict | None:
        """Dict .get"""
        return self._metadata.get(signal_id, default)

    def update(self, info: Metadata) -> None:
        """Dict .update equivalent. Info needs to respect {signal_id:dset} format."""
        self._metadata.update(info.items)

    def save(self, path) -> None:
        """Save the metadata to path, in original epigeec_json format."""
        self._save_metadata(path)

    @property
    def signal_ids(self) -> KeysView:
        """Return a signal IDs view (like dict.keys())."""
        return self._metadata.keys()

    @property
    def datasets(self) -> ValuesView:
        """Return a datasets view (like dict.values())."""
        return self._metadata.values()

    @property
    def items(self) -> ItemsView:
        """Return a (signal_id, datasets) view (like dict.items())."""
        return self._metadata.items()

    def to_df(self):
        """Return a dataframe with one file per row.
        Index is md5sum.
        """
        df = pd.DataFrame.from_records(list(self.datasets))
        df.set_index("md5sum", inplace=True)
        return df

    def _load_metadata(self, path):
        """Return signal_id:dataset dict."""
        with open(path, "r", encoding="utf-8") as file:
            meta_raw = json.load(file)

        self._rest = {k: v for k, v in meta_raw.items() if k != "datasets"}
        return {dset["md5sum"]: dset for dset in meta_raw["datasets"]}

    def _save_metadata(self, path):
        """Save the metadata to path, in original epigeec_json format.

        Only saves dataset information.
        """
        to_save = {"datasets": list(self.datasets)}
        to_save.update(self._rest)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(to_save, file)

    def apply_filter(self, meta_filter=lambda item: True):
        """Apply a filter on items (signal_id:dataset)."""
        self._metadata = dict(filter(meta_filter, self.items))

    def remove_missing_labels(self, label_category: str):
        """Remove datasets where the metadata category is missing."""
        filt = lambda item: label_category in item[1]
        self.apply_filter(filt)  # type: ignore

    def ids_per_class(self, label_category: str) -> Dict[str, List[str]]:
        """Return {label/class: signal_id list} dict for a given metadata category.

        Can fail if remove_missing_labels has not been ran before.
        """
        sorted_ids = sorted(self.signal_ids)
        data = defaultdict(list)
        for signal_id in sorted_ids:
            label = self[signal_id][label_category]
            data[label].append(signal_id)
        return data

    def remove_small_classes(
        self, min_class_size: int, label_category: str, verbose=True
    ):
        """Remove classes with less than min_class_size examples
        for a given metadata category.

        Returns string of class ratio left if verbose.
        """
        data = self.ids_per_class(label_category)
        nb_class = len(data)

        nb_removed_class = 0
        for label, size in self.label_counter(label_category).most_common():
            if size < min_class_size:
                nb_removed_class += 1
                for signal_id in data[label]:
                    del self[signal_id]

        if verbose:
            remaining = nb_class - nb_removed_class
            ratio = f"{remaining}/{nb_class}"
            print(
                f"{ratio} labels left from {label_category} "
                f"after removing classes with less than {min_class_size} signals."
            )

    def _check_label_category(self, label_category: str):
        """Raise ValueError if label_category does not exist."""
        cats = self.get_categories()
        if label_category not in cats:
            ratios = []
            s1 = label_category
            for s2 in cats:
                ratios.append(SM(None, s1, s2).ratio())

            top5 = sorted(zip(cats, ratios), key=lambda x: x[1], reverse=True)[:5]
            top5 = [(label, f"{ratio:0.4f}") for label, ratio in top5]
            raise KeyError(
                f"Label category '{label_category}' not in categories. "
                f"Did you mean: {top5}"
            )

    def select_category_subsets(self, label_category: str, labels: Iterable[str]):
        """Select only datasets which possess the given labels
        for the given label category.

        Raises ValueError if label_category does not exist
        """
        if isinstance(labels, str):
            labels = [labels]
        self._check_label_category(label_category)
        filt = lambda item: item[1].get(label_category) in set(labels)
        self.apply_filter(filt)  # type: ignore

    def remove_category_subsets(self, label_category: str, labels: Iterable[str]):
        """Remove datasets which possess the given labels
        for the given label category.

        Raises ValueError if label_category does not exist
        """
        if isinstance(labels, str):
            labels = [labels]
        self._check_label_category(label_category)
        filt = lambda item: item[1].get(label_category) not in set(labels)
        self.apply_filter(filt)  # type: ignore

    def label_counter(self, label_category: str, verbose=True) -> Counter[str]:
        """Return a Counter() with label count from the given category.
        Ignores missing labels.
        """
        counter = Counter([dset.get(label_category) for dset in self.datasets])

        if verbose:
            print(f"{counter[None]} labels missing and ignored from count")
        del counter[None]

        return counter

    def unique_classes(self, label_category: str) -> List[str]:
        """Return sorted list of unique classes currently existing for the given category."""
        sorted_ids = sorted(self.signal_ids)
        uniq = set()
        for signal_id in sorted_ids:
            val = self[signal_id].get(label_category)

            # Everything should be a string, this was added because there was a bug with a nan object treated as a float
            if not isinstance(val, str):
                print(
                    f"signal_id: {signal_id} has non-string label of type {type(val)}: {val}",
                    file=sys.stderr,
                )
                raise ValueError(f"Non-string label for {label_category} at {signal_id}.")

            uniq.add(self[signal_id].get(label_category))
        uniq.discard(None)
        return sorted(list(uniq))

    def display_labels(self, label_category: str):
        """Print number of examples for each label in given category."""
        print(f"\nLabel breakdown for {label_category}")
        i = 0
        label_counter = self.label_counter(label_category)
        for label, count in label_counter.most_common():
            print(f"{label}: {count}")
            i += count
        print(f"For a total of {i} examples in {len(label_counter)} classes\n")

    def get_categories(self) -> list[str]:
        """Return a list of all metadata categories sorted by lowercase."""
        categories = {key for dset in self.datasets for key in dset.keys()}
        return sorted(categories, key=str.lower)

    def convert_classes(self, category: str, converter: Dict[str, str]):
        """Convert classes labels in the given category using the converter mapping.

        Can be used to merge classes.
        """
        for dataset in self.datasets:
            label = dataset.get(category, None)
            if label in converter:
                dataset[category] = converter[label]

    def save_marshal(self, path: Path | str) -> None:
        """Save the metadata to path, in marshal format. Only saves dataset information."""
        with open(path, "wb") as file:
            marshal.dump(self._metadata, file)


class UUIDMetadata(Metadata):
    """Metadata class for UUID datasets, e.g. epiatlas."""

    @classmethod
    def from_dict(
        cls, metadata: Dict[str, dict], allow_variable_length_id: bool = False
    ) -> UUIDMetadata:
        """Creates an object from a dict conforming to {signal_id:dset} format."""
        first_key = list(metadata.keys())[0]
        if len(first_key) != 32:
            message = f"Unexpected signal ID format (expected 32-char string). First key is: {first_key}"
            if not allow_variable_length_id:
                raise ValueError(
                    f"Unexpected signal ID format (expected 32-char string). First key is: {first_key}"
                )
            print(message, file=sys.stderr)

        obj = cls.__new__(cls)
        obj._metadata = copy.deepcopy(metadata)
        obj._rest = {}
        return obj

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> UUIDMetadata:
        """Create UUIDMetadata from Metadata."""
        meta = dict(metadata.items)
        return cls.from_dict(meta)

    @classmethod
    def from_marshal(cls, path: Path | str) -> UUIDMetadata:
        """Load a metadata dict from a marshal file format."""
        with open(path, "rb") as file:
            metadata_dict = marshal.load(file)

        first_key = list(metadata_dict.keys())[0]
        if len(first_key) != 32:
            raise ValueError(
                f"Unexpected signal ID format (expected 32-char string). Is: {first_key}"
            )

        obj = cls.__new__(cls)
        obj._metadata = metadata_dict
        obj._rest = {}
        return obj

    def __eq__(self, other):
        if isinstance(other, UUIDMetadata):
            return self._metadata == other._metadata and self._rest == other._rest
        return False

    def uuid_per_class(self, label_category: str) -> Dict[str, set[str]]:
        """Return {label/class:uuid list} dict for a given metadata category.

        Can fail if remove_missing_labels has not been ran before.
        """
        uuid_dict = defaultdict(set)
        for signal_id in self._metadata:
            track_type = self._metadata[signal_id]["track_type"]
            label = self._metadata[signal_id][label_category]
            uuid = self._metadata[signal_id]["uuid"]

            # Special case for ctl_raw, same uuid as other tracks, but counts as unique experiment
            if track_type == "ctl_raw":
                uuid += "_ctl"

            uuid_dict[label].add(uuid)
        return uuid_dict

    def uuid_counter(self, label_category: str, verbose=True) -> Counter[str]:
        """Return a Counter() with uuid count from the given category.
        Ignores missing labels.
        """
        uuid_dict = self.uuid_per_class(label_category)
        uuid_counter = Counter({label: len(uuid_dict[label]) for label in uuid_dict})

        if verbose:
            print(f"{uuid_counter[None]} uuid missing and ignored from count")  # type: ignore
        del uuid_counter[None]

        return uuid_counter

    def display_uuid_per_class(self, label_category: str) -> None:
        """Display uuid_per_class for a given metadata category."""
        uuid_dict = self.uuid_per_class(label_category)
        uuid_counter = Counter({label: len(uuid_dict[label]) for label in uuid_dict})
        print(f"{label_category} label breakdown for unique experiments (uuid):")

        for label, c in uuid_counter.most_common():
            print(f"{label}: {c}")

        print(
            f"For {sum(uuid_counter.values())} unique experiments in {len(uuid_dict)} classes\n"
        )

    def uuid_to_signal_ids(self) -> Dict[str, Dict[str, str]]:
        """Return uuid to {track_type:signal_id} mapping { uuid : {track_type1:signal_id, track_type2:signal_id, ...} }"""
        uuid_to_ids = defaultdict(dict)
        for dset in self.datasets:
            uuid = dset["uuid"]
            uuid_to_ids[uuid].update({dset["track_type"]: dset["md5sum"]})
        return uuid_to_ids

    def remove_small_classes(
        self,
        min_class_size: int,
        label_category: str,
        verbose=True,
        using_uuid: bool = True,
    ):
        """Remove classes with less than min_class_size examples
        for a given metadata category.

        Counts unique uuids if using_uuid=True, else counts signal IDs.

        Returns string of class ratio left if verbose.
        """
        nb_class_init = len(self.unique_classes(label_category))

        if not using_uuid:
            ids_per_class = self.ids_per_class(label_category)
            for label, size in self.label_counter(label_category).most_common():
                if size < min_class_size:
                    for signal_id in ids_per_class[label]:
                        del self[signal_id]
        else:
            uuid_to_ids = self.uuid_to_signal_ids()
            for label, uuids in self.uuid_per_class(label_category).items():
                if len(uuids) < min_class_size:
                    for uuid in uuids:
                        for signal_id in uuid_to_ids[uuid].values():
                            del self[signal_id]

        if verbose:
            remaining = len(self.unique_classes(label_category))
            ratio = f"{remaining}/{nb_class_init}"
            print(
                f"{ratio} labels left from {label_category} "
                f"after removing classes with less than {min_class_size} signals."
            )


def env_filtering(metadata: Metadata, category: str) -> List[str]:
    """Filter metadata using environment variables.
    Return the list of classes/labels to consider.

    Currently supports:
    EXCLUDE_LIST
    ASSAY_LIST
    LABEL_LIST
    REMOVE_TRACKS
    """
    print("Checking environment variables.")
    # fmt: off
    name = "ASSAY_LIST"
    if os.getenv(name) is not None:
        assay_list = json.loads(os.environ[name])
        print(f"{name}: {assay_list}")
        print(f"Filtering metadata: Only keeping examples with targets/assay {assay_list}")

        assay_category_label = list(set(["assay_epiclass", "assay"]) & set(metadata.get_categories()))
        if len(assay_category_label) == 0:
            raise ValueError(f"Assay category not found in metadata categories: {metadata.get_categories()}")

        metadata.select_category_subsets(assay_category_label[0], assay_list)

    name = "EXCLUDE_LIST"
    if os.getenv(name) is not None:
        exclude_list = json.loads(os.environ[name])
        print(f"{name}: {exclude_list}")
        print(f"Filtering metadata: Removing labels {exclude_list} from category '{category}'.")
        metadata.remove_category_subsets(label_category=category, labels=exclude_list)

    name = "LABEL_LIST"
    if os.getenv(name) is not None:
        label_list = json.loads(os.environ[name])
        print(f"{name}: {label_list}")
        print(f"Filtering metadata: Only keeping examples with labels {label_list} from category '{category}'")
        metadata.select_category_subsets(category, label_list)
    else:
        label_list = metadata.unique_classes(category)
        print(f"No label list, considering all left classes : {label_list}")
    # fmt: on

    name = "REMOVE_TRACKS"
    if os.getenv(name) is not None:
        track_list = json.loads(os.environ[name])
        print(f"{name}: {track_list}")

        print(f"Filtering metadata: Removing examples with track type {track_list}.")
        track_type_category_label = "track_type"
        metadata.remove_category_subsets(track_type_category_label, track_list)

    return label_list


class HealthyCategory:
    """Create/Represent/manipulate the "healthy" metadata category"""

    def __init__(self):
        self.pairs_file = Path(__file__).parent / "healthy_category.tsv"
        self.healthy_dict = self.read_healthy_pairs()

    @staticmethod
    def get_healthy_pairs(datasets: Iterable[Dict]) -> Set[Tuple[str, str]]:
        """Return set of (disease, donor_health_status) pairs."""
        pairs = set([])
        for dataset in datasets:
            disease = dataset.get("disease", "--empty--")
            donor_health_status = dataset.get("donor_health_status", "--empty--")
            pairs.add((disease, donor_health_status))
        return pairs

    def list_healthy_pairs(self, datasets: Iterable[Dict]) -> None:
        """List unique (disease, donor_health_status) pairs."""
        for x1, x2 in sorted(self.get_healthy_pairs(datasets)):
            print(f"{x1}\t{x2}")

    def read_healthy_pairs(self) -> Dict[Tuple[str, str], str]:
        """Return a (disease, donor_health_status):healthy dict defined in
        a tsv file with disease|donor_health_status|healthy columns.
        """
        healthy_dict = {}
        with open(self.pairs_file, "r", encoding="utf-8") as tsv_file:
            next(tsv_file)  # skip header
            for line in tsv_file:
                disease, donor_health_status, healthy = line.rstrip("\n").split("\t")
                healthy_dict[(disease, donor_health_status)] = healthy
        return healthy_dict

    def get_healthy_status(self, dataset: Dict) -> str:
        """Return "y", "n" or "?" depending of the healthy status of the dataset."""
        disease = dataset.get("disease", "--empty--")
        donor_health_status = dataset.get("donor_health_status", "--empty--")
        return self.healthy_dict[(disease, donor_health_status)]

    @staticmethod
    def create_healthy_category(metadata: Metadata) -> None:
        """Combine "disease" and "donor_health_status" to create a "healthy" category.

        When a dataset has pairs with unknow correspondance, it does not add
        the category, and so these datasets are ignored through remove_missing_labels().
        """
        healthy_category = HealthyCategory()
        for dataset in metadata.datasets:
            healthy = healthy_category.get_healthy_status(dataset)
            if healthy == "?":
                continue
            dataset["healthy"] = healthy
