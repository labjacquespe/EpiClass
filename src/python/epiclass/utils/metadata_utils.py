"""Metadata analysis basic utility functions. Also defines some constants."""
import typing
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from epiclass.core.data.data import KnownData
from epiclass.core.metadata import Metadata

EPIATLAS_CATS = set(
    [
        "uuid",
        "inputs",
        "inputs_ctl",
        "original_read_length",
        "original_read_length_ctl",
        "trimmed_read_length",
        "trimmed_read_length_ctl",
        "software_version",
        "chastity_passed",
        "paired_end_mode",
        "antibody_nan",
        "gembs_config",
        "rna_seq_type",
        "analyzed_as_stranded",
        "paired",
        "antibody",
        "upload_date",
    ]
)

EPIATLAS_ASSAYS = [
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k4me1",
    "h3k4me3",
    "h3k9me3",
    "input",
    "rna_seq",
    "mrna_seq",
    "wgbs",
    "wgbs-standard",
    "wgbs-pbat",
]

DP_ASSAYS = [
    "chromatin_acc",
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k4me1",
    "h3k4me3",
    "h3k9me3",
    "input",
    "mrna_seq",
    "rna_seq",
    "wgb_seq",
]


def make_table(my_metadata: Metadata, cat1: str, cat2: str, filepath: str):
    """Write metadata content tsv table for given metadata categories"""
    counter = count_pairs(my_metadata, cat1, cat2)
    triplets = [(pair[0], pair[1], int(count)) for pair, count in sorted(counter.items())]

    df = pd.DataFrame(triplets, columns=[cat1, cat2, "count"])
    table = df.pivot_table(
        values="count",
        index=cat1,
        columns=cat2,
        aggfunc=np.sum,
        fill_value=0,
        margins=True,
        margins_name="Total",
    )
    table = table.T.sort_values(by="Total", ascending=False)
    table.to_csv(Path(filepath).with_suffix(".csv"), sep=",")


def count_pairs(
    my_metadata: Metadata, cat1: str, cat2: str, use_uuid: bool = False
) -> typing.Counter[Tuple[str, str]]:
    """
    Return label pairs (label_cat1, label_cat2) counter from the given metadata categories.

    Args:
        my_metadata (Metadata): experiment metadata to analyze.
        cat1 (str): The first category to count.
        cat2 (str): The second category to count.
        use_uuid (bool): If True, count pairs by uuid; otherwise, by md5sum.

    Returns:
        Counter (Tuple[str, str], int): A counter object that counts label pairs.
    """
    id_label = "md5sum"
    if use_uuid:
        id_label = "uuid"

    unique_examples = defaultdict(set)
    for dset in my_metadata.datasets:
        dset_id: str = dset[id_label]
        cat1_label: str = dset.get(cat1, "--empty--")
        cat2_label: str = dset.get(cat2, "--empty--")
        unique_examples[(cat1_label, cat2_label)].add(dset_id)

    counter = Counter(
        {label_pair: len(ids) for label_pair, ids in unique_examples.items()}
    )
    return counter


def count_combinations(
    my_metadata: Metadata, categories: Iterable[str], use_uuid: bool = False
) -> typing.Counter[Tuple[str, ...]]:
    """
    Return label combinations counter from the given metadata categories.

    Args:
        my_metadata (Metadata): Experiment metadata to analyze.
        categories (List[str]): A list of categories to count.
        use_uuid (bool): If True, count combinations by uuid; otherwise, by md5sum.

    Returns:
        Counter (Tuple[str, ...], int): A counter object that counts label combinations.
    """
    id_label = "uuid" if use_uuid else "md5sum"

    unique_examples = defaultdict(set)
    for dset in my_metadata.datasets:
        dset_id: str = dset[id_label]
        labels = tuple(dset.get(category, "--empty--") for category in categories)
        unique_examples[labels].add(dset_id)

    counter = Counter(
        {
            label_combination: len(ids)
            for label_combination, ids in unique_examples.items()
        }
    )
    return counter


def print_pairs(my_metadata: Metadata, cat1: str, cat2: str):
    """Print label pairs from the given metadata categories."""
    counter = count_pairs(my_metadata, cat1, cat2)
    for pair, count in sorted(counter.items()):
        print(pair, count)


def count_labels_from_dset(dset: KnownData, label_category: str, from_uuid):
    """
    Returns a Counter object containing the counts of all examples of a given label category in a dataset.

    This function processes the dataset and organizes the samples by the specified label category,
    counting the number of occurrences for each unique label.

    Args:
        dset (KnownData): The dataset object containing the samples and metadata.
        label_category (str): The specific label category to count (e.g., "assay", "cell_type").
        from_uuid (bool): A flag to determine whether to use "uuid" or "md5sum" as the identifying label.

    Returns:
        collections.Counter: A Counter object with the unique labels as keys and the counts as values.
    """
    label_samples = defaultdict(list)
    id_label = "uuid" if from_uuid else "md5sum"
    meta = dset.metadata

    for md5 in dset.ids:
        sample_meta = meta[md5]
        label = sample_meta.get(label_category, "--empty--")
        label_samples[label].append(sample_meta[id_label])

    counter = Counter({label: len(ids) for label, ids in label_samples.items()})
    return counter
