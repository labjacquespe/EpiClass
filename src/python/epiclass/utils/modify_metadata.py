"""Functions to perform more complex operations on the metadata."""
# pylint: disable=too-many-positional-arguments
import collections
import copy
import datetime
import random
from typing import TypeVar

import numpy as np

from epiclass.core.metadata import Metadata, UUIDMetadata
from epiclass.utils.metadata_utils import DP_ASSAYS, count_pairs

Meta = TypeVar("Meta", Metadata, UUIDMetadata)

merge_fetal_tissues = {
    "fetal_intestine_large": "fetal_intestine",
    "fetal_intestine_small": "fetal_intestine",
    "fetal_lung_left": "fetal_lung",
    "fetal_lung_right": "fetal_lung",
    "fetal_muscle_arm": "fetal_muscle",
    "fetal_muscle_back": "fetal_muscle",
    "fetal_muscle_leg": "fetal_muscle",
    "fetal_renal_cortex": "fetal_kidney",
    "fetal_renal_pelvis": "fetal_kidney",
}
merge_molecule = {"rna": "total_rna", "polyadenylated_mrna": "polya_rna"}


merge_not_cancer = {
    "Disease": "non-cancer",
    "Healthy/None": "non-cancer",
    "Cancer": "cancer",
}


def keep_major_cell_types(my_metadata: Metadata):
    """Remove datasets which are not part of a cell_type which has
    at least 10 signals in two assays. Those assays must also have
    at least two cell_type.
    """
    # First pass to remove useless classes
    for category in ["assay", "cell_type"]:
        my_metadata.remove_small_classes(10, category)

    # Find big enough cell types in each assay
    ids_per_assay = my_metadata.ids_per_class("assay")
    cell_types_in_assay = {}
    for assay, signal_ids in ids_per_assay.items():
        # count cell_type occurence in assay
        cell_types_count = collections.Counter(
            my_metadata[sid]["cell_type"] for sid in signal_ids
        )

        # remove small classes from counter
        cell_types_count = {
            cell_type: size for cell_type, size in cell_types_count.items() if size >= 10
        }

        # bad assay if only one class left with more than 10 examples
        if len(cell_types_count) == 1:
            cell_types_count = {}

        # delete small classes from metadata
        for sid in signal_ids:
            if my_metadata[sid]["cell_type"] not in cell_types_count:
                del my_metadata[sid]

        # keep track of big enough classes
        if cell_types_count:
            cell_types_in_assay[assay] = set(cell_types_count.keys())

    # Count in how many assay each passing cell_type is
    cell_type_counter = collections.Counter()
    for cell_type_set in cell_types_in_assay.values():
        for cell_type in cell_type_set:
            cell_type_counter[cell_type] += 1

    # Remove signals which are not part of common+big cell_types
    for signal_id in list(my_metadata.signal_ids):
        dset = my_metadata[signal_id]
        good_cell_type = cell_type_counter.get(dset["cell_type"], 0) > 1
        if not good_cell_type:
            del my_metadata[signal_id]

    return my_metadata


def keep_major_cell_types_alt(my_metadata: Metadata):
    """Return a filtered metadata with certain assays. Datasets which are
    not part of a cell_type which has at least 10 signals are removed.
    """
    my_meta = copy.deepcopy(my_metadata)

    # remove useless assays and cell_types
    my_meta.select_category_subsets("assay", DP_ASSAYS)
    my_meta.remove_small_classes(10, "cell_type")
    my_meta.convert_classes("tissue_type", merge_fetal_tissues)

    new_meta = copy.deepcopy(my_meta)
    new_meta.empty()
    for assay in DP_ASSAYS:
        temp_meta = copy.deepcopy(my_meta)
        temp_meta.select_category_subsets("assay", [assay])
        temp_meta.remove_small_classes(10, "cell_type")
        for signal_id in temp_meta.signal_ids:
            new_meta[signal_id] = temp_meta[signal_id]

    return new_meta


def five_cell_types_selection(my_metadata: Metadata):
    """Return a filtered metadata with 5 major cell_types and certain assays."""
    cell_types = [
        "monocyte",
        "cd4_positive_helper_t_cell",
        "macrophage",
        "skeletal_muscle_tissue",
        "thyroid",
    ]
    my_metadata.select_category_subsets("cell_type", cell_types)
    my_metadata.select_category_subsets("assay", DP_ASSAYS)
    return my_metadata


def filter_by_pairs(
    my_metadata: Meta,
    assay_cat: str = "assay",
    cat2: str = "cell_type",
    nb_pairs: int = 5,
    min_per_pair: int = 10,
    use_uuid: bool = True,
) -> Meta:
    """Returns filtered metadata keeping only certain classes from cat2 based on pairs conditions with assay_cat.

    1) Remove (assay_cat, cat2) pairs that have less than 'min_per_pair' signals.
    2) Only keep cat2 that have at least 'nb_pairs' different pairings still non-zero.
    """
    my_metadata = copy.deepcopy(my_metadata)
    print("Applying metadata filter function 'filter_by_pairs'.")
    to_ignore = ["other", "--", "", "na"]
    for cat in assay_cat, cat2:
        my_metadata.remove_missing_labels(cat)
        my_metadata.remove_category_subsets(cat, to_ignore)

    full_metadata = copy.deepcopy(my_metadata)

    nb_assay = len(full_metadata.label_counter(assay_cat, verbose=False))
    if nb_assay < nb_pairs:
        print(
            f"WARNING: {nb_assay} assays when nb_pairs asked is {nb_pairs}. Using {nb_assay} instead."
        )
        nb_pairs = nb_assay

    # Remove (assay, cat2) pairs with less than X unique examples (min_per_pair condition)
    # Use uuid to count unique experiments
    counter = count_pairs(my_metadata, assay_cat, cat2, use_uuid=use_uuid)
    ok_pairs = set(pair for pair, i in counter.most_common() if i >= min_per_pair)

    for dset in list(full_metadata.datasets):
        pair = (dset[assay_cat], dset[cat2])
        if pair not in ok_pairs:
            del full_metadata[dset["md5sum"]]

    # Remove cat2 labels that don't have enough different assays (nb_pairs condition)
    # First get assays per cat2.
    cell_type_assays = collections.defaultdict(set)
    for pair in count_pairs(full_metadata, assay_cat, cat2, use_uuid=use_uuid).keys():
        assay_label, cat2_label = pair
        cell_type_assays[cat2_label].add(assay_label)
    print(
        f"After min_per_pair >= {min_per_pair} condition: {len(cell_type_assays)} {cat2} labels left."
    )

    # Then count assay, and filter cat2.
    ok_ct = []
    for cell_type, assays in cell_type_assays.items():
        if len(assays) >= nb_pairs:
            ok_ct.append(cell_type)
    print(f"After nb_pairs >= {nb_pairs} condition: {len(ok_ct)} {cat2} labels left.")
    if len(ok_ct) == 0:
        raise ValueError(
            f"No {cat2} labels left after filtering by pairs. Try with less strict conditions."
        )

    full_metadata.select_category_subsets(cat2, ok_ct)
    full_metadata.display_labels(assay_cat)
    full_metadata.display_labels(cat2)
    if isinstance(full_metadata, UUIDMetadata) and use_uuid:
        full_metadata.display_uuid_per_class(assay_cat)
        full_metadata.display_uuid_per_class(cat2)
    print("'filter_by_pairs' finished.")

    return full_metadata


def special_case(my_metadata):
    """Return a filtered metadata with only rna_seq examples,
    but also add 3 thyroid (for model construction).

    Made to evaluate an already trained model,
    works with min_class_size=3 and oversample=False.
    """
    my_metadata = five_cell_types_selection(my_metadata)

    # get some thyroid signal IDs, there are none in rna_seq
    temp_meta = copy.deepcopy(my_metadata)
    temp_meta.select_category_subsets("assay", ["h3k9me3"])
    thyroid_ids = temp_meta.ids_per_class("cell_type")["thyroid"][0:3]

    # select only rna_seq examples + 3 thyroid examples for model making
    my_metadata.select_category_subsets("assay", ["rna_seq"])
    for signal_id in thyroid_ids:
        my_metadata[signal_id] = temp_meta[signal_id]

    return my_metadata


def special_case_2(my_metadata):
    """Return a filtered metadata without 2 examples
    from all assay/cell_type pairs, and all mrna_seq.
    """
    my_metadata = five_cell_types_selection(my_metadata)
    my_metadata.remove_category_subsets(
        "assay",
        ["mrna_seq"],
    )

    cell_types = my_metadata.ids_per_class("cell_type").keys()
    to_del = []
    for cell_type in cell_types:
        temp_meta = copy.deepcopy(my_metadata)
        temp_meta.select_category_subsets("cell_type", [cell_type])
        for signal_ids in temp_meta.ids_per_class("assay").values():
            to_del.extend(signal_ids[0:2])

    for signal_id in to_del:
        del my_metadata[signal_id]

    return my_metadata


def keep_major_assays_2019(my_metadata):
    """Combine rna_seq and polr2a classes pairs in the assay category.
    Written for the 2019-11 release.
    """
    print("Filtering, removing smrna_seq, and then merging rna/polr2a similar labels")
    my_metadata.remove_small_classes(10, "assay")
    my_metadata.remove_category_subsets(
        "assay",
        ["smrna_seq"],
    )
    for dataset in my_metadata.datasets:
        assay = dataset.get("assay", None)
        if assay == "mrna_seq":
            dataset["assay"] = "rna_seq"
        elif assay == "polr2aphosphos5":
            dataset["assay"] = "polr2a"

    return my_metadata


def keep_major_cell_types_2019(my_metadata):
    """Select 20 cell types in the major assays signal subset.
    A cell type needs to have at least 10 signals in one assay.
    Selection choices made out of the code.
    Written for the 2019-11 release.
    """
    my_metadata = keep_major_assays_2019(my_metadata)

    brain_labels = set(
        [
            "brain_occipetal_lobe_right",
            "brain_right_temporal",
            "brain_temporal_lobe_left",
            "brain",
            "brain_frontal_lobe_left",
        ]
    )
    hepatocyte_labels = set(["hepatocytes", "hepatocyte"])
    large_intestin_colon_labels = set(
        [
            "large_intestine_colon_ascending_(right)",
            "large_intestine_colon",
            "large_intestine_colon_rectosigmoid",
        ]
    )

    for dataset in my_metadata.datasets:
        cell_type = dataset.get("cell_type", None)
        if cell_type in brain_labels:
            dataset["cell_type"] = "brain"
        elif cell_type in hepatocyte_labels:
            dataset["cell_type"] = "hepatocytes"
        elif cell_type in large_intestin_colon_labels:
            dataset["cell_type"] = "large_intestine_colon"

    selected_cell_types = set(
        [
            "myeloid_cell",
            "venous_blood",
            "monocyte",
            "thyroid",
            "mature_neutrophil",
            "macrophage",
            "b_cells",
            "cd14_positive,_cd16_negative_classical_monocyte",
            "normal_human_colon_absorptive_epithelial_cells",
            "cd4_positive,_alpha_beta_t_cell",
            "precursor_b_cell",
            "naive_b_cell",
            "stomach",
            "lymph_node",
            "muscle_of_leg",
            "neoplastic_plasma_cell",
            "plasma_cell",
            "brain",
            "hepatocytes",
            "large_intestine_colon",
        ]
    )
    my_metadata.select_category_subsets(
        "cell_type",
        selected_cell_types,
    )

    return my_metadata


def merge_pair_end_info(metadata: Metadata):
    """Merge info from 'paired' and 'pair_end_mode' categories.

    Convert FALSE/TRUE to 'single_end' and 'paired_end'
    """
    rna_dset = []
    for signal_id, dset in metadata.items:
        try:
            rna_dset.append((signal_id, dset["paired"]))
        except KeyError:
            continue

    for signal_id, label in rna_dset:
        metadata[signal_id]["paired_end_mode"] = label

    converter = {"TRUE": "paired_end", "FALSE": "single_end"}
    metadata.convert_classes("paired_end_mode", converter)


def fix_roadmap(metadata: Metadata):
    """Merge info from 'data_generating_centre' category.

    Convert 'NIH Roadmap Epigenomics' to 'Roadmap'.
    """
    converter = {"NIH Roadmap Epigenomics": "Roadmap"}
    metadata.convert_classes("data_generating_centre", converter)


def add_fake_epiatlas_metadata(metadata: Metadata) -> None:
    """Add uuid and track_type info to non epiatlas metadata.
    uuid will be the signal ID, and track_type will be raw.
    """
    for signal_id, dset in list(metadata.items):
        if "uuid" not in dset:
            metadata[signal_id]["uuid"] = signal_id
        if "track_type" not in dset:
            metadata[signal_id]["track_type"] = "raw"


def add_formated_date(metadata: Metadata) -> None:
    """Add 'upload_date_2' category to dsets with YYYY-MM format."""
    for signal_id, dset in metadata.items:
        upload_date = dset.get("upload_date")
        try:
            date = str(datetime.datetime.strptime(upload_date, "%m-%d-%Y").date())
        except TypeError:
            continue

        date = date.rsplit("-", 1)[0]
        metadata[signal_id]["upload_date_2"] = date


def add_random_group(metadata: Metadata, seed=42, n_split=23) -> str:
    """Add a 'random_seed{seed}_{n_split}splits' category made out of
    n_splits random separations withing uuids (different tracks have same val).

    Return the name of the new category.
    """
    uuids = list(metadata.label_counter("uuid", verbose=False).keys())
    new_category = f"random_seed{seed}_{n_split}splits"

    random.seed(a=seed)
    random.shuffle(uuids)

    splits = np.array_split(np.array(uuids), n_split)
    mapping = {uuid: i for i, a_list in enumerate(splits) for uuid in a_list}

    for dset in metadata.datasets:
        group_id = mapping[dset["uuid"]]
        dset[new_category] = group_id

    return new_category
