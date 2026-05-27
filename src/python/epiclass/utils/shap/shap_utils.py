"""Module containing utility functions for shap files handling and a bit of analysis."""
# pylint: disable=use-dict-literal, too-many-positional-arguments
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from epiclass.core.metadata import Metadata


def get_archives(shap_values_dir: str | Path) -> Tuple[Dict, Dict]:
    """
    Extracts SHAP values and explainer background information from .npz files in a specified directory.

    This function searches for files in the provided directory, specifically looking for files that match
    the patterns "*evaluation*.npz" and "*explainer_background*.npz". It loads these .npz files as dictionaries
    and returns them. The function raises a FileNotFoundError if the required files are not found in the directory.

    Args:
        shap_values_dir (str | Path): The directory path where the .npz files are located.

    Returns:
        Tuple[Dict, Dict]: The first dictionary contains the SHAP values extracted from the "*evaluation*.npz" file,
        and the second contains the explainer background information extracted from the "*explainer_background*.npz" file.

    Raises:
        FileNotFoundError: If either the SHAP values file or the explainer background file is not found
        in the specified directory.
    """
    shap_values_dir = Path(shap_values_dir)
    try:
        shap_values_path = next(shap_values_dir.glob("*evaluation*.npz"))
        background_info_path = next(shap_values_dir.glob("*explainer_background*.npz"))
    except StopIteration as err:
        raise FileNotFoundError(
            f"Could not find shap values or explainer background archives in {shap_values_dir}"
        ) from err

    with open(shap_values_path, "rb") as f:
        shap_values_archive = np.load(f)
        shap_values_archive = dict(shap_values_archive.items())

    with open(background_info_path, "rb") as f:
        explainer_background = np.load(f)
        explainer_background = dict(explainer_background.items())

    return shap_values_archive, explainer_background


def extract_shap_values_and_info(
    shap_logdir: str | Path, verbose: bool = True
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
    """Extract and print basic statistics about SHAP values from an archive.

    Args:
        shap_logdir (str): The directory where the SHAP values archive is located.
        verbose (bool): Whether to print basic statistics about the SHAP values.

    Returns:
        shap_matrices (np.ndarray): SHAP matrices.
        eval_signal_ids (List[str]): List of evaluation signal IDs.
        classes (List[Tuple[str, str]]): List of classes. Each class is a tuple containing the class index and the class label.
    """
    # Extract shap values and signal IDs from archive
    shap_values_archive, _ = get_archives(shap_logdir)
    try:
        eval_signal_ids: List[str] = shap_values_archive["evaluation_md5s"]
    except KeyError:
        eval_signal_ids: List[str] = shap_values_archive["evaluation_ids"]
    shap_matrices: np.ndarray = shap_values_archive["shap_values"]

    # Print basic statistics about the loaded SHAP values
    if verbose:
        print(f"nb classes: {len(shap_matrices)}")
        print(f"nb samples: {len(eval_signal_ids)}")
        print(f"dim shap value matrix: {shap_matrices[0].shape}")
        print(f"Output classes of classifier:\n {shap_values_archive['classes']}")

    return shap_matrices, eval_signal_ids, shap_values_archive["classes"]


def select_random_shap_samples(
    shap_dict: Dict[str, List[np.ndarray]], n: int
) -> Dict[str, List[np.ndarray]]:
    """
    Selects a random subset of SHAP values and their corresponding IDs from a given dictionary.

    This function randomly selects 'n' samples from the provided SHAP values. It ensures that the selection
    is non-repetitive. The function is designed to work with a dictionary containing SHAP values and their
    corresponding IDs. The resulting subset contains both SHAP values and IDs, maintaining their association.

    Args:
        shap_dict (Dict[str, List[np.ndarray]]): A dictionary with two keys: 'shap' and 'ids'. 'shap' should be
            a list of numpy arrays containing SHAP values, and 'ids' should be a list of identifiers corresponding
            to each SHAP value.
        n (int): The number of random samples to select. If 'n' is larger than the total number of samples available,
            all samples are returned without duplication.

    Returns:
        Dict[str, List[np.ndarray]]: A dictionary containing two keys: 'shap' and 'ids'. 'shap' is a list of
            numpy arrays representing the randomly selected SHAP values, and 'ids' is a list of the corresponding
            identifiers. The length of the lists equals 'n', or the total number of samples if 'n' is larger than
            the available samples.

    Raises:
        ValueError: If 'n' is negative.
        IndexError: If the provided 'shap_dict' does not contain the required keys ('shap' and 'ids').
    """
    selected_shap_samples = {"shap": [], "ids": []}
    total_samples = len(shap_dict["ids"])
    selected_indices = np.random.choice(total_samples, n, replace=False)

    for class_shap_values in shap_dict["shap"]:
        selected_shap_samples["shap"].append(class_shap_values[selected_indices, :])

    selected_shap_samples["ids"] = [shap_dict["ids"][idx] for idx in selected_indices]

    return selected_shap_samples


def subsample_signal_ids(
    signal_ids: List[str],
    metadata: Metadata,
    category_label: str,
    labels: List[str],
    copy_metadata: bool = True,
) -> List[int]:
    """Subsample signal_ids index based on metadata filtering provided, for a given category and filtering labels.

    Args:
        signal_ids (list): A list of signal IDs.
        metadata (Metadata): A metadata object containing the data to be filtered.
        category_label (str): The category label to be used for filtering the metadata.
        labels (list): A list of labels to be used for selecting category subsets in the metadata.

    Returns:
        list: A list of indices corresponding to the selected signal IDs.
    """
    if copy_metadata:
        meta = copy.deepcopy(metadata)
    else:
        meta = metadata

    meta.select_category_subsets(category_label, labels)
    chosen_idxs = []
    for i, sid in enumerate(signal_ids):
        if sid in meta:
            chosen_idxs.append(i)
    return chosen_idxs


def get_shap_matrix(
    meta: Metadata,
    shap_matrices: np.ndarray,
    eval_signal_ids: List[str],
    label_category: str,
    selected_labels: List[str],
    class_idx: int,
    copy_meta: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """Generates a SHAP matrix corresponding to a selected subset of samples.

    This function selects a subset of samples based on specified criteria
    and then generates a SHAP matrix for these selected samples. It filters
    the metadata if a specific target subsample is provided, and selects a
    subset of samples that are identified by their signal ID. It then selects
    the SHAP values of these samples under the matrix of the given class number.

    Args:
        meta (metadata.Metadata): Metadata object containing information about the samples.
        shap_matrices (np.ndarray): Array of SHAP matrices for each class.
        eval_signal_ids (List[str]): List of signal IDs identifying the evaluation samples.
        label_category (str): Name of the category in the metadata that contains the desired labels.
        selected_labels (List[str]): Name of the classes for which samples will be considered.
        class_idx (int): Index of the class for which the shap values matrix will be used.

    Returns:
        np.ndarray: The selected SHAP matrix for the selected class and for the
                    chosen samples based on the provided criteria.
        List[int]: The indices of the chosen samples in the original SHAP matrix.

    Raises:
        IndexError: If the `class_idx` is out of bounds for the `shap_matrices`.
    """
    if copy_meta:
        my_meta = copy.deepcopy(meta)
    else:
        my_meta = meta

    chosen_idxs = subsample_signal_ids(
        signal_ids=eval_signal_ids,
        metadata=my_meta,
        category_label=label_category,
        labels=selected_labels,
        copy_metadata=copy_meta,
    )
    if len(shap_matrices.shape) == 3:  # deepSHAP
        try:
            class_shap = shap_matrices[class_idx]
        except IndexError as err:
            raise IndexError(f"Class index {class_idx} is out of bounds.") from err

        selected_class_shap = np.array(class_shap[chosen_idxs, :])
    else:  # TreeExplainer 2D
        class_shap = shap_matrices
        selected_class_shap = class_shap[chosen_idxs]  # type: ignore
    print(
        f"Shape of selected class ({selected_labels}) shap values: {selected_class_shap.shape}"
    )
    print(f"Chose {len(chosen_idxs)} samples from {class_shap.shape[0]} samples")
    return selected_class_shap, chosen_idxs


def n_most_important_features(sample_shaps: np.ndarray, n: int) -> np.ndarray:
    """Return indices of features with the highest absolute shap values.

    Args:
        sample_shaps (np.ndarray): Array of SHAP values for a single sample.
        n (int): Number of top features to return.

    Returns:
        np.ndarray: Indices of the top `n` features with the highest absolute SHAP values.
    """
    return np.flip(np.argsort(np.absolute(sample_shaps)))[:n]


# def verify_subsample_coherence(
#     shap_matrices: np.ndarray, chosen_idxs: List[int], class_int: int
# ) -> None:
#     """Verify if the subsampling is coherent with the SHAP values.

#     This function calculates the mean absolute SHAP values for the samples
#     identified by chosen_idxs in each class' SHAP matrix. It then checks
#     if the class index for which the subsampling was done (class_int) has
#     the highest mean absolute SHAP value.

#     Args:
#         shap_matrices (np.ndarray): Array of SHAP matrices for each class.
#         chosen_idxs (List[int]): Indices of the samples chosen during subsampling.
#         class_int (int): The class index for which the subsampling was performed.

#     Returns:
#         None: Prints out the results.
#     """

#     # Calculate the mean of absolute SHAP values for each class for selected samples.
#     avg_abs_shap_per_class = [
#         np.mean(np.abs(shap_matrices[i][chosen_idxs, :]))
#         for i in range(len(shap_matrices))
#     ]

#     # Find the index of the class with highest average absolute SHAP value
#     highest_shap_class_idx = np.argmax(avg_abs_shap_per_class)

#     print(f"Average absolute SHAP values per class: {avg_abs_shap_per_class}")
#     print(f"Class with highest average absolute SHAP value: {highest_shap_class_idx}")

#     # Compare the index with class_int to check if they are same.
#     if highest_shap_class_idx == class_int:
#         print(
#             f"The subsampling for class index {class_int} is coherent with SHAP values."
#         )
#     else:
#         print(
#             f"Warning: The subsampling for class index {class_int} may not be coherent with SHAP values. Highest SHAP values belong to class index {highest_shap_class_idx}."
#         )
