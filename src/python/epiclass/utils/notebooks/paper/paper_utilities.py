"""Utility functions for the paper notebooks."""
# pylint: disable=too-many-branches, too-many-lines, too-many-positional-arguments

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from IPython.display import display
except ImportError:
    display = print
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from epiclass.core.metadata import Metadata

ASSAY: str = "assay_epiclass"
CELL_TYPE: str = "harmonized_sample_ontology_intermediate"
SEX: str = "harmonized_donor_sex"
LIFE_STAGE: str = "harmonized_donor_life_stage"
BIOMATERIAL_TYPE: str = "harmonized_biomaterial_type"
CANCER: str = "harmonized_sample_cancer_high"
DISEASE: str = "harmonized_sample_disease_high"
TRACK: str = "track_type"

ASSAY_MERGE_DICT: Dict[str, str] = {
    "mrna_seq": "rna_seq",
    "wgbs-pbat": "wgbs",
    "wgbs-standard": "wgbs",
}
ASSAY_ORDER = [
    "h3k4me3",
    "h3k27ac",
    "h3k4me1",
    "h3k36me3",
    "h3k27me3",
    "h3k9me3",
    "input",
    "rna_seq",
    "wgbs",
]

EPIATLAS_16_CT: List[str] = [
    ct.lower()
    for ct in [
        "T cell",
        "neutrophil",
        "brain",
        "monocyte",
        "lymphocyte of B lineage",
        "myeloid cell",
        "venous blood",
        "macrophage",
        "mesoderm-derived structure",
        "endoderm-derived structure",
        "colon",
        "connective tissue cell",
        "hepatocyte",
        "mammary gland epithelial cell",
        "muscle organ",
        "extraembryonic cell",
    ]
]

LIFESTAGE_REMAPPER = {
    "adult": "adult",
    "embryo": "perinatal",
    "embryonic": "perinatal",
    "fetal": "perinatal",
    "newborn": "perinatal",
    "perinatal": "perinatal",
    "child": "child",
    "unknown": "unknown",
    "indeterminate": "unknown",
    np.nan: "unknown",
    pd.NA: "unknown",
}


class IHECColorMap:
    """Class to handle IHEC color map."""

    def __init__(self, base_fig_dir: Path):
        self.base_fig_dir = base_fig_dir
        self.ihec_colormap_name = "IHEC_EpiATLAS_IA_colors_Apl01_2024.json"
        self.ihec_color_map = self.get_IHEC_color_map(
            base_fig_dir, self.ihec_colormap_name
        )
        self.assay_color_map = self.create_assay_color_map()
        self.cell_type_color_map = self.create_cell_type_color_map()
        self.sex_color_map = self.create_sex_color_map()

    @staticmethod
    def get_IHEC_color_map(folder: Path, name: str) -> List[Dict]:
        """Get the IHEC color map."""
        color_map_path = folder / name
        with open(color_map_path, "r", encoding="utf8") as color_map_file:
            ihec_color_map = json.load(color_map_file)
        return ihec_color_map

    def _create_color_map(self, label_category: str) -> Dict[str, str]:
        """Create a rbg color map from IHEC rgb strings"""
        color_dict = [elem for elem in self.ihec_color_map if label_category in elem][0][
            label_category
        ][0]
        for name, color in list(color_dict.items()):
            rbg = color.split(",")
            color_dict[name] = f"rgb({rbg[0]},{rbg[1]},{rbg[2]})"
            color_dict[name.lower()] = color_dict[name]
            color_dict[name.lower().replace("-", "_")] = color_dict[name]
        return color_dict

    def create_sex_color_map(self) -> Dict[str, str]:
        """Create a rbg color map for ihec sex label category."""
        color_dict = self._create_color_map(SEX)
        return color_dict

    def create_assay_color_map(self) -> Dict[str, str]:
        """Create a rbg color map for ihec assays."""
        category_label = "experiment"
        color_dict = self._create_color_map(category_label)
        color_dict["mrna_seq"] = color_dict["rna_seq"]
        for assay in ["wgbs-pbat", "wgbs-standard"]:
            color_dict[assay] = color_dict["wgbs"]
            color_dict[assay.replace("-", "_")] = color_dict["wgbs"]
        return color_dict

    def create_cell_type_color_map(self) -> Dict[str, str]:
        """Create the rbg color map for ihec cell types."""
        return self._create_color_map(CELL_TYPE)


def merge_similar_assays(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to merge rna-seq/wgbs categories, included prediction score.

    Expects the dataframe to have the following columns:
    - rna_seq
    - mrna_seq
    - wgbs-standard
    - wgbs-pbat

    The output dataframe will have the following columns:
    - rna_seq
    - wgbs

    The dataframe is modified in place.

    Expecteds "True class" or "Expected class", and "Predicted class" to be present.

    A ValueError is raised if the columns are not present.
    """
    df = df.copy(deep=True)

    if "wgbs" in df.columns:
        return df

    try:
        df["rna_seq"] = df["rna_seq"] + df["mrna_seq"]
        df["wgbs"] = df["wgbs-standard"] + df["wgbs-pbat"]
    except KeyError as exc:
        raise ValueError("Wrong results dataframe, rna or wgbs columns missing.") from exc

    df.drop(columns=["mrna_seq", "wgbs-standard", "wgbs-pbat"], inplace=True)

    if "True class" in df.columns:
        true_label = "True class"
        df.replace({true_label: ASSAY_MERGE_DICT}, inplace=True)
    elif "Expected class" in df.columns:
        true_label = "Expected class"
        df.replace({true_label: ASSAY_MERGE_DICT}, inplace=True)
    else:
        raise ValueError("Wrong results dataframe, True or Expected class missing.")

    df.replace({"Predicted class": ASSAY_MERGE_DICT}, inplace=True)

    try:
        df.replace({ASSAY: ASSAY_MERGE_DICT}, inplace=True)
    except KeyError:
        pass

    # Recompute Max pred if it exists
    classes = list(df[true_label].unique()) + list(df["Predicted class"].unique())
    if "Max pred" in df.columns:
        df["Max pred"] = df[classes].max(axis=1)
    return df


class MetadataHandler:
    """Class to handle Metadata objects."""

    def __init__(self, paper_dir: Path | str):
        self.paper_dir = Path(paper_dir)
        self.data_dir = self.paper_dir / "data"

        self.version_names = {
            "v1": "hg38_2023-epiatlas_dfreeze_formatted_JR.json",
            "v2": "hg38_2023-epiatlas-dfreeze-pospurge-nodup_filterCtl.json",
            "v2-encode": "hg38_2023-epiatlas-dfreeze_v2.1_w_encode_noncore_2.json",
        }

    def load_metadata(self, version: str) -> Metadata:
        """Return metadata for a specific version.

        Args:
            version (str): The version of the metadata to load.
            One of 'v1', 'v2', 'v2-encode'.

        Example of epiRR unique to v1: IHECRE00003355.2
        """
        if version not in self.version_names:
            raise ValueError(f"Version must be one of {self.version_names.keys()}")

        metadata = Metadata(
            self.data_dir / "metadata" / "epiatlas" / self.version_names[version]
        )
        return metadata

    def load_any_metadata(
        self, path: Path | str, as_dataframe: bool = False
    ) -> Metadata | pd.DataFrame:
        """Return metadata for a specific file."""
        metadata = Metadata(Path(path))
        if as_dataframe:
            metadata = metadata.to_df()
        return metadata

    def load_metadata_df(self, version: str, merge_assays: bool = True) -> pd.DataFrame:
        """Load a metadata dataframe for a given version.

        merge_assays: Merge similar assays (rna 2x / wgb 2x)
        """
        metadata = self.load_metadata(version)
        metadata_df = self.metadata_to_df(metadata, merge_assays)
        return metadata_df

    @staticmethod
    def metadata_to_df(metadata: Metadata, merge_assays: bool = True) -> pd.DataFrame:
        """Convert the metadata to a dataframe.

        merge_assays: Merge similar assays (rna 2x / wgb 2x)
        """
        metadata_df = pd.DataFrame.from_records(list(metadata.datasets))
        metadata_df.set_index("md5sum", inplace=True)
        if merge_assays:
            metadata_df.replace({ASSAY: ASSAY_MERGE_DICT}, inplace=True)
        return metadata_df

    @staticmethod
    def join_metadata(df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        """Join the metadata to the results dataframe."""
        metadata_df = pd.DataFrame(metadata.datasets)
        metadata_df.set_index("md5sum", inplace=True)
        metadata_df["md5sum"] = metadata_df.index

        diff_set = set(df.index) - set(metadata_df.index)
        if diff_set:
            err_df = pd.DataFrame(diff_set, columns=["md5sum"])
            print(err_df.to_string(), file=sys.stderr)
            raise AssertionError(
                f"{len(diff_set)} md5sums in the results dataframe are not present in the metadata dataframe. Saved error md5sums to join_missing_md5sums.csv."
            )

        merged_df = df.merge(
            metadata_df,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=(None, "_delete"),
        )
        if len(merged_df) != len(df):
            raise AssertionError(
                "Merged dataframe has different length than original dataframe"
            )
        to_drop = [col for col in merged_df.columns if "_delete" in col]
        merged_df.drop(columns=to_drop, inplace=True)

        return merged_df

    @staticmethod
    def uniformize_metadata_for_plotting(
        epiatlas_metadata: Metadata,
        ca_pred_df: pd.DataFrame | None = None,
        enc_pred_df: pd.DataFrame | None = None,
        recount3_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Simplify metadata with chip-atlas metadata for plotting."""
        columns_to_keep = ["id", ASSAY, "track_type", "source", "plot_label"]
        to_concat = []

        epiatlas_df = pd.DataFrame.from_records(list(epiatlas_metadata.datasets))
        epiatlas_df["source"] = ["epiatlas"] * len(epiatlas_df)
        epiatlas_df["id"] = epiatlas_df["md5sum"]
        if "plot_label" not in epiatlas_df.columns:
            epiatlas_df["plot_label"] = [None] * len(epiatlas_df)
        epiatlas_df = epiatlas_df[columns_to_keep]

        to_concat.append(epiatlas_df)

        if ca_pred_df is not None:
            ca_df = ca_pred_df.copy(deep=True)
            ca_df["source"] = ["C-A"] * len(ca_df)
            ca_df[ASSAY] = ca_df["manual_target_consensus"].str.lower()
            ca_df["track_type"] = ["raw"] * len(ca_df)
            ca_df["id"] = ca_df["Experimental-id"]
            if "plot_label" not in ca_df.columns:
                ca_df["plot_label"] = [None] * len(ca_df)

            ca_df = ca_df[columns_to_keep]
            to_concat.append(ca_df)

        if enc_pred_df is not None:
            enc_df = enc_pred_df.copy(deep=True)
            enc_df["source"] = ["encode"] * len(enc_df)
            enc_df[ASSAY] = enc_df[ASSAY].str.lower()
            enc_df["track_type"] = ["pval"] * len(enc_df)
            enc_df["id"] = enc_df["FILE_accession"]
            if "plot_label" not in enc_df.columns:
                enc_df["plot_label"] = [None] * len(enc_df)

            enc_df = enc_df[columns_to_keep]
            to_concat.append(enc_df)

        if recount3_metadata is not None:
            recount3_df = recount3_metadata.copy(deep=True)
            recount3_df["source"] = ["recount3"] * len(recount3_df)
            recount3_df[ASSAY] = recount3_df["harmonized_assay"].str.lower()
            recount3_df["track_type"] = ["unique_raw"] * len(recount3_df)
            recount3_df["id"] = recount3_df["ID"]
            if "plot_label" not in recount3_df.columns:
                recount3_df["plot_label"] = [None] * len(recount3_df)

            recount3_df = recount3_df[columns_to_keep]
            to_concat.append(recount3_df)

        return pd.concat(to_concat)


class SplitResultsHandler:
    """Class to handle split results."""

    @staticmethod
    def add_max_pred(
        df: pd.DataFrame,
        target_label: str = "True class",
        expected_classes: List[str] | None = None,
    ) -> pd.DataFrame:
        """Add the max prediction column ("Max pred") to the results dataframe.

        The dataframe needs to not contain extra metadata columns.

        Args:
            df: The results dataframe.
            target_label: Column to ascertain output classes columns.
            expected_classes: If the classes cannot be ascertained from the target_label and "Predicted class" columns, they can be given directly.

        Returns:
            The results dataframe with the "Max pred" column added.
        """
        if "Max pred" not in df.columns:
            df = df.copy(deep=True)
            classes_test = (
                df[target_label].astype(str).unique().tolist()
                + df["Predicted class"].astype(str).unique().tolist()
            )
            classes_test = list(set(classes_test))

            if expected_classes:
                classes_test = expected_classes

            classes = list(df.columns[2:])
            if any(label in classes for label in ["TRUE", "FALSE"]):
                print(
                    "WARNING: Found TRUE or FALSE in pred vector columns. Changing column names."
                )
                df.rename(columns={"TRUE": "True", "FALSE": "False"}, inplace=True)
                classes = list(df.columns[2:])

            for col in ["md5sum", "split", "Same?", "Predicted class", "True class"]:
                try:
                    classes.remove(col)
                except ValueError:
                    pass
            for class_label in classes:
                if class_label not in classes_test:
                    raise ValueError(
                        f"""Dataframe contains extra metadata columns, cannot ascertain classes: {classes}.
                        Column '{class_label}' is not in '{classes_test}'"""
                    )
            df["Max pred"] = df[classes].max(axis=1)
        return df

    @staticmethod
    def verify_md5_consistency(dfs: Dict[str, pd.DataFrame]) -> None:
        """Verify that all dataframes have the same md5sums.

        Used to compare the md5sums of the same split across different classifiers methods.

        Args:
            dfs (Dict[str, pd.DataFrame]): {classifier_name: results_df}
                Results dataframes need to have md5sums as index.
        """
        md5s = {}
        for key, df in dfs.items():
            md5s[key] = set(df.index)

        first_key = list(dfs.keys())[0]
        base_md5s = set(dfs[first_key].index)
        if not base_md5s.intersection(*list(md5s.values())) == base_md5s:
            raise AssertionError("Not all dataframes have the same md5sums")

    @staticmethod
    def compute_acc_per_assay(
        split_results: Dict[str, pd.DataFrame], metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute accuracy per assay for each split.

        Args:
            split_results: {split_name: results_df}.
            metadata_df: The metadata dataframe.
        """

        assay_acc = defaultdict(dict)
        for split_name, split_result_df in split_results.items():
            # Merge metadata
            split_result_df = split_result_df.merge(
                metadata_df, left_index=True, right_index=True
            )

            # Compute accuracy per assay
            assay_groupby = split_result_df.groupby(ASSAY)
            for assay, assay_df in assay_groupby:
                assay_acc[assay][split_name] = np.mean(
                    assay_df["True class"].astype(str).str.lower()
                    == assay_df["Predicted class"].astype(str).str.lower()
                )

        return pd.DataFrame(assay_acc)

    @staticmethod
    def gather_split_results_across_methods(
        results_dir: Path, label_category: str, only_NN: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Gather split results for each classifier type.

        Args:
            results_dir: The directory containing the results. Child directories should be task/classifer names.
            label_category: The label category for the results.
            only_NN: A boolean flag to only gather the NN results.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {split_name:{classifier_name: results_df}}
        """
        base_NN_path = results_dir / f"{label_category}_1l_3000n"
        if label_category == ASSAY:
            NN_csv_path_template = base_NN_path / "11c"
            if not NN_csv_path_template.exists():
                NN_csv_path_template = base_NN_path
        elif label_category == CELL_TYPE:
            NN_csv_path_template = base_NN_path
        else:
            raise ValueError("Label category not supported.")

        NN_csv_path_template = str(
            NN_csv_path_template
            / "10fold-oversampling"
            / "{split}"
            / "validation_prediction.csv"
        )
        all_split_dfs = {}
        for split in [f"split{i}" for i in range(10)]:
            # Get the csv paths
            NN_csv_path = Path(NN_csv_path_template.format(split=split))  # type: ignore
            other_csv_root = results_dir / f"{label_category}" / "predict-10fold"

            if not only_NN:
                if not other_csv_root.exists():
                    raise FileNotFoundError(f"Could not find {other_csv_root}")
                other_csv_paths = other_csv_root.glob(
                    f"*/*_{split}_validation_prediction.csv"
                )

                other_csv_paths = list(other_csv_paths)
                if len(other_csv_paths) != 4:
                    raise AssertionError(
                        f"Expected 4 other_csv_paths, got {len(other_csv_paths)}"
                    )

            # Load the dataframes
            dfs: Dict[str, pd.DataFrame] = {}
            dfs["NN"] = pd.read_csv(NN_csv_path, header=0, index_col=0, low_memory=False)

            if not only_NN:
                for path in other_csv_paths:
                    category = path.name.split("_", maxsplit=1)[0]
                    dfs[category] = pd.read_csv(
                        path, header=0, index_col=0, low_memory=False
                    )

                # Verify md5sum consistency
                SplitResultsHandler.verify_md5_consistency(dfs)

            all_split_dfs[split] = dfs

        return all_split_dfs

    @staticmethod
    def read_split_results(parent_dir: Path) -> Dict[str, pd.DataFrame]:
        """Read split results from the given parent directory.

        Args:
            parent_dir: The parent directory containing the split results.

        Returns:
            Dict[str, pd.DataFrame]: {split_name: results_df}
        """
        csv_path_template = "split*/validation_prediction.csv"
        experiment_dict = {}
        for split_result_csv in parent_dir.glob(csv_path_template):
            split_name = split_result_csv.parent.name
            df = pd.read_csv(split_result_csv, header=0, index_col=0, low_memory=False)
            experiment_dict[split_name] = df
        return experiment_dict

    @staticmethod
    def gather_split_results_across_categories(
        parent_results_dir: Path, verbose: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Gather NN split results for each classification task in the given folder children.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {general_name:{split_name: results_df}}
        """
        all_dfs = {}

        for category_dir in parent_results_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for experiment_dir in category_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue
                experiment_name = experiment_dir.name
                category_name = category_dir.name
                general_name = f"{category_name}_{experiment_name}"
                if verbose:
                    print(f"Reading {general_name} from {experiment_dir}")

                experiment_dict = SplitResultsHandler.read_split_results(experiment_dir)
                if verbose:
                    print(
                        f"Found {len(experiment_dict)} split results for {general_name}"
                    )

                all_dfs[general_name] = experiment_dict

        return all_dfs

    @staticmethod
    def concatenate_split_results(
        split_dfs: Dict[str, Dict[str, pd.DataFrame]] | Dict[str, pd.DataFrame],
        concat_first_level: bool = False,
        depth: int = 2,
    ) -> Dict[str, pd.DataFrame] | pd.DataFrame:
        """Concatenate split results for each different classifier or split name based on the order.

        This method supports two structures of input dictionaries:
        1. {split_name: {classifier_name: results_df}} when concat_first_level is False.
        2. {classifier_name: {split_name: results_df}} when concat_first_level is True.

        Args:
            split_dfs: A nested dictionary of DataFrames to be concatenated. The structure
                       depends on the concat_first_level flag.
            concat_first_level: A boolean flag that indicates the structure of the split_dfs dictionary.
                                If True, the first level is the classifier name. Otherwise, the first
                                level is the split name.
            depth: The depth of the dictionary structure. Must be 1 or 2.

        Returns:
            out (Dict[str, pd.DataFrame]): {classifier_name: concatenated_dataframe}

        Raises:
            AssertionError: If the index of any concatenated DataFrame is not of type str, indicating
                            an unexpected index format.
        """
        if depth not in [1, 2]:
            raise ValueError(f"Depth must be 1 or 2, got {depth}")

        # Handle basic case, code wasn't made for that, so a hack is done
        if depth == 1:
            to_concat = {"classifier": split_dfs}
            return SplitResultsHandler.concatenate_split_results(to_concat, concat_first_level=True, depth=2)["classifier"]  # type: ignore

        # Check for user error
        if depth == 2 and isinstance(next(iter(split_dfs.values())), pd.DataFrame):
            raise ValueError(
                "Depth given is two, but the input is not a nested dictionary."
            )

        if concat_first_level:
            # Reverse the nesting of the dictionary if concatenating by the first level.
            reversed_dfs = defaultdict(dict)
            for outer_key, inner_dict in split_dfs.items():
                for inner_key, df in inner_dict.items():
                    reversed_dfs[inner_key][outer_key] = df
            split_dfs = reversed_dfs

        to_concat_dfs = defaultdict(list)
        for split_name, dfs in split_dfs.items():
            for classifier, df in dfs.items():
                try:
                    df = df.assign(split=int(split_name.split("split")[-1]))
                except ValueError as e:
                    if "base 10" in str(e):
                        raise ValueError(
                            "Wrong concat_first_level value was probably used"
                        ) from e
                    raise e
                to_concat_dfs[classifier].append(df)

        concatenated_dfs = {
            classifier: pd.concat(dfs, axis=0)
            for classifier, dfs in to_concat_dfs.items()
        }

        # Verify index is still md5sum
        for df in concatenated_dfs.values():
            if not isinstance(df.index[0], str):
                raise AssertionError("Index is not md5sum")
            df["md5sum"] = df.index
            df.index.name = "md5sum"

        return concatenated_dfs

    @staticmethod
    def calculate_metrics_for_single_df(
        df: pd.DataFrame,
        task_name: str = "unknown task",
        logging_name: str = "df",
    ) -> Dict[str, float]:
        """Compute metrics for a single task's DataFrame.

        Expects a DataFrame with the following columns:
            md5sum, True class, predicted class, predicted probabilities

        Output is a dictionary of metrics with the following keys:
            Accuracy, F1_macro, AUC_micro, AUC_macro, count
        """
        df = df.copy()

        # Ensure 'True class' labels match (e.g. int or bool class labels)
        df["True class"] = df["True class"].astype(str).str.lower()
        df.columns = list(df.columns[:2]) + [
            label.lower() for i, label in enumerate(df.columns.str.lower()) if i > 1
        ]

        # One hot encode true class and get predicted probabilities
        # Reindex to ensure that the order of classes is consistent
        classes_order = df.columns[2:]
        onehot_true = (
            pd.get_dummies(df["True class"], dtype=int)
            .reindex(columns=classes_order, fill_value=0)
            .values
        )
        pred_probs = df[classes_order].values

        # Argmax encodings for accuracy/F1
        ravel_true = np.argmax(onehot_true, axis=1)
        ravel_pred = np.argmax(pred_probs, axis=1)

        # Core metrics
        task_metrics_dict: Dict[str, float | int] = {
            "Accuracy": accuracy_score(ravel_true, ravel_pred),
            "F1_macro": f1_score(
                ravel_true, ravel_pred, average="macro", zero_division="warn"
            ),  # type: ignore
            "count": len(df),
        }

        # --- AUC Handling ---
        unique_classes = np.unique(ravel_true)
        n_classes = unique_classes.shape[0]

        if n_classes < 2:
            logging.warning(
                "Cannot compute ROC AUC: only one class present in %s for %s (%s)",
                logging_name,
                task_name,
                df["True class"].unique(),
            )
            task_metrics_dict.update({"AUC_micro": np.nan, "AUC_macro": np.nan})
            return task_metrics_dict

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

                if len(classes_order) == 2:
                    # Binary classification: compute a single AUC directly.
                    # With complementary probabilities, per-class AUCs are identical,
                    # so micro/macro distinction is a meaningless artifact of the
                    # multiclass API. Report the same correct value for both.
                    binary_auc = roc_auc_score(onehot_true[:, 0], pred_probs[:, 0])
                    task_metrics_dict.update(
                        {  # type: ignore
                            "AUC_micro": binary_auc,
                            "AUC_macro": binary_auc,
                        }
                    )
                else:
                    # Multiclass: micro and macro are meaningfully different
                    task_metrics_dict.update(
                        {  # type: ignore
                            "AUC_micro": roc_auc_score(
                                onehot_true,
                                pred_probs,
                                multi_class="ovr",
                                average="micro",
                            ),
                            "AUC_macro": roc_auc_score(
                                onehot_true,
                                pred_probs,
                                multi_class="ovr",
                                average="macro",
                            ),
                        }
                    )
        except ValueError as err:
            if "multiclass" in str(err) or "Only one class" in str(err):
                logging.warning(
                    "Cannot compute ROC AUC for %s/%s: %s",
                    logging_name,
                    task_name,
                    err,
                )
                task_metrics_dict.update({"AUC_micro": np.nan, "AUC_macro": np.nan})
            else:
                err_msg = f"Unexpected AUC error in {logging_name} for {task_name}."
                logging.error(err_msg)
                logging.debug("columns: %s", df.columns)
                logging.debug("True class values: %s\n", df["True class"].value_counts())
                raise ValueError(err_msg) from err

        return task_metrics_dict

    @staticmethod
    def compute_split_metrics(
        all_split_dfs: Dict[str, Dict[str, pd.DataFrame]],
        concat_first_level: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute desired metrics for each split and classifier, accommodating different dictionary structures.

        This method supports two structures of input dictionaries:
        1. {split_name: {classifier_name: results_df}} when concat_first_level is False.
        2. {classifier_name: {split_name: results_df}} when concat_first_level is True.

        Args:
            all_split_dfs: A nested dictionary of DataFrames. The structure
                           depends on the concat_first_level flag. (Docstring corrected to use all_split_dfs)
            concat_first_level: A boolean flag that indicates the structure of the all_split_dfs dictionary.
                                If True, the first level is the classifier name. Otherwise, the first
                                level is the split name.

        Returns:
            A nested dictionary with metrics computed for each classifier and split. The structure is
            {split_name: {classifier_name: {metric_name: value}}}.
        """
        all_split_dfs = copy.deepcopy(all_split_dfs)
        split_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Reorganize the dictionary to always work with {split_name: {classifier_name: DataFrame}}
        if concat_first_level:
            temp_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
            for classifier, splits in all_split_dfs.items():
                for split_val, df_val in splits.items():
                    if split_val not in temp_dict:
                        temp_dict[split_val] = {}
                    temp_dict[split_val][classifier] = df_val

            all_split_dfs = temp_dict

        for split in [f"split{i}" for i in range(10)]:
            dfs = all_split_dfs[split]

            metrics: Dict[str, Dict[str, float]] = {}

            for task_name, df in dfs.items():
                metrics[
                    task_name
                ] = SplitResultsHandler().calculate_metrics_for_single_df(
                    df=df, task_name=task_name, logging_name=split
                )
                # --- END OF REFACTORED PART ---

            split_metrics[split] = metrics

        return split_metrics

    @staticmethod
    def invert_metrics_dict(
        metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Invert metrics dict so classifier name is parent key.
        Before: {split_name:{classifier_name:{metric_name:val}}}
        After: {classifier_name:{split_name:{metric_name:val}}}
        """
        # Check for correct input
        first_keys = list(metrics.keys())
        if not all("split" in key for key in first_keys):
            raise ValueError("Wrong input dict: first level keys don't contain 'split'.")

        # Invert
        new_metrics = defaultdict(dict)
        for split_name, split_metrics in metrics.items():
            for classifier_name, classifier_metrics in split_metrics.items():
                new_metrics[classifier_name][split_name] = classifier_metrics
        return dict(new_metrics)

    @staticmethod
    def extract_count_from_metrics(metrics) -> Dict[str, int]:
        """Extract total file count from metrics dict (sums on each split).

        Returns a dict {classifier_name: count}
        """
        new_metrics = copy.deepcopy(metrics)

        # Check for correct input
        first_keys = list(new_metrics.keys())
        if all("split" in key for key in first_keys):
            new_metrics = SplitResultsHandler.invert_metrics_dict(new_metrics)

        counts = defaultdict(int)
        for classifier_name, all_split_metrics in new_metrics.items():
            for _, split_metrics in all_split_metrics.items():
                counts[classifier_name] += split_metrics["count"]  # type: ignore

        return dict(counts)

    @staticmethod
    def general_split_metrics(
        results_dir: Path,
        merge_assays: bool,
        exclude_categories: List[str] | None = None,
        exclude_names: List[str] | None = None,
        include_categories: List[str] | None = None,
        include_names: List[str] | None = None,
        return_type: str = "both",
        mislabel_corrections: Tuple[Dict[str, str], Dict[str, Dict[str, str]]]
        | None = None,
        oversampled_only: bool | None = True,
        verbose: bool | None = False,
        min_pred_score: float | None = None,
    ) -> (
        Dict[str, Dict[str, Dict[str, float]]]
        | Dict[str, Dict[str, pd.DataFrame]]
        | Tuple[
            Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, pd.DataFrame]]
        ]
    ):
        """Create the content data for figure 2a. (get metrics for each task)

        Args:
            results_dir (Path): Directory containing the results. Needs to be parent over category folders.
            merge_assays (bool): Merge similar assays (rna-seq x2, wgbs x2)
            exclude_categories (List[str]): Task categories to exclude (first level directory names).
            exclude_names (List[str]): Names of folders to exclude (ex: 7c or no-mix).
            include_categories (List[str]): Task categories to include (first level directory names).
            include_names (List[str]): Names of folders to include (ex: 7c or no-mix).
            return_type (str): Type of data to return ('metrics', 'split_results', 'both').
            mislabel_corrections (Tuple[Dict[str, str], Dict[str, Dict[str, str]]]): ({md5sum: EpiRR_no-v},{label_category: {EpiRR_no-v: corrected_label}})
            oversampled_only (bool): Only include oversampled runs.
            verbose (bool): Print additional information.
            min_pred_score (float): Minimum prediction score to consider. Affects the metrics. Defaults to None.

        Returns:
            Union[Dict[str, Dict[str, Dict[str, float]]],
                Dict[str, Dict[str, pd.DataFrame]],
                Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, pd.DataFrame]]]]
                Depending on return_type, it returns:
                - 'metrics': A metrics dictionary with the structure {split_name: {task_name: metrics_dict}}
                - 'split_results': A split results dictionary with the structure {task_name: {split_name: split_results_df}}
                - 'both': A tuple with both dictionaries described above
        """
        if return_type not in ["metrics", "split_results", "both"]:
            raise ValueError(
                f"Invalid return_type: {return_type}. Choose from 'metrics', 'split_results', or 'both'."
            )

        all_split_results = {}
        split_results_handler = SplitResultsHandler()

        if mislabel_corrections:
            md5sum_to_epirr, epirr_to_corrections = mislabel_corrections
        else:
            md5sum_to_epirr = {}
            epirr_to_corrections = {}

        for parent, _, _ in os.walk(results_dir, followlinks=True):
            # Looking for oversampling only results
            parent = Path(parent)
            if "10fold" not in parent.name:
                if verbose:
                    print(f"Skipping {parent}: not 10fold")
                continue
            if parent.name != "10fold-oversampling" and oversampled_only:
                if verbose:
                    print(f"Skipping {parent}: not oversampled")
                continue

            if verbose:
                print(f"Checking {parent}")

            # Get the category + filter
            relpath = parent.relative_to(results_dir)
            category = relpath.parts[0].replace("_1l_3000n", "")
            if verbose:
                print(f"Checking category: {category}")
            if include_categories is not None:
                if not any(include_str in category for include_str in include_categories):
                    if verbose:
                        print(f"Skipping {category}: not in {include_categories}")
                    continue
            if exclude_categories is not None:
                if any(exclude_str in category for exclude_str in exclude_categories):
                    if verbose:
                        print(f"Skipping {category}: in {exclude_categories}")
                    continue

            # Get the rest of the name, ignore certain runs
            rest_of_name = list(relpath.parts[1:])
            for dir_name in ["10fold", "10fold-oversampling"]:
                try:
                    rest_of_name.remove(dir_name)
                except ValueError:
                    pass

            if len(rest_of_name) > 1:
                raise ValueError(
                    f"Too many parts in the name: {rest_of_name}. Path: {relpath}"
                )
            if rest_of_name:
                rest_of_name = rest_of_name[0]
            if verbose:
                print(f"Rest of name: {rest_of_name}")

            # Filter out certain runs
            if include_names is not None:
                if not any(name in rest_of_name for name in include_names):
                    if verbose:
                        print(
                            f"Skipping {category} {rest_of_name}: not in {include_names}"
                        )
                    continue
            if exclude_names is not None:
                if any(name in rest_of_name for name in exclude_names):
                    if verbose:
                        print(f"Skipping {category} {rest_of_name}: in {exclude_names}")
                    continue

            full_task_name = category
            if rest_of_name:
                full_task_name += f"_{rest_of_name}"

            # Get the split results
            if verbose:
                print(f"Getting split results for {full_task_name}")
            split_results = split_results_handler.read_split_results(parent)
            if not split_results:
                raise ValueError(f"No split results found in {parent}")

            if (
                ("sex" in full_task_name) or ("life_stage" in full_task_name)
            ) and mislabel_corrections:
                corrections = epirr_to_corrections[category]
                for split_name, results_df in list(split_results.items()):
                    current_true_class = results_df["True class"].to_dict()
                    new_true_class = {
                        k: corrections.get(md5sum_to_epirr[k], v)
                        for k, v in current_true_class.items()
                    }
                    results_df["True class"] = new_true_class.values()

                    split_results[split_name] = results_df

            if ("assay" in full_task_name) and merge_assays:
                for split_name, df in split_results.items():
                    try:
                        split_result_df = merge_similar_assays(df)
                    except ValueError as e:
                        print(f"Skipping {full_task_name} assay merging: {e}")
                        break
                    split_results[split_name] = split_result_df

            if min_pred_score:
                for split_name, df in split_results.items():
                    tmp_df = df.copy()
                    tmp_df = SplitResultsHandler.add_max_pred(tmp_df)
                    tmp_df = tmp_df[tmp_df["Max pred"] >= min_pred_score]
                    tmp_df = tmp_df.drop(columns=["Max pred"])
                    split_results[split_name] = tmp_df

            all_split_results[full_task_name] = split_results

        if return_type in ["metrics", "both"]:
            try:
                split_results_metrics = split_results_handler.compute_split_metrics(
                    all_split_results, concat_first_level=True
                )
            except KeyError as e:
                logging.error("KeyError: %s", e)
                logging.error("all_split_results: %s", all_split_results)
                logging.error("check folder: %s", results_dir)
                raise e

        if return_type == "metrics":
            # pylint: disable=possibly-used-before-assignment
            return split_results_metrics
        if return_type == "split_results":
            return all_split_results

        # the default return type is 'both'
        return split_results_metrics, all_split_results

    def obtain_all_feature_set_data(
        self,
        parent_folder: Path,
        merge_assays: bool,
        return_type: str,
        include_sets: List[str] | None = None,
        include_categories: List[str] | None = None,
        exclude_names: List[str] | None = None,
        oversampled_only: bool | None = True,
        verbose: bool | None = False,
    ) -> (
        Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
        | Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
    ):
        """
        Obtain either metrics or split results for all feature sets based on the specified return_type.

        Args:
            parent_folder (Path): The parent folder containing all feature set folders.
                Needs to be the parent of feature set folders.
            merge_assays (bool): Whether to merge similar assays (e.g., RNA-seq x2, WGBS x2).
            return_type (str): Type of data to return, either "metrics" or "split_results".
            include_sets (List[str] | None): Feature sets to include.
            include_categories (List[str] | None): Task categories to include.
            exclude_names (List[str] | None): Names of folders to exclude (e.g., 7c or no-mix).
            oversampled_only (bool): Only include oversampled runs.
            verbose (bool): Print additional information.

        Returns:
            Union[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
            A dictionary containing either metrics or results for all feature sets.
            Format for metrics: {feature_set: {task_name: {split_name: metric_dict}}}
            Format for split results: {feature_set: {task_name: {split_name: results_dataframe}}}
        """
        valid_return_types = ["metrics", "split_results"]
        if return_type not in valid_return_types:
            raise ValueError(
                f"Invalid return_type: {return_type}. Choose from {valid_return_types}."
            )

        all_data = {}
        if verbose:
            print(f"Checking folder: {parent_folder}")
        for folder in parent_folder.iterdir():
            if not folder.is_dir():
                continue

            feature_set = folder.name
            if include_sets is not None and feature_set not in include_sets:
                if verbose:
                    print(f"Skipping {feature_set}: not in 'include_sets' list.")
                continue

            try:
                # Fetch either metrics or split results based on the return_type
                split_data = self.general_split_metrics(
                    results_dir=folder,
                    merge_assays=merge_assays,
                    return_type=return_type,
                    include_categories=include_categories,
                    exclude_names=exclude_names,
                    oversampled_only=oversampled_only,
                    verbose=verbose,
                )

                # Invert if metrics, otherwise keep as is
                if return_type == "metrics":
                    inverted_data = self.invert_metrics_dict(split_data)  # type: ignore
                    all_data[feature_set] = inverted_data
                elif return_type == "split_results":
                    all_data[feature_set] = split_data  # type: ignore

            except ValueError as err:
                raise ValueError(f"Problem with {feature_set}") from err

        return all_data


def load_split_predictions(
    data_dir: Path,
    metadata_handler: MetadataHandler,
    metadata: Metadata,
    merge_assays: bool = False,
    expected_classes: list | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load, concatenate, and enrich split results in one call, for one classification task.

    If merge_assays is True, expects assay prediction scores columns.
    If it fails to find them, it will simply try to merge labels under the 'assay_epiclass' column.

    Args:
        data_dir: The directory containing the split results.
        metadata_handler: An instance of the MetadataHandler class.
        metadata: The metadata to join with the split results.
        merge_assays: Whether to merge similar assays.
        expected_classes: The expected classes for the predictions.
        depth: The depth to concatenate the split results.
    """
    split_dfs = SplitResultsHandler.read_split_results(data_dir)
    concat_df: pd.DataFrame = SplitResultsHandler.concatenate_split_results(split_dfs, depth=1)  # type: ignore
    concat_df = SplitResultsHandler.add_max_pred(
        concat_df, expected_classes=expected_classes
    )
    df = metadata_handler.join_metadata(concat_df, metadata)
    if merge_assays:
        try:
            df = merge_similar_assays(df)
        except ValueError as e:
            if "Wrong results dataframe" in str(e):
                if verbose:
                    print(
                        "Could not find column, trying to merge only assay labels and not prediction values."
                    )
                df = df.replace({ASSAY: ASSAY_MERGE_DICT}, inplace=False)
    return df


def create_mislabel_corrector(
    paper_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Obtain information necessary to correct sex and life_stage mislabels.

    Returns:
    - Dict[str, str]: {md5sum: EpiRR_no-v}
    - Dict[str, Dict[str, str]]: {label_category: {EpiRR_no-v: corrected_label}}
    """
    epirr_no_v = "EpiRR_no-v"
    # Associate epirrs to md5sums
    metadata = MetadataHandler(paper_dir).load_metadata("v2")
    metadata_df = pd.DataFrame.from_records(list(metadata.datasets))
    md5sum_to_epirr = metadata_df.set_index("md5sum")[epirr_no_v].to_dict()

    # Load mislabels
    epirr_to_corrections = {}
    metadata_dir = (
        paper_dir / "data" / "metadata" / "epiatlas" / "official" / "BadQual-mislabels"
    )

    sex_mislabeled = pd.read_csv(metadata_dir / "official_Sex_mislabeled.csv")
    epirr_to_corrections[SEX] = sex_mislabeled.set_index(epirr_no_v)[
        "EpiClass_pred_Sex"
    ].to_dict()

    life_stage_mislabeled = pd.read_csv(
        metadata_dir / "official_Life_stage_mislabeled.csv"
    )
    epirr_to_corrections[LIFE_STAGE] = life_stage_mislabeled.set_index(epirr_no_v)[
        "EpiClass_pred_Life_stage"
    ].to_dict()

    return md5sum_to_epirr, epirr_to_corrections


def extract_data_from_files(
    parent_folder: Path,
    file_pattern: str,
    search_line: str,
    extract_pattern: str,
    type_cast: type = int,
    unique: bool = True,
) -> Dict[str, Set[Any]]:
    """
    Extracts data from files matching a specific pattern within each directory of a parent folder.

    Args:
        parent_folder (Path): The directory containing subdirectories to search.
        file_pattern (str): Glob pattern for files to search within each subdirectory.
        search_line (str): Line identifier to search for data extraction.
        extract_pattern (str): Regex pattern to extract the desired data from the identified line.
        type_cast (type): Type to cast the extracted data to.

    Returns:
        Dict[str, Set[type]]: A dictionary with subdirectory names as keys and sets of extracted data as values.
    """
    data = defaultdict(set)
    for folder in parent_folder.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.rglob(file_pattern):
            with open(file, "r", encoding="utf8") as f:
                # print(f"Reading {file.name}")
                lines = [l.rstrip() for l in f if search_line in l]

            if not lines:
                # print(f"Skipping {file.name}, no relevant data found.")
                continue
            if len(lines) > 1 and unique:
                if len(set(lines)) == 1:
                    pass
                else:
                    raise ValueError(
                        f"Incorrect file reading, captured more than one unique line in {file.name}: {lines}"
                    )
            matches = [re.match(pattern=extract_pattern, string=line) for line in lines]
            try:
                extracted_data = [type_cast(match.group(1)) for match in matches]  # type: ignore
                # print(f"Extracted data: {extracted_data}")
            except AttributeError as err:
                raise ValueError(
                    f"Could not extract data from {file.name} using pattern {extract_pattern}."
                ) from err
            data[folder.name].update(extracted_data)
    return dict(data)


def extract_input_sizes_from_output_files(parent_folder: Path) -> Dict[str, Set[int]]:
    """Extracts model input sizes from output (.o) files."""
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.o",
        search_line="(1): Linear",
        extract_pattern=r".*in_features=(\d+).*",
        type_cast=int,
    )


def extract_node_jobs_from_error_files(parent_folder: Path) -> Dict[str, Set[int]]:
    """Extracts SLURM job IDs from error (.e) files."""
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.e",
        search_line="SLURM_JOB_ID",
        extract_pattern=r".*SLURM_JOB_ID\s+: (\d+)$",
        type_cast=int,
    )


def extract_experiment_keys_from_output_files(parent_folder: Path) -> Dict[str, Set[str]]:
    """Extracts experiment keys from output (.o) files."""
    search_line = "The current experiment key is"
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.o",
        search_line=search_line,
        extract_pattern=search_line + r" (\w{32})$",
        type_cast=str,
        unique=False,
    )


def add_second_highest_prediction(df: pd.DataFrame, pred_cols: List[str]) -> pd.DataFrame:
    """Return the DataFrame with a columns for the second highest prediction class.

    Adds columns:
    - '2nd pred class': The class with the second highest prediction.
    - '1rst/2nd prob diff': The difference between the highest and second highest prediction probabilities.
    - '1rst/2nd prob ratio': The ratio of the highest to second highest prediction probabilities.
    """
    # Convert the relevant columns to a numpy array
    predictions = df[pred_cols].values

    # Get the indices of the sorted values
    sorted_indices = np.argsort(predictions, axis=1)

    # The second highest will be at position -2 (second to last) in the sorted order
    second_highest_indices = sorted_indices[:, -2]

    # Map indices to column names
    second_highest_columns = np.array(pred_cols)[second_highest_indices]
    df["2nd pred class"] = second_highest_columns

    # Calculate the difference and ratio
    highest_probs = np.max(predictions, axis=1)
    second_highest_probs = predictions[
        np.arange(len(predictions)), second_highest_indices
    ]

    df["1st/2nd prob diff"] = highest_probs - second_highest_probs
    df["1st/2nd prob ratio"] = highest_probs / second_highest_probs
    return df


def display_perc(df: pd.DataFrame | pd.Series) -> None:
    """Display a DataFrame with percentages."""
    # pylint: disable=consider-using-f-string
    with pd.option_context("display.float_format", "{:.3f}".format):
        display(df)


def check_label_coherence(
    dataframe, categories, column_templates: Dict[str, str] | None = None
):
    """
    Check that the predicted and true labels for each category are consistent.

    Args:
        dataframe: DataFrame containing the predictions and true classes.
        categories: List of categories to check.
        column_templates: Dictionary mapping column types to template strings.
            Requires "True", "Predicted"
            If None, uses default templates.

    Raises ValueError if the predicted and true labels are not consistent.
    """
    if column_templates is None:
        column_templates = {
            "True": "True class ({})",
            "Predicted": "Predicted class ({})",
        }

    df = dataframe.copy()

    for col in categories:
        pred_col = column_templates["Predicted"].format(col)
        true_col = column_templates["True"].format(col)

        try:
            predicted_labels = df[pred_col].unique()
        except KeyError:
            print("KeyError:", pred_col)
            print("Attempting to use merged column name.")
            pred_col = f"Predicted class ({col}_merged)"
            predicted_labels = df[pred_col].unique()

        print(df[pred_col].value_counts(dropna=False), "\n")

        try:
            expected_labels = df[true_col].unique()
        except KeyError:
            print("KeyError:", true_col)
            print("Attempting to use merged column name.")
            true_col = column_templates["True"].format(f"{col}_merged")
            expected_labels = df[true_col].unique()

        print(df[true_col].value_counts(dropna=False), "\n")

        predicted_labels = set(predicted_labels)
        expected_labels = set(expected_labels)
        for unknown_label in ["unknown", "other", "indeterminate"]:
            expected_labels.discard(unknown_label)

        if any(l not in predicted_labels for l in expected_labels):
            if "life" in col:
                print("Need to merge expected or predicted life stages")
                print("Predicted life stages:", predicted_labels)
                print("Expected life stages:", expected_labels)
            else:
                raise ValueError(
                    f"Expected labels not matching with predicted labels in `{col}`:\nPredicted labels: {predicted_labels}\nExpected labels: {expected_labels}"
                )


def merge_life_stages(
    df: pd.DataFrame,
    lifestage_column_name: str = LIFE_STAGE,
    column_name_templates: List[str] | None = None,
    exact_replace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge perinatal stages into one category, for given column names.

    New columns for f"{LIFE_STAGE}_merged" will be added for each column name template.

    Args:
        df (pd.DataFrame): DataFrame to merge columns in.
        lifestage_column_name (str): Name of the life stage category.
        column_name_templates (List[str] | None): List of column name templates to merge.
            ex: ["{}", "True class ({})", "Predicted class ({})"]
        exact_replace (bool): If True, raise error if column not found or replacement already exists.
        verbose (bool): If True, print the column names that are being merged or skipped.

    Returns:
        pd.DataFrame: DataFrame with merged columns.

    Raises:
        KeyError: If exact_replace=True and column operations fail.
        ValueError: If remapping fails due to unmapped values.
    """
    df = df.copy(deep=True)

    # Set default column templates if not provided
    if column_name_templates is None:
        column_name_templates = [
            "{}",
            "True class ({})",
            "Predicted class ({})",
            "Max pred ({})",
        ]

    def _is_score_column(column_name: str) -> bool:
        """Check if column contains prediction scores that should be copied, not remapped."""
        return "max" in column_name.lower()

    def _check_column_availability(old_col: str, new_col: str) -> bool:
        """
        Check if column processing should proceed.

        Returns:
            bool: True if processing should continue, False if should skip.

        Raises:
            KeyError: If exact_replace=True and conditions aren't met.
        """
        # Case 1: Neither column exists
        if old_col not in df.columns and new_col not in df.columns:
            if exact_replace:
                raise KeyError(
                    f"Column `{old_col}` not found in dataframe, and replacement "
                    f"column `{new_col}` not found in dataframe."
                )
            if verbose:
                print(f"Column `{old_col}` not found in dataframe. Skipping.")
            return False

        # Case 2: Source column doesn't exist
        if old_col not in df.columns:
            if exact_replace:
                raise KeyError(f"Column `{old_col}` not found in dataframe.")
            if verbose:
                print(f"Column `{old_col}` not found in dataframe. Skipping.")
            return False

        # Case 3: Target column already exists
        if new_col in df.columns:
            if exact_replace:
                raise KeyError(f"Column `{new_col}` already exists in dataframe.")
            if verbose:
                print(f"Column `{new_col}` already exists. Skipping.")
            return False

        return True

    # Process each column template
    for column_template in column_name_templates:
        old_column_name = column_template.format(lifestage_column_name)
        new_column_name = old_column_name.replace(
            lifestage_column_name, f"{lifestage_column_name}_merged"
        )

        # Check if we should process this column
        if not _check_column_availability(old_column_name, new_column_name):
            continue

        if verbose:
            print(f"Processing: {old_column_name} -> {new_column_name}")

        # Handle prediction score columns (copy without remapping)
        if _is_score_column(old_column_name):
            df[new_column_name] = df[old_column_name].copy()
            if verbose:
                print("Copied scores (no remapping applied)")
            continue

        # Apply life stage remapping
        df[new_column_name] = df[old_column_name].map(LIFESTAGE_REMAPPER)

        # Validate that all values were successfully mapped
        unmapped_mask = df[new_column_name].isnull() & df[old_column_name].notnull()
        if unmapped_mask.any():
            unmapped_values = df.loc[unmapped_mask, old_column_name].unique().tolist()
            available_keys = list(LIFESTAGE_REMAPPER.keys())

            raise ValueError(
                f"Failed to remap all values in column `{old_column_name}`. "
                f"Unmapped values: {unmapped_values}. "
                f"Available remapper keys: {available_keys}"
            )

        if verbose:
            value_counts = df[new_column_name].value_counts(dropna=False)
            print(f"Remapped successfully. New categories:\n{value_counts}\n")

    return df


class TemporaryLogFilter:
    """Context manager for adding a filter to a logger."""

    def __init__(self, filter_obj, logger=None):
        self.logger = logger or logging.getLogger()
        self.filter = filter_obj

    def __enter__(self):
        self.logger.addFilter(self.filter)

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.removeFilter(self.filter)


def set_file_id(
    df: pd.DataFrame, input_col: str = "Unnamed: 0", output_col: str = "md5sum"
) -> pd.DataFrame:
    """Standardizes a filename column by extracting the prefix and ensuring it is the first column.

    This function renames a given column (`input_col`) by extracting the prefix (before `_`)
    and moves it to the first position in the DataFrame. If `input_col` and `output_col`
    are the same, the column is updated in place and repositioned.

    Handled filename cases:
    - recount3: sra.base_sums.SRP076599_SRR3669968.ALL_[resolution]_[filters].hdf5 -> SRR3669968
    - ENCODE/ChIP-Atlas: [file_db_accession]_[resolution]_[filters].hdf5 -> [file_db_accession]
    - EpiATLAS: [md5sum]_[resolution]_[filters].hdf5 -> [md5sum]


    Args:
        df (pd.DataFrame): The input DataFrame.
        input_col (str, optional): The name of the column to process. Defaults to "Unnamed: 0".
        output_col (str, optional): The name of the resulting column. Defaults to "md5sum".

    Returns:
        pd.DataFrame: A modified DataFrame with the processed column as the first column.

    Raises:
        KeyError: If `input_col` is not found in the DataFrame.
    """
    df = df.copy(deep=True)

    try:
        input_vals = df[input_col]
    except KeyError as err:
        raise KeyError(f"Column {input_col} not found.") from err

    if output_col in df.columns:
        df = df.drop(output_col, axis=1)

    new_ids = [
        input_val.split("_")[0]
        if input_val[0:3] != "sra"
        else input_val.split(".")[2].split("_")[1]
        for input_val in input_vals
    ]

    if len(set(new_ids)) != len(new_ids):
        raise ValueError("Produce non-unique ids. Review input column and code.")

    # Remove input_col and reinsert as the first column, regardless of whether input_col == output_col
    df[input_col] = new_ids
    df.insert(0, output_col, df.pop(input_col))

    return df


def print_column_content(df: pd.DataFrame, col: str) -> None:
    """Print absolute and relative count of a string column."""
    label_count_df = df[col].value_counts(dropna=False).to_frame()

    label_count_df["relative"] = label_count_df["count"] / len(df)

    label_count_df.loc["Total", "count"] = len(df)
    label_count_df.loc["Total", "relative"] = 1

    style_map = {
        "count": "{:.0f}",
        "relative": "{:.2%}",
    }

    display(label_count_df.style.format(style_map))  # type: ignore


def format_labels(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    For given column label, format the labels by replacing spaces by underscores,
    and converting to lowercase.
    """
    for col in columns:
        try:
            df[col] = df[col].str.replace(" ", "_").str.lower()
        except KeyError as err:
            raise KeyError(f"Column {col} not found.") from err
    return df


def rename_columns(
    df, remapper: Dict[str, str], exact_match: bool = False, verbose: bool = False
) -> pd.DataFrame:
    """Rename columns in a DataFrame using a dictionary of old to new column names.

    If exact_match is True, the columns will be renamed exactly as specified in the dictionary.
    If exact_match is False, the columns will be renamed using a regular expression to replace the all occurrences of a string to a new string.
    """
    if exact_match:
        if verbose:
            for old, new in remapper.items():
                if old in df.columns:
                    print(f"Renaming {old} to {new}")
                    df.rename(columns={old: new}, inplace=True)
                else:
                    print(f"Column {old} not found. Skipping.")
        else:
            df = df.rename(columns=remapper)
        return df

    for column in list(df.columns):
        for old, new in remapper.items():
            if re.search(old, column):
                new_column = re.sub(old, new, column)
                if verbose:
                    print(f"Renaming {column} to {new_column}")
                df = df.rename(columns={column: new_column})

    return df


def find_target_recall(
    df: pd.DataFrame,
    category_name: str,
    col_templates: Dict[str, str],
    class_of_interest: str,
    target_recall: float = 0.9,
    verbose: bool = False,
    iterations: int = 250,
    minimum_threshold: float = 0.5,
):
    """
    Find the first threshold such that recall ≥ target_recall
    among predictions with prediction score ≥ threshold.

    Args:
        df: DataFrame
        category_name: metadata category (e.g. 'biomaterial_type')
        col_templates: Column templates for 'true_col', 'pred_col', and 'max_pred' columns.
        class_of_interest: Label of class within category
        target_recall: The target recall value
        verbose: Print intermediate results
        iterations: Number of thresholds to try in between minimum_threshold and 1

    Returns:
        threshold, recall
    """
    if target_recall > 1.0 or target_recall < 0:
        raise ValueError(f"target_recall not between 0 and 1: {target_recall}")

    if minimum_threshold > 1.0 or minimum_threshold < 0:
        raise ValueError(f"minimum_threshold not between 0 and 1: {minimum_threshold}")

    if iterations < 10:
        raise ValueError(f"iterations must be at least 10: {iterations}")

    true_col = col_templates["True"].format(category_name)
    pred_col = col_templates["Predicted"].format(category_name)
    max_pred_col = col_templates["Max pred"].format(category_name)

    thresholds = np.linspace(minimum_threshold, 1.0, iterations)

    for t in thresholds:
        # Filtered subset
        sub_df = df[df[max_pred_col] >= t]

        # Number of true positives for class_of_interest in filtered set
        tp = (
            (sub_df[true_col] == class_of_interest)
            & (sub_df[pred_col] == class_of_interest)
        ).sum()

        # Number of all true class_of_interest samples in filtered set
        total = (sub_df[true_col] == class_of_interest).sum()

        recall = tp / total

        if verbose:
            print(f"{t:.3f}:{recall:.3f}")

        if recall >= target_recall:
            return t, recall

    return None, 0.0


def filter_biomat_LS(
    df: pd.DataFrame,
    biomaterial_cat_name: str,
    col_templates: Dict[str, str],
    predScore_threshold: float = 0.8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter biomaterials for to enable more life stage predictions.

    Remove samples expected to be from cell lines.

    For unknown biomat, retain samples where biomat predictions are not
    cell line with predScore > predScore_threshold.

    Args:
        df: DataFrame with biomaterial type predictions.
        biomaterial_cat_name: metadata category label (e.g. 'biomaterial_type')
        col_templates: Column templates for 'true_col', 'pred_col', and 'max_pred' columns.
        predScore_threshold: threshold for minimum prediction score
        verbose: Print intermediate results
    """
    unknown_values = ["unknown", "other", "indeterminate"]

    if verbose:
        print("Filtering biomaterial types for life stage predictions.")

    true_col = col_templates["True"].format(biomaterial_cat_name)
    pred_col = col_templates["Predicted"].format(biomaterial_cat_name)
    max_pred_col = col_templates["Max pred"].format(biomaterial_cat_name)

    if verbose:
        print(
            f"Before:\n{df[true_col].value_counts(dropna=False)}\n\n{df[pred_col].value_counts(dropna=False)}\n"
        )

    # First filter out cell lines
    df = df[~df[true_col].isin(["cell_line"])]

    # Then retain unknown that are probably not cell lines
    unknown_subset = df[df[true_col].isin(unknown_values)]

    if verbose:
        print(
            f"Unknown subset pre predScore filter:\n{unknown_subset[true_col].value_counts(dropna=False)}\n\n{unknown_subset[pred_col].value_counts(dropna=False)}\n"
        )

    # Only consider samples with predScore > predScore_threshold
    unknown_subset = unknown_subset[unknown_subset[max_pred_col] > predScore_threshold]

    if verbose:
        print(
            f"Unknown subset post 'predScore > {predScore_threshold}' filter:\n{unknown_subset[true_col].value_counts(dropna=False)}\n\n{unknown_subset[pred_col].value_counts(dropna=False)}\n"
        )

    unknown_subset = unknown_subset[(unknown_subset[pred_col] != "cell_line")]

    df = pd.concat([df, unknown_subset])

    return df


class PathChecker:
    """Class to verify path existences."""

    @staticmethod
    def _validate_path(
        path: Path | str, check_file: bool = False, check_dir: bool = False
    ) -> None:
        """Internal helper to validate paths."""
        path_obj = Path(path)

        if not path_obj.exists():
            type_name = "file" if check_file else "directory" if check_dir else "path"
            raise FileNotFoundError(f"{type_name.title()} does not exist: {path}")

        if check_file and not path_obj.is_file():
            raise IsADirectoryError(f"Not a file: {path}")
        if check_dir and not path_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

    @staticmethod
    def check_path_exists(path: Path | str) -> None:
        """Check if a path exists."""
        PathChecker._validate_path(path)

    @staticmethod
    def check_file(path: Path | str) -> None:
        """Check if a file exists."""
        PathChecker._validate_path(path, check_file=True)

    @staticmethod
    def check_directory(path: Path | str) -> None:
        """Check if a directory exists."""
        PathChecker._validate_path(path, check_dir=True)


def save_figure(
    fig: Any,  # go.Figure, but avoid importing plotly here
    logdir: Path | str,
    filename: str,
    scale: float = 2,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Write a Plotly figure to html + svg + png."""
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    fig.write_html(logdir / f"{filename}.html")
    fig.write_image(logdir / f"{filename}.svg", width=width, height=height)
    fig.write_image(logdir / f"{filename}.png", width=width, height=height, scale=scale)
