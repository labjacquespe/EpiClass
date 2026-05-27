"""Workbook to quantify bias present in metadata
Q: Can you identify certain labels by using other metadata
e.g. find cell type using project+assay+other
"""

# pylint: disable=import-error, use-dict-literal, invalid-name
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.utils.notebooks.paper.paper_utilities import (
    ASSAY,
    CELL_TYPE,
    LIFE_STAGE,
    SEX,
    MetadataHandler,
    SplitResultsHandler,
    create_mislabel_corrector,
)


def define_input_bias_categories(target_category: str) -> List[List[str]]:
    """Define bias categories used for bias analysis.

    Args:
        target_category (str): Classification target category. Is excluded from input lists.

    Returns:
        List[List[str]]: List of bias categories.
    """
    bias_categories_1 = [ASSAY, "project", "harmonized_biomaterial_type", CELL_TYPE]
    bias_categories_2 = [
        ASSAY,
        "project",
        "harmonized_biomaterial_type",
        CELL_TYPE,
        LIFE_STAGE,
    ]
    bias_categories_3 = [
        ASSAY,
        "project",
        "harmonized_biomaterial_type",
        CELL_TYPE,
        SEX,
    ]
    bias_categories_4 = [
        ASSAY,
        "project",
        "harmonized_biomaterial_type",
        CELL_TYPE,
        SEX,
        LIFE_STAGE,
    ]

    all_bias_categories = [
        bias_categories_1,
        bias_categories_2,
        bias_categories_3,
        bias_categories_4,
    ]
    for bias_categories in all_bias_categories:
        try:
            bias_categories.remove(target_category)
        except ValueError:
            pass
    return all_bias_categories


def create_models() -> List:
    """Create models for bias analysis."""
    lr_model_1 = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    lr_model_2 = OneVsRestClassifier(
        LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(kernel="linear", random_state=42)
    svm_model_rbf = SVC(kernel="rbf", random_state=42)
    return [lr_model_1, lr_model_2, rf_model, svm_model, svm_model_rbf]


def filter_samples(
    metadata_df: pd.DataFrame,
    target_category: str,
    md5_set: Set[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter samples based on the output category to match the original training set."""
    df = metadata_df.copy(deep=True)

    if "md5sum" not in df.columns:
        df["md5sum"] = df.index

    df = df[df["md5sum"].isin(md5_set)]

    if verbose:
        print("Metadata shape:", metadata_df.shape)
        print("Filtered shape:", df.shape)
        print(df[target_category].value_counts())

    return df  # type: ignore


def find_max_bias(
    filtered_metadata_df: pd.DataFrame, target_category: str, verbose: bool = True
) -> Dict[Tuple[str, ...], float]:
    """Find the bias categories that provide the highest accuracy for the target category."""
    max_bias_dict = {}
    for bias_categories in define_input_bias_categories(target_category):
        print(f"Using bias categories: {bias_categories}")
        X = filtered_metadata_df[bias_categories]
        y = filtered_metadata_df[target_category]

        # one-hot encode the data
        X_encoded = OneHotEncoder().fit_transform(X).toarray()  # type: ignore
        y_encoded = LabelEncoder().fit_transform(y)

        max_acc = 0
        for model in create_models():
            scores = cross_val_score(
                model, X_encoded, y_encoded, cv=10, scoring="accuracy", n_jobs=-1
            )
            if verbose:
                print(f"Model: {model}")
                print(f"Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
            if np.mean(scores) > max_acc:
                max_acc = np.mean(scores)
                max_bias_dict[tuple(bias_categories)] = max_acc

    return max_bias_dict


def compute_all_max_bias(
    metadata_df: pd.DataFrame,
    target_categories: List[str],
    md5s_to_include: Dict[str, Set[str]],
    avg_observed_acc: Dict[str, float],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute the max metadata bias for all target categories."""
    final_results: Dict[str, Any] = {}
    for target_category in target_categories:
        if verbose:
            print(f"Target category: {target_category}")

        filtered_metadata_df = filter_samples(
            metadata_df=metadata_df,
            target_category=target_category,
            md5_set=md5s_to_include[target_category],
        )

        max_bias_dict = find_max_bias(filtered_metadata_df, target_category)
        max_bias_cats, max_bias_acc = max(max_bias_dict.items(), key=lambda x: x[1])
        if verbose:
            print(f"Max bias categories: {max_bias_cats}")
            print(f"Max bias acc: {max_bias_acc}\n")

        MLP_acc = avg_observed_acc[target_category]

        acc_to_compare = [
            acc for cat, acc in avg_observed_acc.items() if cat in max_bias_cats
        ]
        avg_MLP_acc = np.mean(acc_to_compare)
        max_acc_with_bias = max_bias_acc * avg_MLP_acc

        if verbose:
            print("CLASSIFICATION ACCURACY")
            print(f"Average {target_category} observed acc: {MLP_acc:.1%}")
            print(f"Average MLP acc on bias categories: {avg_MLP_acc:.1%}")
            print(
                f"Max avg acc with bias from ({max_bias_cats}): {max_acc_with_bias:.1%}"
            )
            print(f"Not accounted for: {MLP_acc - max_acc_with_bias:.1%}\n")

        final_results[target_category] = {
            "max_bias_cats": max_bias_cats,
            "max_bias_acc": max_bias_acc,
            "MLP_acc": MLP_acc,
            "bias_avg_MLP_acc": avg_MLP_acc,
            "max_bias_acc_corrected": max_acc_with_bias,
            "acc_diff": MLP_acc - max_acc_with_bias,
        }

    return final_results


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "training_results_dir", type=DirectoryChecker(), help="Directory where models cross-validation results are stored."
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="A directory for the results."
    )
    arg_parser.add_argument(
        "paper_dir", type=DirectoryChecker(),
        help="Paper data directory (contains data/metadata/epiatlas/... and BadQual-mislabels/)."
    )
    arg_parser.add_argument(
        "--metadata-version", default="v2", choices=["v1", "v2", "v2-encode"],
        help="Metadata version key understood by MetadataHandler. Default: v2."
    )
    arg_parser.add_argument("--input-only", action="store_true", default=False)
    # fmt: on
    return arg_parser.parse_args()


def main():
    """Main function."""
    cli_args = parse_arguments()
    training_results_dir = cli_args.training_results_dir
    logdir = cli_args.logdir
    paper_dir = cli_args.paper_dir
    metadata_version = cli_args.metadata_version

    metadata_handler = MetadataHandler(paper_dir)
    metadata_df = metadata_handler.load_metadata_df(metadata_version)
    metadata = metadata_handler.load_metadata(metadata_version)

    split_results_handler = SplitResultsHandler()

    # Gather observed MLP results
    exclusion = ["cancer", "random", "track", "disease", "second", "end"]
    exclude_names = ["chip", "no-mixed", "ct", "7c"]

    all_split_results = split_results_handler.general_split_metrics(
        results_dir=training_results_dir,
        exclude_categories=exclusion,
        exclude_names=exclude_names,
        merge_assays=True,
        mislabel_corrections=create_mislabel_corrector(paper_dir),
        return_type="split_results",
    )

    concat_split_results: Dict[
        str, pd.DataFrame
    ] = split_results_handler.concatenate_split_results(
        all_split_results, concat_first_level=True  # type: ignore
    )

    for cat_name, df in list(concat_split_results.items()):
        new_df = metadata_handler.join_metadata(df, metadata)
        concat_split_results[cat_name] = new_df

    # Only keep input samples
    if cli_args.input_only:
        for cat_name, df in list(concat_split_results.items()):
            new_df = df[df[ASSAY] == "input"]
            concat_split_results[cat_name] = new_df

    avg_input_acc = {}
    for cat_name, df in list(concat_split_results.items()):
        filtered_df = df
        acc = (filtered_df["True class"] == filtered_df["Predicted class"]).sum() / len(
            filtered_df
        )
        avg_input_acc[cat_name] = acc

    print("Avg MLP acc:", avg_input_acc)

    avg_input_acc[SEX] = avg_input_acc["harmonized_donor_sex_w-mixed"]
    concat_split_results[SEX] = concat_split_results["harmonized_donor_sex_w-mixed"]

    avg_input_acc[ASSAY] = avg_input_acc["assay_epiclass_11c"]
    concat_split_results[ASSAY] = concat_split_results["assay_epiclass_11c"]

    # Compute max bias and compare
    target_categories = [
        "project",
        "harmonized_biomaterial_type",
        CELL_TYPE,
        SEX,
        LIFE_STAGE,
    ]

    md5s_to_include = {
        cat: set(concat_split_results[cat]["md5sum"]) for cat in target_categories
    }
    final_results = compute_all_max_bias(
        metadata_df=metadata_df,
        target_categories=target_categories,
        md5s_to_include=md5s_to_include,
        avg_observed_acc=avg_input_acc,
    )

    final_results_df = pd.DataFrame.from_dict(final_results, orient="index")

    filename = "metadata_bias_analysis_results"
    if cli_args.input_only:
        filename += "_input_only"
    final_results_df.to_csv(logdir / f"{filename}.csv")


if __name__ == "__main__":
    main()
