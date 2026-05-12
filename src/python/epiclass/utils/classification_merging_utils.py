"""Utility functions for merging classification results."""
# pylint: disable=too-many-branches
from __future__ import annotations

import numpy as np
import pandas as pd


def sjoin(x):
    """join columns if column is not null"""
    return ";".join(x[x.notnull()].astype(str))


# Helper function to format numbers without decimal points for whole numbers
def clean_format(x: object) -> str:
    """
    Format a value to string, removing decimal points for whole numbers.
    Args:
        x: Any value that needs string formatting
    Returns:
        Formatted string representation of the value
    """
    if isinstance(x, (int, float)):
        try:
            if float(x).is_integer():
                return str(int(x))
        except (ValueError, AttributeError):
            pass
    return str(x)


def merge_dataframes(
    df1: pd.DataFrame, df2: pd.DataFrame, on: str = "md5sum", verbose: bool = False
) -> pd.DataFrame:
    """
    Merge two DataFrames by concatenating along the given column,
    otherwise it attemps to merge on md5sum, filename.
    It attempts to merge by aligning common columns
    and appending non-common columns.

    Column with same names get combined with ';' value separator.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame
    df2 (pd.DataFrame): The second DataFrame
    on (str, optional): The column to merge on. Defaults to "md5sum".
    verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Raises:
        ValueError: If no merge is possible.

    Returns:
    pd.DataFrame: Merged DataFrame. Index name is preserved if it was the same.
    """
    if verbose:
        print(f"Entering shapes: {df1.shape}, {df2.shape}")
    if on == "index" and df1.index.name != df2.index.name:
        raise ValueError(
            f"Index names are different: {df1.index.name} != {df2.index.name}"
        )

    successful_merge = False
    # Create a prioritized list of columns to try
    merge_cols = [on]
    for col in ["md5sum", "filename"]:
        if on != col:
            merge_cols.append(col)

    for merge_col in merge_cols:
        try:
            if merge_col == "index":
                result = pd.merge(
                    df1,
                    df2,
                    left_index=True,
                    right_index=True,
                    how="outer",
                    suffixes=("", "_merge"),
                )
            else:
                result = pd.merge(
                    df1, df2, on=merge_col, how="outer", suffixes=("", "_merge")
                )
            successful_merge = True
            break
        except (KeyError, ValueError):
            continue
    if not successful_merge:
        raise ValueError(f"Could not merge on any of the columns: {merge_cols}")

    if verbose:
        print(f"Output shape 1 (After pd.merge): {result.shape}")

    # Combine different values with a separator
    dup_cols = [name for name in result.columns if name.endswith("_merge")]
    for dup_col in dup_cols:
        normal_col = dup_col.replace("_merge", "")

        # For rows where both columns have values and they're different
        # combine them with a semicolon
        both_exist_mask = result[normal_col].notna() & result[dup_col].notna()
        different_vals_mask = both_exist_mask & (result[normal_col] != result[dup_col])

        if different_vals_mask.any():
            # Explicitly cast to object to avoid FutureWarning when assigning string values
            # (the semicolon separator) to a potentially numeric column
            result[normal_col] = result[normal_col].astype(object)

        # Combine different values with semicolon separator and clean formatting
        result.loc[different_vals_mask, normal_col] = (
            result.loc[different_vals_mask, normal_col].apply(clean_format)
            + ";"
            + result.loc[different_vals_mask, dup_col].apply(clean_format)
        )

        # Fill missing values in normal column from duplicate column
        missing_mask = result[normal_col].isna() & result[dup_col].notna()
        if missing_mask.any():
            # Also respect stricter dtypes
            if result[normal_col].dtype != object and result[dup_col].dtype == object:
                result[normal_col] = result[normal_col].astype(object)

        result.loc[missing_mask, normal_col] = result.loc[missing_mask, dup_col]

        # Drop the duplicate column
        result = result.drop(columns=[dup_col])

    if verbose:
        print(f"Output shape 2 (After merging dups columns): {result.shape}")

    return result


def merge_two_columns(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Return update IN-PLACE dataframe that merged values of col1 and col2, only if they are complementary."""
    df[col1].replace(np.nan, None, inplace=True)
    df[col2].replace(np.nan, None, inplace=True)
    for idx, (val1, val2) in enumerate(df[[col1, col2]].values):
        if val1 is not None and val2 is not None and val1 != val2:
            raise ValueError(
                f"Both {col1} and {col2} are not None in some rows: {idx} - {val1} != {val2}"
            )
    df[col1].update(df.pop(col2))
    return df


def remove_pred_vector(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove the prediction vector from a result dataframe.

    If the "files/epiRR" columns does not exist, it will remove everything after the "1rst/2nd prob ratio" column.
    If there is any metadata after that column, it will also be removed.
    """

    # Prediction vector should be between these columns
    col1 = "1rst/2nd prob ratio".lower()
    col2 = "files/epiRR".lower()

    # column and prediction are transfwred to lowercase to avoid weird mismatches (e.g. with boolean labels)
    column_names = df.columns.str.lower()

    cut_pos_1 = column_names.get_loc(col1)
    try:
        cut_pos_2 = column_names.get_loc(col2)
    except KeyError:
        # Try to determine if the dataframe is already reduced
        predict_labels = df["Predicted class"].astype(str).str.lower().unique()
        cut_pos_2 = None
        for predict_label in predict_labels:
            if predict_label in column_names:
                # Not a reduced dataframe
                cut_pos_2 = df.shape[1]
                if verbose:
                    print(
                        f"No files/epiRR column, removing everything after '{col1}' column"
                    )
                break

        if cut_pos_2 is None:
            if verbose:
                print("df seems already reduced")
            return df

    df = df.drop(df.columns[cut_pos_1 + 1 : cut_pos_2], axis=1)
    return df
