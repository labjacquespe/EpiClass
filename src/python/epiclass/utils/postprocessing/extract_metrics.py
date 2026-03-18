"""Extract metrics for other classifiers output files, or csv output of cometML for neural networks."""
import argparse
import re
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Add an argument parser"""
    # fmt: off
    parser = argparse.ArgumentParser(description="Extract validation metrics from files.")
    parser.add_argument(
        "input_file", type=Path, help="Input file path"
        )
    parser.add_argument(
        "file_type", type=str, choices=["other", "NN"],
        help='Type of file ("other" (for other classifiers) or "NN" (for neural network cometML export))',
    )
    parser.add_argument(
        "-o", "--output_file",
        type=Path,
        default=None,
        help="Output file path (default: metrics.csv)",
    )
    # fmt: on
    return parser.parse_args()


def extract_other_estimators_metrics(file_path: str) -> pd.DataFrame:
    """
    Extracts validation metrics for multiple classifiers from a log file.

    This function reads a log file, identifies lines that contain validation metrics,
    and stores these metrics in a DataFrame. The log file is expected to contain one
    or more sections for each classifier, each section containing lines for different
    splits and their associated validation metrics.

    Args:
        file_path (str): The path to the log file.

    Returns:
        pd.DataFrame: A DataFrame with one row for each metric of each split of each
            classifier. The DataFrame contains the following columns: 'classifier',
            'split', 'metric', and 'value'.
    """
    # Initialize an empty list to store the metrics
    metrics = []

    # Initialize a variable to store the current classifier
    classifier = None

    # Open the file and iterate over each line
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if the line matches the pattern "Using {classifier}"
            classifier_match = re.match(r"Using (\w+)\.", line)
            if classifier_match:
                classifier = classifier_match.group(1)
            # Check if the line matches the pattern "Split {number} metrics"
            split_match = re.match(r"Split (\d+) metrics:", line)
            if split_match:
                split_number = int(split_match.group(1))
                # The next five lines contain the metrics
                for j in range(1, 6):
                    metric_line = lines[i + j]
                    # The line has the format "{metric name}: {metric value}"
                    # Use a regular expression to extract the metric name and value
                    metric_match = re.match(r"Validation (\w+): (\d+\.\d+)", metric_line)
                    metric_name = metric_match.group(1)
                    metric_value = float(metric_match.group(2))
                    # Store the classifier, split number, metric name, and metric value
                    metrics.append(
                        {
                            "classifier": classifier,
                            "split": split_number,
                            "metric": metric_name,
                            "value": metric_value,
                        }
                    )
                # Skip the next five lines
                i += 6
            else:
                # Move to the next line
                i += 1

    # Convert the list of metrics into a DataFrame
    metrics_df = pd.DataFrame(metrics)

    return metrics_df


def extract_neural_network_metrics(file_path: str) -> pd.DataFrame:
    """
    Extracts validation metrics for a Neural Network classifier from a CSV file (comet ML export)

    This function reads a CSV file, extracts the split number and validation metrics,
    and stores these metrics in a DataFrame. The CSV file is expected to contain
    columns for different validation metrics and a column 'Name' that includes
    information about the split number.

    Args:
        file_path (str): The path to the CSV file.
        output_file_path (str): The path of the output file.

    Returns:
        pd.DataFrame: A DataFrame with one row for each metric of each split. The
            DataFrame contains the following columns: 'classifier', 'split', 'metric',
            and 'value'.
    """
    data_new = pd.read_csv(file_path)

    # Extract the split number from the 'Name' column
    data_new["split"] = data_new["Name"].str.extract(r"split(\d+)").astype(int)

    # Select the relevant columns and rename them to match the previous DataFrame
    metrics_df_new = data_new[
        [
            "split",
            "val_Accuracy",
            "val_Precision",
            "val_Recall",
            "val_F1Score",
            "val_MatthewsCorrCoef",
        ]
    ]

    # Unpivot the DataFrame from wide to long format
    metrics_df_new = metrics_df_new.melt(
        id_vars="split", var_name="metric", value_name="value"
    )

    # Remove the 'val_' prefix from the metric names
    metrics_df_new["metric"] = metrics_df_new["metric"].str.replace("val_", "")
    metrics_df_new["metric"].replace(
        {"F1Score": "F1_score", "MatthewsCorrCoef": "MCC"}, inplace=True
    )

    # Add the classifier type
    metrics_df_new["classifier"] = "Neural Network"

    # Reorder the columns to match the previous DataFrame
    metrics_df_new = metrics_df_new[["classifier", "split", "metric", "value"]]

    return metrics_df_new


def main():
    """Main function"""
    # Parse the arguments
    cli = parse_arguments()

    input_file = cli.input_file
    if not input_file.is_file():
        raise FileNotFoundError(f"File not found: {input_file}")

    if cli.output_file is None:
        output_file = input_file.parent / "metrics.csv"
    else:
        output_file = cli.output_file

    # Call the appropriate function based on the file type
    if cli.file_type == "other":
        df = extract_other_estimators_metrics(input_file)
    elif cli.file_type == "NN":
        df = extract_neural_network_metrics(input_file)
    else:
        raise ValueError(f"Unknown file type: {cli.file_type}")

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
