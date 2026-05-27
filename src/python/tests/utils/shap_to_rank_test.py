"""test module for shap_to_rank.py"""
from pathlib import Path

import numpy as np
import pytest

from epiclass.utils.shap.shap_to_rank import main


@pytest.fixture(name="temp_input_folder")
def temp_input_structure(tmpdir) -> Path:
    """Create a temporary input structure for testing."""
    parent_dir = Path(tmpdir)

    # Create input structure
    for split_nb in range(3):
        split_dir = parent_dir / f"split{split_nb}" / "shap"
        split_dir.mkdir(parents=True)

        # Generate test data
        N_samples = 3
        N_classes = 5
        evaluation_signal_ids = [f"md5_{split_nb}_{j:02d}" for j in range(N_samples)]
        classes = [(str(k), chr(97 + k)) for k in range(N_classes)]
        shap_values = [
            np.array(
                [
                    range(split_nb * 42 + j * 1000, split_nb * 42 + (j + 1) * 1000)
                    for j in range(N_samples)
                ]
            )
            for _ in range(N_classes)
        ]

        # Save npz file
        np.savez_compressed(
            file=split_dir / f"shap_evaluation_split{split_nb}.npz",
            evaluation_md5s=evaluation_signal_ids,
            shap_values=shap_values,
            classes=classes,
        )

        np.savez_compressed(
            file=split_dir / f"explainer_background_split{split_nb}.npz",
            background_md5s=evaluation_signal_ids,
            background_expectation=list(range(len(classes))),
            classes=classes,
        )

    return parent_dir


def test_shap_rank_conversion(temp_input_folder: Path):
    """Test the conversion of SHAP values to ranks."""
    if not temp_input_folder.exists():
        raise FileNotFoundError(f"Input folder {temp_input_folder} does not exist")

    # Run the main function
    main(temp_input_folder)

    # Check if the output file exists
    output_file = temp_input_folder / "shap_ranks" / "all_shap_abs_ranks.npz"
    assert output_file.exists()

    # Load the output file
    output_data = np.load(output_file, allow_pickle=True)

    # Check if all expected keys are present
    assert set(output_data.keys()) == {"ranks", "signal_ids", "classes"}

    # Verify concatenation order
    N_samples = 3
    N_classes = 5
    N_splits = 3
    expected_signal_ids = [
        f"md5_{i}_{j:02d}" for i in range(N_splits) for j in range(N_samples)
    ]
    assert np.array_equal(output_data["signal_ids"], expected_signal_ids)

    # Verify classes
    expected_classes = [(str(k), chr(97 + k)) for k in range(N_classes)]
    assert np.array_equal(output_data["classes"], expected_classes)

    # Verify ranks
    ranks = output_data["ranks"]
    total_samples = N_samples * N_splits
    assert ranks.shape == (
        N_classes,
        total_samples,
        1000,
    )  # 5 classes, 30 samples (3 splits * 10 samples), 1000 features

    # Check if ranks are properly computed
    for class_idx in range(N_classes):
        for sample_idx in range(total_samples):
            sample_ranks = ranks[class_idx, sample_idx]
            expected_ranks = np.argsort(
                np.argsort(-np.abs(range(sample_idx * 42, sample_idx * 42 + 1000)))
            )
            assert np.array_equal(sample_ranks, expected_ranks)

    # Check that last value of each sample has the highest rank (0)
    assert np.all(ranks[:, :, -1] == 0)

    # Check that first value of each sample has the lowest rank (999)
    assert np.all(ranks[:, :, 0] == 999)

    # # print first 10 ranks of each sample for each class
    # for class_idx in range(N_classes):
    #     print(f"Class {class_idx}")
    #     for sample_idx in range(total_samples):
    #         print(ranks[class_idx, sample_idx, :10])
    #     print()
