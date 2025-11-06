"""Test module for the ConfusionMatrixWriter class."""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from epiclass.core.confusion_matrix import ConfusionMatrixWriter

THIS_FILE = Path(__file__).resolve()


@pytest.fixture
def output_dir():
    """Fixture to create the output directory, optionally cleaned up after the test."""
    dir_path = THIS_FILE.parent / "test_matrix"
    dir_path.mkdir(exist_ok=True)
    yield dir_path  # provide the path to the test

    # Optional cleanup: only remove if KEEP_TEST_OUTPUT is not set
    keep_output = os.getenv("KEEP_TEST_OUTPUT", "0")  # default is "0" (False)
    if keep_output.lower() not in ("1", "true", "yes") and dir_path.exists():
        print(f"Removing {dir_path}")
        shutil.rmtree(dir_path)


@pytest.fixture(params=[2, 5, 15, 25])
def sklearn_confusion_matrix(request):
    """
    Generates a confusion matrix for a simulated classification problem with the given number of classes.

    Args:
        num_classes (int): The number of classes in the classification problem.
        num_samples (int): The number of samples to simulate.

    Returns:
        np.ndarray: The confusion matrix.
    """
    num_samples = 500
    num_classes = request.param
    true_labels = np.random.randint(0, num_classes, size=num_samples)
    predicted_labels = np.random.randint(0, num_classes, size=num_samples)
    return confusion_matrix(true_labels, predicted_labels)


@pytest.mark.skip(reason="Visual test - enable when needed")
@pytest.mark.parametrize("name_base", ["class", "reallylonglabelnameforrealaaaahhhh"])
def test_to_png(sklearn_confusion_matrix: np.ndarray, name_base: str, output_dir: Path):
    """Tests the to_png method of the ConfusionMatrixWriter class."""
    labels = [f"{name_base} {i}" for i in range(sklearn_confusion_matrix.shape[0])]
    cm = ConfusionMatrixWriter(labels, sklearn_confusion_matrix)

    output = output_dir / f"test_c{len(labels)}_{name_base}.png"
    print(f"Writing confusion matrix to: {output}")
    cm.to_png(output)

    assert output.exists()
