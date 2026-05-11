"""Test SHAP related modules."""
# pylint: disable=import-error
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from epiclass.core.data.dataset import DataSet
from epiclass.core.shap_values import NN_SHAP_Handler


class Test_NN_SHAP_Handler:
    """Class to test NN_SHAP_Handler class."""

    @pytest.fixture
    def logdir(self, mk_logdir) -> Path:
        """Test logdir"""
        return mk_logdir("shap")

    @pytest.fixture
    def handler(self, logdir: Path, test_NN_model) -> NN_SHAP_Handler:
        """NN_SHAP_Handler instance"""
        return NN_SHAP_Handler(test_NN_model, logdir)

    @pytest.fixture
    def mock_shap_values(self, test_epiatlas_dataset: DataSet) -> List[np.ndarray]:
        """Mock shape values for evaluation on two examples."""
        val_signals, _ = test_epiatlas_dataset.validation.materialize()
        shap_values = [np.zeros(val_signals.shape) for _ in test_epiatlas_dataset.classes]
        return shap_values

    @pytest.fixture
    def fake_ids(self, test_epiatlas_dataset: DataSet):
        """Fake signal ids"""
        num_signals = test_epiatlas_dataset.validation.num_examples
        return [f"id{i}" for i in range(num_signals)]

    @pytest.mark.slow
    def test_compute_shaps(
        self, handler: NN_SHAP_Handler, test_epiatlas_dataset: DataSet
    ):
        """Test shapes of return of compute_shaps method.

        With SHAP 0.45+, for models with one input and multiple outputs,
        shap_values changed from list to np.ndarray:

        Old format (< 0.45):
            List of arrays, one per class: [array(n_samples, n_features), ...]

        New format (>= 0.45):
            Single array: (n_samples, n_features, n_classes)

        Test validates that:
            - Output is a numpy array (not list)
            - Shape matches (n_samples, n_features, n_classes)
            - Can access individual sample SHAP values via shap_values[i]
        """
        dset = test_epiatlas_dataset
        _, shap_values = handler.compute_shaps(
            background_dset=dset.train, evaluation_dset=dset.validation, save=False
        )

        val_signals, _ = dset.validation.materialize()
        n_samples, n_features = val_signals.shape
        n_classes = len(handler.model_classes)

        # New SHAP 0.45+ format: single numpy array
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == (n_samples, n_features, n_classes)

        # Accessing first sample gives SHAP values for all features and classes
        assert shap_values[0].shape == (n_features, n_classes)

        print(f"shap_values.shape = {shap_values.shape}")
        print(f"shap_values[0].shape = {shap_values[0].shape}")

    def test_save_load_csv(self, handler: NN_SHAP_Handler, mock_shap_values, fake_ids):
        """Test pickle save/load methods."""
        shaps = mock_shap_values[0]
        path = handler.saver.save_to_csv(shaps, fake_ids, name="test")

        data = handler.saver.load_from_csv(path)
        assert list(data.index) == fake_ids
        assert np.array_equal(shaps, data.values)

    def test_save_to_csv_list_input(
        self, handler: NN_SHAP_Handler, mock_shap_values, fake_ids
    ):
        """Test effect of list input."""
        shap_values_matrix = [mock_shap_values[0]]
        name = "test_csv"

        with pytest.raises(ValueError):
            handler.saver.save_to_csv(shap_values_matrix, fake_ids, name)  # type: ignore

    def test_create_filename(self, handler: NN_SHAP_Handler):
        """Test filename creation method. Created by GPT4 lol."""
        ext = "pickle"
        name = "test_name"

        filename = handler.saver._create_filename(  # pylint: disable=protected-access
            ext, name
        )
        assert filename.name.startswith(f"shap_{name}_")
        assert filename.name.endswith(f".{ext}")
        assert filename.parent == Path(handler.logdir)
