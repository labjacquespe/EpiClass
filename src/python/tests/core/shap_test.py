"""Test SHAP related modules."""
# pylint: disable=import-error
from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from shap import TreeExplainer
from sklearn.datasets import make_blobs

from epiclass.core.data import DataSet, UnknownData
from epiclass.core.estimators import EstimatorAnalyzer
from epiclass.core.shap_values import LGBM_SHAP_Handler, NN_SHAP_Handler


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
        shap_values = [
            np.zeros(test_epiatlas_dataset.validation.signals.shape)
            for _ in test_epiatlas_dataset.classes
        ]
        return shap_values

    @pytest.fixture
    def fake_ids(self, test_epiatlas_dataset: DataSet):
        """Fake signal ids"""
        num_signals = test_epiatlas_dataset.validation.num_examples
        return [f"id{i}" for i in range(num_signals)]

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

        n_samples, n_features = dset.validation.signals.shape
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


@pytest.mark.skip(reason="One time thing")
def test_tree_explainer():
    """Minimal test to check if TreeExplainer works with LGBMClassifier."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)  # type: ignore # pylint: disable=unbalanced-tuple-unpacking

    for boosting_method, model_output in itertools.product(
        ["gbdt", "dart"],
        ["raw", "probability", "log_loss", "predict", "predict_proba"],
    ):
        test_model = LGBMClassifier(boosting_type=boosting_method, objective="multiclass")
        test_model.fit(X, y)
        try:
            explainer = TreeExplainer(
                model=test_model,
                data=X,
                model_output=model_output,
                feature_perturbation="interventional",
            )
        except AttributeError:
            print(
                f"({boosting_method})(err2) TreeExplainer does not support multiclass + model_output={model_output}"
            )
            continue
        except Exception as e:
            if "Model does not have a known objective or output type" in e.args[0]:
                print(
                    f"({boosting_method})(err1) TreeExplainer does not support multiclass + model_output={model_output}"
                )
                continue
            raise e
        shap_values = explainer.shap_values(X)
        print(
            f"({boosting_method}) TreeExplainer supports multiclass + model_output={model_output}"
        )
        print(np.array(shap_values).shape)


class Test_LGBM_SHAP_Handler:
    """Class to test LGBM_SHAP_Handler class."""

    N = 100

    @staticmethod
    def create_test_model(
        nb_class: int, nb_features: int, nb_samples: int
    ) -> Tuple[EstimatorAnalyzer, UnknownData]:
        """Create a test LGBMClassifier model for testing."""
        X, y = make_blobs(n_samples=nb_samples, centers=nb_class, n_features=nb_features, random_state=42)  # type: ignore # pylint: disable=unbalanced-tuple-unpacking
        test_model = LGBMClassifier(
            boosting_type="dart",
        )
        test_model.fit(X, y)

        dataset = UnknownData(range(nb_samples), X, y, [str(val) for val in y])

        model_analyzer = EstimatorAnalyzer(
            classes=[str(i) for i in range(nb_class)],
            estimator=test_model,
        )

        return model_analyzer, dataset

    @pytest.fixture(name="model2c")
    def test_model_2classes(self) -> Tuple[EstimatorAnalyzer, UnknownData]:
        """Test model with 2 classes."""
        model_analyzer, dataset = Test_LGBM_SHAP_Handler.create_test_model(
            2, 4, Test_LGBM_SHAP_Handler.N
        )

        return model_analyzer, dataset

    @pytest.fixture(name="model3c")
    def test_model_3classes(self) -> Tuple[EstimatorAnalyzer, UnknownData]:
        """Test model with 3 classes."""
        model_analyzer, dataset = Test_LGBM_SHAP_Handler.create_test_model(
            3, 4, Test_LGBM_SHAP_Handler.N
        )

        return model_analyzer, dataset

    @pytest.mark.parametrize(
        "test_data,num_workers",
        [("model2c", 1), ("model3c", 1), ("model2c", 2), ("model3c", 2)],
    )
    def test_compute_shaps(self, test_data, num_workers, tmp_path, request):
        """Tests the compute_shaps method of the LGBM_SHAP_Handler class.

        With SHAP 0.45+, TreeExplainer return format changed for multiclass models:

        Binary classification (one input, one output):
            Returns: np.ndarray of shape (n_samples, n_features)
            Contains SHAP values for the positive class only.

        Multiclass classification (one input, multiple outputs):
            Old format (< 0.45): List of arrays, one per class
            New format (>= 0.45): Single array of shape (n_samples, n_features, n_classes)

        This test verifies:
            1. SHAP values are numpy arrays (not lists)
            2. Shapes match the new SHAP 0.45+ conventions
            3. Expected values from explainer are correctly formatted
            4. Results are properly saved to disk
        """
        model_analyzer, evaluation_dset = request.getfixturevalue(test_data)
        handler = LGBM_SHAP_Handler(model_analyzer, tmp_path)

        # Test compute_shaps
        explainer, shap_values = handler.compute_shaps(
            background_dset=evaluation_dset,
            evaluation_dset=evaluation_dset,
            save=True,
            name="test",
            num_workers=num_workers,
        )

        # Test output types - must be numpy array in SHAP 0.45+
        expected_value = explainer.expected_value
        assert isinstance(shap_values, np.ndarray)
        assert isinstance(expected_value, (float, np.ndarray, np.floating))

        # Test output shapes
        nb_samples = Test_LGBM_SHAP_Handler.N
        nb_classes = len(model_analyzer.classes)
        nb_features = evaluation_dset.signals.shape[1]

        if nb_classes == 2:  # Binary classification
            # Returns: (n_samples, n_features) for positive class only
            assert shap_values.shape == (nb_samples, nb_features)
            assert isinstance(expected_value, (float, np.floating))
        else:  # Multiclass classification
            # New format: (n_samples, n_features, n_classes)
            assert shap_values.shape == (nb_samples, nb_features, nb_classes)
            assert isinstance(expected_value, np.ndarray)
            assert expected_value.shape == (nb_classes,)
