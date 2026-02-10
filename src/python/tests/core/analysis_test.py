"""Tests for analysis.py, specifically logger=None (no CometML) support."""
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from epiclass.core.analysis import Analysis
from epiclass.core.data.dataset import DataSet
from epiclass.core.data.eager import KnownData
from epiclass.core.metadata import Metadata
from epiclass.core.model_pytorch import LightningDenseClassifier


@pytest.fixture(name="small_model")
def fixture_small_model():
    """Create a small model for testing."""
    mapping = {0: "classA", 1: "classB"}
    return LightningDenseClassifier(
        input_size=10,
        output_size=2,
        mapping=mapping,
        hparams={"learning_rate": 1e-3},
        hl_units=8,
        nb_layer=1,
    )


@pytest.fixture(name="dummy_datasets")
def fixture_dummy_datasets():
    """Create dummy train/val DataSet and TensorDatasets."""
    n_train, n_val, n_features = 20, 10, 10
    classes = ["classA", "classB"]

    rng = np.random.RandomState(42)
    train_x = rng.randn(n_train, n_features).astype(np.float32)
    train_y = rng.randint(0, 2, size=n_train)
    train_y_str = [classes[y] for y in train_y]
    train_ids = [f"md5_train_{i:03d}" for i in range(n_train)]

    val_x = rng.randn(n_val, n_features).astype(np.float32)
    val_y = rng.randint(0, 2, size=n_val)
    val_y_str = [classes[y] for y in val_y]
    val_ids = [f"md5_val_{i:03d}" for i in range(n_val)]

    meta = {
        md5: {"label": label}
        for md5, label in zip(train_ids + val_ids, train_y_str + val_y_str)
    }

    metadata = Metadata.from_dict(meta, allow_non_md5sum_index=True)

    train_data = KnownData(
        ids=train_ids, x=train_x, y=train_y, y_str=train_y_str, metadata=metadata
    )
    val_data = KnownData(
        ids=val_ids, x=val_x, y=val_y, y_str=val_y_str, metadata=metadata
    )

    dataset = DataSet(
        training=train_data,
        validation=val_data,
        test=KnownData.empty_collection(),
        sorted_classes=classes,
    )

    train_tensor = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_tensor = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    return dataset, train_tensor, val_tensor


class TestAnalysisWithoutLogger:
    """Test Analysis works fully with logger=None."""

    def test_metrics_no_logger(self, small_model, dummy_datasets):
        """Metrics should compute and print without a logger."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
        )
        train_metrics = analyzer.get_training_metrics(verbose=True)
        val_metrics = analyzer.get_validation_metrics(verbose=True)
        assert train_metrics is not None
        assert val_metrics is not None
        assert "MulticlassAccuracy" in train_metrics
        assert "MulticlassAccuracy" in val_metrics

    def test_write_prediction_no_logger(self, small_model, dummy_datasets, tmp_path):
        """Predictions should be written to disk without a logger."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
            save_dir=tmp_path,
        )
        analyzer.write_validation_prediction()
        pred_file = tmp_path / "validation_prediction.csv"
        assert pred_file.is_file()
        assert pred_file.stat().st_size > 0

    def test_write_prediction_explicit_path(self, small_model, dummy_datasets, tmp_path):
        """Predictions should be written to an explicit path without a logger or save_dir."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
        )
        pred_path = tmp_path / "my_predictions.csv"
        analyzer.write_validation_prediction(path=pred_path)
        assert pred_path.is_file()

    def test_write_prediction_no_path_no_save_dir_raises(
        self, small_model, dummy_datasets
    ):
        """Should raise ValueError when no path and no save_dir are set."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
        )
        with pytest.raises(ValueError, match="no path given and no save_dir"):
            analyzer.write_validation_prediction()

    def test_confusion_matrix_no_logger(self, small_model, dummy_datasets, tmp_path):
        """Confusion matrix should be saved without a logger."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
            save_dir=tmp_path,
        )
        analyzer.validation_confusion_matrix()

        # Check that CSV and PNG files were created
        csv_files = list(tmp_path.glob("*confusion_matrix*.csv"))
        png_files = list(tmp_path.glob("*confusion_matrix*.png"))
        assert len(csv_files) >= 1
        assert len(png_files) >= 1

    def test_confusion_matrix_explicit_path(self, small_model, dummy_datasets, tmp_path):
        """Confusion matrix should be saved to an explicit path without save_dir."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
        )
        mat_path = tmp_path / "my_confusion_matrix"
        analyzer.validation_confusion_matrix(path=mat_path)
        csv_files = list(tmp_path.glob("my_confusion_matrix*.csv"))
        assert len(csv_files) >= 1

    def test_confusion_matrix_no_path_no_save_dir_raises(
        self, small_model, dummy_datasets
    ):
        """Should raise ValueError when no path and no save_dir are set."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
        )
        with pytest.raises(ValueError, match="no path given and no save_dir"):
            analyzer.validation_confusion_matrix()

    def test_save_dir_parameter(self, small_model, dummy_datasets, tmp_path):
        """save_dir should be used as default output directory."""
        dataset, train_tensor, val_tensor = dummy_datasets
        analyzer = Analysis(
            model=small_model,
            datasets_info=dataset,
            logger=None,
            train_dataset=train_tensor,
            val_dataset=val_tensor,
            save_dir=tmp_path,
        )
        # Both should use tmp_path as output
        analyzer.write_validation_prediction()
        analyzer.validation_confusion_matrix()

        assert (tmp_path / "validation_prediction.csv").is_file()
        assert len(list(tmp_path.glob("*confusion_matrix*"))) >= 1
