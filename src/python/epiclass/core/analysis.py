"""Module containing result analysis code.

Logs results to an experiment logger (e.g. CometML) when available,
and writes confusion matrices and prediction tables to files.
All functionality works with logger=None (no remote logging).
"""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor
from torch.utils.data import Dataset

from epiclass.core.confusion_matrix import ConfusionMatrixWriter
from epiclass.core.data.dataset import DataSet
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.types import TensorData


class Analysis:
    """Class containing main analysis methods desired.

    Works with or without a logger. When logger is None, metrics are
    printed but not logged, and assets are written to disk but not uploaded.
    """

    def __init__(
        self,
        model: LightningDenseClassifier,
        datasets_info: DataSet,
        logger: Optional[pl.loggers.CometLogger] = None,  # type: ignore
        train_dataset: Optional[TensorData] = None,
        val_dataset: Optional[TensorData] = None,
        test_dataset: Optional[TensorData] = None,
        save_dir: Optional[Path] = None,
    ):
        self._model = model
        self._classes = sorted(list(self._model.mapping.values()))
        self._logger = logger

        # Determine save directory: explicit > logger > None
        if save_dir is not None:
            self._save_dir = Path(save_dir)
        elif self._logger is not None and hasattr(self._logger, "save_dir"):
            self._save_dir = Path(self._logger.save_dir)
        else:
            self._save_dir = None

        # Per-split sample ids; only used to label rows in the prediction CSV.
        self._ids_dict = {
            name: list(split.ids) if split.num_examples else []
            for name, split in (
                ("training", datasets_info.train),
                ("validation", datasets_info.validation),
                ("test", datasets_info.test),
            )
        }

        # Torch Dataset objects driving inference (TensorDataset, LazyHdf5Dataset, ...).
        self._train = train_dataset
        self._val = val_dataset
        self._test = test_dataset

    def _log_metrics(self, metric_dict, prefix=""):
        """Log metrics to experiment logger. (key: tensor(val))"""
        if self._logger is None:
            return
        for metric, val in metric_dict.items():
            name = f"{prefix[0:3]}_{metric}"
            self._logger.experiment.log_metric(name, val.item())

    @staticmethod
    def print_metrics(metric_dict, name):
        """Print metrics from TorchMetrics dict."""
        print(f"--- {name} METRICS ---")
        vals = []
        for metric, val in metric_dict.items():
            str_val = f"{val.item():.3f}"
            print(metric, str_val)
            vals.append(str_val)
        print(*vals)

    def _generic_metrics(self, dataset, name, verbose):
        """General treatment to compute and print metrics"""
        if dataset is None:
            print(f"Cannot compute {name} metrics : No {name} dataset given")
            metrics_dict = None
        else:
            metrics_dict = self._model.compute_metrics(dataset)
            if self._logger is not None:
                self._log_metrics(metrics_dict, prefix=name)
            if verbose:
                Analysis.print_metrics(metrics_dict, name=f"{name} set")
        return metrics_dict

    def get_training_metrics(self, verbose=True):
        """Compute and print training set metrics."""
        return self._generic_metrics(self._train, "training", verbose)

    def get_validation_metrics(self, verbose=True):
        """Compute and print validation set metrics."""
        return self._generic_metrics(self._val, "validation", verbose)

    def get_test_metrics(self, verbose=True):
        """Compute and print test set metrics."""
        return self._generic_metrics(self._test, "test", verbose)

    def _generic_write_prediction(
        self, to_predict: TensorData | None, name, path, verbose=True
    ) -> Optional[Path]:
        """General treatment to write predictions
        Name can be {training, validation, test}.

        Returns path to written file.

        to_predict: Object that contains samples to predict.
        """
        if path is None:
            if self._save_dir is None:
                raise ValueError(
                    f"Cannot write {name} predictions: no path given and no save_dir available."
                )
            path = self._save_dir / f"{name}_prediction.csv"

        if to_predict is None:
            print(f"Cannot compute {name} predictions : No {name} dataset given")
            return None

        if isinstance(to_predict, Dataset):
            preds, targets = self._model.compute_predictions_from_dataset(to_predict)
            str_targets = [self._model.mapping[int(val.item())] for val in targets]
        elif isinstance(to_predict, Tensor):
            preds = self._model.compute_predictions_from_features(to_predict)
            str_targets = ["Unknown" for _ in range(to_predict.size(dim=1))]
        else:
            raise ValueError(
                f"Cannot compute {name} predictions : to_predict should be either Dataset or Tensor, but got {type(to_predict)}"
            )

        write_pred_table(
            predictions=preds,
            str_preds=[
                self._model.mapping[int(val.item())]
                for val in torch.argmax(preds, dim=-1)
            ],
            str_targets=str_targets,
            signal_ids=self._ids_dict[name],
            classes=self._classes,
            path=path,
        )
        if self._logger is not None:
            self._logger.experiment.log_asset(
                file_data=path, file_name=f"{name}_prediction"
            )

        if verbose:
            print(f"'{path.name}' written to '{path.parent}'")

        return path

    def write_training_prediction(self, path=None):
        """Compute and write training predictions to file."""
        self._generic_write_prediction(self._train, name="training", path=path)

    def write_validation_prediction(self, path=None):
        """Compute and write validation predictions to file."""
        self._generic_write_prediction(self._val, name="validation", path=path)

    def write_test_prediction(self, path=None):
        """Compute and write test predictions to file.
        Test predictions do not include any "True class" column, as the true labels are unknown.
        """
        pred_path = self._generic_write_prediction(self._test, name="test", path=path)
        # Remove 'True class' which is just the first class repeated
        if pred_path is not None:
            df = pd.read_csv(pred_path, index_col=0)
            df.drop(columns=["True class"], inplace=True)
            df.to_csv(pred_path, encoding="utf8")

    def _generic_confusion_matrix(self, dataset: TensorData | None, name) -> np.ndarray:
        """General treatment to write confusion matrices."""
        if dataset is None:
            raise ValueError(
                f"Cannot compute {name} confusion matrix : No {name} dataset given"
            )
        if isinstance(dataset, Tensor):
            raise ValueError(
                f"Cannot compute {name} confusion matrix : No targets in given dataset."
            )

        preds, targets = self._model.compute_predictions_from_dataset(dataset)

        final_pred = torch.argmax(preds, dim=-1)

        mat = torchmetrics.functional.confusion_matrix(
            preds=final_pred,
            target=targets,
            num_classes=len(self._classes),
            normalize=None,
            task="multiclass",
        )
        return mat.detach().cpu().numpy()

    def _save_matrix(self, mat: ConfusionMatrixWriter, set_name, path: Path | None):
        """Save matrix to files"""
        if path is None:
            if self._save_dir is None:
                raise ValueError(
                    f"Cannot save {set_name} confusion matrix: no path given and no save_dir available."
                )
            parent = self._save_dir
            name = f"{set_name}_confusion_matrix"
        else:
            parent = path.parent
            name = path.with_suffix("").name
        csv, csv_rel, png = mat.to_all_formats(logdir=parent, name=name)
        if self._logger is not None:
            self._logger.experiment.log_asset(file_data=csv, file_name=f"{csv.name}")
            self._logger.experiment.log_asset(file_data=csv_rel, file_name=f"{csv_rel.name}")  # fmt: skip
            self._logger.experiment.log_asset(file_data=png, file_name=f"{png.name}")

    def train_confusion_matrix(self, path=None):
        """Compute and write train confusion matrix to file."""
        set_name = "train"
        mat = self._generic_confusion_matrix(self._train, name=set_name)
        mat = ConfusionMatrixWriter(labels=self._classes, confusion_matrix=mat)
        self._save_matrix(mat, set_name, path)

    def validation_confusion_matrix(self, path=None):
        """Compute and write validation confusion matrix to file."""
        set_name = "validation"
        mat = self._generic_confusion_matrix(self._val, name=set_name)
        mat = ConfusionMatrixWriter(labels=self._classes, confusion_matrix=mat)
        self._save_matrix(mat, set_name, path)

    def test_confusion_matrix(self, path=None):
        """Compute and write test confusion matrix to file."""
        set_name = "test"
        mat = self._generic_confusion_matrix(self._test, name=set_name)
        mat = ConfusionMatrixWriter(labels=self._classes, confusion_matrix=mat)
        self._save_matrix(mat, set_name, path)


# TODO: Insert "ID" in header, and make sure subsequent script use that (e.g. the bash one liner, for sorting)
def write_pred_table(predictions, str_preds, str_targets, signal_ids, classes, path):
    """Write to "path" a csv containing class probability predictions.

    pred : Prediction vectors
    str_preds : List of predictions, but in string form
    str_targets : List of corresponding targets, but in string form
    signal_ids : List of corresponding signal IDs
    classes : Ordered list of the output classes
    path : Where to write the file
    """
    df = pd.DataFrame(data=predictions, index=signal_ids, columns=classes)

    df.insert(loc=0, column="True class", value=str_targets)
    df.insert(loc=1, column="Predicted class", value=str_preds)

    df.to_csv(path, encoding="utf8")
