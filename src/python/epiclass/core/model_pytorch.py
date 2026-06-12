"""Model creation module"""
# pylint: disable=unused-argument, arguments-differ, too-many-positional-arguments
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Dict, Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchmetrics import (
    Accuracy,
    F1Score,
    MatthewsCorrCoef,
    MetricCollection,
    Precision,
    Recall,
)

from epiclass.core.model_checkpoint import restore_from_checkpoint_list


# pylint: disable=too-many-ancestors
class LightningDenseClassifier(pl.LightningModule):
    """Simple dense network handler"""

    def __init__(
        self, input_size, output_size, mapping, hparams, hl_units=3000, nb_layer=1
    ):
        """Metrics expect probabilities and not logits"""
        super().__init__()
        # this is recommended Lightning way to save model arguments
        # it saves everything passed into __init__
        # and allows you to access it as self.myparam1, self.myparam2
        self.save_hyperparameters()  # saves values given to __init__

        # -- general structure --
        self._x_size = input_size
        self._y_size = output_size
        self._hl_size = hl_units  # hl = hidden layer
        self._nb_layer = nb_layer  # number of intermediary/hidden layers
        if self._nb_layer < 1:
            raise AssertionError("Number of layers cannot be less than 1.")

        self._mapping = mapping

        # -- hyperparameters --
        self.l1_scale = hparams.get("l1_scale", 0)
        self.l2_scale = hparams.get("l2_scale", 0.01)
        self.dropout_rate = 1 - hparams.get("keep_prob", 0.5)
        self.learning_rate = hparams.get("learning_rate", 1e-5)

        self._pt_model = self.define_model()

        # Used Metrics
        metrics = MetricCollection(
            [
                Accuracy(task="multiclass", num_classes=self._y_size, average="micro"),
                Precision(task="multiclass", num_classes=self._y_size, average="macro"),
                Recall(task="multiclass", num_classes=self._y_size, average="macro"),
                F1Score(task="multiclass", num_classes=self._y_size, average="macro"),
                MatthewsCorrCoef(task="multiclass", num_classes=self._y_size),
            ]
        )
        self.metrics = metrics
        self.train_acc = Accuracy(
            task="multiclass", num_classes=self._y_size, average="micro"
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=self._y_size, average="micro"
        )

    @property
    def model(self) -> nn.Module:
        """Return the pytorch model."""
        return self._pt_model

    @property
    def mapping(self) -> Dict[int, str]:
        """Return {output index:label} mapping."""
        return self._mapping

    @property
    def invert_mapping(self) -> Dict[str, int]:
        """Return {label:output index} mapping."""
        return {val: key for key, val in self._mapping.items()}

    # --- Define general model structure ---
    def define_model(self):
        """ref : https://stackoverflow.com/questions/62937388/pytorch-dynamic-amount-of-layers"""
        layer_list = []
        # See the layers as matrix operations, as the weights, not the neurons.

        # input layer
        # Note: inplace True can save some memory but is not compatible with SHAP computation
        # https://github.com/shap/shap/issues/2204#issuecomment-933036857
        layer_list.append(nn.Dropout(0.1, inplace=False))  # drop part of input
        layer_list.append(nn.Linear(self._x_size, self._hl_size))
        layer_list.append(nn.Dropout(self.dropout_rate, inplace=False))
        layer_list.append(nn.ReLU(inplace=False))

        # hidden layers
        for _ in range(0, self._nb_layer - 1):
            layer_list.append(nn.Linear(self._hl_size, self._hl_size))
            layer_list.append(nn.ReLU(inplace=False))
            # in case of ReLU, dropout should be applied before for computational efficiency,
            # swapping them gives same result
            # https://sebastianraschka.com/faq/docs/dropout-activation.html

        # output layer
        layer_list.append(nn.Linear(self._hl_size, self._y_size))

        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
        model = nn.Sequential(*layer_list)

        return model

    def configure_optimizers(self):
        """https://pytorch.org/docs/stable/optim.html"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_scale
        )
        return optimizer

    # --- Define format of output ---
    def forward(self, x: Tensor):
        """Return logits."""
        return self._pt_model(x)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return probabilities"""
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def predict_class(self, x: Tensor) -> Tensor:
        """Return class"""
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    # --- Define how training and validation is done, what loss is used ---
    def training_step(self, train_batch, batch_idx):
        """Return training loss and co."""
        x, y = train_batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        if self.l1_scale > 0:
            l1_norm = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            loss += self.l1_scale * l1_norm

        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.train_acc(preds, y)

        # Log directly in training_step
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        """Return validation loss and co."""
        x, y = val_batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.valid_acc(preds, y)

        # Log directly in validation_step
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # --- Model saving and loading ---
    @classmethod
    def restore_model(cls, model_dir, verbose=True):
        """Load the checkpoint of the best model from the last run."""
        return restore_from_checkpoint_list(cls, model_dir, verbose)

    # --- Analysis related functions ---
    # These are not used during training but can be used for analysis after training,
    # e.g. to compute metrics on a test set or to print a model summary.
    # Force CPU to avoid device mismatch issues when reloading a model trained on GPU.
    def print_model_summary(self, batch_size=1):
        """Print torchinfo summary."""
        print("--MODEL SUMMARY--")
        summary(
            model=self,
            input_size=(batch_size, self._x_size),
            col_names=["input_size", "output_size", "num_params"],
        )

    def compute_metrics(self, dataset: Dataset, batch_size: int = 256):
        """Return dict of metrics for given dataset.

        Iterates the dataset via a DataLoader rather than slurping it via
        ``dataset[:]`` so it works for streaming lazy datasets without
        materialising the full signal matrix in RAM.
        """
        self.cpu()
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # Reset metric state, then accumulate per-batch.
        self.metrics.reset()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                preds = self(batch_x.cpu())
                self.metrics.update(preds, batch_y.cpu())
        return self.metrics.compute()

    def compute_predictions_from_dataset(
        self, dataset: Dataset, batch_size: int = 256
    ) -> Tuple[Tensor, Tensor]:
        """Return probability predictions and targets from dataset.

        Iterates the dataset via a DataLoader; only the per-sample (n_classes)
        probability vector is kept in RAM, not the full input matrix.
        """
        self.cpu()
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        prob_chunks: list[Tensor] = []
        target_chunks: list[Tensor] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                prob_chunks.append(self.predict_proba(batch_x.cpu()))
                target_chunks.append(batch_y.cpu())
        return torch.cat(prob_chunks, dim=0), torch.cat(target_chunks, dim=0)

    def compute_predictions_from_features(self, features: Tensor) -> Tensor:
        """Return probability predictions from features."""
        self.cpu()
        self.eval()
        with torch.no_grad():
            probs = self.predict_proba(features.cpu())
        return probs
