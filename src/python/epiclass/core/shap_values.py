"""Module containing shap values related code (e.g. handling computation, analysing results)."""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

import multiprocessing

# ---------------------------------------------------------------------
# Set the start method to 'spawn' for multiprocessing.
# This is crucial for PyTorch and CUDA to avoid deadlocks when using
# ProcessPoolExecutor. 'fork' (the default on Linux) can cause hangs
# by incorrectly sharing CUDA context with child processes.
# 'force=True' is needed because pytest might initialize the context.
try:
    multiprocessing.set_start_method("spawn", force=True)
    print("--- Multiprocessing start method set to 'spawn' ---")
except RuntimeError:
    # This can happen if the context is already set and cannot be changed.
    # In most cases, the try block will succeed.
    pass
# ---------------------------------------------------------------------

import concurrent.futures
import copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
import torch
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline

from epiclass.core.estimators import EstimatorAnalyzer
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.types import SomeData
from epiclass.utils.time import time_now_str


class SHAP_Saver:
    """Handle shap data saving/loading."""

    def __init__(self, logdir: Path | str):
        self.logdir = logdir
        self.filename_template = "shap_{name}_{time}.{ext}"

    def _create_filename(self, ext: str, name="") -> Path:
        """Create a filename with the given extension and name, and a timestamp."""
        filename = self.filename_template.format(name=name, ext=ext, time=time_now_str())
        filename = Path(self.logdir) / filename
        return filename

    def save_to_csv(
        self, shap_values_matrix: np.ndarray, ids: List[str], name: str
    ) -> Path:
        """Save a single shap value matrix (shape (n_samples, #features)) to csv.
        Giving a name is mandatory.

        Returns path of saved file.
        """
        if isinstance(shap_values_matrix, list):
            raise ValueError(
                f"Expected 'shap_values_matrix' to be a numpy array of shape (n_samples, #features), but got a list instead: {shap_values_matrix}"  # pylint: disable=line-too-long
            )
        filename = self._create_filename(name=name, ext="csv")

        n_dims = shap_values_matrix.shape[1]
        df = pd.DataFrame(data=shap_values_matrix, index=ids, columns=range(n_dims))

        print(f"Saving SHAP values to: {filename}")
        df.to_csv(filename)

        return filename

    def save_to_npz(self, name: str, verbose=True, **kwargs):
        """Save kwargs to numpy compressed npz file. Transforms everything into numpy arrays."""
        filename = self._create_filename(name=name, ext="npz")
        if verbose:
            print(f"Saving SHAP values to: {filename}")
        np.savez_compressed(
            file=filename,
            **kwargs,  # type: ignore
        )

    @staticmethod
    def load_from_csv(path: Path | str) -> pd.DataFrame:
        """Return pandas dataframe of shap values for loaded file."""
        return pd.read_csv(path, index_col=0)


class NN_SHAP_Handler:
    """Handle shap computations and data saving/loading."""

    def __init__(self, model: LightningDenseClassifier, logdir: Path | str):
        self.model = model
        self.model.eval()
        self.model_classes = list(self.model.mapping.items())
        self.logdir = logdir
        self.saver = SHAP_Saver(logdir=logdir)

    def compute_shaps(
        self,
        background_dset: SomeData,
        evaluation_dset: SomeData,
        save=True,
        name="",
        num_workers: int = 4,
    ) -> Tuple[shap.DeepExplainer, np.ndarray]:
        """Compute shap values of deep learning model on evaluation dataset
        by creating an explainer with background dataset.

        Returns:
            Tuple of (explainer, shap_values) where shap_values is np.ndarray of shape:
                (#samples, #features, #classes)
        """
        model = self.model
        data = torch.from_numpy(background_dset.signals).float()
        explainer = shap.DeepExplainer(model=model, data=data)

        if save:
            self.saver.save_to_npz(
                name=name + "_explainer_background",
                background_md5s=background_dset.ids,
                background_expectation=explainer.expected_value,  # type: ignore
                classes=self.model_classes,
            )

        signals = torch.from_numpy(evaluation_dset.signals).float()
        shap_values = NN_SHAP_Handler._compute_shap_values_parallel(
            model=model,
            background_data=data,
            signals=signals,
            num_workers=num_workers,
        )

        if save:
            self.saver.save_to_npz(
                name=name + "_evaluation",
                evaluation_md5s=evaluation_dset.ids,
                shap_values=shap_values,
                classes=self.model_classes,
            )

        return explainer, shap_values  # type: ignore

    @staticmethod
    def _nn_shap_worker(
        args: Tuple[LightningDenseClassifier, torch.Tensor, torch.Tensor]
    ) -> np.ndarray:
        """
        Static worker function for parallel SHAP computation. Can be pickled.
        """
        model, background_data, signal_chunk = args
        model.eval()
        local_explainer = shap.DeepExplainer(model, background_data)
        return local_explainer.shap_values(signal_chunk)  # type: ignore

    # Can't define _nn_shap_worker as inner function because it needs to be picklable for ProcessPoolExecutor
    @staticmethod
    def _compute_shap_values_parallel(
        model: LightningDenseClassifier,
        background_data: torch.Tensor,
        signals: torch.Tensor,
        num_workers: int,
    ) -> np.ndarray:
        """Compute SHAP values in parallel using a ProcessPoolExecutor."""
        model.to(torch.device("cpu"))

        signal_chunks = torch.tensor_split(signals, num_workers)
        tasks = [(model, background_data, chunk) for chunk in signal_chunks]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            shap_values_chunks = list(
                executor.map(NN_SHAP_Handler._nn_shap_worker, tasks)
            )

        shap_values = np.concatenate(shap_values_chunks, axis=0)
        return shap_values


class LGBM_SHAP_Handler:
    """Handle shap computations and data saving/loading."""

    def __init__(self, model_analyzer: EstimatorAnalyzer, logdir: Path | str):
        self.logdir = logdir
        self.saver = SHAP_Saver(logdir=logdir)
        self.model_classes = list(model_analyzer.mapping.items())
        self.model: LGBMClassifier = LGBM_SHAP_Handler._check_model_is_lgbm(
            model_analyzer
        )

    @staticmethod
    def _check_model_is_lgbm(model_analyzer: EstimatorAnalyzer) -> LGBMClassifier:
        """Return lightgbm classifier if found, else raise ValueError."""
        model = model_analyzer.classifier
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
        if not isinstance(model, LGBMClassifier):
            raise ValueError(
                f"Expected model to be a lightgbm classifier, but got {model} instead."
            )
        return model

    def compute_shaps(
        self,
        background_dset: SomeData,
        evaluation_dset: SomeData,
        save=True,
        name="",
        num_workers: int = 4,
    ) -> Tuple[shap.TreeExplainer, np.ndarray]:
        """Compute shap values of lgbm model on evaluation dataset.

        Args:
            background_dset: Background dataset for SHAP explainer.
            evaluation_dset: Dataset to compute SHAP values for.
            save: Whether to save the results.
            name: Name prefix for saved files.
            num_workers: Number of parallel workers.

        Returns:
            Tuple[explainer, shap_values] where shap_values is np.ndarray of shape:
                - Binary classification: (#samples, #features)
                - Multiclass: (#samples, #features, #classes)
        """
        explainer = shap.TreeExplainer(
            model=self.model,
            data=background_dset.signals,
            model_output="raw",
            feature_perturbation="interventional",
        )

        if save:
            self.saver.save_to_npz(
                name=name + "_explainer_background",
                background_md5s=background_dset.ids,
                background_expectation=explainer.expected_value,  # type: ignore
                classes=self.model_classes,
            )

        shap_values = LGBM_SHAP_Handler._compute_shap_values_parallel(
            explainer=explainer,
            signals=evaluation_dset.signals,
            num_workers=num_workers,
        )

        if save:
            self.saver.save_to_npz(
                name=name + "_evaluation",
                evaluation_md5s=evaluation_dset.ids,
                shap_values=shap_values,
                expected_value=explainer.expected_value,
                classes=self.model_classes,
            )

        return explainer, shap_values

    @staticmethod
    def _compute_shap_values_parallel(
        explainer: shap.TreeExplainer,
        signals: ArrayLike,
        num_workers: int,
    ) -> np.ndarray:
        """Compute SHAP values in parallel using a ThreadPoolExecutor.

        Args:
            explainer: The SHAP TreeExplainer object.
            signals: The evaluation dataset samples of shape (#samples, #features).
            num_workers: The number of parallel threads to use.

        Returns:
            np.ndarray: SHAP values of shape:
                - Binary classification: (#samples, #features)
                - Multiclass: (#samples, #features, #classes)
        """
        # Split the signals into chunks for parallel processing
        signal_chunks = np.array_split(signals, num_workers)

        def worker(chunk):
            """Compute shap values using an explainer copy."""
            local_explainer = copy.deepcopy(explainer)  # Deep copy to avoid thread issues
            return local_explainer.shap_values(X=chunk, check_additivity=True)

        # Use ThreadPoolExecutor to compute shap_values in parallel
        # The C++ backend (of TreeExplainer) should release the GIL, so multi-core should still work.
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            shap_values_chunks = list(executor.map(worker, signal_chunks))

        # Concatenate along sample dimension (axis=0)
        shap_values = np.concatenate(shap_values_chunks, axis=0)

        return shap_values
