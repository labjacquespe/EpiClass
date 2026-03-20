"""Tests for device robustness of LightningDenseClassifier compute methods.

These tests require a GPU and are skipped on CPU-only machines.
They reproduce a bug where model and data end up on different devices
after checkpoint restoration (model on CPU, data still on GPU),
causing RuntimeError in torch.nn.functional.linear.
"""
# pylint: disable=redefined-outer-name
import pytest
import torch
from torch.utils.data import TensorDataset

from epiclass.core.model_pytorch import LightningDenseClassifier


@pytest.fixture
def dummy_model():
    """A minimal LightningDenseClassifier for testing."""
    mapping = {0: "class_a", 1: "class_b"}
    hparams = {"keep_prob": 0.5, "l2_scale": 0.01, "learning_rate": 1e-5}
    return LightningDenseClassifier(
        input_size=10,
        output_size=2,
        mapping=mapping,
        hparams=hparams,
        hl_units=8,
        nb_layer=1,
    )


@pytest.fixture()
def dummy_dataset():
    """A small TensorDataset with random features and binary targets."""
    features = torch.randn(16, 10)
    targets = torch.randint(0, 2, (16,))
    return TensorDataset(features, targets)


class TestDeviceRobustness:
    """Verify compute methods handle device mismatches gracefully."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_model_cpu_data_gpu(self, dummy_model, dummy_dataset):
        """Reproduces the original bug: model restored to CPU, data still on GPU."""
        features, targets = dummy_dataset[:]
        gpu_dataset = TensorDataset(features.cuda(), targets.cuda())
        dummy_model.cpu()
        result = dummy_model.compute_metrics(gpu_dataset)
        assert isinstance(result, dict)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_model_gpu_data_cpu(self, dummy_model, dummy_dataset):
        """Inverse mismatch: model on GPU, data on CPU."""
        dummy_model.cuda()
        result = dummy_model.compute_metrics(dummy_dataset)
        assert isinstance(result, dict)
