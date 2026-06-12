"""Unit tests for the hybrid AVE model (LightningAVE).

Self-contained: uses small synthetic TensorDatasets so the tests stay fast and
do not depend on HDF5 fixtures. The end-to-end pipeline is exercised separately
in tests/mains/ave_training_test.py.
"""
# pylint: disable=redefined-outer-name, protected-access, duplicate-code
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from epiclass.core.model_ave import LightningAVE, bounded_sublinear_sizes

INPUT_SIZE = 40
LATENT_DIM = 8


@pytest.fixture
def hparams() -> dict:
    """Small AVE hyperparameters for fast CPU tests."""
    return {
        "learning_rate": 1e-3,
        "l2_scale": 0.0,
        "dropout": 0.0,
        "kl_weight": 0.1,
        "fusion_weight": 0.5,
    }


@pytest.fixture
def model(hparams) -> LightningAVE:
    """A small AVE with an explicit latent dim and hidden sizes."""
    return LightningAVE(
        input_size=INPUT_SIZE,
        hparams=hparams,
        ae_hidden=(16, 8),
        vae_hidden=(16, 8),
        latent_dim=LATENT_DIM,
    )


@pytest.fixture
def dataset() -> TensorDataset:
    """Random features with dummy labels (labels are ignored by the AVE)."""
    features = torch.randn(32, INPUT_SIZE)
    targets = torch.randint(0, 2, (32,))
    return TensorDataset(features, targets)


def test_bounded_sublinear_sizes_caps():
    """Large input dimensions are capped at the report's bounds."""
    assert bounded_sublinear_sizes(30_000) == (1024, 512)
    # Tiny inputs hit the lower bounds.
    assert bounded_sublinear_sizes(10) == (256, 128)


def test_forward_shape(model):
    """forward() reconstructs to the input dimension."""
    x = torch.randn(6, INPUT_SIZE)
    recon = model(x)
    assert recon.shape == (6, INPUT_SIZE)


def test_encode_decode_shapes(model):
    """_encode_decode returns reconstruction plus latent params of latent_dim."""
    x = torch.randn(6, INPUT_SIZE)
    recon, mu, log_var = model._encode_decode(x)
    assert recon.shape == (6, INPUT_SIZE)
    assert mu.shape == (6, LATENT_DIM)
    assert log_var.shape == (6, LATENT_DIM)


def test_losses_finite_and_nonnegative(model):
    """Reconstruction and KL terms are finite and non-negative."""
    x = torch.randn(8, INPUT_SIZE)
    total, recon_loss, kl_loss = model._losses(x)
    for value in (total, recon_loss, kl_loss):
        assert torch.isfinite(value)
    assert recon_loss.item() >= 0.0
    assert kl_loss.item() >= 0.0


def test_training_reduces_loss(model, dataset):
    """A few optimizer steps decrease the reconstruction+KL loss."""
    model.train()
    optimizer = model.configure_optimizers()
    features, _ = dataset[:]

    initial = model._losses(features)[0].item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = model._losses(features)[0]
        loss.backward()
        optimizer.step()
    final = model._losses(features)[0].item()

    assert final < initial


def test_kl_annealing_scales_weight(hparams):
    """The effective KL weight ramps from 0 toward kl_weight when annealing."""
    annealed = LightningAVE(
        input_size=INPUT_SIZE,
        hparams={**hparams, "kl_anneal_epochs": 10},
        latent_dim=LATENT_DIM,
    )
    # current_epoch defaults to 0 before any training -> weight starts at 0.
    assert annealed._current_kl_weight() == pytest.approx(0.0)


def test_reconstruction_errors_one_per_sample(model, dataset):
    """Scoring returns one error per sample, aligned with dataset order."""
    errors = model.reconstruction_errors(dataset, batch_size=8)
    assert errors.shape == (len(dataset),)
    assert np.all(np.isfinite(errors))


def test_contamination_threshold_flags_expected_count(model):
    """The contamination threshold flags ~rate fraction of samples."""
    errors = np.arange(100, dtype=np.float32)
    threshold = model.threshold_from_contamination(errors, rate=0.1)
    flags = model.predict_outliers(errors, threshold)
    # 10% of 100 -> the 10 largest values are flagged (cutoff is the 10th largest).
    assert flags.sum() == 10


def test_contamination_threshold_flags_at_least_one(model):
    """Even a tiny set / small rate flags at least one sample."""
    errors = np.array([0.1, 0.2, 0.3, 5.0], dtype=np.float32)
    threshold = model.threshold_from_contamination(errors, rate=0.01)
    flags = model.predict_outliers(errors, threshold)
    assert flags.sum() >= 1


def test_iqr_threshold_above_median(model):
    """The IQR upper fence sits above the bulk of the data."""
    errors = np.array([1, 2, 3, 4, 5, 100], dtype=np.float32)
    threshold = model.threshold_from_iqr(errors)
    assert threshold > np.median(errors)


def test_invalid_output_activation_raises(hparams):
    """An unknown output activation is rejected at construction."""
    with pytest.raises(ValueError):
        LightningAVE(
            input_size=INPUT_SIZE,
            hparams={**hparams, "output_activation": "tanh"},
            latent_dim=LATENT_DIM,
        )
