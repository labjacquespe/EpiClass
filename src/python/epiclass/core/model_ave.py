"""Hybrid Autoencoder-Variational-Autoencoder (AVE) model for anomaly detection.

PyTorch Lightning port of the hybrid AVE proposed by Daoud et al. (2025,
*Scientific Reports*). Two encoders -- a deterministic autoencoder (AE) branch
and a probabilistic variational (VAE) branch -- are trained jointly and feed a
single shared decoder. Their latent vectors are fused by a weighted average

    z = omega * z_AE + (1 - omega) * z_VAE

before decoding. Training is unsupervised: the loss combines reconstruction
(MSE) and a KL-divergence regularizer on the VAE branch. Samples are scored as
outliers by their reconstruction error against an adaptive threshold.

Layer sizes default to a bounded sub-linear sizing heuristic (widely used for
autoencoders on high-dimensional genomic data) so that epigenomic signals of
tens of thousands of bins do not blow up the first projection's parameter count.
"""
# pylint: disable=arguments-differ, too-many-positional-arguments, too-many-instance-attributes
# pylint: disable=unused-argument
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

from epiclass.core.model_checkpoint import restore_from_checkpoint_list

# Bounded sub-linear sizing caps for the first projections: compress sparse,
# redundant features early and keep the parameter footprint bounded for genomic
# inputs (tens of thousands of features).
_H1_MAX, _H1_MIN, _H1_DIV = 1024, 256, 2
_H2_MAX, _H2_MIN, _H2_DIV = 512, 128, 4
_DEFAULT_LATENT_DIM = 128


def bounded_sublinear_sizes(input_dim: int) -> Tuple[int, int]:
    """Return ``(H1, H2)`` hidden sizes from the bounded sub-linear heuristic.

    ``H1 = min(1024, max(256, Din // 2))`` and ``H2 = min(512, max(128, Din // 4))``.
    This compresses sparse, redundant features in the very first projection,
    keeping the parameter footprint bounded for ``Din`` in the tens of thousands.

    Rationale: in the high-dimensional / low-sample-size (HDLSS) regime
    (``Din >> N``), classical fractional halving (``H1 = Din / 2`` unbounded)
    gives the first layer enough capacity to learn a near-identity mapping,
    collapsing the reconstruction error gap that outlier scoring depends on.
    Capping the early layers is the established remedy in genomic autoencoders,
    which compress 20k-30k+ input genes into small bounded hidden layers rather
    than scaling proportionally. The exact caps used here follow a preprint
    heuristic; the bounded sub-linear *philosophy* is well established in
    peer-reviewed work, e.g.:

    - Eraslan et al. (2019), "Single-cell RNA-seq denoising using a deep count
      autoencoder", Nature Communications 10:390 -- DCA compresses ~20k-30k
      genes through a 64-32-64 bottleneck rather than scaling with input width.
    - Tian et al. (2019), "Clustering single-cell RNA-seq data with a model-based
      deep learning approach" (scDeepCluster), Nature Machine Intelligence 1:191
      -- 256-64-32 encoder over tens of thousands of genes.

    These confirm the heavily bounded, sub-linear first projection used here,
    even though the specific min/max caps are a design choice, not a fixed rule.
    """
    h1 = min(_H1_MAX, max(_H1_MIN, input_dim // _H1_DIV))
    h2 = min(_H2_MAX, max(_H2_MIN, input_dim // _H2_DIV))
    return h1, h2


# pylint: disable=too-many-ancestors
class LightningAVE(pl.LightningModule):
    """Hybrid AE+VAE anomaly-detection model (shared decoder, weighted fusion)."""

    def __init__(
        self,
        input_size: int,
        hparams: Dict,
        mapping: Optional[Dict[int, str]] = None,
        ae_hidden: Optional[Tuple[int, int]] = None,
        vae_hidden: Optional[Tuple[int, int]] = None,
        latent_dim: Optional[int] = None,
    ):
        """Build the dual-encoder / shared-decoder network.

        Args:
            input_size: Number of input features (genomic bins).
            hparams: Hyperparameter dict (see module docstring for keys).
            mapping: Optional ``{index: label}`` map, kept for reference only;
                labels are not used by the unsupervised loss.
            ae_hidden / vae_hidden: Optional ``(H1, H2)`` overrides per branch.
                When omitted, the bounded sub-linear heuristic is used.
            latent_dim: Optional latent dimension override (default 128). The AE
                and VAE latents share this size so the weighted average is defined.
        """
        super().__init__()
        self.save_hyperparameters()

        self._x_size = input_size
        self._mapping = mapping

        # -- hyperparameters: kl_weight, learning_rate and fusion_weight follow
        #    the reference AVE implementation; weight decay and dropout are tuned
        #    for high-dimensional genomic input --
        self.learning_rate = hparams.get("learning_rate", 1e-4)
        self.l2_scale = hparams.get("l2_scale", 0.001)  # AdamW weight decay
        self.dropout_rate = hparams.get("dropout", 0.1)
        self.kl_weight = hparams.get("kl_weight", 0.1)
        self.fusion_weight = hparams.get("fusion_weight", 0.5)  # omega
        self.kl_anneal_epochs = hparams.get("kl_anneal_epochs", 0)
        self.contamination_rate = hparams.get("contamination_rate", 0.05)
        self.output_activation_name = hparams.get("output_activation", "linear")

        # -- layer sizing --
        default_h1, default_h2 = bounded_sublinear_sizes(input_size)
        self._ae_h1, self._ae_h2 = ae_hidden or (default_h1, default_h2)
        self._vae_h1, self._vae_h2 = vae_hidden or (default_h1, default_h2)
        self._latent_dim = latent_dim or _DEFAULT_LATENT_DIM

        # -- network --
        self.ae_encoder = self._build_encoder(self._ae_h1, self._ae_h2, self._latent_dim)
        self.vae_encoder = self._build_encoder(
            self._vae_h1, self._vae_h2, self._latent_dim * 2
        )
        self.decoder = self._build_decoder(default_h1, default_h2)

    # --- Network construction ---
    def _build_encoder(self, h1: int, h2: int, out_dim: int) -> nn.Sequential:
        """Return an encoder trunk ``Din -> h1 -> h2 -> out_dim``."""
        return nn.Sequential(
            nn.Linear(self._x_size, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=False),
            nn.Dropout(self.dropout_rate, inplace=False),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=False),
            nn.Dropout(self.dropout_rate, inplace=False),
            nn.Linear(h2, out_dim),
        )

    def _build_decoder(self, h1: int, h2: int) -> nn.Sequential:
        """Return the shared decoder ``latent -> h2 -> h1 -> Din`` (+ activation)."""
        layers: List[nn.Module] = [
            nn.Linear(self._latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=False),
            nn.Dropout(self.dropout_rate, inplace=False),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=False),
            nn.Dropout(self.dropout_rate, inplace=False),
            nn.Linear(h1, self._x_size),
        ]
        if self.output_activation_name == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.output_activation_name != "linear":
            raise ValueError(
                f"Unknown output_activation '{self.output_activation_name}'. "
                "Expected 'linear' or 'sigmoid'."
            )
        return nn.Sequential(*layers)

    # --- Properties ---
    @property
    def mapping(self) -> Optional[Dict[int, str]]:
        """Return the ``{index: label}`` mapping (unused by the loss)."""
        return self._mapping

    # --- Forward / latent logic ---
    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample ``z = mu + eps * exp(0.5 * log_var)`` with ``eps ~ N(0, I)``."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_decode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(reconstruction, mu, log_var)`` for input batch ``x``."""
        z_ae = self.ae_encoder(x)
        mu, log_var = torch.chunk(self.vae_encoder(x), 2, dim=1)
        z_vae = self.reparameterize(mu, log_var)
        z_hybrid = self.fusion_weight * z_ae + (1.0 - self.fusion_weight) * z_vae
        reconstruction = self.decoder(z_hybrid)
        return reconstruction, mu, log_var

    def forward(self, x: Tensor) -> Tensor:
        """Return the reconstruction of ``x`` (stochastic VAE sampling)."""
        reconstruction, _, _ = self._encode_decode(x)
        return reconstruction

    def _current_kl_weight(self) -> float:
        """Return the (possibly annealed) KL weight for the current epoch."""
        if self.kl_anneal_epochs and self.kl_anneal_epochs > 0:
            frac = min(1.0, self.current_epoch / self.kl_anneal_epochs)
            return self.kl_weight * frac
        return self.kl_weight

    def _losses(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(total_loss, reconstruction_loss, kl_loss)`` for batch ``x``."""
        reconstruction, mu, log_var = self._encode_decode(x)
        recon_loss = F.mse_loss(reconstruction, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total = recon_loss + self._current_kl_weight() * kl_loss
        return total, recon_loss, kl_loss

    # --- Training / validation ---
    def configure_optimizers(self):
        """Use AdamW so weight decay is decoupled from the gradient update."""
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_scale
        )

    def training_step(self, train_batch, batch_idx):
        """Reconstruction + KL loss; labels (if present) are ignored."""
        x = train_batch[0]
        total, recon_loss, kl_loss = self._losses(x)
        self.log("train_loss", total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", recon_loss, on_step=False, on_epoch=True)
        self.log("train_kl", kl_loss, on_step=False, on_epoch=True)
        return total

    def validation_step(self, val_batch, batch_idx):
        """Validation reconstruction + KL loss (monitored as ``valid_loss``)."""
        x = val_batch[0]
        total, recon_loss, kl_loss = self._losses(x)
        self.log("valid_loss", total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_recon", recon_loss, on_step=False, on_epoch=True)
        self.log("valid_kl", kl_loss, on_step=False, on_epoch=True)
        return total

    # --- Scoring / thresholding ---
    def reconstruction_errors(
        self, dataset: Dataset, batch_size: int = 256
    ) -> np.ndarray:
        """Return the per-sample mean squared reconstruction error.

        Iterates ``dataset`` (unshuffled) via a DataLoader so it works for
        streaming lazy datasets without materialising the full signal matrix.
        The returned array is aligned with the dataset's index order.
        """
        self.cpu()
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        errors: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].cpu()
                reconstruction = self(x)
                err = torch.mean((reconstruction - x) ** 2, dim=1)
                errors.append(err.numpy())
        return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)

    @staticmethod
    def threshold_from_contamination(errors: np.ndarray, rate: float) -> float:
        """Return the error cutoff such that ~``rate`` of samples are flagged.

        Mirrors the reference: sort the errors and take the value at the
        ``rate`` quantile from the top, flagging at least one sample.
        """
        n_samples = len(errors)
        if n_samples == 0:
            raise ValueError("Cannot compute a threshold from empty errors.")
        n_flagged = max(1, int(round(rate * n_samples)))
        return float(np.sort(errors)[-n_flagged])

    @staticmethod
    def threshold_from_iqr(errors: np.ndarray, k: float = 1.5) -> float:
        """Return the Tukey upper-fence threshold ``Q3 + k * IQR``."""
        if len(errors) == 0:
            raise ValueError("Cannot compute a threshold from empty errors.")
        q1, q3 = np.percentile(errors, [25, 75])
        return float(q3 + k * (q3 - q1))

    @staticmethod
    def predict_outliers(errors: np.ndarray, threshold: float) -> np.ndarray:
        """Return a 0/1 array flagging errors at or above ``threshold``.

        Using ``>=`` (rather than strict ``>``) means a contamination threshold
        set to the ``k``-th largest error flags exactly ``k`` samples.
        """
        return (errors >= threshold).astype(int)

    # --- Model saving / loading ---
    @classmethod
    def restore_model(cls, model_dir, verbose: bool = True) -> "LightningAVE":
        """Load the checkpoint of the best model from the last run."""
        return restore_from_checkpoint_list(cls, model_dir, verbose)

    # --- Analysis ---
    def print_model_summary(self, batch_size: int = 2):
        """Print a torchinfo summary (in eval mode so BatchNorm accepts it)."""
        print("--MODEL SUMMARY--")
        was_training = self.training
        self.eval()
        try:
            summary(
                model=self,
                input_size=(batch_size, self._x_size),
                col_names=["input_size", "output_size", "num_params"],
            )
        finally:
            self.train(was_training)
