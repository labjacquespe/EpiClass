"""Trainer class extensions module"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as pl
from lightning.pytorch import callbacks as pl_callbacks


class MyTrainer(pl.Trainer):
    """Personalized trainer"""

    def __init__(self, general_log_dir: str, model, **kwargs):
        """Metrics expect probabilities and not logits."""
        super().__init__(**kwargs)

        self.best_checkpoint_file = Path(general_log_dir) / "best_checkpoint.list"
        self.my_model = model
        self.batch_size = None

    def fit(self, *args, verbose=True, **kwargs):
        """Base pl.Trainer.fit function, but also prints the batch size."""
        self.batch_size = kwargs["train_dataloaders"].batch_size
        if verbose:
            print(f"Training batch size : {self.batch_size}")
        super().fit(*args, **kwargs)

    def save_model_path(self):
        """Save best checkpoint path to a file.

        Falls back to `last_model_path` when `best_model_path` is empty —
        ModelCheckpoint with `monitor=None` (no-validation training) never
        populates `best_model_path`, only `last_model_path` via `save_last=True`.
        """
        try:
            model_path = self.checkpoint_callback.best_model_path  # type: ignore
            if not model_path:
                model_path = self.checkpoint_callback.last_model_path  # type: ignore
            if not model_path:
                print("Cannot save model, no checkpoint was created.")
                return
            print(f"Saving model to '{model_path}'")
            with open(self.best_checkpoint_file, "a", encoding="utf-8") as ckpt_file:
                ckpt_file.write(f"{model_path} {datetime.now()}\n")
        except AttributeError:
            print("Cannot save model, no checkpoint callback.")

    def print_hyperparameters(self):
        """Print training hyperparameters."""
        print("--TRAINING HYPERPARAMETERS--")
        print(f"L2 scale : {self.my_model.l2_scale}")
        print(f"Dropout rate : {self.my_model.dropout_rate}")
        print(f"Learning rate : {self.my_model.learning_rate}")
        try:
            stop_callback = self.early_stopping_callback
            print(f"Patience : {stop_callback.patience}")  # type: ignore
            print(f"Monitored value : {stop_callback.monitor}")  # type: ignore
        except AttributeError:
            print("No early stopping.")


def define_callbacks(
    early_stop_limit: int | None,
    show_summary=True,
    show_progress_bar=True,
    monitor: str = "valid_acc",
    mode: str = "max",
):
    """Returns list of PyTorch trainer callbacks.
    RichProgressBar, RichModelSummary, EarlyStopping, ModelCheckpoint

    Will only save last epoch model if there is no early stopping.

    `monitor` / `mode` select the metric EarlyStopping and ModelCheckpoint
    track. Defaults suit classifiers ("valid_acc" maximised); the AVE passes
    monitor="valid_loss", mode="min".
    """
    callbacks = []

    if show_progress_bar:
        callbacks.append(pl_callbacks.RichProgressBar(leave=True))

    if show_summary:
        callbacks.append(pl_callbacks.RichModelSummary(max_depth=3))

    monitored_value = monitor  # have same name as the logged metric

    if early_stop_limit is not None:
        callbacks.append(
            pl_callbacks.EarlyStopping(
                monitor=monitored_value,
                mode=mode,
                patience=early_stop_limit,
                check_on_train_epoch_end=False,
            )
        )

        callbacks.append(
            pl_callbacks.ModelCheckpoint(
                monitor=monitored_value,
                mode=mode,
                save_last=True,
                auto_insert_metric_name=True,
                every_n_epochs=1,
                save_top_k=2,
                save_on_train_epoch_end=False,
            )
        )
    else:
        callbacks.append(
            pl_callbacks.ModelCheckpoint(
                monitor=None,
                save_last=True,
                save_top_k=0,
            )
        )

    return callbacks


class ValidationTrendMonitor(pl_callbacks.Callback):
    """Watch a validation metric and report on it, without ever stopping training.

    ``EarlyStopping`` (and monitor-based ``ModelCheckpoint``) makes the surviving
    model depend on the validation fold it watches: which epoch survives is
    chosen by a score computed on those same samples. How much that matters
    depends on the metric. Picking the highest of a noisy ``valid_acc`` lands on
    a lucky epoch, and that luck inflates the accuracy you then report for it.
    Picking the lowest of a ``valid_loss`` that falls every epoch just returns
    the last epoch, choosing nothing at all. This callback is the
    diagnostic half of ``EarlyStopping`` with the intervening half removed: it
    never touches ``trainer.should_stop``, it only reports.

    With a fixed epoch budget and no stopping, two things can go wrong, and both
    are read off the same validation curve:

    - *degraded*: the metric stayed worse than its best value for ``patience``
      consecutive checks, so the kept last-epoch model may be past its peak;
    - *undertrained*: the metric still improved by more than
      ``min_rel_improvement`` over the final ``window`` checks, so the budget
      ran out before the curve flattened.

    Findings are printed, raised as ``UserWarning``, tagged on any logger
    exposing ``experiment.add_tag``, and written to ``valid_trend_report.json``
    under ``save_dir`` -- the console is unreliable across many SLURM folds and
    offline Comet logs nothing, so the on-disk report is the auditable record.
    The full metric history is in that report as well, next to the scores CSV it
    qualifies.
    """

    def __init__(
        self,
        monitor: str = "valid_loss",
        *,
        mode: str = "min",
        patience: int = 5,
        window: int = 5,
        min_rel_improvement: float = 0.01,
        save_dir: str | Path | None = None,
        report_name: str = "valid_trend_report.json",
    ):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.window = window
        self.min_rel_improvement = min_rel_improvement
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.report_name = report_name

        self.history: List[Dict[str, float]] = []
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.degraded_at_epoch: Optional[int] = None
        self._streak = 0

    @property
    def degraded(self) -> bool:
        """True when the metric stayed worse than its best for `patience` checks."""
        return self.degraded_at_epoch is not None

    def _is_better(self, value: float, reference: float) -> bool:
        """Is `value` an improvement over `reference` under the configured mode?"""
        if self.mode == "min":
            return value < reference
        return value > reference

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: D102
        if trainer.sanity_checking:
            return
        raw = trainer.callback_metrics.get(self.monitor)
        if raw is None:
            return
        value = float(raw)
        epoch = int(trainer.current_epoch)
        self.history.append({"epoch": epoch, "value": value})

        if self.best_value is None or self._is_better(value, self.best_value):
            self.best_value = value
            self.best_epoch = epoch
            self._streak = 0
            return

        self._streak += 1
        if self._streak == self.patience and not self.degraded:
            self.degraded_at_epoch = epoch
            self._report(
                trainer,
                tag=f"{self.monitor}Degraded",
                message=(
                    f"{self.monitor} has not improved on its best value "
                    f"({self.best_value:.6g} at epoch {self.best_epoch}) for "
                    f"{self.patience} consecutive validation checks "
                    f"(now {value:.6g} at epoch {epoch}). Training continues on "
                    f"purpose -- the last-epoch model is still what gets kept -- "
                    f"but this fold's model and its validation scores should be "
                    f"inspected before use."
                ),
            )

    def on_fit_end(self, trainer, pl_module) -> None:  # noqa: D102
        undertrained, rel_improvement = self._final_improvement()
        if undertrained:
            self._report(
                trainer,
                tag=f"{self.monitor}Undertrained",
                message=(
                    f"{self.monitor} still improved by "
                    f"{rel_improvement:.1%} over the final {self.window} "
                    f"validation checks (threshold {self.min_rel_improvement:.1%}). "
                    f"The epoch budget likely ended before the curve flattened; "
                    f"consider raising 'training_epochs'."
                ),
            )
        self._write_report(undertrained=undertrained, rel_improvement=rel_improvement)

    def _final_improvement(self) -> tuple[bool, float]:
        """Relative gain of the last `window` checks over everything before them.

        Returns ``(still_improving, relative_improvement)``. Needs at least
        ``window + 1`` points to say anything; below that it abstains.
        """
        if len(self.history) < self.window + 1:
            return False, 0.0
        values = [entry["value"] for entry in self.history]
        prior, recent = values[: -self.window], values[-self.window :]
        pick = min if self.mode == "min" else max
        prior_best, recent_best = pick(prior), pick(recent)

        gain = (
            prior_best - recent_best if self.mode == "min" else recent_best - prior_best
        )
        scale = abs(prior_best)
        rel = gain / scale if scale > 1e-12 else gain
        return rel > self.min_rel_improvement, rel

    def _report(self, trainer, tag: str, message: str) -> None:
        """Print, warn, and tag every logger that accepts tags."""
        banner = "=" * 78
        print(f"\n{banner}\nWARNING [{tag}] {message}\n{banner}\n", flush=True)
        warnings.warn(f"[{tag}] {message}", UserWarning)
        for logger in getattr(trainer, "loggers", []) or []:
            try:
                logger.experiment.add_tag(tag)
            except (AttributeError, TypeError):  # CSVLogger, offline Comet, stubs
                continue

    def _write_report(self, undertrained: bool, rel_improvement: float) -> None:
        """Persist the curve and both verdicts next to the fold's other outputs."""
        if self.save_dir is None:
            return
        report: Dict[str, Any] = {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "window": self.window,
            "min_rel_improvement": self.min_rel_improvement,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "final_value": self.history[-1]["value"] if self.history else None,
            "final_epoch": self.history[-1]["epoch"] if self.history else None,
            "degraded": self.degraded,
            "degraded_at_epoch": self.degraded_at_epoch,
            "undertrained": undertrained,
            "final_rel_improvement": rel_improvement,
            "history": self.history,
        }
        out_path = self.save_dir / self.report_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote validation trend report to '{out_path}'")
