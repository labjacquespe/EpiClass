"""Unit tests for ValidationTrendMonitor (core/trainer.py).

The callback is driven directly with a stub trainer: its whole job is reading a
metric off `callback_metrics` and deciding what the curve means, so a real
Lightning loop would only add minutes without adding coverage.
"""
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch

from epiclass.core.trainer import ValidationTrendMonitor


@contextmanager
def _no_warnings():
    """Assert that no UserWarning escapes the block."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


class _FakeLogger:
    """Logger stub recording the tags the callback pushes at it."""

    def __init__(self):
        self.tags: List[str] = []
        self.experiment = SimpleNamespace(add_tag=self.tags.append)


def _run(monitor: ValidationTrendMonitor, values, logger=None) -> _FakeLogger:
    """Feed `values` to the callback as one validation check per epoch."""
    logger = logger if logger is not None else _FakeLogger()
    trainer = SimpleNamespace(
        sanity_checking=False,
        callback_metrics={},
        current_epoch=0,
        loggers=[logger],
    )
    for epoch, value in enumerate(values):
        trainer.current_epoch = epoch
        trainer.callback_metrics[monitor.monitor] = torch.tensor(value)
        monitor.on_validation_end(trainer, None)
    monitor.on_fit_end(trainer, None)
    return logger


def test_monotone_decrease_is_silent(tmp_path: Path):
    """The assumed-normal case: no warning, but a report is still written."""
    mon = ValidationTrendMonitor(patience=3, window=3, save_dir=tmp_path)
    with _no_warnings():
        logger = _run(mon, [10.0, 8.0, 6.5, 6.0, 5.99, 5.985, 5.98])

    assert not mon.degraded
    assert logger.tags == []
    report = json.loads((tmp_path / "valid_trend_report.json").read_text())
    assert report["degraded"] is False
    assert report["undertrained"] is False
    assert report["best_epoch"] == 6
    assert len(report["history"]) == 7


def test_rising_loss_warns_but_never_stops(tmp_path: Path):
    """A sustained rise trips the degraded verdict; training is not touched."""
    mon = ValidationTrendMonitor(patience=3, window=3, save_dir=tmp_path)
    with pytest.warns(UserWarning, match="valid_lossDegraded"):
        logger = _run(mon, [10.0, 5.0, 5.5, 5.6, 5.7, 5.8])

    assert mon.degraded
    assert mon.degraded_at_epoch == 4  # third consecutive non-improving check
    assert mon.best_epoch == 1
    assert "valid_lossDegraded" in logger.tags
    assert json.loads((tmp_path / "valid_trend_report.json").read_text())["degraded"]


def test_single_blip_does_not_warn(tmp_path: Path):
    """Epoch-to-epoch noise below `patience` must not trip the warning."""
    mon = ValidationTrendMonitor(patience=3, window=2, save_dir=tmp_path)
    with _no_warnings():
        _run(mon, [10.0, 5.0, 5.2, 4.8, 4.9, 4.79])
    assert not mon.degraded


def test_still_improving_at_last_epoch_warns(tmp_path: Path):
    """Budget too short: the curve is still descending steeply at the end."""
    mon = ValidationTrendMonitor(patience=3, window=3, save_dir=tmp_path)
    with pytest.warns(UserWarning, match="valid_lossUndertrained"):
        logger = _run(mon, [100.0, 80.0, 60.0, 40.0, 20.0, 10.0])

    assert "valid_lossUndertrained" in logger.tags
    report = json.loads((tmp_path / "valid_trend_report.json").read_text())
    assert report["undertrained"] is True
    assert report["final_rel_improvement"] > 0.01


def test_max_mode_tracks_the_other_direction(tmp_path: Path):
    """`mode='max'` flips both verdicts; used if a classifier ever opts in."""
    mon = ValidationTrendMonitor(
        monitor="valid_acc", mode="max", patience=2, window=2, save_dir=tmp_path
    )
    with pytest.warns(UserWarning, match="valid_accDegraded"):
        _run(mon, [0.5, 0.9, 0.85, 0.84])
    assert mon.best_value == pytest.approx(0.9)


def test_sanity_check_values_are_ignored(tmp_path: Path):
    """Lightning's sanity pass must not enter the history."""
    mon = ValidationTrendMonitor(save_dir=tmp_path)
    trainer = SimpleNamespace(
        sanity_checking=True,
        callback_metrics={"valid_loss": torch.tensor(999.0)},
        current_epoch=0,
        loggers=[],
    )
    mon.on_validation_end(trainer, None)
    assert not mon.history


def test_short_run_abstains_on_the_budget_verdict(tmp_path: Path):
    """Fewer than `window + 1` points is not enough to judge the budget."""
    mon = ValidationTrendMonitor(patience=5, window=5, save_dir=tmp_path)
    with _no_warnings():
        _run(mon, [10.0, 1.0])
    report = json.loads((tmp_path / "valid_trend_report.json").read_text())
    assert report["undertrained"] is False


def test_bad_mode_rejected():
    """A typo in `mode` fails at construction, not silently at epoch 1."""
    with pytest.raises(ValueError, match="mode must be"):
        ValidationTrendMonitor(mode="minimum")
