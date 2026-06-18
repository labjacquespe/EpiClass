"""Naming and discovery helpers for prediction CSVs.

Prediction CSVs are written per cross-validation fold into a ``splitX`` / ``fold_X``
directory (see ``core/analysis.py``). Two concerns live here:

* **Provenance tag** -- re-running training into the same ``splitX`` produces a new
  comet-ml experiment and checkpoint but historically overwrote ``validation_prediction.csv``.
  ``build_prediction_tag`` derives a ``"<cometid>_<ckptstem>"`` suffix so each run's file is
  uniquely named and traceable.
* **Discovery** -- once filenames carry a tag, downstream consumers can no longer glob the
  exact ``split*/validation_prediction.csv``. ``resolve_split_prediction_csvs`` returns the
  newest prediction CSV per fold, tolerating both the legacy and tagged names.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

# Fold sub-directory naming used by the training scripts: ``split0..`` (epiatlas_training)
# and ``fold_0..`` (general_training).
SPLIT_DIR_GLOBS = ("split*", "fold_*")

# A comet-ml experiment key is a 32-character lowercase hex string.
_COMET_KEY_RE = re.compile(r"^[0-9a-f]{32}$")


def experiment_id_from_checkpoint(checkpoint_path: Optional[Path | str]) -> Optional[str]:
    """Recover the original *training* comet-ml experiment id from a checkpoint path.

    Comet-logged training (``epiatlas_training``) saves checkpoints under
    ``<split>/<project>/<experiment_id>/checkpoints/<file>.ckpt`` -- the experiment id is the
    checkpoint's grandparent directory. Returns it when that grandparent looks like a comet
    key (32 hex chars), else ``None`` (e.g. ``general_training``'s CSVLogger uses
    ``logs/version_N/checkpoints`` instead, which carries no experiment id).
    """
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if path.parent.name != "checkpoints":
        return None
    candidate = path.parent.parent.name
    return candidate if _COMET_KEY_RE.match(candidate) else None


def build_prediction_tag(
    experiment_id: Optional[str], checkpoint_path: Optional[Path | str]
) -> str:
    """Join the non-empty provenance parts into a filename-safe tag.

    Returns ``"<cometid>_<ckptstem>"``, ``"<ckptstem>"``, ``"<cometid>"`` or ``""`` depending
    on which parts are available. Empty string means "no tag" -- callers fall back to the
    bare ``{name}_prediction.csv``.
    """
    parts = []
    if experiment_id:
        parts.append(str(experiment_id))
    if checkpoint_path:
        parts.append(Path(checkpoint_path).stem)
    return "_".join(parts)


def resolve_split_prediction_csvs(
    run_dir: Path | str, set_name: str = "validation"
) -> Dict[str, Path]:
    """Return ``{fold_dir_name: newest_prediction_csv}`` for a CV run directory.

    For each ``split*`` / ``fold_*`` sub-directory of ``run_dir``, globs
    ``{set_name}_prediction*.csv`` and keeps the newest match by modification time (so a
    re-trained fold's latest tagged file wins deterministically). Folds with no matching
    file are skipped. Result is ordered by fold directory name.
    """
    run_dir = Path(run_dir)
    resolved: Dict[str, Path] = {}
    seen: set[Path] = set()
    for glob in SPLIT_DIR_GLOBS:
        for split_dir in sorted(run_dir.glob(glob)):
            if not split_dir.is_dir() or split_dir in seen:
                continue
            seen.add(split_dir)
            candidates = list(split_dir.glob(f"{set_name}_prediction*.csv"))
            if not candidates:
                continue
            resolved[split_dir.name] = max(candidates, key=lambda p: p.stat().st_mtime)
    return dict(sorted(resolved.items()))
