"""Resolve the mmap cache directory used by training/prediction mains.

See also ``inspect_mmap.py`` for diagnosing an existing mmap cache after the
fact (corruption, reuse hangs).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def resolve_mmap_dir(cli_mmap_dir: Path | None) -> Generator[Path, None, None]:
    """Resolve the mmap cache directory for a training/prediction main.

    Priority: explicit ``--mmap_dir`` (kept as-is, never cleaned up here) >
    ``$SLURM_TMPDIR/mmap_cache`` (node-local scratch, wiped by SLURM at job end,
    not cleaned up here either) > a fresh ``tempfile.mkdtemp()`` directory, removed
    on exit. The cache is a purely temporary artifact, so nothing should default to
    landing under ``<logdir>`` where it would linger and get re-saved every run.
    """
    if cli_mmap_dir is not None:
        yield cli_mmap_dir
        return

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        yield Path(slurm_tmpdir) / "mmap_cache"
        return

    tmp_dir = tempfile.mkdtemp(prefix="epiclass_mmap_")
    try:
        yield Path(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
