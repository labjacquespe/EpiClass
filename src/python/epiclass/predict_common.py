"""Shared data-loading helpers for the prediction entry points.

``predict.py`` (single model) and ``predict_CV.py`` (ensemble over all CV fold models) both
turn a list of unlabeled samples into a ``TensorDataset`` the exact same way -- the only thing
that differs is how many models consume it. Keeping the loader / dataset construction here lets
the ensemble build the data **once** and reuse it across every fold model.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import signal
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import analysis
from epiclass.core.blas_guard import check_blas_backend
from epiclass.core.data.dataset import DataSet
from epiclass.core.lazy.chunked_hdf5_loader import ChunkedHdf5Loader
from epiclass.core.lazy.lazy_data_classes import LazyUnknownData, SignalLoader
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.core.model_pytorch import LightningDenseClassifier


def add_data_arguments(
    arg_parser: argparse.ArgumentParser, hdf5_flag: bool = False
) -> None:
    """Register the input-format / data-loading CLI args shared by predict entry points.

    ``hdf5_flag`` controls how the input is exposed: as the required ``--hdf5`` flag
    (``predict.py``, all-flag interface) when True, or as a positional (``predict_CV.py``)
    when False. Either way the value lands on ``cli.hdf5``.
    """
    hdf5_help = (
        "For single format: file listing HDF5 paths. "
        "For chunked format: directory or file of chunk HDF5s."
    )
    # fmt: off
    if hdf5_flag:
        arg_parser.add_argument(
            "--hdf5", dest="hdf5", type=Path, required=True, help=hdf5_help,
        )
    else:
        arg_parser.add_argument("hdf5", type=Path, help=hdf5_help)
    arg_parser.add_argument(
        "--chromsize", type=Path,
        help="Chromosome sizes file. Required for single-sample HDF5 format.",
    )
    arg_parser.add_argument(
        "--chunked", action="store_true",
        help="Input is chunked HDF5 format (e.g. produced by convert_to_chunked.py). "
             "If not set, single-sample HDF5 format is assumed.",
    )
    arg_parser.add_argument(
        "--mmap_dir", type=Path, default=None,
        help="Directory for the mmap cache (single format only). "
             "Defaults to a 'mmap_cache' dir under the output directory. "
             "On HPC, set to $SLURM_TMPDIR.",
    )
    arg_parser.add_argument(
        "--hdf5_dir", type=Path,
        help="Override HDF5 file paths to this directory (single format). "
             "Useful when HDF5s are copied to $SLURM_TMPDIR.",
    )
    arg_parser.add_argument(
        "--batch-size", type=int, default=256, dest="batch_size",
        help="Inference batch size. Default: 256.",
    )
    # fmt: on


def _env_flag(name: str) -> bool:
    """Return True if environment variable ``name`` is set to a truthy value."""
    return os.getenv(name, "") not in ("", "0", "false", "False")


def enable_diagnostics() -> None:
    """Install on-demand traceback dumping when ``EPICLASS_DIAG`` is set.

    Off by default — does nothing unless the env var is set, so normal runs are
    untouched. When enabled it wires up two opt-in debug probes:

      - ``faulthandler.enable()``: a fatal signal (SIGSEGV/SIGBUS) prints the
        Python stack to stderr.
      - ``SIGUSR1`` handler: ``kill -USR1 <pid>`` dumps the current stack of every
        thread, so a running process (crashing, hung, or just slow) can be
        inspected without py-spy. The PID is printed at startup for convenience.
    """
    if not _env_flag("EPICLASS_DIAG"):
        return
    faulthandler.enable()
    if hasattr(faulthandler, "register"):  # POSIX only
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        print(
            f"[diagnostics] PID {os.getpid()}: "
            f"`kill -USR1 {os.getpid()}` to dump the Python stack."
        )


def configure_inference_backend() -> None:
    """Optionally disable torch's oneDNN/MKLDNN backend when ``EPICLASS_DISABLE_MKLDNN`` is set.

    Off by default. NOTE: this is *not* a fix for the cluster CPU-inference segfault. That
    crash is an out-of-bounds read in BLIS's fp32 sgemm kernel (FlexiBLAS backend), not in
    oneDNN; disabling MKLDNN was tested and did not prevent it. The real workaround is to
    select a different BLAS backend, e.g. ``export FLEXIBLAS=openblas``. Kept only as a
    generic toggle for experimenting with the oneDNN dispatch path.
    """
    if _env_flag("EPICLASS_DISABLE_MKLDNN"):
        torch.backends.mkldnn.enabled = False
        print("EPICLASS_DISABLE_MKLDNN set: torch oneDNN/MKLDNN backend disabled.")


def prepare_inference_runtime() -> None:
    """Prepare the runtime for the predict entry points.

    Always guards against the known BLIS < 1.1 CPU-inference segfault
    (:func:`check_blas_backend`). Also bundles the opt-in :func:`enable_diagnostics` and
    :func:`configure_inference_backend`, both no-ops unless their env vars are set, so
    ``predict.py`` and ``predict_CV.py`` call one thing at startup.
    """
    enable_diagnostics()
    configure_inference_backend()
    check_blas_backend()


def build_loader(
    cli: argparse.Namespace, mmap_default_dir: Path
) -> tuple[SignalLoader, list[str]]:
    """Return (loader, ordered sample_ids) for the selected input format.

    ``mmap_default_dir`` is the fallback mmap cache directory used for the single-sample
    format when ``--mmap_dir`` is not given.
    """
    if cli.chunked:
        loader = ChunkedHdf5Loader()
        loader.register_chunked_hdf5s(cli.hdf5, strict=True)
        return loader, loader.sample_ids

    if cli.chromsize is None:
        raise ValueError(
            "--chromsize is required for single-sample HDF5 format. "
            "Use --chunked if your data is in chunked format."
        )
    mmap_dir = cli.mmap_dir if cli.mmap_dir is not None else mmap_default_dir
    loader = LazyHdf5Loader(
        chrom_file=cli.chromsize,
        normalization=True,
        mmap_dir=mmap_dir,
    )
    # strict=True opens every HDF5 to validate it exists — necessary when the mmap
    # cache must be built, but pointless (and very slow on Lustre: thousands of
    # small-file opens) when the cache already exists, since the data is then read
    # from the .npy. On reuse, preload_all's row-count check verifies the cache
    # matches the file list instead.
    loader.register_hdf5s(
        cli.hdf5, hdf5_dir=cli.hdf5_dir, strict=not loader.mmap_exists()
    )
    loader.preload_all()
    return loader, list(loader.file_paths.keys())


def build_test_dataset(
    cli: argparse.Namespace, mmap_default_dir: Path
) -> Tuple[LazyUnknownData, TensorDataset, DataSet]:
    """Build the unlabeled test data once: (LazyUnknownData, TensorDataset, DataSet).

    The ``TensorDataset`` drives inference; the ``DataSet`` carries the sample ids used to
    label prediction-CSV rows. Raises ``ValueError`` when no samples are registered.
    """
    loader, sample_ids = build_loader(cli, mmap_default_dir)
    n = len(sample_ids)

    test_data = LazyUnknownData(
        ids=sample_ids,
        loader=loader,
        y=np.zeros(n, dtype=np.int64),
        y_str=[""] * n,
    )
    if test_data.num_examples == 0:
        raise ValueError("Trying to predict without any test data.")

    signals, labels = test_data.materialize()
    test_dataset = TensorDataset(
        torch.from_numpy(signals).float(),
        torch.from_numpy(labels).int(),
    )

    datasets = DataSet.empty_collection(data_class=LazyUnknownData)
    datasets.set_test(test_data)
    return test_data, test_dataset, datasets


def write_test_predictions(
    model: LightningDenseClassifier,
    datasets: DataSet,
    test_dataset: TensorDataset,
    path: Path,
    batch_size: int = 256,
) -> None:
    """Score ``test_dataset`` with ``model`` and write the test-prediction CSV to ``path``.

    Inference does not use an experiment logger (``logger=None``): metrics/CSVs are written to
    disk only.
    """
    analyzer = analysis.Analysis(
        model,
        datasets_info=datasets,
        logger=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=test_dataset,
    )
    analyzer.write_test_prediction(path=path, batch_size=batch_size)


# Re-exported so callers needing a directory-validating CLI type don't import argparseutils.
__all__ = [
    "add_data_arguments",
    "check_blas_backend",
    "configure_inference_backend",
    "enable_diagnostics",
    "prepare_inference_runtime",
    "build_loader",
    "build_test_dataset",
    "write_test_predictions",
    "DirectoryChecker",
]
