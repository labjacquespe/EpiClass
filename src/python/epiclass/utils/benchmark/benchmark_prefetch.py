"""Benchmark the effect of DataLoader prefetching parameters on lazy training.

Sweeps DataLoader knobs (num_workers, prefetch_factor, pin_memory,
persistent_workers, batch_size) *and* the number of physical CPU cores actually
available to the workers, on a realistic short EpiAtlas training run, to find the
cheapest configuration that keeps the GPU fed.

Design:

- One HPC job requests up to N cores (``--cpus-per-task``) and this script sweeps
  *subsets* internally by capping CPU affinity per run
  (``os.sched_setaffinity``) instead of launching one job per allocation.
- Each configuration runs in a **fresh subprocess** so affinity, DataLoader
  workers, CUDA and page-cache state never bleed between runs.
- DataLoader params reach the pipeline through the production seam
  (``EPICLASS_*`` env vars read by ``utils.torch_data._create_lazy``).
- The dataset mmap page cache is **evicted** (``posix_fadvise(DONTNEED)``) before
  the timed region so the benchmark measures the data-``>``-RAM regime where
  prefetching actually matters, not a warm best-case.

Usage::

    python -m epiclass.utils.benchmark.benchmark_prefetch \
        --config config.json --logdir out/

Prefetching only matters with a real GPU; pass ``--allow-cpu`` only for smoke
tests. The ``--single-run`` mode is internal (spawned per configuration).

Implementation note -- ``import-outside-toplevel`` is disabled module-wide on
purpose: torch / lightning / pandas / epiclass-core are imported inside the
functions that use them, not at module top, for two reasons. (1) The
orchestrator process (and the config-expansion unit tests) never train, so they
must not pay the heavy DL-stack import cost or require a GPU-enabled torch just
to spawn subprocesses / expand a sweep. (2) In a child, ``torch`` must be
imported *after* ``cap_cpus()`` sets CPU affinity (with OMP/MKL thread caps
already in the env) so its intra-op thread pools are sized against the capped
core set. This mirrors the existing deferral in ``utils/torch_data.py``.
"""
# pylint: disable=import-outside-toplevel, too-many-positional-arguments
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.utils.check_dir import create_dirs

# Order defines the Cartesian-product iteration order and the result columns.
SWEEP_KEYS = [
    "num_workers",
    "num_physical_cpus",
    "prefetch_factor",
    "pin_memory",
    "persistent_workers",
    "batch_size",
]


# ---------------------------------------------------------------------------
# Config expansion (orchestrator)
# ---------------------------------------------------------------------------


def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Force prefetch/persistent off when there are no workers.

    ``num_workers == 0`` runs data loading in the main process; torch rejects
    ``prefetch_factor`` / ``persistent_workers`` there, and the pipeline forces
    them off anyway, so collapse them to canonical values to avoid spurious
    duplicate configurations.
    """
    cfg = dict(cfg)
    if int(cfg.get("num_workers", 0)) == 0:
        cfg["prefetch_factor"] = None
        cfg["persistent_workers"] = False
    return cfg


def expand_configs(sweep: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand a sweep dict into the de-duplicated list of configurations."""
    keys = [k for k in SWEEP_KEYS if k in sweep] + [
        k for k in sweep if k not in SWEEP_KEYS
    ]
    combos: List[Dict[str, Any]] = []
    seen = set()
    for values in itertools.product(*(sweep[k] for k in keys)):
        cfg = _normalize_config(dict(zip(keys, values)))
        # Keys are unique, so sorting by (key, value) never compares heterogen.
        # values against each other -- safe despite mixed None/bool/int.
        signature = tuple(sorted(cfg.items()))
        if signature in seen:
            continue
        seen.add(signature)
        combos.append(cfg)
    return combos


# ---------------------------------------------------------------------------
# Page-cache control (shared)
# ---------------------------------------------------------------------------


def resolve_mmap_path(mmap_dir: Path | str) -> Optional[Path]:
    """Return the preloaded signals mmap under ``mmap_dir`` (or None)."""
    matches = sorted(Path(mmap_dir).glob("signals*.npy"))
    return matches[0] if matches else None


def drop_page_cache(path: Optional[Path]) -> None:
    """Evict ``path``'s clean pages from the OS page cache (no root needed)."""
    if path is None or not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
        finally:
            os.close(fd)
    except OSError as err:
        print(f"Warning: could not drop page cache for {path}: {err}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _child_env(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Build the subprocess env carrying the DataLoader + CPU-cap knobs.

    ``OMP``/``MKL`` thread caps must be set here (before the child imports
    numpy/torch) to take effect; affinity is applied inside the child.
    """
    env = os.environ.copy()
    env["EPICLASS_NUM_WORKERS"] = str(cfg["num_workers"])
    env["EPICLASS_PIN_MEMORY"] = "1" if cfg["pin_memory"] else "0"
    env["EPICLASS_PERSISTENT_WORKERS"] = "1" if cfg["persistent_workers"] else "0"
    if cfg.get("prefetch_factor") is not None:
        env["EPICLASS_PREFETCH_FACTOR"] = str(cfg["prefetch_factor"])
    else:
        env.pop("EPICLASS_PREFETCH_FACTOR", None)
    cpus = str(cfg["num_physical_cpus"])
    env["EPICLASS_CPU_LIMIT"] = cpus
    env["OMP_NUM_THREADS"] = cpus
    env["MKL_NUM_THREADS"] = cpus
    return env


def _run_one_subprocess(
    config_id: str,
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    run_cfg: Dict[str, Any],
    mmap_dir: Path,
    logdir: Path,
    allow_cpu: bool,
) -> Dict[str, Any]:
    """Run a single configuration in a fresh subprocess; return its result row."""
    run_dir = logdir / config_id
    create_dirs(run_dir)
    spec_path = run_dir / "run_spec.json"
    result_path = run_dir / "result.json"
    spec = {
        "config_id": config_id,
        "config": cfg,
        "data": data,
        "run": run_cfg,
        "mmap_dir": str(mmap_dir),
        "logdir": str(run_dir),
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-run",
        "--run-spec",
        str(spec_path),
        "--result-file",
        str(result_path),
    ]
    if allow_cpu:
        cmd.append("--allow-cpu")

    print(f"\n{'='*70}\n[{config_id}] {cfg}\n{'='*70}", flush=True)
    completed = subprocess.run(cmd, env=_child_env(cfg), check=False)

    row: Dict[str, Any] = {"config_id": config_id, **cfg}
    if completed.returncode != 0 or not result_path.exists():
        row["error"] = f"subprocess exit {completed.returncode}"
        print(f"[{config_id}] FAILED (exit {completed.returncode})", file=sys.stderr)
        return row
    row.update(json.loads(result_path.read_text(encoding="utf-8")))
    return row


def _append_result_csv(results: List[Dict[str, Any]], logdir: Path) -> None:
    """Rewrite the results CSV (cheap; keeps partial progress on crash)."""
    import pandas as pd

    pd.DataFrame(results).to_csv(logdir / "benchmark_results.csv", index_label="row")


def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print the GPU-bound floor and flag each config GPU- vs I/O-bound."""
    ok = [r for r in results if r.get("steady_epoch_s")]
    if not ok:
        print("\nNo successful runs to summarize.")
        return
    floor = min(r["steady_epoch_s"] for r in ok)
    tol = 1.10  # within 10% of the fastest => GPU no longer waiting
    print("\n=== Summary (cold steady-state epoch time) ===")
    print(f"GPU-bound floor (fastest steady epoch): {floor:.2f}s\n")
    for r in sorted(ok, key=lambda x: x["steady_epoch_s"]):
        bound = "GPU-bound" if r["steady_epoch_s"] <= floor * tol else "I/O-bound"
        print(
            f"  {r['config_id']}: cpus={r['num_physical_cpus']} "
            f"workers={r['num_workers']} prefetch={r.get('prefetch_factor')} "
            f"batch={r['batch_size']} -> {r['steady_epoch_s']:.2f}s "
            f"({r.get('samples_per_s', 0):.0f} samples/s) [{bound}]"
        )


def _validate_data_paths(data: Dict[str, Any]) -> None:
    """Fail fast if any input file in the config is missing."""
    for key in ("hyperparameters", "hdf5_list", "chromsize", "metadata"):
        path = Path(data[key])
        if not path.is_file():
            raise FileNotFoundError(f"config data.{key} not found: {path}")


def run_orchestrator(
    config: Dict[str, Any],
    logdir: Path,
    allow_cpu: bool,
    mmap_dir_override: Optional[Path] = None,
) -> None:
    """Expand the sweep and run every configuration in its own subprocess.

    ``mmap_dir_override`` (from ``--mmap-dir``) wins over the config's
    ``mmap_dir``; on HPC set it to ``$SLURM_TMPDIR`` for fast node-local disk.
    """
    data = config["data"]
    _validate_data_paths(data)
    run_cfg = config.get("run", {})
    repeats = int(run_cfg.get("repeats", 1))
    mmap_dir_cfg = mmap_dir_override or config.get("mmap_dir")
    mmap_dir = Path(mmap_dir_cfg) if mmap_dir_cfg else logdir / "mmap_cache"
    create_dirs(mmap_dir)

    avail: int = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()  # type: ignore
    )
    configs = expand_configs(config["sweep"])
    print(
        f"{len(configs)} unique configuration(s) x {repeats} repeat(s); "
        f"{avail} CPU core(s) available to this job."
    )

    results: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(configs):
        if int(cfg["num_physical_cpus"]) > avail:
            print(
                f"  [skip] cfg{idx:03d}: num_physical_cpus="
                f"{cfg['num_physical_cpus']} > {avail} available"
            )
            continue
        for rep in range(repeats):
            config_id = f"cfg{idx:03d}_rep{rep}"
            row = _run_one_subprocess(
                config_id, cfg, data, run_cfg, mmap_dir, logdir, allow_cpu
            )
            results.append(row)
            _append_result_csv(results, logdir)

    _print_summary(results)
    print(f"\nResults written to {logdir / 'benchmark_results.csv'}")


# ---------------------------------------------------------------------------
# Single-run (child)
# ---------------------------------------------------------------------------


def cap_cpus() -> int:
    """Restrict this process (and forked workers) to EPICLASS_CPU_LIMIT cores.

    Returns the number of cores actually pinned. DataLoader worker processes are
    forked and inherit this affinity mask, so a run configured for k physical
    CPUs genuinely contends for k cores even when the job allocated more.
    """
    limit = os.getenv("EPICLASS_CPU_LIMIT")
    if limit is None or not hasattr(os, "sched_setaffinity"):
        return os.cpu_count() or 1
    k = int(limit)
    avail = sorted(os.sched_getaffinity(0))
    use = avail[:k] if 0 < k < len(avail) else avail
    os.sched_setaffinity(0, set(use))
    print(f"CPU affinity capped to {len(use)} core(s): {sorted(os.sched_getaffinity(0))}")
    return len(use)


def build_epiatlas_fold(data: Dict[str, Any], mmap_dir: Path):
    """Build the first EpiAtlas CV fold, mirroring epiatlas_training.run prep.

    Category-specific remappings (paired/random/ontology pairs) are intentionally
    omitted -- the benchmark only needs a realistic fold, not the exact training
    label set.
    """
    from epiclass.core import metadata as md
    from epiclass.core.data_source import EpiDataSource
    from epiclass.core.lazy.lazy_fold_factory import (
        LazyEpiAtlasFoldFactory as EpiAtlasFoldFactory,
    )

    datasource = EpiDataSource(
        Path(data["hdf5_list"]), Path(data["chromsize"]), Path(data["metadata"])
    )
    with open(data["hyperparameters"], "r", encoding="utf-8") as file:
        hparams: Dict[str, Any] = json.load(file)

    category = data["category"]
    my_metadata = md.UUIDMetadata(datasource.metadata_file)
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )
    my_metadata.remove_missing_labels(category)
    label_list = md.env_filtering(my_metadata, category)

    # MIN_CLASS_SIZE env wins over the hparam, mirroring epiatlas_training.run.
    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = hparams.get("min_class_size", 10)

    factory = EpiAtlasFoldFactory.from_datasource(
        datasource,
        category,
        label_list,
        n_fold=hparams.get("n_fold", 10),
        test_ratio=0,
        min_class_size=min_class_size,
        force_filter=True,
        metadata=my_metadata,
        mmap_dir=mmap_dir,
    )
    oversample = hparams.get("oversample", hparams.get("oversampling", True))
    my_data = next(iter(factory.yield_split(oversample=oversample)))
    return my_data, hparams


def _make_epoch_timer(mmap_path: Optional[Path], drop_between: bool):
    """Return a Lightning Callback recording per-epoch wall times.

    When ``drop_between`` is set it evicts the mmap page cache at the start of
    every epoch, keeping each measured epoch cold (working set >> RAM).
    """
    from lightning.pytorch.callbacks import Callback

    class _EpochTimer(Callback):
        def __init__(self) -> None:
            self.epoch_times: List[float] = []
            self._t0: float = 0.0

        def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: D401
            if drop_between:
                drop_page_cache(mmap_path)
            self._t0 = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            self.epoch_times.append(time.perf_counter() - self._t0)

    return _EpochTimer()


def _time_dataloader_pass(loader, n_batches: int) -> float:
    """Time one bare (model-free) iteration over ``loader``, up to n_batches."""
    t0 = time.perf_counter()
    for i, _ in enumerate(loader):
        if i + 1 >= n_batches:
            break
    return time.perf_counter() - t0


def _build_model(my_data, hparams: Dict[str, Any], logdir: Path):
    """Build the classifier, sizes derived from the fold (as do_one_experiment)."""
    from epiclass.core.model_pytorch import LightningDenseClassifier

    mapping_file = Path(logdir) / "training_mapping.tsv"
    my_data.save_mapping(mapping_file)
    mapping = my_data.load_mapping(mapping_file)
    return LightningDenseClassifier(
        input_size=my_data.train.signal_length,
        output_size=len(my_data.classes),
        mapping=mapping,
        hparams=hparams,
        hl_units=int(os.getenv("LAYER_SIZE", "3000")),
        nb_layer=int(os.getenv("NB_LAYER", "1")),
    )


def run_single(spec_path: Path, result_file: Path, allow_cpu: bool) -> None:
    """Execute one configuration end to end and write its result JSON."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    cfg = spec["config"]
    data = spec["data"]
    run_cfg = spec["run"]
    mmap_dir = Path(spec["mmap_dir"])
    logdir = Path(spec["logdir"])

    n_cores = cap_cpus()

    import torch

    torch.set_num_threads(n_cores)
    gpu = torch.cuda.is_available()
    if not gpu and not allow_cpu:
        raise RuntimeError(
            "No GPU detected. Prefetch benchmarking is only meaningful on a GPU "
            "node; pass --allow-cpu to run anyway (results not representative)."
        )

    from epiclass.core.trainer import MyTrainer
    from epiclass.utils.torch_data import create_torch_datasets

    my_data, hparams = build_epiatlas_fold(data, mmap_dir)
    mmap_path = resolve_mmap_path(mmap_dir)
    batch_size = int(cfg["batch_size"])

    dsets = create_torch_datasets(my_data, batch_size)  # reads EPICLASS_* env vars
    _, train_loader = dsets["training"]

    n_batches = len(train_loader)
    limit = run_cfg.get("limit_train_batches")
    if isinstance(limit, int):
        n_batches = min(n_batches, limit)

    # Model-free cold delivery pass isolates pure I/O throughput from GPU compute.
    dataloader_only_s: Optional[float] = None
    if run_cfg.get("warmup_dataloader_pass", True):
        drop_page_cache(mmap_path)
        dataloader_only_s = _time_dataloader_pass(train_loader, n_batches)

    model = _build_model(my_data, hparams, logdir)
    epochs = int(run_cfg.get("epochs", hparams.get("training_epochs", 4)))
    drop_between = bool(run_cfg.get("drop_cache_between_epochs", True))
    timer = _make_epoch_timer(mmap_path, drop_between)

    trainer_kwargs: Dict[str, Any] = {
        "general_log_dir": str(logdir),
        "model": model,
        "max_epochs": epochs,
        "limit_train_batches": limit if isinstance(limit, int) else 1.0,
        "limit_val_batches": 0,  # validation excluded: measure training I/O only
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "callbacks": [timer],
        "accelerator": "gpu" if gpu else "cpu",
        "devices": 1,
    }
    if gpu:
        trainer_kwargs["precision"] = 16
    trainer = MyTrainer(**trainer_kwargs)

    drop_page_cache(mmap_path)  # cold epoch 0 even when drop_between is False
    fit_start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, verbose=False)
    total_fit_s = time.perf_counter() - fit_start

    epoch_times = timer.epoch_times
    first_epoch_s = epoch_times[0] if epoch_times else None
    steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    steady_epoch_s = sum(steady) / len(steady) if steady else None
    samples_per_s = n_batches * batch_size / steady_epoch_s if steady_epoch_s else None
    mmap_bytes = mmap_path.stat().st_size if mmap_path and mmap_path.exists() else None

    result = {
        "n_train_samples": my_data.train.num_examples,
        "n_batches": n_batches,
        "epochs": epochs,
        "first_epoch_s": first_epoch_s,
        "steady_epoch_s": steady_epoch_s,
        "total_fit_s": total_fit_s,
        "samples_per_s": samples_per_s,
        "dataloader_only_s": dataloader_only_s,
        "mmap_bytes": mmap_bytes,
        "cores_pinned": n_cores,
        "device": torch.cuda.get_device_name() if gpu else "cpu",
    }
    Path(result_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResult: {json.dumps(result)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse orchestrator / single-run arguments."""
    arg_parser = ArgumentParser()
    # fmt: off
    arg_parser.add_argument(
        "--config", type=Path, help="Benchmark config JSON (sweep + data + run).",
    )
    arg_parser.add_argument(
        "--logdir", type=Path, help="Output directory (created if absent).",
    )
    arg_parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Permit running without a GPU (smoke tests only; not representative).",
    )
    arg_parser.add_argument(
        "--mmap-dir", type=Path, default=None,
        help="Override the config's mmap_dir (shared across runs). On HPC set to "
             "$SLURM_TMPDIR for fast node-local disk.",
    )
    # Internal flags: the orchestrator re-invokes this script per config via subprocess
    # See module docstring for design details
    arg_parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    arg_parser.add_argument("--run-spec", type=Path, help=argparse.SUPPRESS)
    arg_parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    # fmt: on
    return arg_parser.parse_args()


def main() -> None:
    """Entry point: dispatch to orchestrator or single-run mode."""
    cli = parse_arguments()

    if cli.single_run:
        if cli.run_spec is None or cli.result_file is None:
            raise SystemExit("--single-run requires --run-spec and --result-file.")
        run_single(cli.run_spec, cli.result_file, cli.allow_cpu)
        return

    if cli.config is None or cli.logdir is None:
        raise SystemExit("--config and --logdir are required.")
    create_dirs(cli.logdir)
    with open(cli.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    run_orchestrator(config, cli.logdir, cli.allow_cpu, mmap_dir_override=cli.mmap_dir)


if __name__ == "__main__":
    main()
