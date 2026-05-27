# CLAUDE.md

This file provides guidance to coding agents (e.g. Claude Code) when working with code in this repository.

## Project Overview

EpiClass (Epigenomic Classifier) is a framework for training neural network models to classify and label epigenomic data. It uses PyTorch Lightning for training and supports both single-sample and chunked HDF5 input formats.

## Repository Layout

- `src/python/` — Main Python package root (where `pyproject.toml` lives)
  - `epiclass/` — Core package
    - `core/` — Data loading (`lazy/`), model (`model_pytorch.py`), training (`trainer.py`), metadata (`metadata.py`), analysis (`analysis.py`)
    - `core/lazy/` — Lazy data layer: `lazy_hdf5_loader.py`, `chunked_hdf5_loader.py`, `lazy_data_classes.py`, `lazy_fold_factory.py`, `lazy_torch_dataset.py`
    - `utils/` — Analysis scripts, notebooks, HDF5 utilities, SHAP analysis
    - Top-level scripts: `epiatlas_training.py`, `epiatlas_training_no_valid.py`, `predict.py`, `compute_shaps.py`, `general_training.py`
  - `tests/` — pytest test suite (mirrors `core/`, `mains/`, `utils/` structure)
  - `requirements/` — Pinned dependency files per Python version (`req_test-pyX.Y.txt`)
- `src/R/` — R scripts for metadata handling (ENCODE, recount3, ChIP-Atlas)
- `src/bash_utils/` — SLURM job launcher templates
- `input-format/` — Example input files (HDF5 lists, metadata JSON, hyperparameters, chrom sizes)

## Common Commands

All Python commands should be run from `src/python/`.

### Installation

```bash
cd src/python
python install.py          # auto-detects CPU/GPU for torch
pip install -e .           # base install (editable)
pip install -e .[test]     # install with test dependencies
pip install -e .[dev]      # install with dev tools (black, isort, pylint, pre-commit)
```

### Running Tests

Tests live in `src/python/tests/` and mirror the package layout
(`core/`, `mains/`, `utils/`). pytest's config lives in
`src/python/pyproject.toml` under `[tool.pytest.ini_options]`, and
pytest finds it automatically from any cwd as long as you pass a path
under `src/python/tests/` — so any of these work:

```bash
# One-time: uncompress fixtures
cd src/python/tests && tar -xf fixtures.tar.xz

# All three invocation styles resolve the same rootdir + configfile:
cd src/python      && pytest tests -n auto
cd src/python/tests && pytest -n auto
pytest src/python/tests -n auto                # from repo root

# Examples (paths relative to src/python/):
pytest tests/path/to/file_test.py -v           # single test file
pytest tests/path/to/file_test.py::TestClass::test_method
pytest tests -m slow -s                        # slow-marked tests, show prints
pytest tests -m "not slow" -n auto             # skip slow tests
```

Avoid bare `pytest` (no path) from the repo root — it would try to
collect from the whole repo with no config.

Test markers (declared in `src/python/pyproject.toml`):

- `slow`: integration / long-running tests (skipped unless explicitly selected)
- `embedding`: PCA / UMAP smoke tests (JIT-heavy, dominate suite time)

The `tests/justfile` provides multi-version test orchestration via `uv`:

```bash
cd src/python/tests
just test 3.11              # run tests for Python 3.11
just test 3.11 "test_name"  # filter tests
just test-all               # run tests for Python 3.10, 3.11, 3.12 in parallel
```

### Pre-commit checks (run after editing)

**IMPORTANT — Run pre-commit from the repo root.** If you pass `--files`
paths that don't resolve from the repo root, pre-commit silently reports
"no files to check" for every hook and exits 0 — looking exactly like a
successful run. Either `cd` to the repo root or pass absolute paths.

After modifying any `.py` or `.ipynb` file, validate it with the same
pylint config the pre-commit hook uses — ad-hoc `pylint <file>` skips
the project rcfile and gives different results.

```bash
# Validate only the files you touched (recommended after each task):
pre-commit run --files <path1> <path2> ...

# Or just pylint, same args as the hook:
pylint --rcfile=src/python/pyproject.toml --recursive=y -sn <paths>

# Notebooks have a separate hook (nbqa-pylint); --files runs it too:
pre-commit run --files src/python/.../some_notebook.ipynb
```

`pre-commit run --files ...` also exercises isort, black, nbqa, and the
file-hygiene hooks — catches everything the commit will catch, without
having to actually attempt the commit.

## Code Style

- **Formatter**: Black (line length 90)
- **Import sorting**: isort (black-compatible profile, line length 90)
- **Linter**: pylint (config in `pyproject.toml`)
- Pre-commit hooks enforce formatting on both `.py` files and notebooks (via nbQA)
- Python 3.10–3.12 supported; originally developed with 3.8

## Architecture Notes

- **Data pipeline**: Epigenomic signal data is stored in HDF5 files (created by epigeec tool). Two input formats are supported:
  - *Single-sample HDF5*: one file per sample with per-chromosome datasets. Requires a chromosome sizes file (`--chromsize`). Loaded via `LazyHdf5Loader`, which preloads all samples into a single memory-mapped `.npy` file (`preload_all()`) and exposes it via `as_mmap()`. Chromosome sizes files + resolution define the genomic bins.
  - *Chunked HDF5* (`--chunked`): multi-sample HDF5 files produced by `utils/preprocessing/hdf5_chunks_creation.py`. No chromosome sizes needed. Loaded via `ChunkedHdf5Loader`, which streams chunk-by-chunk (true streaming for PCA; materializes for UMAP due to random-access requirement).
- **Lazy data layer** (`core/lazy/`): all data access goes through this stack. `LazyKnownData` / `LazyUnknownData` hold sample IDs + a loader reference; signals are read on demand. `LazyEpiAtlasFoldFactory` drives cross-validation. `LazyHdf5Dataset` wraps them as a PyTorch `Dataset`. `as_mmap(mmap_mode="c")` is required for UMAP (copy-on-write so numba's jitted kernels accept the array); `mmap_mode="r"` is sufficient for everything else.
- **Metadata**: The `Metadata` class (`core/metadata.py`) handles label management. Any value (including empty strings) in a label category is treated as a valid label — use `remove_missing_labels()` to clean. Track/assay constants live in `core/epiatlas_constants.py`.
- **Training flow**: `epiatlas_training.py` does cross-validation; `epiatlas_training_no_valid.py` trains without validation (for final models). Both use PyTorch Lightning via `trainer.py`. `general_training.py` is the shared backbone used by both.
- **Experiment tracking**: Comet-ML integration (supports `--offline` mode).
- **Models are published** to Hugging Face under the "EpiClass models" collection.
