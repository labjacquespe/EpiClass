# CLAUDE.md

This file provides guidance to coding agents (e.g. Claude Code) when working with code in this repository.

## Project Overview

EpiClass (Epigenomic Classifier) is a framework for training machine learning models (neural networks, LGBM, Random Forest, etc.) to classify and label epigenomic data. It uses PyTorch Lightning for neural network training and scikit-learn/LightGBM for traditional ML models.

## Repository Layout

- `src/python/` — Main Python package root (where `pyproject.toml` lives)
  - `epiclass/` — Core package
    - `core/` — Data loading (`loaders/`, `data/`), model (`model_pytorch.py`), training (`trainer.py`), metadata (`metadata.py`), estimators (`estimators.py`)
    - `utils/` — Analysis scripts, notebooks, HDF5 utilities, SHAP analysis
    - Top-level scripts: `epiatlas_training.py`, `predict.py`, `compute_shaps.py`, `other_estimators.py`
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

```bash
# Uncompress fixtures first (only needed once)
cd src/python/tests && tar -xf fixtures.tar.xz

# Run all tests (from src/python/)
pytest tests
```

The `tests/justfile` provides multi-version test orchestration via `uv`:

```bash
cd src/python/tests
just test 3.11              # run tests for Python 3.11
just test 3.11 "test_name"  # filter tests
just test-all               # run tests for Python 3.10, 3.11, 3.12 in parallel
```

## Code Style

- **Formatter**: Black (line length 90)
- **Import sorting**: isort (black-compatible profile, line length 90)
- **Linter**: pylint (config in `pyproject.toml`)
- Pre-commit hooks enforce formatting on both `.py` files and notebooks (via nbQA)
- Python 3.10–3.12 supported; originally developed with 3.8

## Architecture Notes

- **Data pipeline**: Epigenomic signal data is stored in HDF5 files (created by epigeec tool). `hdf5_loader.py` loads them, `dataset_factory.py` creates PyTorch datasets. Chromosome sizes files + resolution define the genomic bins.
- **Metadata**: The `Metadata` class (`core/metadata.py`) handles label management. Any value (including empty strings) in a label category is treated as a valid label — use `remove_missing_labels()` to clean.
- **Training flow**: `epiatlas_training.py` does cross-validation; `epiatlas_training_no_valid.py` trains without validation (for final models). Both use PyTorch Lightning via `trainer.py`.
- **Non-NN models**: `other_estimators.py` supports LinearSVC, Random Forest, Logistic Regression, and LGBM with Bayesian hyperparameter search.
- **Experiment tracking**: Comet-ML integration (supports `--offline` mode).
- **Models are published** to Hugging Face under the "EpiClass models" collection.
