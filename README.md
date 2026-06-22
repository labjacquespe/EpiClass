# EpiClass - Epigenomic Classifier

EpiClass trains neural network models to classify and label epigenomic data.

## Publication

This repository contains most of the code used to obtain results for the following paper:
[Leveraging the largest harmonized epigenomic data collection for metadata prediction validated and augmented over 350,000 public epigenomic datasets](https://doi.org/10.1101/2025.09.04.670545)

To explore the interactive figures, use the [Quarto website](https://labjacquespe.github.io/EpiClass/epiclass-paper/index.html). This website is generated from an alternative version of the Python notebooks used to create the figures (notebooks are at `src/python/epiclass/utils/notebooks/paper/paper-final/fig*.ipynb`).

See [Key Scripts](#key-scripts) section for the training code.

## Model Availability – Neural Networks Trained on EpiATLAS

Models trained on the EpiATLAS dataset and used for inference on other datasets (as part of the associated publication) are available on Hugging Face under the ["EpiClass models" collection](https://huggingface.co/collections/KatLeChat/epiclass-models-68adb5ce65c8f2fb93322e59).

## Setup

The code has been tested with Python 3.11 and 3.12. Python 3.10 is no longer supported following the upgrade to scikit-learn >=1.8.0.

### Installation for training

To install the environment for training:

1. Clone this repository.
2. Create and activate a virtual environment.
3. From the Python code root (where `pyproject.toml` is located), run:

```bash
python install.py # torch cpu install by default, will detect nvidia gpu if available

# or run the default installation command directly, which might install nvidia packages even on cpu-only systems
pip install -e . # you can also use 'uv'
```

This installs the base requirements needed for training models, and EpiClass (`.`) as an editable package. (`-e`) It is suggested to install the package in editable mode to facilitate personal modifications.

### Installation for analysis notebooks or running tests

```bash
pip install -e .[extra_name] # adds requirements notebooks and utility scripts
```

The available `extra_name` options are:

- `utils`: for utility scripts and notebooks (`src/python/epiclass/utils/`)
- `test`: for running tests (includes `utils`)
- `dev`: for development tools (includes all of the above)

## Quick Start & Demo

To verify your installation and see the training pipeline in action, we provide a `Makefile` for convenience.

**Prerequisite:** Ensure you have installed the test dependencies (`pip install -e .[test]`).

You can run a demonstration that trains a simple MLP classifier on sample *S. cerevisiae* (sacCer3) data. This runs on the CPU and typically completes in a few minutes:

```bash
cd src/python/tests
tar -xf fixtures.tar.xz
make demo-test
```

You should see a progress bar for two-fold cross-validation training, and multiple performance metrics on the training and validation set, once a fold has finished training.

The output will specify where the trained model and predictions are saved, allowing you to inspect the results and understand the file structure immediately.

The prediction files contain the file ID (the 'md5sum' entry from the metadata), the predicted class, and the probability scores (softmax) for each output class.

### Dependencies

The base requirements are listed in `src/python/requirements/req_core.in`. Additional dependencies (for `utils` and `test`) are defined in `pyproject.toml`.

If you encounter issues installing or running the code, try using a Python version–specific requirements file. These files are located in `src/python/requirements/` and follow the naming pattern `req_test-pyX.Y.txt`, where `X` and `Y` are the major and minor Python version numbers (for example, `req_test-py3.11.txt` for Python 3.11). Install the requirements and then install EpiClass in editable mode:

```bash
pip install -r src/python/requirements/req_test-pyX.Y.txt
pip install -e .
```

The test suite has been confirmed to pass with all of these fixed-dependency files.

## Troubleshooting

**Segmentation fault during CPU prediction or training (BLIS backend).** On HPC clusters whose PyTorch build routes CPU matrix multiplies through FlexiBLAS, a segfault inside `nn.Linear` / `addmm` can be caused by a bug in the active BLIS backend rather than by EpiClass (observed with `bliscore/0.9.0` on the Digital Research Alliance of Canada clusters: an out-of-bounds read in a single-precision Haswell `sgemm` kernel; it does not occur with BLIS 2.0 or OpenBLAS, both verified). The reliable fix is to switch the FlexiBLAS backend to OpenBLAS:

```bash
export FLEXIBLAS=openblas
```

Loading a newer BLIS module does **not** help on its own: `module load blis/2.0` leaves FlexiBLAS still loading its bundled `bliscore/0.9.0` backend (the buggy one), regardless of which `blis` module is active. To actually run a fixed BLIS, point FlexiBLAS at the library by absolute path:

```bash
module load blis/2.0
export FLEXIBLAS=$EBROOTBLIS/lib64/libblis.so
```

Check the available backends with `flexiblas list`, and confirm which library is actually loaded with `FLEXIBLAS_VERBOSE=1` (or `LD_DEBUG=libs ... | grep libblis`). Set the export before launching the prediction/training job (e.g. in your SLURM script).

## Input Format & Job Launching

- See the `input-format/` folder for examples of required input files.
- The `src/bash_utils/` folder contains SLURM-compatible job launcher templates.
- Main training scripts are in `src/python/epiclass/`.

### Key Scripts

- `epiatlas_training.py`: Performs cross-validation training and evaluation.
- `epiatlas_training_no_valid.py`: Trains the model without validation (e.g. final model for inference).
- `epiatlas_training.sh`: Job submission template supporting both training modes. Update variables as needed.
- `predict.py`: Uses a trained model to generate predictions on new data.
- `compute_shaps.py`: Computes SHAP values using a trained model and a representative background set.
- `general_training.py`: A more general training script that can be used for non-EpiATLAS datasets.

## Metadata Handling

The `Metadata` class provides a convenient API for modifying metadata during preprocessing and training.

Notable methods:

- `select_category_subsets()`
- `remove_category_subsets()`

These allow dynamic relabeling or filtering of specific categories.

**Important notes:**

- Once a label category exists, any value (including `""`, `"--"`, or `"NA"`) is interpreted as a valid label.
- If your dataset may contain inconsistent keys, use `remove_missing_labels()` on the relevant categories.

For more details, refer to the [documentation](https://labjacquespe.github.io/EpiClass/epiclass/index.html). It was automatically generated by pdoc, and so might be lacking in some aspects. Do not hesitate to open an issue with questions or suggestions.

For advanced metadata manipulation, use `pandas` directly.

## Command-Line Interfaces

### `general_training.py`

```text
usage: general_training.py [-h] [--n_fold N_FOLD] [--hl_units HL_UNITS] [--nb_layer NB_LAYER] [--min_class_size MIN_CLASS_SIZE]
                           category hyperparameters hdf5_list chromsize metadata logdir

positional arguments:
  category              The metadata category to classify (e.g. assay).
  hyperparameters       JSON file with model hyperparameters.
  hdf5_list             Text file containing absolute paths to HDF5 files.
  chromsize             Chromosome sizes file.
  metadata              Metadata JSON file.
  logdir                Output directory.

options:
  -h, --help            show this help message and exit
  --n_fold N_FOLD       Number of CV folds (default: 4).
  --hl_units HL_UNITS   Hidden layer units (default: 1000).
  --nb_layer NB_LAYER   Number of hidden layers (default: 1).
  --min_class_size MIN_CLASS_SIZE
                        Min samples per class (default: 10).
```

### `epiatlas_training.py`

```text
usage: epiatlas_training.py [-h] [--offline] [--restore]
                            category hyperparameters hdf5 chromsize metadata logdir

positional arguments:
  category         The metadata category to analyze.
  hyperparameters  JSON file containing model hyperparameters.
  hdf5             File with HDF5 paths (use absolute paths).
  chromsize        Chromosome sizes file.
  metadata         Metadata JSON file.
  logdir           Output directory.

options:
  -h, --help       Show this help message and exit.
  --offline        Use offline logging. (Note: Comet-ML offline logs cannot currently be merged.)
  --restore        Skip training; restore and reuse existing models from logdir.
```

### `epiatlas_training_no_valid.py`

```text
usage: epiatlas_training_no_valid.py [-h] [--offline] [--restore]
                                     category hyperparameters hdf5 chromsize metadata logdir

(Same arguments and options as above.)
```

### `predict.py`

The model directory should be the folder where the checkpoint `best_checkpoint.list` list is.
The last path of this file will be loaded, so make sure the path points to a model weights file (`.ckpt`) that exists.

```text
usage: predict.py [-h] [--chromsize CHROMSIZE] [--chunked] [--mmap_dir MMAP_DIR] [--hdf5_dir HDF5_DIR] [--model MODEL] [--offline] hdf5 logdir

positional arguments:
  hdf5                  For single format: file listing HDF5 paths. For chunked format: directory or file of chunk HDF5s.
  logdir                Directory for output logs.

options:
  -h, --help            show this help message and exit
  --chromsize CHROMSIZE
                        Chromosome sizes file. Required for single-sample HDF5 format.
  --chunked             Input is chunked HDF5 format (e.g. produced by convert_to_chunked.py). If not set, single-sample HDF5 format is assumed.
  --mmap_dir MMAP_DIR   Directory for the mmap cache (single format only). Defaults to ./mmap_cache. On HPC, set to $SLURM_TMPDIR.
  --hdf5_dir HDF5_DIR   Override HDF5 file paths to this directory (single format). Useful when HDF5s are copied to $SLURM_TMPDIR.
  --model MODEL         Directory from which to load the model. Defaults to logdir.
  --offline             Log offline instead of online.
```

### `compute_shaps.py`

```text
usage: compute_shaps.py [-h] --background_hdf5 background-hdf5 --explain_hdf5 explain-hdf5 --chromsize CHROMSIZE --model_dir MODEL_DIR [-l LOGDIR] [-o --output-name]

Compute SHAP values for a trained neural network. Requires the `shap` package: pip install .[shap]

options:
  -h, --help            show this help message and exit
  --background_hdf5 background-hdf5
                        A file with hdf5 filenames for the explainer background. Use absolute path!
  --explain_hdf5 explain-hdf5
                        A file with hdf5 filenames on which to compute SHAP values. Use absolute path!
  --chromsize CHROMSIZE
                        A file with chrom sizes.
  --model_dir MODEL_DIR
                        Model directory containing 'best_checkpoint.list'.
  -l LOGDIR, --logdir LOGDIR
                        Directory for the output logs.
  -o --output-name, --output_name --output-name
                        Name (not path) of outputted pickle file containing computed SHAP values
```

## Tests

All tests are expected to pass on tagged releases since v0.3.0, provided all requirements are installed (`pip install -e .[test]`).

First, uncompress the test fixtures (only needed once):

```bash
cd src/python/tests && tar -xf fixtures.tar.zstd
```

Then run the full suite in parallel from `src/python/`:

```bash
pytest tests # you can add `-n auto` if pytest-xdist is installed
```

Known exceptions:

- **Skipped (GPU required)**: one test only runs when a CUDA-capable GPU is available.
- **Skipped (visual)**: confusion matrix graph tests require manual visual inspection.
- **Not yet implemented**: fold splitting for track-type classification (e.g. raw/pval/fold-change) is not yet correct.

## License

This work is licensed under the GNU General Public License v3.0 (GPLv3)
