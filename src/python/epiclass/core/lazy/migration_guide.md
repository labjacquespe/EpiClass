# Migration Guide: From Eager to Lazy Loading

## Overview

This guide covers transitioning from the old eager-loading codebase (all HDF5 files loaded into RAM) to the new lazy-loading architecture, which supports two input formats:

- **Single-file format**: One HDF5 file per sample, with per-chromosome datasets (the existing EpiATLAS format). Handled by `LazyHdf5Loader`, which converts to a single memory-mapped `.npy` file via `preload_all()`.
- **Chunked format**: Multiple pre-concatenated samples per HDF5 file. Handled by `ChunkedHdf5Loader`, which reads directly from the HDF5 files with persistent file handles.

Both loaders expose the same interface (`load_signal(sample_id) → np.ndarray`), so everything downstream — `LazyKnownData`, `LazyHdf5Dataset`, DataLoaders, training loops — is format-agnostic.

## Architecture

```
Training script
    │
    ├─ Single-file format ──→ LazyHdf5Loader
    │                         ├── register_hdf5s()
    │                         ├── preload_all()  →  single .npy mmap
    │                         └── load_signal(sample_id)
    │
    └─ Chunked format ──────→ ChunkedHdf5Loader
                              ├── register_chunked_hdf5s()
                              └── load_signal(sample_id)  →  direct HDF5 read
                    │
                    v
             LazyKnownData (format-agnostic)
                    │
                    v
             LazyHdf5Dataset (PyTorch Dataset)
                    │
                    v
             DataLoader → Training
```

The caller picks the loader. No auto-detection, no hybrid wrapper.

## New Modules

| Module | Purpose |
|--------|---------|
| `lazy_hdf5_loader.py` | Single-file lazy loader with mmap conversion |
| `chunked_hdf5_loader.py` | Chunked HDF5 loader with direct reads |
| `lazy_data_classes.py` | `LazyKnownData`, `LazyUnknownData` — loader-backed data containers |
| `lazy_epidata.py` | `DataSetFactory` for single-file format |
| `lazy_torch_dataset.py` | `LazyHdf5Dataset` and `create_lazy_dataloaders` |
| `lazy_fold_factory.py` | `LazyEpiAtlasFoldFactory` for cross-validation |
| `convert_to_chunked.py` | Standalone script to convert single-file → chunked format |

## Step-by-Step Migration (Single-File Format)

### 1. Replace Hdf5Loader with LazyHdf5Loader

**Before:**
```python
from hdf5_loader import Hdf5Loader

loader = Hdf5Loader(chrom_file, normalization=True)
loader.load_hdf5s(data_file, md5s=md5_list, strict=True)
all_signals = loader.signals  # Everything in RAM
```

**After:**
```python
from lazy_hdf5_loader import LazyHdf5Loader

loader = LazyHdf5Loader(
    chrom_file,
    normalization=True,
    mmap_dir="./mmap_cache",  # Directory for the .npy mmap file
)
loader.register_hdf5s(data_file, md5s=md5_list, strict=True)
loader.preload_all()  # Converts all HDF5s into a single mmap file
# Access signals via loader.load_signal(sample_id)
```

`preload_all()` reads every registered HDF5, concatenates chromosomes, normalizes (if enabled), and writes the result into a single memory-mapped `.npy` file. Subsequent `load_signal()` calls read from this mmap. The OS page cache handles the rest — if the dataset fits in RAM, it stays cached automatically; if not, pages are evicted and re-read as needed.

On HPC, set `mmap_dir` to `$SLURM_TMPDIR` for fast local storage. `preload_all()` checks available disk space before writing and skips if the mmap file already exists, so re-running is safe.

### 2. Replace Data Classes

**Before:**
```python
from data import KnownData

data = KnownData(
    ids=md5s,
    x=loaded_signals,  # numpy array with all signals
    y=encoded_labels,
    y_str=original_labels,
    metadata=metadata
)
signal = data.get_signal(0)
```

**After:**
```python
from lazy_data_classes import LazyKnownData

data = LazyKnownData(
    ids=sample_ids,
    loader=loader,       # Any loader with load_signal(sample_id)
    y=encoded_labels,
    y_str=original_labels,
    metadata=metadata
)
signal, label, label_str = data[0]  # Loads from mmap via loader
```

### 3. Replace DataSet Creation

**Before:**
```python
from data import DataSetFactory

dataset = DataSetFactory.from_epidata(
    datasource=datasource,
    metadata=metadata,
    label_category="assay",
    oversample=True,
    normalization=True,
)
# All signals now in RAM
```

**After:**
```python
from lazy_epidata import DataSetFactory

dataset = DataSetFactory.from_epidata(
    datasource=datasource,
    metadata=metadata,
    label_category="assay",
    oversample=True,
    normalization=True,
)
# Only metadata in RAM; signals loaded on-demand
```

### 4. Replace PyTorch Dataset Creation

**Before:**
```python
from torch.utils.data import TensorDataset, DataLoader

train_dset = TensorDataset(
    torch.from_numpy(dataset.train.signals).float(),
    torch.from_numpy(dataset.train.encoded_labels),
)
train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
```

**After:**
```python
from lazy_torch_dataset import LazyHdf5Dataset, create_lazy_dataloaders

# Option 1: Direct creation
train_dataset = LazyHdf5Dataset(dataset.train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

# Option 2: Helper function (recommended)
dataloaders = create_lazy_dataloaders(
    train_data=dataset.train,
    val_data=dataset.validation,
    test_data=dataset.test,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
train_loader = dataloaders["training"][1]
```

### 5. Replace Cross-Validation

**Before:**
```python
from fold_creation import EpiAtlasFoldFactory

fold_factory = EpiAtlasFoldFactory.from_datasource(
    datasource=datasource,
    label_category="assay",
)
for fold_dataset in fold_factory.yield_split(oversample=True):
    # fold_dataset.train.signals already in memory
    pass
```

**After:**
```python
from lazy_fold_factory import LazyEpiAtlasFoldFactory

fold_factory = LazyEpiAtlasFoldFactory.from_datasource(
    datasource=datasource,
    label_category="assay",
)
for fold_dataset in fold_factory.yield_split(oversample=True):
    dataloaders = create_lazy_dataloaders(
        train_data=fold_dataset.train,
        val_data=fold_dataset.validation,
        batch_size=32,
        num_workers=4,
    )
```

## Using the Chunked Format

If your data is already in chunked HDF5 files (or you've converted it), use `ChunkedHdf5Loader` instead. The downstream code is identical.

### Chunked HDF5 Structure

Each chunk file contains:
```
chunk_0000.h5
├── signals:    shape (n_samples, signal_length), dtype float32
├── sample_ids: shape (n_samples,), variable-length str
└── attrs (signal_length, normalized, source_chrom_file)
```

### Using ChunkedHdf5Loader

```python
from chunked_hdf5_loader import ChunkedHdf5Loader

# Register chunk files (no data loaded)
loader = ChunkedHdf5Loader()
loader.register_chunked_hdf5s(
    Path("data/chunks"),   # Directory, single file, or list of files
    strict=True,
)

# Load signals — reads directly from HDF5 with persistent file handles
signal = loader.load_signal("sample_0001")

# Batch loading groups by file and uses sorted index access
batch = loader.load_batch(["sample_0001", "sample_0002", "sample_0003"])
```

No `preload_all()` needed. The chunked files already contain pre-concatenated, pre-normalized signals. The OS page cache handles repeat access across epochs.

### Plugging into the Pipeline

Both loaders work interchangeably with `LazyKnownData`:

```python
from lazy_data_classes import LazyKnownData

# Works with either loader
data = LazyKnownData(
    ids=sample_ids,
    loader=loader,  # LazyHdf5Loader or ChunkedHdf5Loader
    y=encoded_labels,
    y_str=original_labels,
    metadata=metadata,
)

# Everything downstream is the same
train_dataset = LazyHdf5Dataset(data)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, ...)
```

### Converting Single-File to Chunked

Use the standalone conversion script:

```bash
# Basic conversion (10000 samples per chunk)
python convert_to_chunked.py convert \
    --hdf5-list file_list.txt \
    --chrom-file chroms.txt \
    --output-dir data/chunks/ \
    --samples-per-chunk 10000

# With normalization
python convert_to_chunked.py convert \
    --hdf5-list file_list.txt \
    --chrom-file chroms.txt \
    --output-dir data/chunks/ \
    --samples-per-chunk 10000 \
    --normalize

# Dry run to check sizes first
python convert_to_chunked.py convert \
    --hdf5-list file_list.txt \
    --chrom-file chroms.txt \
    --output-dir data/chunks/ \
    --dry-run

# Read from $SLURM_TMPDIR instead of original paths
python convert_to_chunked.py convert \
    --hdf5-list file_list.txt \
    --chrom-file chroms.txt \
    --output-dir data/chunks/ \
    --hdf5-dir "$SLURM_TMPDIR/hdf5s"

# Convert only a subset of samples
python convert_to_chunked.py convert \
    --hdf5-list file_list.txt \
    --chrom-file chroms.txt \
    --output-dir data/chunks/ \
    --sample-ids-file wanted_ids.txt

# Verify chunk integrity after conversion
python convert_to_chunked.py verify data/chunks/
python convert_to_chunked.py verify data/chunks/ --expected-samples 50000
```

The script is standalone (no `epiclass` dependency, just `h5py` and `numpy`) so it can run in minimal HPC batch environments. It skips already-written chunks, so interrupted conversions can be resumed.

## Complete Example: Training Loop

### Before (Eager Loading)
```python
from data_source import EpiDataSource
from metadata import Metadata
from data import DataSetFactory
from torch.utils.data import TensorDataset, DataLoader

datasource = EpiDataSource(...)
metadata = Metadata(datasource.metadata_file)
dataset = DataSetFactory.from_epidata(
    datasource, metadata, label_category="assay"
)

train_dset = TensorDataset(
    torch.from_numpy(dataset.train.signals).float(),
    torch.from_numpy(dataset.train.encoded_labels),
)
train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    for batch_signals, batch_labels in train_loader:
        outputs = model(batch_signals.to(device))
        # ...
```

### After: Single-File Format
```python
from data_source import EpiDataSource
from metadata import Metadata
from lazy_epidata import DataSetFactory
from lazy_torch_dataset import create_lazy_dataloaders

datasource = EpiDataSource(...)
metadata = Metadata(datasource.metadata_file)
dataset = DataSetFactory.from_epidata(
    datasource, metadata, label_category="assay"
)

dataloaders = create_lazy_dataloaders(
    train_data=dataset.train,
    val_data=dataset.validation,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
train_loader = dataloaders["training"][1]

for epoch in range(num_epochs):
    for batch_signals, batch_labels in train_loader:
        outputs = model(batch_signals.to(device))
        # ... training code unchanged
```

### After: Chunked Format
```python
from chunked_hdf5_loader import ChunkedHdf5Loader
from lazy_data_classes import LazyKnownData
from lazy_torch_dataset import create_lazy_dataloaders

loader = ChunkedHdf5Loader()
loader.register_chunked_hdf5s(Path("data/chunks"), strict=True)

# Build LazyKnownData from your metadata + the loader
# (or use a factory that does this for you)
train_data = LazyKnownData(
    ids=train_ids,
    loader=loader,
    y=train_labels,
    y_str=train_label_strs,
    metadata=train_metadata,
)

dataloaders = create_lazy_dataloaders(
    train_data=train_data,
    val_data=val_data,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
train_loader = dataloaders["training"][1]

for epoch in range(num_epochs):
    for batch_signals, batch_labels in train_loader:
        outputs = model(batch_signals.to(device))
        # ... training code unchanged
```

## Memory and Performance

### Memory Usage Comparison

| Phase | Eager | Single-file lazy | Chunked lazy |
|-------|-------|------------------|--------------|
| Initialization | Loads all data | Metadata only | Metadata only |
| After preload | N/A | Mmap file on disk; OS pages in/out | N/A |
| Training | Dataset + batch buffers | Workers × batch + OS page cache | Workers × batch + OS page cache |
| Total RAM floor | Full dataset | Batch size × num_workers | Batch size × num_workers |

For datasets that fit in RAM, both lazy approaches reach eager-loading speed after the first epoch — the OS page cache keeps the mmap / HDF5 data in memory. For datasets larger than RAM, pages are evicted and re-read as needed, with no OOM risk.

### DataLoader Tuning

These settings apply to both formats:

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,            # Async I/O workers
    pin_memory=True,          # Faster GPU transfer
    prefetch_factor=2,        # Batches prefetched per worker
    persistent_workers=True,  # Keep workers alive between epochs
)
```

`persistent_workers=True` is particularly important for the chunked format — each worker keeps its HDF5 file handles open across batches, avoiding repeated open/close overhead.

For worker count, start with 4 and increase if GPU utilization is low. On Narval, the number of available cores depends on your SLURM allocation.

### HPC Considerations

For `LazyHdf5Loader`, set `mmap_dir` to `$SLURM_TMPDIR` so the `.npy` file is written to local NVMe. `preload_all()` checks disk space and skips if the file already exists.

For `ChunkedHdf5Loader`, copy your chunk files to `$SLURM_TMPDIR` before training for fastest reads. The `convert_to_chunked.py` script supports `--hdf5-dir` to read source files from `$SLURM_TMPDIR` during conversion.

## Troubleshooting

**Training is too slow** — Increase `num_workers` and `prefetch_factor`. Check that data is on fast local storage (`$SLURM_TMPDIR`), not Lustre. For single-file format, ensure `preload_all()` completed successfully.

**Out of memory** — Decrease `batch_size` or `num_workers`. Each worker holds its own batch in memory.

**DataLoader hangs** — Set `num_workers=0` to debug without multiprocessing. This eliminates worker-related issues and runs everything in the main process.

**HDF5 file access errors** — Ensure files are on local storage. Lustre + HDF5 can have locking issues with multiple readers. Copy to `$SLURM_TMPDIR` first.

**Mmap disk space error** — `preload_all()` checks available disk before writing. Either free space on `$SLURM_TMPDIR` or request a larger allocation.

## Summary of Changes

| Component | Old | New (single-file) | New (chunked) |
|-----------|-----|-------------------|---------------|
| Loader | `Hdf5Loader` | `LazyHdf5Loader` | `ChunkedHdf5Loader` |
| Registration | `.load_hdf5s()` | `.register_hdf5s()` + `.preload_all()` | `.register_chunked_hdf5s()` |
| Signal access | `loader.signals[md5]` | `loader.load_signal(sample_id)` | `loader.load_signal(sample_id)` |
| Data class | `KnownData(ids, x, y, ...)` | `LazyKnownData(ids, loader, y, ...)` | `LazyKnownData(ids, loader, y, ...)` |
| PyTorch Dataset | `TensorDataset` | `LazyHdf5Dataset` | `LazyHdf5Dataset` |
| Factory | `DataSetFactory` (eager) | `DataSetFactory` (lazy) | Direct construction |
| Folds | `EpiAtlasFoldFactory` | `LazyEpiAtlasFoldFactory` | Direct construction |
| Storage | All in RAM | Single `.npy` mmap + OS page cache | Direct HDF5 reads + OS page cache |
