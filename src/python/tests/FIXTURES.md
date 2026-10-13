# Test fixtures

`fixtures/` is gitignored; only `fixtures.tar.zstd` (with its `fixtures.tar.index` listing) is committed. Unpack and repack instructions are in the repository's `CLAUDE.md` ("Running Tests"). Trained-model fixtures under `fixtures/models/` and how to regenerate them are described in `fixtures_gen/README.md`.

## epigeec single-sample HDF5 layouts

The single-sample HDF5 fixtures were written by epigeec, and they cover the two attribute layouts epigeec has used. Code that reads, copies or writes such files (e.g. `utils/preprocessing/sum_stranded_rna_hdf5.py`) must handle both. Fixture paths are relative to `fixtures/`.

| | older layout | newer layout |
| --- | --- | --- |
| fixture | `saccer3/hdf5/saccer3_2016-07` (1055 files, sacCer3, 10 kb) | `89a0dcb635f0e9740f587931437b69f1_100kb_all_none_value.hdf5` (hg38, 100 kb) |
| files created | 2018-01 | 2020-12 |
| root attributes | `bin`, `chrom_sizes_hash`, `hash` | `bin`, `chrom_sizes_filename`, `signal_filename` |
| `sumX` / `sumXX` | float32 | float64 |

The `2016-07` in the saccer3 directory name is the data release, not when the files were written. epigeec does not record its own version in the files; the creation dates above are the object creation times HDF5 stores in each file:

```python
h5py.h5o.get_info(dataset.id).ctime  # seconds since epoch
```

Common to both layouts:

- One top-level group, named after the file's ID; the ID attribute (`hash` or `signal_filename`) holds the same value. Group attributes: none.
- One dataset per chromosome: float32, contiguous (no chunking, no compression).
- Per-chromosome `sumX` (sum of values) and `sumXX` (sum of squared values), shape `(1,)`. `epigeec correlate` uses them to speed up large correlation matrices, so they must describe the dataset they are attached to.
- Root `bin`: int32, shape `(1,)` — the resolution in bp, read by `EpiDataSource.get_file_hdf5_resolution`.
- String attributes are fixed-length ASCII, NUL-terminated, sized to the value + 1 byte. h5py's default for `bytes` values is NUL-padded with no terminator byte, so writing a string with plain `attrs[name] = value` silently changes its HDF5 type; write it with an explicit type instead.
