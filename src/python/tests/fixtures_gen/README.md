# Fixture generators

Scripts that (re)generate the **trained model** fixtures under `fixtures/models/`
(and the paper run tree under `fixtures/run_info/`, see the last section).
They exist so a fixture model can be rebuilt from scratch instead of being an
opaque binary nobody remembers how to reproduce.

Run them from `src/python/` so `tests` is importable:

```bash
cd src/python
python -m tests.fixtures_gen.make_saccer3_mlp_fixture   # fixtures/models/saccer3_mlp/
python -m tests.fixtures_gen.make_saccer3_ave_fixture   # fixtures/models/saccer3_ave/
```

Each script trains a short CPU cross-validation run on the saccer3 fixture data,
then installs fold 0's checkpoint into the fixture directory together with a
`best_checkpoint_template.list`. That template holds `THIS_FOLDER` placeholders;
`pytest_sessionstart` expands them into the absolute `best_checkpoint.list` the
restore code reads, which is why the absolute list is never committed.

Checkpoints are filed under `<logger>/<32-hex run id>/checkpoints/`. That shape is
what `experiment_id_from_checkpoint` (`core/prediction_files.py`) parses to tag
prediction CSVs with their training provenance, and the predict tests assert on it.
The ids are pinned constants in each script so re-running does not churn the paths.

`fixtures/` is gitignored — only `fixtures.tar.zstd` is committed. After regenerating,
repack:

```bash
cd src/python/tests
bash pack_fixtures.sh
bash pack_fixtures.sh --check
```

## Paper run tree (`fixtures/run_info/`)

`make_run_info_fixture.py` rebuilds the fixture of `tests/utils/notebooks/paper/run_info_test.py` from the paper's real `training_results/dfreeze_v2` folder, which only exists on the machines that hold the paper data:

```bash
cd src/python
python -m tests.fixtures_gen.make_run_info_fixture <path/to/training_results/dfreeze_v2>
```

It copies the run folder tree with only the SLURM logs, and in them only the lines `run_info` parses (experiment key, model input layer, checkpoint paths, SLURM job id); the cluster path prefix is replaced by `/CLUSTER_LOGDIR/`. Folder symlinks are kept as relative symlinks. It also writes `expected_*.tsv`, the values `run_info` gives on the real tree, and refuses to write the fixture if the reduced logs give different values.

Regenerate it when the paper runs change, or when a `run_info` change is meant to change its results: the `git diff` of the `expected_*.tsv` files (after unpacking) then shows which runs are affected. Repack afterwards, as above.
