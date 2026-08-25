# Fixture generators

Scripts that (re)generate the **trained model** fixtures under `fixtures/models/`.
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
